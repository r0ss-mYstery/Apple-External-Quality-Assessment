# local_size_track_select_hough.py
# ------------------------------------------------------------
# Revised from the RANSAC workflow to test HOUGH CIRCLES.
# Keeps the same overall pipeline:
#   - YOLO segmentation (Ultralytics) on ZED frames
#   - Stable fruit_id tracking across frames (3D + 2D IoU, Hungarian)
#   - Hough-circle diameter estimation from mask-boundary ROI
#   - Per-frame quality Q
#   - Best frame selection per fruit + median over top-K frames
#   - Debug frames saved (optional)
#
# Requirements:
#   pip install ultralytics opencv-python numpy scipy
#   ZED SDK + pyzed installed
# ------------------------------------------------------------

from __future__ import annotations
import os
import time
import csv
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
from ultralytics import YOLO
import pyzed.sl as sl

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# CONFIG
# -----------------------------
MODE = "SVO"
SVO_PATH = r"C:/Users/kmh7116/PhD_Projects/Fruit_dect/Data_raw_image/size_est/ZED/HD1080_SN58421738_15-53-35.svo2"
# SVO_PATH = r"C:/Users/kmh7116/PhD_Projects/Fruit_dect/Data_raw_image/Localfruit/Fuji/ZED/HD1080_SN58421738_19-40-35.svo2"
YOLO_WEIGHTS = r"C:/Users/kmh7116/PhD_Projects/Fruit_dect/yolov12/runs/segment/train2/weights/best.pt"

# Use a separate output directory so it does not overwrite RANSAC results.
OUTPUT_DIR = "./local_size_outputs_track_select_hough14"
PER_FRAME_CSV = os.path.join(OUTPUT_DIR, "fruit_size_per_frame.csv")
BEST_FRAME_CSV = os.path.join(OUTPUT_DIR, "fruit_best_frame.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "run_summary.json")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_frames")

# Inference
CONF_THRES = 0.25
IMG_SZ = 960
NMS_IOU = 0.7
FRAME_STRIDE = 10

# Mask
MASK_THRESH = 0.40
MASK_ALPHA = 0.35
MASK_ERODE_PX = 0
MIN_MASK_AREA_PX = 150

# Depth
PIXEL_SAMPLE_STEP = 2
MIN_VALID_POINTS = 30
MAX_DEPTH_M = 15.0

# ZED
PREFER_NEURAL = True
USE_BGRA = True

# Display / Debug
SHOW_WINDOW = True
WINDOW_NAME = "Local Size (Track + Select) - Hough"
ROTATE_DISPLAY_CCW_90 = True
SAVE_DEBUG_FRAMES = True
SAVE_EVERY_N = 10

# Optional class filter
APPLE_CLASS_ID = None

# -----------------------------
# Hough circle parameters
# -----------------------------
# Main tuning knobs:
#   - If Hough misses circles: lower HOUGH_PARAM2 (e.g., 18 -> 12)
#   - If Hough finds too many wrong circles: increase HOUGH_PARAM2 (e.g., 18 -> 24)
#   - Radius bounds are derived from the mask bbox for each fruit.
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 18
HOUGH_EDGE_LO = 50
HOUGH_EDGE_HI = 150
HOUGH_BLUR_K = 5
HOUGH_MIN_R_FRAC = 0.25   # minRadius = frac * min(ROI width, ROI height)
HOUGH_MAX_R_FRAC = 0.70   # maxRadius = frac * min(ROI width, ROI height)
HOUGH_TARGET_R_FRAC = 0.45

# Selection
TOPK = 5
MIN_Q_KEEP = 0.15
MIN_DEPTH_VALID_RATIO = 0.25

# Tracking (3D+2D)
TRACK_ENABLE = True
TRACK_GATE3D_M = 0.08
TRACK_MIN_IOU = 0.10
TRACK_MAX_MISSES = 12
TRACK_SMOOTH = 0.75
TRACK_W3D = 1.0
TRACK_W2D = 0.4

# Center estimation (more stable than full-mask median)
CENTROID_WIN = 7
CENTROID_MIN_PTS = 10


# -----------------------------
# Utility
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -----------------------------
# ZED helpers
# -----------------------------
def open_zed() -> sl.Camera:
    zed = sl.Camera()
    init = sl.InitParameters()
    if MODE.upper() == "SVO":
        init.set_from_svo_file(SVO_PATH)
        init.svo_real_time_mode = False

    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    if PREFER_NEURAL and hasattr(sl.DEPTH_MODE, "NEURAL"):
        init.depth_mode = sl.DEPTH_MODE.NEURAL
    elif hasattr(sl.DEPTH_MODE, "QUALITY"):
        init.depth_mode = sl.DEPTH_MODE.QUALITY
    else:
        init.depth_mode = sl.DEPTH_MODE.ULTRA

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {err}")
    return zed


def zed_4c_to_bgr_no_rotate(img_4c: np.ndarray) -> np.ndarray:
    if USE_BGRA:
        return cv2.cvtColor(img_4c, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img_4c, cv2.COLOR_RGBA2BGR)


def get_fx_fy(zed: sl.Camera) -> Tuple[float, float]:
    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters
    return float(calib.left_cam.fx), float(calib.left_cam.fy)


# -----------------------------
# Depth helpers
# -----------------------------
def median_depth_from_mask_absZ(point_cloud: sl.Mat, mask_bool: np.ndarray,
                                sample_step: int, min_pts: int) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None

    step = max(1, int(sample_step))
    ys = ys[::step]
    xs = xs[::step]

    zs = []
    total = len(xs)
    for y, x in zip(ys, xs):
        err, p = point_cloud.get_value(int(x), int(y))
        if err != sl.ERROR_CODE.SUCCESS:
            continue
        Z = float(p[2])
        if not np.isfinite(Z):
            continue
        z_abs = abs(Z)
        if z_abs < 1e-3 or z_abs > MAX_DEPTH_M:
            continue
        zs.append(z_abs)

    if len(zs) < int(min_pts):
        return None

    zs = np.asarray(zs, dtype=np.float64)
    return float(np.median(zs)), float(len(zs) / max(total, 1))


def center_xyz_from_mask_centroid_window(point_cloud: sl.Mat, mask_bool: np.ndarray,
                                         win: int = 7, min_pts: int = 10) -> Optional[np.ndarray]:
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    cy = int(np.median(ys))
    cx = int(np.median(xs))

    half = max(1, int(win))
    pts = []
    for y in range(cy - half, cy + half + 1):
        for x in range(cx - half, cx + half + 1):
            err, p = point_cloud.get_value(int(x), int(y))
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            X, Y, Z, _ = p
            if not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z):
                continue
            z_abs = abs(float(Z))
            if z_abs < 1e-3 or z_abs > MAX_DEPTH_M:
                continue
            pts.append((float(X), float(Y), float(Z)))

    if len(pts) < min_pts:
        return None
    return np.median(np.asarray(pts, dtype=np.float64), axis=0)


# -----------------------------
# Tracking (3D + IoU, Hungarian)
# -----------------------------
@dataclass
class Track:
    tid: int
    center_xyz: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    last_t: float
    hits: int = 0
    misses: int = 0


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
    area_b = max(1, (bx2 - bx1)) * max(1, (by2 - by1))
    return float(inter / (area_a + area_b - inter))


class MultiFruitTracker:
    def __init__(self,
                 gate3d_m: float,
                 min_iou: float,
                 max_misses: int,
                 smooth: float,
                 w3d: float,
                 w2d: float):
        self.gate3d = float(gate3d_m)
        self.min_iou = float(min_iou)
        self.max_misses = int(max_misses)
        self.smooth = float(smooth)
        self.w3d = float(w3d)
        self.w2d = float(w2d)

        self.tracks: List[Track] = []
        self.next_id = 1

    def _cleanup(self):
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

    def update(self, detections: List[Dict[str, Any]], now: float) -> List[int]:
        self._cleanup()

        D = len(detections)
        if D == 0:
            for tr in self.tracks:
                tr.misses += 1
            self._cleanup()
            return []

        if len(self.tracks) == 0:
            ids = []
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append(Track(tid=tid,
                                         center_xyz=det["center_xyz"].copy(),
                                         bbox_xyxy=det["bbox_xyxy"],
                                         last_t=now, hits=1, misses=0))
                ids.append(tid)
            return ids

        T = len(self.tracks)
        cost = np.full((T, D), 1e6, dtype=np.float64)

        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                d3d = float(np.linalg.norm(tr.center_xyz - det["center_xyz"]))
                iou = bbox_iou(tr.bbox_xyxy, det["bbox_xyxy"])
                if d3d > self.gate3d:
                    continue
                if iou < self.min_iou:
                    continue
                cost[ti, di] = self.w3d * d3d + self.w2d * (1.0 - iou)

        ids = [-1] * D
        assigned_tracks = set()
        assigned_dets = set()

        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e5:
                    continue
                assigned_tracks.add(r)
                assigned_dets.add(c)
                ids[c] = self.tracks[r].tid

                new_c = detections[c]["center_xyz"]
                self.tracks[r].center_xyz = self.smooth * self.tracks[r].center_xyz + (1.0 - self.smooth) * new_c
                self.tracks[r].bbox_xyxy = detections[c]["bbox_xyxy"]
                self.tracks[r].last_t = now
                self.tracks[r].hits += 1
                self.tracks[r].misses = 0
        else:
            used_tracks = set()
            for di, det in enumerate(detections):
                best_t = None
                best_c = 1e6
                for ti, tr in enumerate(self.tracks):
                    if ti in used_tracks:
                        continue
                    d3d = float(np.linalg.norm(tr.center_xyz - det["center_xyz"]))
                    iou = bbox_iou(tr.bbox_xyxy, det["bbox_xyxy"])
                    if d3d > self.gate3d or iou < self.min_iou:
                        continue
                    cst = self.w3d * d3d + self.w2d * (1.0 - iou)
                    if cst < best_c:
                        best_c = cst
                        best_t = ti
                if best_t is not None:
                    used_tracks.add(best_t)
                    ids[di] = self.tracks[best_t].tid
                    self.tracks[best_t].center_xyz = self.smooth * self.tracks[best_t].center_xyz + (1.0 - self.smooth) * det["center_xyz"]
                    self.tracks[best_t].bbox_xyxy = det["bbox_xyxy"]
                    self.tracks[best_t].last_t = now
                    self.tracks[best_t].hits += 1
                    self.tracks[best_t].misses = 0
                    assigned_tracks.add(best_t)
                    assigned_dets.add(di)

        for ti, tr in enumerate(self.tracks):
            if ti not in assigned_tracks:
                tr.misses += 1

        for di in range(D):
            if di not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append(Track(tid=tid,
                                         center_xyz=detections[di]["center_xyz"].copy(),
                                         bbox_xyxy=detections[di]["bbox_xyxy"],
                                         last_t=now, hits=1, misses=0))
                ids[di] = tid

        self._cleanup()
        return ids


# -----------------------------
# Hough circle diameter estimation
# -----------------------------
def hough_circle_from_mask_roi(mask_u8_255: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> Optional[Tuple[float, float, float, float]]:
    """
    Run HoughCircles on a Canny edge map created from the binary mask boundary.

    Returns:
        (cx_full, cy_full, r_px, radius_penalty)
        where center/radius are in full-image coordinates.
    """
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(x1 + 2, x2)
    y2 = max(y1 + 2, y2)

    roi = mask_u8_255[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # Build a boundary edge map from the mask, not from RGB, to reduce clutter from leaves/branches.
    edge = cv2.Canny(roi, HOUGH_EDGE_LO, HOUGH_EDGE_HI)
    k = int(HOUGH_BLUR_K)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    edge = cv2.GaussianBlur(edge, (k, k), 0)

    h, w = roi.shape[:2]
    min_dim = max(1, min(h, w))
    r_min = int(max(5, HOUGH_MIN_R_FRAC * min_dim))
    r_max = int(max(r_min + 2, HOUGH_MAX_R_FRAC * min_dim))

    circles = cv2.HoughCircles(
        edge,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=r_min,
        maxRadius=r_max,
    )
    if circles is None:
        return None

    circles = circles[0]
    target_r = HOUGH_TARGET_R_FRAC * min_dim

    # Choose the candidate whose radius is closest to the expected mask/bbox scale.
    # This is simple and keeps Hough-only testing deterministic.
    penalties = np.abs(circles[:, 2] - target_r) / max(target_r, 1e-6)
    j = int(np.argmin(penalties))
    cx, cy, r = circles[j]
    penalty = float(np.clip(penalties[j], 0.0, 1.0))
    return float(cx + x1), float(cy + y1), float(r), penalty


# -----------------------------
# Quality score Q for Hough
# -----------------------------
def compute_quality_hough(score_yolo: float,
                          depth_valid_ratio: float,
                          hough_ok: int,
                          radius_penalty: float) -> float:
    """
    Hough does not provide RANSAC-style inlier ratio/arc coverage.
    Q therefore combines:
      - YOLO confidence
      - depth validity
      - whether a circle was found
      - radius-scale sanity penalty
    """
    s = float(np.clip(score_yolo, 0.0, 1.0))
    dv = float(np.clip(depth_valid_ratio, 0.0, 1.0))
    ok = 1.0 if int(hough_ok) == 1 else 0.0
    rp = float(np.clip(radius_penalty, 0.0, 1.0))
    Q = 0.30 * s + 0.35 * dv + 0.25 * ok + 0.10 * (1.0 - rp)
    return float(np.clip(Q, 0.0, 1.0))


# -----------------------------
# Visualization
# -----------------------------
def overlay_mask(vis_bgr: np.ndarray, mask_bool: np.ndarray, alpha: float) -> np.ndarray:
    H, W = vis_bgr.shape[:2]
    if mask_bool.shape[:2] != (H, W):
        m = (mask_bool.astype(np.uint8) * 255)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = m > 0
    overlay = vis_bgr.copy()
    overlay[mask_bool] = (0, 255, 0)
    return cv2.addWeighted(overlay, alpha, vis_bgr, 1.0 - alpha, 0)


def draw_box_label(vis: np.ndarray, bbox_xyxy: Tuple[int, int, int, int], text: str) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(vis, text, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(DEBUG_DIR)

    model = YOLO(YOLO_WEIGHTS)
    print("model.task:", getattr(model, "task", None))
    print("model.names:", getattr(model, "names", None))

    zed = open_zed()
    fx, fy = get_fx_fy(zed)
    print("fx, fy:", fx, fy)

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    point_cloud = sl.Mat()

    tracker = None
    if TRACK_ENABLE:
        tracker = MultiFruitTracker(
            gate3d_m=TRACK_GATE3D_M,
            min_iou=TRACK_MIN_IOU,
            max_misses=TRACK_MAX_MISSES,
            smooth=TRACK_SMOOTH,
            w3d=TRACK_W3D,
            w2d=TRACK_W2D,
        )
        if not _HAS_SCIPY:
            print("[WARN] SciPy not available; using greedy one-to-one assignment fallback. Install scipy for best tracking.")

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fieldnames = [
        "timestamp", "frame_idx",
        "fruit_id", "inst_id",
        "class_id", "score_yolo",
        "hough_ok",
        "hough_cx", "hough_cy", "hough_r_px", "hough_radius_penalty",
        "depth_median_m", "depth_valid_ratio",
        "D_hough_mm",
        "Q",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
    ]

    per_fruit_rows: Dict[int, List[Dict[str, Any]]] = {}

    with open(PER_FRAME_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        frame_idx = 0
        prev = time.time()

        while True:
            err = zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                if MODE.upper() == "SVO" and err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    print("[INFO] End of SVO reached.")
                else:
                    print("[WARN] grab error:", err)
                break

            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            img_bgr = zed_4c_to_bgr_no_rotate(image.get_data())
            vis = img_bgr.copy()
            now = time.time()

            # Ultralytics: NumPy input -> pass BGR
            r = model.predict(source=img_bgr, conf=CONF_THRES, iou=NMS_IOU, imgsz=IMG_SZ, verbose=False)[0]
            n_boxes = 0 if r.boxes is None else len(r.boxes)

            wrote = 0

            if r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
                masks = (r.masks.data.cpu().numpy() > MASK_THRESH)
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)

                H, W = img_bgr.shape[:2]
                kernel = None
                if MASK_ERODE_PX > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_ERODE_PX + 1, 2 * MASK_ERODE_PX + 1))

                # 1) Build per-frame detections for tracker
                dets_for_tracker: List[Dict[str, Any]] = []
                meta_for_dets: List[Dict[str, Any]] = []

                for i in range(masks.shape[0]):
                    score_yolo = float(confs[i])
                    cid = int(clss[i])
                    if score_yolo < CONF_THRES:
                        continue
                    if APPLE_CLASS_ID is not None and cid != int(APPLE_CLASS_ID):
                        continue

                    inst_id = i

                    m = masks[i].astype(np.uint8)
                    if m.shape[:2] != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    if kernel is not None:
                        m = cv2.erode(m, kernel, iterations=1)
                    if int(m.sum()) < MIN_MASK_AREA_PX:
                        continue

                    mask_bool = m.astype(bool)

                    # contour + bbox
                    mask_u8_255 = (m * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_u8_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if not contours:
                        continue
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) < MIN_MASK_AREA_PX:
                        continue
                    x, y, bw, bh = cv2.boundingRect(c)
                    bbox = (int(x), int(y), int(x + bw), int(y + bh))

                    # stable 3D center for tracking
                    center_xyz = center_xyz_from_mask_centroid_window(point_cloud, mask_bool, win=CENTROID_WIN, min_pts=CENTROID_MIN_PTS)
                    if center_xyz is None:
                        continue

                    dets_for_tracker.append({"center_xyz": center_xyz, "bbox_xyxy": bbox})
                    meta_for_dets.append({
                        "inst_id": inst_id,
                        "cid": cid,
                        "score_yolo": score_yolo,
                        "mask_bool": mask_bool,
                        "mask_u8_255": mask_u8_255,
                        "bbox": bbox,
                    })

                # 2) Assign stable IDs (one-to-one)
                if tracker is not None:
                    fruit_ids = tracker.update(dets_for_tracker, now)
                else:
                    fruit_ids = [m["inst_id"] for m in meta_for_dets]

                # 3) Process each detection with its fruit_id using Hough
                for meta, fruit_id in zip(meta_for_dets, fruit_ids):
                    inst_id = int(meta["inst_id"])
                    cid = int(meta["cid"])
                    score_yolo = float(meta["score_yolo"])
                    mask_bool = meta["mask_bool"]
                    mask_u8_255 = meta["mask_u8_255"]
                    bbox = meta["bbox"]

                    # overlay
                    vis = overlay_mask(vis, mask_bool, MASK_ALPHA)

                    # depth stats
                    depth_out = median_depth_from_mask_absZ(point_cloud, mask_bool, PIXEL_SAMPLE_STEP, MIN_VALID_POINTS)
                    if depth_out is None:
                        continue
                    depth_med, depth_vr = depth_out

                    hough_ok = 0
                    cx = cy = r_px = ""
                    radius_penalty = 1.0
                    D_hough_mm = ""

                    circ = hough_circle_from_mask_roi(mask_u8_255, bbox)
                    if circ is not None:
                        hough_ok = 1
                        cx, cy, r_px, radius_penalty = circ
                        D_px = 2.0 * float(r_px)
                        D_hough_mm = (D_px * depth_med / fx) * 1000.0

                        # draw Hough circle
                        cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r_px)), (255, 0, 0), 2)

                    # quality score
                    Q = compute_quality_hough(score_yolo, depth_vr, hough_ok, radius_penalty)

                    label = f"ID:{fruit_id} H Q:{Q:.2f} D:{'' if D_hough_mm=='' else int(round(D_hough_mm))}"
                    draw_box_label(vis, bbox, label)

                    row = {
                        "timestamp": now,
                        "frame_idx": frame_idx,
                        "fruit_id": int(fruit_id),
                        "inst_id": inst_id,
                        "class_id": cid,
                        "score_yolo": score_yolo,
                        "hough_ok": hough_ok,
                        "hough_cx": cx,
                        "hough_cy": cy,
                        "hough_r_px": r_px,
                        "hough_radius_penalty": radius_penalty,
                        "depth_median_m": depth_med,
                        "depth_valid_ratio": depth_vr,
                        "D_hough_mm": D_hough_mm,
                        "Q": Q,
                        "bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3],
                    }
                    writer.writerow(row)
                    per_fruit_rows.setdefault(int(fruit_id), []).append(row)
                    wrote += 1

            # HUD
            fps = 1.0 / max(time.time() - prev, 1e-6)
            prev = time.time()
            cv2.putText(vis, f"frame:{frame_idx} fps:{fps:.1f} boxes:{n_boxes} wrote:{wrote}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            vis_show = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE) if ROTATE_DISPLAY_CCW_90 else vis

            if SAVE_DEBUG_FRAMES and (frame_idx % SAVE_EVERY_N == 0):
                cv2.imwrite(os.path.join(DEBUG_DIR, f"frame_{frame_idx:06d}.png"), vis_show)

            if SHOW_WINDOW:
                cv2.imshow(WINDOW_NAME, vis_show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1

    # -----------------------------
    # Best-frame selection per fruit
    # -----------------------------
    with open(BEST_FRAME_CSV, "w", newline="", encoding="utf-8") as f:
        out_fields = [
            "fruit_id",
            "n_frames_total",
            "n_frames_kept",
            "best_frame_idx",
            "D_best_mm",
            "max_Q",
            "mean_Q_kept",
            "D_median_topK_mm",
            "K_used"
        ]
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()

        for fruit_id, rows in sorted(per_fruit_rows.items(), key=lambda x: x[0]):
            kept = []
            for r in rows:
                if float(r["Q"]) < MIN_Q_KEEP:
                    continue
                if int(r["hough_ok"]) != 1:
                    continue
                if float(r["depth_valid_ratio"]) < MIN_DEPTH_VALID_RATIO:
                    continue
                if r["D_hough_mm"] == "" or r["D_hough_mm"] is None:
                    continue
                kept.append(r)

            if not kept:
                w.writerow({
                    "fruit_id": fruit_id,
                    "n_frames_total": len(rows),
                    "n_frames_kept": 0,
                    "best_frame_idx": "",
                    "D_best_mm": "",
                    "max_Q": "",
                    "mean_Q_kept": "",
                    "D_median_topK_mm": "",
                    "K_used": 0
                })
                continue

            kept_sorted = sorted(kept, key=lambda r: float(r["Q"]), reverse=True)
            best = kept_sorted[0]
            best_frame_idx = int(best["frame_idx"])
            D_best = float(best["D_hough_mm"])

            Qs = np.array([float(r["Q"]) for r in kept_sorted], dtype=np.float64)
            max_Q = float(np.max(Qs))
            mean_Q = float(np.mean(Qs))

            top = kept_sorted[:min(TOPK, len(kept_sorted))]
            Ds = np.array([float(r["D_hough_mm"]) for r in top], dtype=np.float64)
            D_med = float(np.median(Ds))

            w.writerow({
                "fruit_id": fruit_id,
                "n_frames_total": len(rows),
                "n_frames_kept": len(kept_sorted),
                "best_frame_idx": best_frame_idx,
                "D_best_mm": D_best,
                "max_Q": max_Q,
                "mean_Q_kept": mean_Q,
                "D_median_topK_mm": D_med,
                "K_used": len(top)
            })

    # Summary
    summary = {
        "svo_path": SVO_PATH,
        "weights": YOLO_WEIGHTS,
        "per_frame_csv": os.path.abspath(PER_FRAME_CSV),
        "best_frame_csv": os.path.abspath(BEST_FRAME_CSV),
        "debug_dir": os.path.abspath(DEBUG_DIR),
        "tracking": {
            "enabled": TRACK_ENABLE,
            "scipy_hungarian": _HAS_SCIPY,
            "gate3d_m": TRACK_GATE3D_M,
            "min_iou": TRACK_MIN_IOU,
            "max_misses": TRACK_MAX_MISSES,
            "smooth": TRACK_SMOOTH,
        },
        "hough": {
            "dp": HOUGH_DP,
            "minDist": HOUGH_MIN_DIST,
            "param1": HOUGH_PARAM1,
            "param2": HOUGH_PARAM2,
            "edge_lo": HOUGH_EDGE_LO,
            "edge_hi": HOUGH_EDGE_HI,
            "blur_k": HOUGH_BLUR_K,
            "min_r_frac": HOUGH_MIN_R_FRAC,
            "max_r_frac": HOUGH_MAX_R_FRAC,
            "target_r_frac": HOUGH_TARGET_R_FRAC,
        },
        "selection": {
            "TOPK": TOPK,
            "MIN_Q_KEEP": MIN_Q_KEEP,
            "MIN_DEPTH_VALID_RATIO": MIN_DEPTH_VALID_RATIO,
        }
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    zed.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("[DONE]")
    print("Per-frame CSV:", os.path.abspath(PER_FRAME_CSV))
    print("Best-frame CSV:", os.path.abspath(BEST_FRAME_CSV))
    print("Debug frames:", os.path.abspath(DEBUG_DIR))
    print("Summary:", os.path.abspath(SUMMARY_JSON))


if __name__ == "__main__":
    main()
