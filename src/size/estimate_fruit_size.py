#!/usr/bin/env python3
"""
Primary RGB-D fruit-size pipeline used for manuscript reproduction.

ZED SVO/LIVE
  -> YOLO26n-seg
  -> fruit mask + median depth
  -> 3-D/2-D Hungarian tracking
  -> outer-arc RANSAC circle fit
  -> per-frame quality score
  -> median diameter of top-5 quality observations

IMPORTANT: processing frames are never rotated because RGB pixels must remain
aligned with ZED point-cloud indexing. Optional rotation is display-only.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

try:
    from .depth_processing import (
        median_depth_from_mask_absZ,
        center_xyz_from_mask_centroid_window,
        diameter_px_to_mm,
    )
    from .fruit_tracking import MultiFruitTracker
    from .geometry_methods import (
        largest_contour, select_outer_arc_points, ransac_circle
    )
    from .frame_quality import ransac_quality, select_final_ransac_size
except ImportError:
    from depth_processing import (
        median_depth_from_mask_absZ,
        center_xyz_from_mask_centroid_window,
        diameter_px_to_mm,
    )
    from fruit_tracking import MultiFruitTracker
    from geometry_methods import (
        largest_contour, select_outer_arc_points, ransac_circle
    )
    from frame_quality import ransac_quality, select_final_ransac_size


def open_zed(mode, svo_path):
    zed = sl.Camera()
    init = sl.InitParameters()
    if mode == "SVO":
        init.set_from_svo_file(str(svo_path))
        init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    if hasattr(sl.DEPTH_MODE, "NEURAL"):
        init.depth_mode = sl.DEPTH_MODE.NEURAL
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {err}")
    return zed


def get_fx(zed):
    info = zed.get_camera_information()
    return float(info.camera_configuration.calibration_parameters.left_cam.fx)


def zed_to_bgr(data, use_bgra=True):
    code = cv2.COLOR_BGRA2BGR if use_bgra else cv2.COLOR_RGBA2BGR
    return cv2.cvtColor(data, code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Field YOLO26n-seg .pt weight")
    ap.add_argument("--mode", choices=["SVO","LIVE"], default="SVO")
    ap.add_argument("--svo", default=None, help="Required when --mode SVO")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--frame_stride", type=int, default=10)
    ap.add_argument("--mask_threshold", type=float, default=0.40)
    ap.add_argument("--min_mask_area", type=int, default=150)
    ap.add_argument("--sample_step", type=int, default=2)
    ap.add_argument("--min_depth_points", type=int, default=30)
    ap.add_argument("--max_depth_m", type=float, default=15.0)
    ap.add_argument("--outer_arc_quantile", type=float, default=0.65)
    ap.add_argument("--ransac_iterations", type=int, default=300)
    ap.add_argument("--ransac_min_inliers", type=int, default=20)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--rotate_display_ccw90", action="store_true")
    ap.add_argument("--use_rgba", action="store_true")
    args = ap.parse_args()

    if args.mode == "SVO" and not args.svo:
        ap.error("--svo is required when --mode SVO")

    out = Path(args.output_dir)
    debug_dir = out / "debug_frames"
    out.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    model = YOLO(args.model)
    zed = open_zed(args.mode, args.svo)
    fx = get_fx(zed)

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    point_cloud = sl.Mat()
    tracker = MultiFruitTracker()

    rows_by_fruit = {}
    per_frame_path = out / "fruit_size_per_frame.csv"
    fields = [
        "frame_idx","fruit_id","inst_id","score_yolo",
        "circle_ok","circle_cx","circle_cy","circle_r_px",
        "circle_inlier_ratio","circle_arc_coverage","circle_rms_px",
        "depth_median_m","depth_valid_ratio","D_circle_mm","Q",
        "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
    ]

    with per_frame_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        frame_idx = 0

        while True:
            err = zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                break

            if frame_idx % args.frame_stride != 0:
                frame_idx += 1
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Never rotate the processing image.
            img = zed_to_bgr(image.get_data(), use_bgra=not args.use_rgba)
            result = model.predict(
                source=img, conf=args.conf, iou=args.iou,
                imgsz=args.imgsz, retina_masks=True, verbose=False
            )[0]

            vis = img.copy()
            detections, metadata = [], []

            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()

                H,W = img.shape[:2]
                for inst_id, (mask_raw, score) in enumerate(zip(masks, confs)):
                    if mask_raw.shape != (H,W):
                        mask_raw = cv2.resize(
                            mask_raw.astype(np.float32), (W,H),
                            interpolation=cv2.INTER_NEAREST
                        )
                    mask = mask_raw > args.mask_threshold
                    if int(mask.sum()) < args.min_mask_area:
                        continue

                    contour = largest_contour(
                        mask.astype(np.uint8)*255, args.min_mask_area
                    )
                    if contour is None:
                        continue

                    x,y,w,h = cv2.boundingRect(contour)
                    bbox = (x,y,x+w,y+h)
                    center = center_xyz_from_mask_centroid_window(
                        point_cloud, mask, max_depth_m=args.max_depth_m
                    )
                    if center is None:
                        continue

                    detections.append({
                        "center_xyz": center, "bbox_xyxy": bbox
                    })
                    metadata.append({
                        "inst_id": inst_id,
                        "score": float(score),
                        "mask": mask,
                        "contour": contour,
                        "bbox": bbox,
                    })

            fruit_ids = tracker.update(detections)

            for fruit_id, meta in zip(fruit_ids, metadata):
                mask = meta["mask"]
                contour = meta["contour"]
                score = meta["score"]
                bbox = meta["bbox"]

                depth = median_depth_from_mask_absZ(
                    point_cloud, mask,
                    sample_step=args.sample_step,
                    min_points=args.min_depth_points,
                    max_depth_m=args.max_depth_m,
                )
                if depth is None:
                    continue
                z, depth_ratio = depth

                points = select_outer_arc_points(
                    contour.reshape(-1,2),
                    keep_quantile=args.outer_arc_quantile
                )
                fit = ransac_circle(
                    points,
                    iterations=args.ransac_iterations,
                    min_inliers=args.ransac_min_inliers,
                    seed=0,
                )

                row = dict(
                    frame_idx=frame_idx, fruit_id=fruit_id,
                    inst_id=meta["inst_id"], score_yolo=score,
                    circle_ok=0, circle_cx="", circle_cy="", circle_r_px="",
                    circle_inlier_ratio=0.0, circle_arc_coverage=0.0,
                    circle_rms_px=0.0,
                    depth_median_m=z, depth_valid_ratio=depth_ratio,
                    D_circle_mm="", Q=0.0,
                    bbox_x1=bbox[0], bbox_y1=bbox[1],
                    bbox_x2=bbox[2], bbox_y2=bbox[3],
                )

                if fit is not None:
                    diameter_mm = diameter_px_to_mm(2.0*fit["r"], z, fx)
                    Q = ransac_quality(
                        score, depth_ratio, fit["inlier_ratio"],
                        fit["arc_coverage"], fit["rms_px"]
                    )
                    row.update(
                        circle_ok=1,
                        circle_cx=fit["cx"], circle_cy=fit["cy"],
                        circle_r_px=fit["r"],
                        circle_inlier_ratio=fit["inlier_ratio"],
                        circle_arc_coverage=fit["arc_coverage"],
                        circle_rms_px=fit["rms_px"],
                        D_circle_mm=diameter_mm,
                        Q=Q,
                    )
                    cv2.circle(
                        vis,
                        (int(round(fit["cx"])),int(round(fit["cy"]))),
                        int(round(fit["r"])), (255,0,0), 2
                    )

                writer.writerow(row)
                rows_by_fruit.setdefault(int(fruit_id), []).append(row)

            if args.show:
                show = (
                    cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    if args.rotate_display_ccw90 else vis
                )
                cv2.imshow("Fruit Size - RANSAC", show)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            frame_idx += 1

    best_path = out / "fruit_best_frame.csv"
    with best_path.open("w", newline="", encoding="utf-8") as f:
        fields_best = [
            "fruit_id","n_frames_total","n_frames_kept","best_frame_idx",
            "D_best_mm","max_Q","D_median_topK_mm","K_used"
        ]
        writer = csv.DictWriter(f, fieldnames=fields_best)
        writer.writeheader()

        for fruit_id, rows in sorted(rows_by_fruit.items()):
            kept, top, final = select_final_ransac_size(
                rows, top_k=args.top_k
            )
            if not top:
                writer.writerow({
                    "fruit_id":fruit_id, "n_frames_total":len(rows),
                    "n_frames_kept":0, "best_frame_idx":"",
                    "D_best_mm":"", "max_Q":"",
                    "D_median_topK_mm":"", "K_used":0,
                })
                continue
            best = top[0]
            writer.writerow({
                "fruit_id":fruit_id,
                "n_frames_total":len(rows),
                "n_frames_kept":len(kept),
                "best_frame_idx":best["frame_idx"],
                "D_best_mm":best["D_circle_mm"],
                "max_Q":best["Q"],
                "D_median_topK_mm":final,
                "K_used":len(top),
            })

    summary = {
        "model": str(Path(args.model)),
        "mode": args.mode,
        "svo": args.svo,
        "fx_px": fx,
        "method": "outer_arc_circle_RANSAC",
        "outer_arc_quantile": args.outer_arc_quantile,
        "ransac_iterations": args.ransac_iterations,
        "top_k": args.top_k,
        "processing_rotation": "none",
    }
    (out/"run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    zed.close()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Per-frame results: {per_frame_path}")
    print(f"Final fruit sizes: {best_path}")


if __name__ == "__main__":
    main()
