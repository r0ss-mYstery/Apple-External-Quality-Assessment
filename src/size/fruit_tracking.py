#!/usr/bin/env python3
"""3-D + 2-D IoU one-to-one fruit tracking."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    tid: int
    center_xyz: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    hits: int = 1
    misses: int = 0


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, ax2-ax1) * max(1, ay2-ay1)
    area_b = max(1, bx2-bx1) * max(1, by2-by1)
    return float(inter / (area_a + area_b - inter))


class MultiFruitTracker:
    """
    Hungarian assignment using gated 3-D distance and 2-D overlap.

    smoothing is the weight retained from the previous center:
        center <- smoothing * previous + (1-smoothing) * new
    """

    def __init__(
        self,
        gate3d_m=0.08,
        min_iou=0.10,
        max_misses=12,
        smoothing=0.75,
        w3d=1.0,
        w2d=0.4,
    ):
        self.gate3d_m = float(gate3d_m)
        self.min_iou = float(min_iou)
        self.max_misses = int(max_misses)
        self.smoothing = float(smoothing)
        self.w3d = float(w3d)
        self.w2d = float(w2d)
        self.tracks: List[Track] = []
        self.next_id = 1

    def _cleanup(self):
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

    def update(self, detections: List[Dict[str, Any]]) -> List[int]:
        self._cleanup()

        if not detections:
            for t in self.tracks:
                t.misses += 1
            self._cleanup()
            return []

        if not self.tracks:
            ids = []
            for d in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append(
                    Track(tid, d["center_xyz"].copy(), d["bbox_xyxy"])
                )
                ids.append(tid)
            return ids

        T, D = len(self.tracks), len(detections)
        cost = np.full((T, D), 1e6, dtype=np.float64)

        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                d3d = float(np.linalg.norm(tr.center_xyz - det["center_xyz"]))
                iou = bbox_iou(tr.bbox_xyxy, det["bbox_xyxy"])
                if d3d <= self.gate3d_m and iou >= self.min_iou:
                    cost[ti, di] = (
                        self.w3d * (d3d / max(self.gate3d_m, 1e-9))
                        + self.w2d * (1.0 - iou)
                    )

        ids = [-1] * D
        assigned_tracks, assigned_dets = set(), set()
        rows, cols = linear_sum_assignment(cost)

        for ti, di in zip(rows, cols):
            if cost[ti, di] >= 1e5:
                continue
            assigned_tracks.add(ti)
            assigned_dets.add(di)
            tr = self.tracks[ti]
            ids[di] = tr.tid
            new_center = detections[di]["center_xyz"]
            tr.center_xyz = (
                self.smoothing * tr.center_xyz
                + (1.0 - self.smoothing) * new_center
            )
            tr.bbox_xyxy = detections[di]["bbox_xyxy"]
            tr.hits += 1
            tr.misses = 0

        for ti, tr in enumerate(self.tracks):
            if ti not in assigned_tracks:
                tr.misses += 1

        for di, det in enumerate(detections):
            if di not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append(
                    Track(tid, det["center_xyz"].copy(), det["bbox_xyxy"])
                )
                ids[di] = tid

        self._cleanup()
        return ids
