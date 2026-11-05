#!/usr/bin/env python3
"""Depth utilities for RGB-D apple size estimation."""

from typing import Optional, Tuple
import numpy as np


def median_depth_from_mask_absZ(
    point_cloud,
    mask_bool: np.ndarray,
    sample_step: int = 2,
    min_points: int = 30,
    max_depth_m: float = 15.0,
) -> Optional[Tuple[float, float]]:
    """
    Return (median |Z| in m, valid-depth ratio) for sampled mask pixels.

    The ZED pipeline uses RIGHT_HANDED_Y_UP, where forward Z may be negative;
    therefore the metric range uses abs(Z).
    """
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None

    step = max(1, int(sample_step))
    ys = ys[::step]
    xs = xs[::step]
    total = len(xs)
    zs = []

    for y, x in zip(ys, xs):
        err, p = point_cloud.get_value(int(x), int(y))
        # pyzed error-code comparison is intentionally avoided here so the
        # function stays testable; unsuccessful reads return non-finite values.
        try:
            z = float(p[2])
        except Exception:
            continue
        if not np.isfinite(z):
            continue
        z_abs = abs(z)
        if 1e-3 <= z_abs <= float(max_depth_m):
            zs.append(z_abs)

    if len(zs) < int(min_points):
        return None

    zs = np.asarray(zs, dtype=np.float64)
    return float(np.median(zs)), float(len(zs) / max(total, 1))


def center_xyz_from_mask_centroid_window(
    point_cloud,
    mask_bool: np.ndarray,
    window_radius: int = 7,
    min_points: int = 10,
    max_depth_m: float = 15.0,
):
    """Robust local 3-D center around the median mask pixel."""
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None

    cy = int(np.median(ys))
    cx = int(np.median(xs))
    h, w = mask_bool.shape
    r = max(1, int(window_radius))
    pts = []

    for y in range(max(0, cy-r), min(h, cy+r+1)):
        for x in range(max(0, cx-r), min(w, cx+r+1)):
            try:
                _, p = point_cloud.get_value(int(x), int(y))
                X, Y, Z = float(p[0]), float(p[1]), float(p[2])
            except Exception:
                continue
            if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
                continue
            if not (1e-3 <= abs(Z) <= float(max_depth_m)):
                continue
            pts.append((X, Y, Z))

    if len(pts) < int(min_points):
        return None

    return np.median(np.asarray(pts, dtype=np.float64), axis=0)


def diameter_px_to_mm(diameter_px: float, depth_m: float, focal_px: float) -> float:
    """Pinhole conversion used by the manuscript: D_mm = 1000 * D_px * z / f."""
    return 1000.0 * float(diameter_px) * float(depth_m) / float(focal_px)
