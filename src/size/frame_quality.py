#!/usr/bin/env python3
"""Observation quality and top-K final-size selection."""

import numpy as np


def ransac_quality(
    segmentation_confidence,
    depth_valid_ratio,
    inlier_ratio,
    arc_coverage,
    rms_px,
):
    s = float(np.clip(segmentation_confidence, 0, 1))
    dv = float(np.clip(depth_valid_ratio, 0, 1))
    ir = float(np.clip(inlier_ratio, 0, 1))
    ac = float(np.clip(arc_coverage, 0, 1))
    rms_norm = float(np.clip(float(rms_px)/5.0, 0, 1))
    return float(np.clip(
        0.25*s + 0.25*dv + 0.45*(ir*ac) - 0.05*rms_norm,
        0, 1
    ))


def select_final_ransac_size(
    rows,
    top_k=5,
    min_quality=0.15,
    min_arc_coverage=0.35,
    min_inlier_ratio=0.50,
    min_depth_valid_ratio=0.25,
):
    """Return quality-ranked observations and median diameter of top K."""
    kept = [
        r for r in rows
        if r.get("circle_ok", 0) == 1
        and float(r.get("Q", 0)) >= min_quality
        and float(r.get("circle_arc_coverage", 0)) >= min_arc_coverage
        and float(r.get("circle_inlier_ratio", 0)) >= min_inlier_ratio
        and float(r.get("depth_valid_ratio", 0)) >= min_depth_valid_ratio
        and r.get("D_circle_mm") not in ("", None)
    ]
    kept.sort(key=lambda r: float(r["Q"]), reverse=True)
    top = kept[:min(int(top_k), len(kept))]
    final = (
        float(np.median([float(r["D_circle_mm"]) for r in top]))
        if top else None
    )
    return kept, top, final
