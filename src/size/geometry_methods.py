#!/usr/bin/env python3
"""Geometric fruit-size estimators used in the manuscript comparison."""

import math
import random
from typing import Optional
import cv2
import numpy as np


def largest_contour(mask_u8_255, min_area_px=150):
    contours, _ = cv2.findContours(
        mask_u8_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px:
        return None
    return contour


def arc_coverage(points_xy, cx, cy):
    angles = np.sort(np.arctan2(points_xy[:, 1]-cy, points_xy[:, 0]-cx))
    if angles.size < 6:
        return 0.0
    gaps = np.diff(angles)
    wrap_gap = (angles[0] + 2*np.pi) - angles[-1]
    return float(np.clip(1.0 - max(np.max(gaps), wrap_gap)/(2*np.pi), 0, 1))


def select_outer_arc_points(contour_xy, keep_quantile=0.65, max_points=800):
    """Keep contour points at or above the radial-distance quantile."""
    pts = contour_xy.astype(np.float64)
    centroid = pts.mean(axis=0)
    radial = np.linalg.norm(pts-centroid, axis=1)
    threshold = np.quantile(radial, float(keep_quantile))
    keep = pts[radial >= threshold]
    if keep.shape[0] > max_points:
        idx = np.linspace(0, keep.shape[0]-1, max_points).astype(int)
        keep = keep[idx]
    return keep


def circle_from_3pts(p1, p2, p3):
    (x1,y1),(x2,y2),(x3,y3) = p1,p2,p3
    a,b = x1-x2, y1-y2
    c,d = x1-x3, y1-y3
    e = ((x1*x1-x2*x2)+(y1*y1-y2*y2))/2.0
    f = ((x1*x1-x3*x3)+(y1*y1-y3*y3))/2.0
    det = a*d-b*c
    if abs(det) < 1e-6:
        return None
    cx = (d*e-b*f)/det
    cy = (-c*e+a*f)/det
    r = math.hypot(cx-x1, cy-y1)
    if not np.isfinite([cx,cy,r]).all() or r <= 1:
        return None
    return float(cx), float(cy), float(r)


def fit_circle_ls(points_xy):
    x, y = points_xy[:,0], points_xy[:,1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x*x + y*y
    a, bb, c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = a/2.0, bb/2.0
    r = math.sqrt(max(c + cx*cx + cy*cy, 0.0))
    return float(cx), float(cy), float(r)


def score_circle(points_xy, cx, cy, r, residual_threshold_px):
    d = np.sqrt((points_xy[:,0]-cx)**2 + (points_xy[:,1]-cy)**2)
    residual = np.abs(d-r)
    inliers = residual <= residual_threshold_px
    inlier_ratio = float(inliers.mean())
    rms = float(np.sqrt(np.mean(residual[inliers]**2))) if inliers.any() else 1e9
    coverage = arc_coverage(points_xy[inliers], cx, cy) if inliers.any() else 0.0
    quality = (inlier_ratio * coverage) / (1.0 + rms)
    return quality, inlier_ratio, coverage, rms, inliers


def ransac_circle(
    points_xy,
    iterations=300,
    min_inliers=20,
    residual_fraction=0.03,
    minimum_residual_px=2.0,
    seed=0,
):
    """Three-point circle RANSAC followed by least-squares inlier refinement."""
    if points_xy.shape[0] < 20:
        return None

    rng = random.Random(seed)
    center0 = points_xy.mean(axis=0)
    r_est = float(np.median(np.linalg.norm(points_xy-center0, axis=1)))
    threshold = max(float(minimum_residual_px), float(residual_fraction)*r_est)

    best = None
    N = points_xy.shape[0]
    for _ in range(int(iterations)):
        i1, i2, i3 = rng.sample(range(N), 3)
        circle = circle_from_3pts(points_xy[i1], points_xy[i2], points_xy[i3])
        if circle is None:
            continue
        cx, cy, r = circle
        q, ir, cov, rms, inliers = score_circle(
            points_xy, cx, cy, r, threshold
        )
        if int(inliers.sum()) < int(min_inliers):
            continue
        if best is None or q > best["quality"]:
            best = dict(cx=cx, cy=cy, r=r, quality=q, inlier_ratio=ir,
                        arc_coverage=cov, rms_px=rms, threshold_px=threshold,
                        inliers=inliers)

    if best is None:
        return None

    cx, cy, r = fit_circle_ls(points_xy[best["inliers"]])
    q, ir, cov, rms, inliers = score_circle(points_xy, cx, cy, r, threshold)
    best.update(cx=cx, cy=cy, r=r, quality=q, inlier_ratio=ir,
                arc_coverage=cov, rms_px=rms, inliers=inliers)
    return best


def min_area_rect(contour):
    if contour is None or len(contour) < 3:
        return None
    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), angle = rect
    if w <= 1 or h <= 1:
        return None
    return {
        "major_px": float(max(w,h)),
        "minor_px": float(min(w,h)),
        "rect": rect,
        "angle_deg": float(angle),
    }


def fit_ellipse(contour):
    if contour is None or len(contour) < 5:
        return None
    try:
        ellipse = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    (_, _), (a,b), angle = ellipse
    if a <= 1 or b <= 1:
        return None
    return {
        "major_px": float(max(a,b)),
        "minor_px": float(min(a,b)),
        "ellipse": ellipse,
        "angle_deg": float(angle),
    }


def hough_circle_from_mask(
    mask_u8_255,
    bbox_xyxy,
    dp=1.2,
    min_dist=30,
    param1=100,
    param2=18,
    edge_lo=50,
    edge_hi=150,
    blur_k=5,
    min_r_frac=0.25,
    max_r_frac=0.70,
    target_r_frac=0.45,
):
    x1,y1,x2,y2 = bbox_xyxy
    roi = mask_u8_255[max(0,y1):max(y1+2,y2), max(0,x1):max(x1+2,x2)]
    if roi.size == 0:
        return None

    edge = cv2.Canny(roi, edge_lo, edge_hi)
    k = max(3, int(blur_k))
    if k % 2 == 0:
        k += 1
    edge = cv2.GaussianBlur(edge, (k,k), 0)

    h,w = roi.shape[:2]
    min_dim = max(1, min(h,w))
    r_min = int(max(5, min_r_frac*min_dim))
    r_max = int(max(r_min+2, max_r_frac*min_dim))
    circles = cv2.HoughCircles(
        edge, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=r_min, maxRadius=r_max,
    )
    if circles is None:
        return None
    circles = circles[0]
    target = target_r_frac*min_dim
    penalties = np.abs(circles[:,2]-target)/max(target,1e-6)
    j = int(np.argmin(penalties))
    cx,cy,r = circles[j]
    return {
        "cx": float(cx+x1),
        "cy": float(cy+y1),
        "r": float(r),
        "radius_penalty": float(np.clip(penalties[j],0,1)),
    }
