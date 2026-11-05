#!/usr/bin/env python3
"""
Compute method-comparison metrics from a CSV containing ground truth and
predictions for RANSAC, Hough, MinAreaRect, and Ellipse.

Required ground-truth column: ground_truth_mm
Prediction columns may be passed explicitly.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def metrics(gt, pred):
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt != 0)
    gt, pred = gt[valid], pred[valid]
    err = pred - gt
    return {
        "n": int(len(gt)),
        "MAE_mm": float(np.mean(np.abs(err))),
        "MAPE_pct": float(np.mean(np.abs(err)/np.abs(gt))*100.0),
        "RMSE_mm": float(np.sqrt(np.mean(err**2))),
        "bias_mm": float(np.mean(err)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--gt", default="ground_truth_mm")
    ap.add_argument(
        "--methods",
        default="ransac_mm,hough_mm,minarearect_mm,ellipse_mm",
        help="Comma-separated prediction columns."
    )
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    gt = df[args.gt].to_numpy(float)
    rows = []

    for col in [x.strip() for x in args.methods.split(",") if x.strip()]:
        if col not in df.columns:
            print(f"[SKIP] Missing column: {col}")
            continue
        out = metrics(gt, df[col].to_numpy(float))
        out["method"] = col
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
