#!/usr/bin/env python3
"""Independent field validation metrics for fruit cheek-width estimation."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--gt", default="ground_truth_mm")
    ap.add_argument("--pred", default="estimated_mm")
    ap.add_argument("--threshold_mm", type=float, default=60.0)
    ap.add_argument("--output_txt", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    gt = df[args.gt].to_numpy(float)
    pred = df[args.pred].to_numpy(float)
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt != 0)
    gt, pred = gt[valid], pred[valid]

    err = pred-gt
    abs_err = np.abs(err)
    gt_pick = gt >= args.threshold_mm
    pred_pick = pred >= args.threshold_mm

    r = float(np.corrcoef(gt,pred)[0,1]) if len(gt) > 1 else float("nan")
    results = {
        "n": len(gt),
        "GT mean (mm)": np.mean(gt),
        "Estimate mean (mm)": np.mean(pred),
        "MAE (mm)": np.mean(abs_err),
        "MAPE (%)": np.mean(abs_err/np.abs(gt))*100.0,
        "RMSE (mm)": np.sqrt(np.mean(err**2)),
        "Median absolute error (mm)": np.median(abs_err),
        "Bias (mm)": np.mean(err),
        "Pearson r": r,
        "R2": r*r,
        "Within 5 mm (%)": np.mean(abs_err<=5)*100,
        "Within 10 mm (%)": np.mean(abs_err<=10)*100,
        "Within 15 mm (%)": np.mean(abs_err<=15)*100,
        f"{args.threshold_mm:g}-mm agreement (%)":
            np.mean(gt_pick==pred_pick)*100,
        "False positives": int(np.sum((~gt_pick)&pred_pick)),
        "False negatives": int(np.sum(gt_pick&(~pred_pick))),
    }

    text = "\n".join(f"{k}: {v:.4f}" if isinstance(v,(float,np.floating))
                     else f"{k}: {v}" for k,v in results.items())
    print(text)

    if args.output_txt:
        p = Path(args.output_txt)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text+"\n", encoding="utf-8")
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
