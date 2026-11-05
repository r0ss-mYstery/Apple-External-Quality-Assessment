# In-Field Apple External Quality Assessment

Reproduction code for the manuscript on apple external quality assessment for
selective robotic harvesting. The repository contains two primary components:

1. **RGB-D fruit size estimation**
2. **Cultivar-specific visible color assessment**

The final public workflow uses **YOLO26n-seg directly** for fruit segmentation.
The earlier YOLO→SAM2 development workflow is not part of the manuscript
reproduction pipeline.

---

## 1. Repository structure

```text
apple_external_quality_assessment/
│
├── models/
│   ├── field_fruit_yolo26n_seg.pt
│   ├── controlled_fruit_yolo26n_seg.pt
│   └── README.md
│
├── src/
│   ├── size/
│   │   ├── estimate_fruit_size.py
│   │   ├── geometry_methods.py
│   │   ├── depth_processing.py
│   │   ├── fruit_tracking.py
│   │   └── frame_quality.py
│   │
│   └── color/
│       ├── gray_reference_calibration.py
│       ├── apply_white_balance.py
│       ├── segment_controlled_fruit_yolo26.py
│       ├── derive_lab_core_band.py
│       ├── derive_hsv_thresholds.py
│       └── color_coverage.py
│
├── analysis/
│   ├── compare_size_methods.py
│   ├── validate_size_estimation.py
│   ├── golden_illumination_normalization.py
│   ├── recompute_golden_normalized_coverage.py
│   └── generate_color_figures.py
│
├── config/
│   ├── size_estimation.json
│   └── color_thresholds.json
│
└── archive/
    └── size_method_development/
```

`archive/` contains development/reference implementations and is not required
for reproducing the final pipeline.

---

## 2. Model weights

Place the pretrained models in `models/`:

```text
models/field_fruit_yolo26n_seg.pt
models/controlled_fruit_yolo26n_seg.pt
```

No training script is required to reproduce inference because the pretrained
models used by the study are provided.

- **Field model:** on-tree segmentation used by RGB-D size estimation and field
  color assessment.
- **Controlled model:** segmentation of controlled color-assessment images.

---

## 3. Environment

Python dependencies:

```bash
pip install -r requirements.txt
```

RGB-D size estimation additionally requires the **Stereolabs ZED SDK** and its
Python package `pyzed`. Install the ZED SDK version compatible with your
camera/system before running the size pipeline.

Test the installation:

```bash
python -c "import cv2, numpy, scipy, ultralytics; print('Python packages OK')"
python -c "import pyzed.sl as sl; print('ZED SDK OK')"
```

---

# Part A — RGB-D Fruit Size Estimation

## 4. Method overview

```text
ZED RGB-D frame
    ↓
YOLO26n-seg fruit mask
    ↓
Median valid |Z| depth inside mask
    ↓
Local 3-D fruit center
    ↓
3-D distance + 2-D IoU Hungarian tracking
    ↓
Largest fruit contour
    ↓
Outer radial contour points
    ↓
Circle RANSAC
    ↓
Per-frame quality score
    ↓
Rank observations by quality
    ↓
Median diameter of top 5 observations
```

The processing image is **never rotated**, because RGB pixel coordinates must
stay aligned with ZED point-cloud coordinates. Display rotation is optional.

The metric conversion is:

```text
D_mm = 1000 × D_px × z / f_x
```

where `D_px = 2r`, `z` is the median valid fruit-mask depth, and `f_x` is the
ZED left-camera focal length in pixels.

## 5. Run the primary RANSAC size pipeline

For a recorded ZED SVO/SVO2:

```bash
python src/size/estimate_fruit_size.py \
  --model models/field_fruit_yolo26n_seg.pt \
  --mode SVO \
  --svo path/to/recording.svo2 \
  --output_dir outputs/size_run
```

Useful optional arguments:

```text
--imgsz 960
--conf 0.25
--iou 0.70
--frame_stride 10
--outer_arc_quantile 0.65
--ransac_iterations 300
--ransac_min_inliers 20
--top_k 5
--show
--rotate_display_ccw90
```

Primary outputs:

```text
outputs/size_run/
├── fruit_size_per_frame.csv
├── fruit_best_frame.csv
└── run_summary.json
```

`fruit_best_frame.csv` contains the final fruit-level estimate in
`D_median_topK_mm`.

### Important RANSAC implementation note

The uploaded final implementation used:

```text
outer-arc radial quantile = 0.65
RANSAC iterations = 300
minimum RANSAC inliers = 20
Top K = 5
```

The residual threshold is adaptive:

```text
max(2 px, 0.03 × estimated radius)
```

If the manuscript currently reports a different iteration count, reconcile the
paper and code before publication.

---

## 6. Geometric method comparison

The study compared:

- RANSAC circle fitting
- Hough circle detection
- MinAreaRect
- Ellipse fitting

The original Hough and MinAreaRect/Ellipse implementations are retained under:

```text
archive/size_method_development/
```

They preserve the same general segmentation, depth, tracking and top-K
framework used during method evaluation.

If the per-fruit predictions from the four methods are assembled in one CSV:

```text
ground_truth_mm,ransac_mm,hough_mm,minarearect_mm,ellipse_mm
...
```

calculate comparison metrics with:

```bash
python analysis/compare_size_methods.py \
  --csv data/size_method_comparison.csv \
  --output_csv outputs/size_method_metrics.csv
```

The script reports `MAE`, `MAPE`, `RMSE`, and bias for each method.

---

## 7. Independent field validation

Prepare a CSV such as:

```text
ground_truth_mm,estimated_mm
88.3,86.5
79.1,82.0
...
```

Then run:

```bash
python analysis/validate_size_estimation.py \
  --csv data/independent_size_validation.csv \
  --threshold_mm 60 \
  --output_txt outputs/independent_size_validation.txt
```

The script reports:

- MAE
- MAPE
- RMSE
- median absolute error
- bias
- Pearson correlation
- R²
- percentage within 5, 10 and 15 mm
- agreement with the 60-mm size criterion
- false positives and false negatives

---

# Part B — Cultivar-Specific Color Assessment

## 8. Controlled-image workflow

```text
Controlled RGB image
    ↓
ColorChecker/gray reference
    ↓
Fixed linear-RGB white balance
    ↓
YOLO26n-seg fruit mask
    ↓
CIELAB analysis
    ↓
65th–98th percentile core color band
    ↓
HSV threshold derivation
```

Fuji uses the CIELAB `a*` channel and Golden Delicious uses `b*`.

## 9. Estimate white-balance gains

```bash
python src/color/gray_reference_calibration.py \
  --image path/to/colorchecker_reference.jpg \
  --out_json outputs/white_balance_gains.json
```

Select only the interior of a neutral gray patch.

For the study dataset, the fixed linear-RGB gains were:

```text
R = 0.970364
G = 1.035937
B = 0.995868
```

## 10. Apply white balance

```bash
python src/color/apply_white_balance.py \
  --input_dir data/controlled_raw \
  --output_dir data/controlled_wb \
  --gains "0.970364,1.035937,0.995868"
```

## 11. Segment controlled fruit with YOLO26n-seg

```bash
python src/color/segment_controlled_fruit_yolo26.py \
  --model models/controlled_fruit_yolo26n_seg.pt \
  --input_dir data/controlled_wb \
  --output_dir outputs/controlled_segmentation \
  --imgsz 1024
```

This script uses **YOLO26n-seg directly**. SAM2 is not required.

## 12. Derive the CIELAB core band

Fuji:

```bash
python src/color/derive_lab_core_band.py \
  --cultivar Fuji \
  --image_dir data/controlled_wb/fuji \
  --mask_dir outputs/controlled_segmentation/fuji/masks \
  --out_json outputs/fuji_lab_band.json
```

Golden Delicious:

```bash
python src/color/derive_lab_core_band.py \
  --cultivar GoldenDelicious \
  --image_dir data/controlled_wb/golden \
  --mask_dir outputs/controlled_segmentation/golden/masks \
  --out_json outputs/golden_lab_band.json
```

Both use the 65th–98th percentile core band by default.

## 13. Derive HSV criteria from the core pixels

```bash
python src/color/derive_hsv_thresholds.py \
  --band_json outputs/fuji_lab_band.json \
  --image_dir data/controlled_wb/fuji \
  --mask_dir outputs/controlled_segmentation/fuji/masks \
  --out_json outputs/fuji_hsv_derived.json
```

The derivation uses:

```text
Hue:       2nd–98th percentile
S_min:    10th percentile
V_min:     5th percentile
V_max:    99th percentile
```

Fuji hue is treated as a circular variable across the OpenCV H=0/180 boundary.

---

## 14. Final fixed HSV criteria reported in the manuscript

The fixed values used for field color assessment are stored in:

```text
config/color_thresholds.json
```

### Fuji

```text
H = [0,10] ∪ [164,180]
S >= 88
V = [79,221]
```

### Golden Delicious

```text
H = [12,43]
S >= 32
V = [129,244]
```

## 15. Calculate visible field color coverage

Fuji example:

```bash
python src/color/color_coverage.py \
  --cultivar Fuji \
  --image_dir data/field_fuji \
  --mask_dir data/field_fuji_masks \
  --config config/color_thresholds.json \
  --output_dir outputs/fuji_field_color
```

Golden Delicious:

```bash
python src/color/color_coverage.py \
  --cultivar GoldenDelicious \
  --image_dir data/field_golden \
  --mask_dir data/field_golden_masks \
  --config config/color_thresholds.json \
  --output_dir outputs/golden_field_color
```

Field color coverage is interpreted as **visible fruit-surface coloration**.
It is not used as a universal whole-fruit pass/fail criterion.

---

## 16. Golden Delicious illumination sensitivity analysis

Normalize dark field frames:

```bash
python analysis/golden_illumination_normalization.py \
  --input_dir data/field_golden \
  --output_dir outputs/golden_normalization \
  --target_v 116 \
  --max_gain 2.0
```

The analysis:

```text
median V < 116  -> gain = 116 / median V
gain <= 2.0
median V >= 116 -> unchanged
```

Only HSV `V` is directly scaled. `H` and `S` are not directly modified.
This is an **illumination/exposure sensitivity analysis**, not formal field
color calibration.

To recompute Golden Delicious coverage using the same masks:

```bash
python analysis/recompute_golden_normalized_coverage.py \
  --original_dir data/field_golden \
  --normalized_dir outputs/golden_normalization/normalized_images \
  --mask_dir data/field_golden_masks \
  --config config/color_thresholds.json \
  --output_csv outputs/golden_normalized_coverage.csv
```

---

## 17. Reproducibility notes

- The provided YOLO weights are sufficient for inference; model retraining is
  not required for the paper reproduction workflow.
- Size estimation requires ZED depth data, not ordinary RGB images alone.
- Do not rotate frames before querying ZED point-cloud pixels.
- Controlled and field color coverage values should not be interpreted as
  direct measurements of the same fruit surface.
- Field color coverage is a visible-surface indicator.
- The 60-mm threshold is applied only to the validated fruit-size output.

---

## 18. Development code excluded from the primary workflow

Legacy YOLO→SAM2 scripts, mask-recovery scripts, and one-off debugging
utilities are intentionally excluded from `src/`.

Original comparison implementations that are useful for documenting the
geometric-method experiments are retained under `archive/`, but the primary
manuscript pipeline is the RANSAC workflow in `src/size/estimate_fruit_size.py`.
