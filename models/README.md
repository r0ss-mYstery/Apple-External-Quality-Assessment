# Model weights

Place the two pretrained segmentation weights used by the manuscript here:

- `field_fruit_yolo26n_seg.pt` — on-tree RGB-D size estimation and field color assessment
- `controlled_fruit_yolo26n_seg.pt` — controlled-image color assessment

The public workflow uses **YOLO26n-seg directly**. No YOLO→SAM2 stage is required.

If the `.pt` files are below GitHub's individual-file size limit, they may be
committed directly. Otherwise provide them through GitHub Releases/Zenodo and
update the main README with the download link.
