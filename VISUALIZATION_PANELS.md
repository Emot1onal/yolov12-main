# Individual visualization panels for WPS

The exporter reuses a fixed final selected-image list. It does not select new images,
alter predicted coordinates/classes, or change class colors. Predictions are rerun
with the supplied model settings on first export. To reproduce an older figure,
use exactly its weights, split, confidence thresholds and inference image size.
The old grid alone does not record these settings.

## NWPU (PowerShell, from yolov12-main)

```powershell
python export_visualization_panels.py --dataset nwpu --split val --selected-list ../runs/paper_figures/nwpu_public_methods_grid.selected_images.txt --panels-out ../runs/paper_figures/nwpu_individual_panels
```

Default weights are `../runs/detect/off_all/weights/best.pt` and
`../runs/detect/exp2_sum_all/weights/best.pt`. Override with `--yolov12` and `--ours`
if the original grid used other checkpoints. Default confidence thresholds are
0.25 (baseline/ours) and 0.35 (public models); default inference size is 640.

## DIOR (RunPod, from /workspace/yolov12-main)

Upload your Times New Roman regular font file (`C:/Windows/Fonts/times.ttf`)
to `/workspace/fonts/times.ttf` first. No substitute font is used.

```bash
python export_visualization_panels.py \
  --dataset dior \
  --selected-list /workspace/runs/paper_figures/dior_public_methods_grid.selected_images.txt \
  --panels-out /workspace/runs/paper_figures/dior_individual_panels \
  --font-path /workspace/fonts/times.ttf
```

Use the final list saved after dropping column d; do not drop it again.
Use the same original CLI options for data, split, weights, confidence and image
size. The exporter accepts the visualization script's options, but never scans
for replacement samples. Missing requested models stop export instead of dropping rows.

## Output and layout

- Each numbered image folder contains `original.png` and one PNG per method/GT.
- `panel_index.csv` records column, scene suggestion, source, method and dimensions.
- `scene_review.png` shows source images for checking scene names.
- `scene_titles.json` maps source filenames to editable scene titles. Suggestions
  based on GT classes need visual review (vehicles can also be in parking lots).
- `predictions.json` records settings and coordinates. Add `--cached` to redraw
  these predictions without running models again. Cached settings remain authoritative.
- To use reviewed titles, pass `--scene-map /path/to/reviewed_titles.json`.

In WPS, insert the numbered PNG panels as pictures, keeping their aspect ratio.
Keep each source image in one column and the method order identical across columns.
Group columns with the same reviewed scene if desired; move entire columns together.
Add scene headings and method names as native text boxes in Times New Roman.
Keep headings outside images. Use alignment and distribution tools for equal spacing.
Export PDF with the highest available image quality and avoid image compression.

Panels retain the complete source field of view, unlike the old grid's square
center crop for non-square inputs. Do not stretch or crop panels just to force a
square layout. Default export is 2x original dimensions, with text/boxes drawn
at that resolution. This improves annotation edges, but does not create additional
photographic detail. PNG annotations remain raster text; WPS text boxes remain editable.

Old grid PNG/PDF files are not overwritten.
