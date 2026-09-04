"""Export fixed visualization samples as individual panels for manual WPS layout."""

import argparse
import csv
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCENES = {
    "airplane": "Airport", "airport": "Airport",
    "ship": "Harbor", "harbor": "Harbor",
    "storage_tank": "Industrial area", "storagetank": "Industrial area",
    "chimney": "Industrial area", "windmill": "Wind farm",
    "bridge": "Bridge", "dam": "Dam",
    "vehicle": "Road traffic", "overpass": "Road interchange",
    "Expressway-Service-area": "Highway service area",
    "Expressway-toll-station": "Highway toll station",
    "trainstation": "Railway station",
    "baseball_diamond": "Sports ground", "baseballfield": "Sports ground",
    "tennis_court": "Sports ground", "tenniscourt": "Sports ground",
    "basketball_court": "Sports ground", "basketballcourt": "Sports ground",
    "ground_track_field": "Sports ground", "groundtrackfield": "Sports ground",
    "stadium": "Sports ground", "golffield": "Golf course",
}


def slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def scene_suggestion(labels, names):
    counts = Counter(SCENES.get(names[int(b[0])], "Other") for b in labels)
    return counts.most_common(1)[0][0] if counts else "Unclassified"


def draw_panel(image, boxes, names, module, font_path, font_scale, scale, gt):
    # Render labels after scaling, instead of enlarging rasterized small text.
    im = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im)
    divisor = 48 if module.__name__.startswith("make_dior") else 42
    minimum = 9 if divisor == 48 else 10
    font = ImageFont.truetype(str(font_path), max(1, round(max(minimum, image.width // divisor) * font_scale * scale)))
    line_width = max(2, image.width // (220 if divisor == 48 else 210)) * scale
    for cls, x1, y1, x2, y2, score in boxes:
        cls = int(cls)
        if not 0 <= cls < len(names):
            raise ValueError(f"Invalid class index {cls} for {names}")
        label = module.compact_name(names[cls]) if hasattr(module, "compact_name") else names[cls]
        if not gt:
            label += f" {score:.2f}"
        x1, y1, x2, y2 = [v * scale for v in (x1, y1, x2, y2)]
        color = module.color_for_class(cls)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        bounds = draw.textbbox((0, 0), label, font=font)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        tx = max(0, min(x1, im.width - tw - 6 * scale))
        ty = max(0, min(y1 - th - 4 * scale, im.height - th - 4 * scale))
        draw.rectangle((tx, ty, tx + tw + 6 * scale, ty + th + 4 * scale), fill=color)
        draw.text((tx + 3 * scale - bounds[0], ty + 2 * scale - bounds[1]), label, font=font, fill="white")
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("nwpu", "dior"), required=True)
    parser.add_argument("--selected-list", type=Path, required=True)
    parser.add_argument("--panels-out", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, default=Path("C:/Windows/Fonts/times.ttf"))
    parser.add_argument("--panel-scale", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--scene-map", type=Path, help="Optional JSON mapping image filenames to scene titles.")
    parser.add_argument("--cached", action="store_true", help="Redraw saved predictions without loading models.")
    export, remaining = parser.parse_known_args()
    if not export.font_path.is_file():
        parser.error("Times New Roman font missing. Supply --font-path /path/to/times.ttf; no font substitution is performed.")
    ImageFont.truetype(str(export.font_path), 16)
    module = importlib.import_module(f"make_{export.dataset}_public_method_visualization")
    sys.argv = [sys.argv[0]] + remaining
    args = module.parse_args()
    if args.images or getattr(args, "drop_cols", None):
        parser.error("Use the final selected list only; --images / --drop-cols would change that selection.")
    selected_names = [s.strip() for s in export.selected_list.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not selected_names or len(set(selected_names)) != len(selected_names):
        parser.error("Selection must contain unique image filenames, one per line.")
    root, data, names = module.read_data_yaml(args.data)
    available = {p.name: p for p in module.list_images(root, data, args.split)}
    missing = set(selected_names) - available.keys()
    if missing:
        parser.error(f"Selected images missing from {args.split}: {sorted(missing)}")
    selected = [available[n] for n in selected_names]
    out = export.panels_out
    cache_path = out / "predictions.json"
    settings = {"dataset": export.dataset, "data": str(args.data.resolve()), "split": args.split,
                "imgsz": args.imgsz, "conf": args.conf, "public_conf": args.public_conf,
                "yolov12": str(args.yolov12.resolve()), "ours": str(args.ours.resolve())}
    if export.cached:
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        if record["selected"] != selected_names or record["settings"]["dataset"] != export.dataset:
            parser.error("Cache selection/dataset differs from this request.")
        names = record["names"]
    else:
        predictors = module.build_predictors(args)
        expected = ["YOLOv12", "HARPNet"]
        if not args.no_public:
            options = (("skip_fasterrcnn", "Faster R-CNN"), ("skip_maskrcnn", "Mask R-CNN"),
                       ("skip_yolov8_public", "YOLOv8n-VHR10")) if export.dataset == "nwpu" else (
                       ("skip_yolov8n_dior", "YOLOv8-DIOR-N"), ("skip_yolov8s_dior", "YOLOv8-DIOR-S"))
            expected += [name for flag, name in options if not getattr(args, flag)]
        if set(expected) != {p.name for p in predictors}:
            raise RuntimeError("Some requested models failed to load. Stopping to avoid silently removing rows.")
        record = {"settings": settings, "selected": selected_names, "names": names,
                  "rows": ["GT"] + [p.name for p in predictors], "images": {}}
        for path in selected:
            with Image.open(path) as image:
                boxes = {"GT": module.gt_to_xyxy(module.load_yolo_labels(path), image.width, image.height)}
                for predictor in predictors:
                    boxes[predictor.name] = predictor.predict(path, args.imgsz)
                record["images"][path.name] = {"size": list(image.size), "boxes": boxes}
            print(f"Predicted {path.name}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    overrides = json.loads(export.scene_map.read_text(encoding="utf-8-sig")) if export.scene_map else {}
    index = []
    suggestions = {}
    contact = Image.new("RGB", (5 * 260, ((len(selected) + 4) // 5) * 290), "white")
    contact_draw = ImageDraw.Draw(contact)
    contact_font = ImageFont.truetype(str(export.font_path), 17)
    for c, path in enumerate(selected, 1):
        scene = overrides.get(path.name) or scene_suggestion(module.load_yolo_labels(path), names)
        suggestions[path.name] = scene
        folder = out / f"{c:02d}_{slug(path.stem)}"
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as source:
            image = source.convert("RGB")
        if list(image.size) != record["images"][path.name]["size"]:
            raise ValueError(f"Source dimensions changed: {path}")
        image.save(folder / "original.png")
        thumb = image.copy()
        thumb.thumbnail((250, 245), Image.Resampling.LANCZOS)
        cx, cy = ((c - 1) % 5) * 260, ((c - 1) // 5) * 290
        contact.paste(thumb, (cx + (260 - thumb.width) // 2, cy))
        contact_draw.text((cx + 5, cy + 248), f"{c:02d} / {path.name}", font=contact_font, fill="black")
        contact_draw.text((cx + 5, cy + 268), scene, font=contact_font, fill="black")
        for r, name in enumerate(record["rows"]):
            panel = draw_panel(image, record["images"][path.name]["boxes"][name], names, module,
                               export.font_path, args.font_scale, export.panel_scale, name == "GT")
            target = folder / f"{r:02d}_{slug(name)}.png"
            panel.save(target, dpi=(300, 300))
            index.append({"column": c, "scene": scene, "scene_status": "user supplied" if path.name in overrides else "suggested; review visually",
                          "source": path.name, "row": r, "method": name, "file": target.relative_to(out).as_posix(),
                          "width": panel.width, "height": panel.height})
    with (out / "panel_index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    (out / "scene_titles.json").write_text(json.dumps(suggestions, indent=2), encoding="utf-8")
    contact.save(out / "scene_review.png")
    print(f"Saved {len(index)} panels: {out.resolve()}")
    print("Scene titles are suggestions inferred from GT classes; confirm visually before publication.")
    print("Background detail stays at source resolution; labels are rendered at export resolution.")


if __name__ == "__main__":
    main()
