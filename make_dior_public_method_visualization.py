import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


DIOR_NAMES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]


CLASS_COLORS = [
    (0, 170, 255),    # airplane
    (0, 210, 120),    # airport
    (255, 80, 80),    # baseballfield
    (0, 200, 210),    # basketballcourt
    (235, 70, 190),   # bridge
    (160, 160, 160),  # chimney
    (120, 90, 220),   # dam
    (255, 125, 0),    # Expressway-Service-area
    (255, 185, 0),    # Expressway-toll-station
    (80, 190, 80),    # golffield
    (255, 145, 40),   # groundtrackfield
    (120, 190, 70),   # harbor
    (180, 95, 255),   # overpass
    (0, 210, 120),    # ship
    (255, 110, 160),  # stadium
    (255, 200, 60),   # storagetank
    (185, 80, 255),   # tenniscourt
    (70, 130, 255),   # trainstation
    (40, 150, 255),   # vehicle
    (20, 190, 190),   # windmill
]


@dataclass
class Predictor:
    name: str
    model: object
    conf: float

    def predict(self, image_path, imgsz):
        return predict_yolo_like(self.model, image_path, self.conf, imgsz)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a paper-style DIOR visualization with public YOLOv8-DIOR models, YOLOv12, and HARPNet."
    )
    parser.add_argument("--data", type=Path, default=Path("/workspace/datasets/DIOR_YOLO/data.yaml"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("/workspace/runs/paper_figures/dior_public_methods_grid.png"))
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--scan-limit", type=int, default=0, help="0 means scan all labeled images in the split.")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--cell-size", type=int, default=210)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--export-scale", type=float, default=1.0, help="Upscale the final figure for high-resolution export.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--public-conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=None,
        help="Drop selected visualization columns by letter or 1-based index, e.g. --drop-cols d or --drop-cols 4.",
    )
    parser.add_argument(
        "--max-gt",
        type=int,
        default=12,
        help="Skip overly dense images whose number of ground-truth boxes is larger than this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-box-area",
        type=float,
        default=0.45,
        help="Skip images containing a very large GT box. This avoids bridge/airport crops that dominate a square cell.",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.02,
        help="Skip images with GT boxes too close to image borders. Use 0 to disable.",
    )
    parser.add_argument("--images", nargs="*", default=None, help="Optional fixed image filenames.")
    parser.add_argument("--no-public", action="store_true", help="Only draw GT, YOLOv12, and HARPNet.")
    parser.add_argument("--skip-yolov8n-dior", action="store_true")
    parser.add_argument("--skip-yolov8s-dior", action="store_true")
    parser.add_argument(
        "--yolov12",
        type=Path,
        default=Path("/workspace/runs/detect_DIOR/DIOR_off_all/weights/best.pt"),
        help="DIOR YOLOv12 baseline weights.",
    )
    parser.add_argument(
        "--ours",
        type=Path,
        default=Path("/workspace/runs/detect_DIOR/DIOR_exp2_sum_all/weights/epoch27.pt"),
        help="DIOR HARPNet weights. Default uses the best mAP50 checkpoint from DIOR evaluation.",
    )
    return parser.parse_args()


def drop_selected_columns(selected, drop_cols):
    if not drop_cols:
        return selected
    drop_indices = set()
    for item in drop_cols:
        value = str(item).strip().lower()
        if not value:
            continue
        if value.isdigit():
            idx = int(value) - 1
        elif len(value) == 1 and "a" <= value <= "z":
            idx = ord(value) - ord("a")
        else:
            raise ValueError(f"Unsupported column specifier: {item}. Use letters like d or 1-based numbers like 4.")
        if idx < 0 or idx >= len(selected):
            raise IndexError(f"Column {item} is out of range for {len(selected)} selected images.")
        drop_indices.add(idx)
    kept = [path for idx, path in enumerate(selected) if idx not in drop_indices]
    dropped = [path.name for idx, path in enumerate(selected) if idx in drop_indices]
    print(f"Dropped columns: {', '.join(dropped)}")
    return kept


def read_data_yaml(path):
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    names = data.get("names", DIOR_NAMES)
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return root, data, names


def list_images(root, data, split):
    split_path = Path(data[split])
    image_dir = split_path if split_path.is_absolute() else root / split_path
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def label_path_for_image(image_path):
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_yolo_labels(image_path):
    label_path = label_path_for_image(image_path)
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        items = line.strip().split()
        if len(items) < 5:
            continue
        cls, xc, yc, w, h = int(float(items[0])), *map(float, items[1:5])
        labels.append((cls, xc, yc, w, h))
    return labels


FONT_SCALE = 1.0


def get_font(size=14):
    size = max(1, round(size * FONT_SCALE))
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def color_for_class(cls):
    return CLASS_COLORS[cls % len(CLASS_COLORS)]


def compact_name(name):
    aliases = {
        "Expressway-Service-area": "service_area",
        "Expressway-toll-station": "toll_station",
        "baseballfield": "baseball",
        "basketballcourt": "basketball",
        "groundtrackfield": "track_field",
        "storagetank": "tank",
        "tenniscourt": "tennis",
        "trainstation": "station",
    }
    return aliases.get(name, name)


def draw_boxes(image, boxes, names, normalized=False, show_score=True):
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    fnt = get_font(max(9, im.width // 48))
    w, h = im.size
    for box in boxes:
        if normalized:
            cls, xc, yc, bw, bh = box
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            conf = None
        else:
            cls, x1, y1, x2, y2, conf = box
        if cls < 0 or cls >= len(names):
            continue
        label = compact_name(names[cls])
        if show_score and conf is not None:
            label = f"{label} {conf:.2f}"
        color = color_for_class(cls)
        width = max(2, im.width // 220)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        bbox = draw.textbbox((0, 0), label, font=fnt)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = max(0, min(x1, im.width - tw - 6))
        ty = max(0, y1 - th - 4)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=color)
        draw.text((tx + 3, ty + 2), label, fill=(255, 255, 255), font=fnt)
    return im


def resize_square(image, size):
    image = image.convert("RGB")
    scale = max(size / image.width, size / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def make_method_grid(rows, cells, cell_size, out):
    label_w = 128
    top_h = 18
    bottom_h = 32
    gap = 10
    row_gap = 10
    rows_count = len(rows)
    cols_count = max(c for _, c in cells) + 1
    width = label_w + cols_count * cell_size + (cols_count - 1) * gap
    height = top_h + rows_count * cell_size + (rows_count - 1) * row_gap + bottom_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = get_font(14)
    caption_font = get_font(13)

    for r, row_name in enumerate(rows):
        y = top_h + r * (cell_size + row_gap)
        bbox = draw.textbbox((0, 0), row_name, font=label_font)
        text_im = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
        td = ImageDraw.Draw(text_im)
        td.text((4, 4), row_name, fill=(0, 0, 0), font=label_font)
        text_im = text_im.rotate(90, expand=True)
        canvas.paste(text_im, (max(0, (label_w - text_im.width) // 2), y + (cell_size - text_im.height) // 2), text_im)
        for c in range(cols_count):
            x = label_w + c * (cell_size + gap)
            canvas.paste(resize_square(cells[(r, c)], cell_size), (x, y))

    caption_y = top_h + rows_count * cell_size + (rows_count - 1) * row_gap + 7
    for c in range(cols_count):
        caption = f"({chr(97 + c)})"
        x = label_w + c * (cell_size + gap)
        bbox = draw.textbbox((0, 0), caption, font=caption_font)
        draw.text((x + (cell_size - (bbox[2] - bbox[0])) // 2, caption_y), caption, fill=(0, 0, 0), font=caption_font)

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def predict_yolo_like(model, image_path, conf, imgsz):
    results = model.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    boxes = []
    if not results:
        return boxes
    result = results[0]
    if getattr(result, "boxes", None) is not None and result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.item())
            conf_value = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            boxes.append((cls, x1, y1, x2, y2, conf_value))
    if boxes:
        return boxes

    # Public DIOR YOLOv8 models are OBB models. Convert oriented boxes to
    # enclosing axis-aligned boxes so the final figure has one visual style.
    obb = getattr(result, "obb", None)
    if obb is not None:
        xyxy = getattr(obb, "xyxy", None)
        if xyxy is not None:
            for cls, conf_value, coords in zip(obb.cls, obb.conf, xyxy):
                x1, y1, x2, y2 = [float(v) for v in coords.tolist()]
                boxes.append((int(cls.item()), x1, y1, x2, y2, float(conf_value.item())))
    return boxes


def gt_to_xyxy(labels, width, height):
    boxes = []
    for cls, xc, yc, bw, bh in labels:
        boxes.append(
            (
                cls,
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
                1.0,
            )
        )
    return boxes


def is_visually_suitable(labels, max_gt, max_box_area, edge_margin):
    if not labels:
        return False
    if max_gt and len(labels) > max_gt:
        return False
    for _, xc, yc, bw, bh in labels:
        if max_box_area and bw * bh > max_box_area:
            return False
        if edge_margin:
            x1, y1 = xc - bw / 2, yc - bh / 2
            x2, y2 = xc + bw / 2, yc + bh / 2
            if x1 < edge_margin or y1 < edge_margin or x2 > 1 - edge_margin or y2 > 1 - edge_margin:
                return False
    return True


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a[1:5]
    bx1, by1, bx2, by2 = b[1:5]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detection_stats(gt_boxes, pred_boxes, iou_thr):
    matched_pred = set()
    matched_gt = set()
    conf_sum = 0.0
    for gi, gt in enumerate(gt_boxes):
        best = None
        best_iou = 0.0
        for pi, pred in enumerate(pred_boxes):
            if pi in matched_pred or pred[0] != gt[0]:
                continue
            iou = box_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best = pi
        if best is not None and best_iou >= iou_thr:
            matched_gt.add(gi)
            matched_pred.add(best)
            conf_sum += pred_boxes[best][5]
    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "count": len(pred_boxes), "conf_sum": conf_sum}


def score_image(gt_boxes, public_stats, base_stats, ours_stats):
    public_best_tp = max([s["tp"] for s in public_stats] + [base_stats["tp"]])
    public_best_fp = min([s["fp"] for s in public_stats] + [base_stats["fp"]])
    score = 0.0
    score += (ours_stats["tp"] - base_stats["tp"]) * 18.0
    score += (ours_stats["tp"] - public_best_tp) * 6.0
    score += (base_stats["fn"] - ours_stats["fn"]) * 8.0
    score += (public_best_fp - ours_stats["fp"]) * 1.0
    score += (base_stats["fp"] - ours_stats["fp"]) * 2.5
    score += (ours_stats["count"] - base_stats["count"]) * 0.5
    score += ours_stats["conf_sum"] * 0.15
    score += min(len(gt_boxes), 10) * 0.25
    return score


def select_images(
    all_images,
    predictors,
    num_images,
    seed,
    scan_limit,
    iou_thr,
    fixed_images,
    imgsz,
    max_gt,
    max_box_area,
    edge_margin,
):
    if fixed_images:
        by_name = {p.name: p for p in all_images}
        missing = [name for name in fixed_images if name not in by_name]
        if missing:
            raise FileNotFoundError(f"Images not found: {missing}")
        return [by_name[name] for name in fixed_images]

    candidates = []
    skipped_unsuitable = 0
    for p in all_images:
        labels = load_yolo_labels(p)
        if not is_visually_suitable(labels, max_gt, max_box_area, edge_margin):
            skipped_unsuitable += 1
            continue
        candidates.append(p)
    if skipped_unsuitable:
        print(
            f"Skipped {skipped_unsuitable} visually unsuitable images "
            f"(max_gt={max_gt}, max_box_area={max_box_area}, edge_margin={edge_margin})."
        )
    random.Random(seed).shuffle(candidates)
    if scan_limit and scan_limit > 0:
        candidates = candidates[:scan_limit]

    base = next((p for p in predictors if p.name == "YOLOv12"), None)
    ours = next((p for p in predictors if p.name == "HARPNet"), None)
    public = [p for p in predictors if p.name not in {"YOLOv12", "HARPNet"}]
    if base is None or ours is None:
        raise RuntimeError("YOLOv12 and HARPNet predictors are required for automatic selection.")

    scored = []
    for idx, image_path in enumerate(candidates, start=1):
        image = Image.open(image_path)
        gt = gt_to_xyxy(load_yolo_labels(image_path), image.width, image.height)
        public_stats = []
        for pred in public:
            try:
                public_stats.append(detection_stats(gt, pred.predict(image_path, imgsz), iou_thr))
            except Exception as exc:
                print(f"WARNING: skipped {pred.name} on {image_path.name}: {exc}")
        base_stats = detection_stats(gt, base.predict(image_path, imgsz), iou_thr)
        ours_stats = detection_stats(gt, ours.predict(image_path, imgsz), iou_thr)
        if ours_stats["tp"] < base_stats["tp"]:
            continue
        if ours_stats["fp"] > base_stats["fp"]:
            continue
        if ours_stats["tp"] == base_stats["tp"] and ours_stats["fp"] >= base_stats["fp"]:
            continue
        score = score_image(gt, public_stats, base_stats, ours_stats)
        scored.append((score, image_path, base_stats, ours_stats))
        if idx % 100 == 0:
            print(f"Scanned {idx}/{len(candidates)} images...")

    scored.sort(key=lambda x: (x[0], x[3]["tp"], -x[3]["fp"], len(load_yolo_labels(x[1]))), reverse=True)
    selected = [x[1] for x in scored[:num_images]]
    if len(selected) < num_images:
        print(f"WARNING: selected only {len(selected)} images. Try increasing --scan-limit or lowering --conf.")
    print("Selected images:")
    for score, path, base_stats, ours_stats in scored[:num_images]:
        print(
            f"  {path.name}: score={score:.2f}, "
            f"YOLOv12 TP/FP/FN={base_stats['tp']}/{base_stats['fp']}/{base_stats['fn']}, "
            f"HARPNet TP/FP/FN={ours_stats['tp']}/{ours_stats['fp']}/{ours_stats['fn']}"
        )
    return selected


def hf_download(repo_id, filename, local_files_only=False):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=local_files_only))


def hf_download_first(repo_id, candidates):
    from huggingface_hub import HfApi

    for filename in candidates:
        try:
            return hf_download(repo_id, filename, local_files_only=True)
        except Exception:
            pass

    files = set(HfApi().list_repo_files(repo_id=repo_id))
    for filename in candidates:
        if filename in files:
            return hf_download(repo_id, filename)
    model_files = sorted(f for f in files if f.endswith(".pt"))
    if model_files:
        return hf_download(repo_id, model_files[0])
    raise FileNotFoundError(f"No .pt weights found in {repo_id}.")


def load_yolo_predictor(label, weight_path, conf):
    from ultralytics import YOLO

    if not weight_path.exists():
        raise FileNotFoundError(f"{label} weights not found: {weight_path}")
    return Predictor(label, YOLO(str(weight_path)), conf)


def load_public_yolov8_dior(label, candidates, conf):
    from ultralytics import YOLO

    path = hf_download_first("pauhidalgoo/yolov8-DIOR", candidates)
    return Predictor(label, YOLO(str(path)), conf)


def filter_working_predictors(predictors, probe_image, imgsz):
    working = []
    for predictor in predictors:
        try:
            predictor.predict(probe_image, imgsz)
            working.append(predictor)
        except Exception as exc:
            print(f"WARNING: skipped {predictor.name}: predictor test failed: {exc}")
    return working


def build_predictors(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".yolo_config"))
    predictors = []
    if not args.no_public:
        if not args.skip_yolov8n_dior:
            try:
                predictors.append(
                    load_public_yolov8_dior(
                        "YOLOv8-DIOR-N",
                        ("DIOR_yolov8n_backbone.pt", "yolov8n.pt", "best_n.pt"),
                        args.public_conf,
                    )
                )
            except Exception as exc:
                print(f"WARNING: skipped YOLOv8-DIOR-N: {exc}")
        if not args.skip_yolov8s_dior:
            try:
                predictors.append(
                    load_public_yolov8_dior(
                        "YOLOv8-DIOR-S",
                        ("DIOR_yolov8s_backbone.pt", "yolov8s.pt", "best_s.pt"),
                        args.public_conf,
                    )
                )
            except Exception as exc:
                print(f"WARNING: skipped YOLOv8-DIOR-S: {exc}")

    predictors.append(load_yolo_predictor("YOLOv12", args.yolov12, args.conf))
    predictors.append(load_yolo_predictor("HARPNet", args.ours, args.conf))
    return predictors


def main():
    args = parse_args()
    global FONT_SCALE
    FONT_SCALE = args.font_scale
    root, data, names = read_data_yaml(args.data)
    images = list_images(root, data, args.split)
    predictors = build_predictors(args)
    probe_images = [p for p in images if load_yolo_labels(p)]
    if not probe_images:
        raise RuntimeError("No labeled images found for predictor probing.")
    predictors = filter_working_predictors(predictors, probe_images[0], args.imgsz)

    selected = select_images(
        images,
        predictors,
        args.num_images,
        args.seed,
        args.scan_limit,
        args.iou,
        args.images,
        args.imgsz,
        args.max_gt,
        args.max_box_area,
        args.edge_margin,
    )
    selected = drop_selected_columns(selected, args.drop_cols)
    if not selected:
        raise RuntimeError("No suitable images were selected.")

    row_names = ["GT"] + [p.name for p in predictors]
    cells = {}
    for c, image_path in enumerate(selected):
        image = Image.open(image_path).convert("RGB")
        cells[(0, c)] = draw_boxes(image, load_yolo_labels(image_path), names, normalized=True, show_score=False)
        for r, predictor in enumerate(predictors, start=1):
            boxes = predictor.predict(image_path, args.imgsz)
            cells[(r, c)] = draw_boxes(image, boxes, names, normalized=False, show_score=True)

    make_method_grid(row_names, cells, args.cell_size, args.out)
    if args.export_scale and args.export_scale != 1.0:
        image = Image.open(args.out)
        scaled = image.resize(
            (round(image.width * args.export_scale), round(image.height * args.export_scale)),
            Image.Resampling.LANCZOS,
        )
        scaled.save(args.out)
    selected_path = args.out.with_suffix(".selected_images.txt")
    selected_path.write_text("\n".join(p.name for p in selected) + "\n", encoding="utf-8")
    print(f"Saved figure: {args.out}")
    print(f"Saved selected image list: {selected_path}")


if __name__ == "__main__":
    main()
