import argparse
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create paper-style dataset and detection visualization figures.")
    sub = parser.add_subparsers(dest="mode", required=True)

    samples = sub.add_parser("samples", help="Draw ground-truth boxes and make an image grid.")
    samples.add_argument("--data", type=Path, required=True)
    samples.add_argument("--split", default="test")
    samples.add_argument("--out", type=Path, required=True)
    samples.add_argument("--cols", type=int, default=5)
    samples.add_argument("--rows", type=int, default=3)
    samples.add_argument("--seed", type=int, default=7)
    samples.add_argument("--image-size", type=int, default=220)
    samples.add_argument("--cell-width", type=int, default=None)
    samples.add_argument("--cell-height", type=int, default=None)
    samples.add_argument("--fit", action="store_true", help="Fit the full image into each cell instead of center-cropping.")

    compare = sub.add_parser("compare", help="Compare GT and model predictions in a grid.")
    compare.add_argument("--data", type=Path, required=True)
    compare.add_argument("--split", default="test")
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--weights", nargs="+", required=True, help="Pairs like YOLOv12=path/to/best.pt Ours=path/to/best.pt")
    compare.add_argument("--images", nargs="*", default=None, help="Optional image filenames to visualize.")
    compare.add_argument("--auto-select", action="store_true", help="Automatically select images where Ours improves over baseline.")
    compare.add_argument("--select-limit", type=int, default=0, help="Maximum number of images to scan. 0 means all images.")
    compare.add_argument("--iou", type=float, default=0.5, help="IoU threshold used for auto-selection scoring.")
    compare.add_argument("--num-images", type=int, default=3)
    compare.add_argument("--seed", type=int, default=3)
    compare.add_argument("--conf", type=float, default=0.25)
    compare.add_argument("--imgsz", type=int, default=640)
    compare.add_argument("--image-size", type=int, default=260)
    compare.add_argument("--cell-width", type=int, default=None)
    compare.add_argument("--cell-height", type=int, default=None)
    compare.add_argument("--fit", action="store_true", help="Fit the full image into each cell instead of center-cropping.")

    return parser.parse_args()


def read_data_yaml(path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("Please install pyyaml or run inside the YOLO environment.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data["path"])
    names = data["names"]
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


def font(size=14):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def color_for_class(cls):
    palette = [
        (0, 160, 255),
        (0, 210, 120),
        (255, 176, 0),
        (255, 84, 84),
        (172, 112, 255),
        (0, 200, 210),
        (255, 120, 0),
        (120, 180, 60),
    ]
    return palette[cls % len(palette)]


def draw_boxes(image, boxes, names, normalized=True, score=False):
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    fnt = font(max(11, im.width // 32))
    w, h = im.size
    for box in boxes:
        if normalized:
            cls, xc, yc, bw, bh = box
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            label = names[cls] if cls < len(names) else str(cls)
        else:
            cls, x1, y1, x2, y2, conf = box
            label = names[cls] if cls < len(names) else str(cls)
            if score:
                label = f"{label} {conf:.2f}"
        color = color_for_class(cls)
        width = max(2, im.width // 180)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        bbox = draw.textbbox((0, 0), label, font=fnt)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=fnt)
    return im


def cell_size_from_args(args):
    width = args.cell_width or args.image_size
    height = args.cell_height or args.image_size
    return width, height


def resize_crop(image, size):
    image = image.convert("RGB")
    width, height = size
    scale = max(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def resize_fit(image, size):
    image = image.convert("RGB")
    width, height = size
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def make_grid(cells, rows, cols, cell_size, headers=None, captions=None, fit=False):
    cell_w, cell_h = cell_size
    header_h = 32 if headers else 0
    caption_h = 26 if captions else 0
    canvas = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h + caption_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fnt = font(15)
    if headers:
        for c, text in enumerate(headers):
            x = c * cell_w
            draw.text((x + 8, 8), text, fill=(0, 0, 0), font=fnt)
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        resized = resize_fit(cell, cell_size) if fit else resize_crop(cell, cell_size)
        canvas.paste(resized, (c * cell_w, header_h + r * cell_h))
    if captions:
        y = header_h + rows * cell_h + 5
        for c, text in enumerate(captions):
            bbox = draw.textbbox((0, 0), text, font=fnt)
            x = c * cell_w + (cell_w - (bbox[2] - bbox[0])) // 2
            draw.text((x, y), text, fill=(0, 0, 0), font=fnt)
    return canvas


def choose_images(images, n, seed, requested=None):
    if requested:
        by_name = {p.name: p for p in images}
        chosen = []
        for name in requested:
            if name not in by_name:
                raise FileNotFoundError(f"Image {name} not found.")
            chosen.append(by_name[name])
        return chosen
    labeled = [p for p in images if load_yolo_labels(p)]
    random.Random(seed).shuffle(labeled)
    return labeled[:n]


def run_samples(args):
    root, data, names = read_data_yaml(args.data)
    images = choose_images(list_images(root, data, args.split), args.rows * args.cols, args.seed)
    cells = []
    for image_path in images:
        image = Image.open(image_path)
        cells.append(draw_boxes(image, load_yolo_labels(image_path), names, normalized=True))
    grid = make_grid(cells, args.rows, args.cols, cell_size_from_args(args), fit=args.fit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"Saved: {args.out}")


def parse_weights(items):
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError("Weights must be written as Label=path/to/weights.pt")
        label, path = item.split("=", 1)
        parsed.append((label, Path(path)))
    return parsed


def predict_boxes(model, image_path, conf, imgsz):
    results = model.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    boxes = []
    if not results:
        return boxes
    result = results[0]
    for box in result.boxes:
        cls = int(box.cls.item())
        conf_value = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        boxes.append((cls, x1, y1, x2, y2, conf_value))
    return boxes


def gt_to_xyxy(labels, width, height):
    boxes = []
    for cls, xc, yc, bw, bh in labels:
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        boxes.append((cls, x1, y1, x2, y2, 1.0))
    return boxes


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
    return {"tp": tp, "fp": fp, "fn": fn, "conf_sum": conf_sum}


def score_difference(gt_boxes, baseline_boxes, ours_boxes, iou_thr):
    base = detection_stats(gt_boxes, baseline_boxes, iou_thr)
    ours = detection_stats(gt_boxes, ours_boxes, iou_thr)
    score = (
        (ours["tp"] - base["tp"]) * 10.0
        + (base["fn"] - ours["fn"]) * 5.0
        + (base["fp"] - ours["fp"]) * 1.5
        + (ours["conf_sum"] - base["conf_sum"]) * 0.5
    )
    return score, base, ours


def auto_select_images(images, models, num_images, conf, imgsz, iou_thr, limit):
    if len(models) < 2:
        raise ValueError("--auto-select needs at least two models: baseline and ours.")
    baseline_model = models[0][1]
    ours_model = models[-1][1]
    candidates = [p for p in images if load_yolo_labels(p)]
    if limit and limit > 0:
        candidates = candidates[:limit]
    scored = []
    for idx, image_path in enumerate(candidates, start=1):
        image = Image.open(image_path)
        gt_boxes = gt_to_xyxy(load_yolo_labels(image_path), image.width, image.height)
        baseline_boxes = predict_boxes(baseline_model, image_path, conf, imgsz)
        ours_boxes = predict_boxes(ours_model, image_path, conf, imgsz)
        score, base, ours = score_difference(gt_boxes, baseline_boxes, ours_boxes, iou_thr)
        scored.append((score, image_path, base, ours))
        if idx % 20 == 0:
            print(f"Scanned {idx}/{len(candidates)} images...")
    scored.sort(
        key=lambda item: (
            item[0],
            item[3]["tp"] - item[2]["tp"],
            item[2]["fn"] - item[3]["fn"],
            item[2]["fp"] - item[3]["fp"],
        ),
        reverse=True,
    )
    selected = [item[1] for item in scored[:num_images]]
    print("Selected images:")
    for score, image_path, base, ours in scored[:num_images]:
        print(
            f"  {image_path.name}: score={score:.3f}, "
            f"baseline TP/FP/FN={base['tp']}/{base['fp']}/{base['fn']}, "
            f"ours TP/FP/FN={ours['tp']}/{ours['fp']}/{ours['fn']}"
        )
    return selected


def run_compare(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".yolo_config"))

    from ultralytics import YOLO

    root, data, names = read_data_yaml(args.data)
    all_images = list_images(root, data, args.split)
    weights = parse_weights(args.weights)
    models = [(label, YOLO(str(path))) for label, path in weights]
    if args.auto_select:
        images = auto_select_images(all_images, models, args.num_images, args.conf, args.imgsz, args.iou, args.select_limit)
    else:
        images = choose_images(all_images, args.num_images, args.seed, args.images)

    headers = ["GT"] + [label for label, _ in models]
    cells = []
    for image_path in images:
        image = Image.open(image_path)
        cells.append(draw_boxes(image, load_yolo_labels(image_path), names, normalized=True))
        for _, model in models:
            boxes = predict_boxes(model, image_path, args.conf, args.imgsz)
            cells.append(draw_boxes(image, boxes, names, normalized=False, score=True))

    rows = len(images)
    cols = 1 + len(models)
    captions = [f"({chr(97 + i)})" for i in range(cols)]
    grid = make_grid(cells, rows, cols, cell_size_from_args(args), headers=headers, captions=captions, fit=args.fit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"Saved: {args.out}")


def main():
    args = parse_args()
    if args.mode == "samples":
        run_samples(args)
    elif args.mode == "compare":
        run_compare(args)


if __name__ == "__main__":
    main()
