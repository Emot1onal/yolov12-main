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

    compare = sub.add_parser("compare", help="Compare GT and model predictions in a grid.")
    compare.add_argument("--data", type=Path, required=True)
    compare.add_argument("--split", default="test")
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--weights", nargs="+", required=True, help="Pairs like YOLOv12=path/to/best.pt Ours=path/to/best.pt")
    compare.add_argument("--images", nargs="*", default=None, help="Optional image filenames to visualize.")
    compare.add_argument("--num-images", type=int, default=3)
    compare.add_argument("--seed", type=int, default=3)
    compare.add_argument("--conf", type=float, default=0.25)
    compare.add_argument("--imgsz", type=int, default=640)
    compare.add_argument("--image-size", type=int, default=260)

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


def resize_crop(image, size):
    image = image.convert("RGB")
    scale = max(size / image.width, size / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def make_grid(cells, rows, cols, cell_size, headers=None, captions=None):
    header_h = 32 if headers else 0
    caption_h = 26 if captions else 0
    canvas = Image.new("RGB", (cols * cell_size, header_h + rows * cell_size + caption_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fnt = font(15)
    if headers:
        for c, text in enumerate(headers):
            x = c * cell_size
            draw.text((x + 8, 8), text, fill=(0, 0, 0), font=fnt)
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        canvas.paste(resize_crop(cell, cell_size), (c * cell_size, header_h + r * cell_size))
    if captions:
        y = header_h + rows * cell_size + 5
        for c, text in enumerate(captions):
            bbox = draw.textbbox((0, 0), text, font=fnt)
            x = c * cell_size + (cell_size - (bbox[2] - bbox[0])) // 2
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
    grid = make_grid(cells, args.rows, args.cols, args.image_size)
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


def run_compare(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".yolo_config"))

    from ultralytics import YOLO

    root, data, names = read_data_yaml(args.data)
    images = choose_images(list_images(root, data, args.split), args.num_images, args.seed, args.images)
    weights = parse_weights(args.weights)
    models = [(label, YOLO(str(path))) for label, path in weights]

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
    grid = make_grid(cells, rows, cols, args.image_size, headers=headers, captions=captions)
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
