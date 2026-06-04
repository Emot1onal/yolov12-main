from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO


WEIGHTS = r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\detect\train11\weights\best.pt"
IMAGE = r"C:\Users\14288\OneDrive\Desktop\Dataset1-NWPU\NWPU_VHR10_YOLO\images\val\006.jpg"
OUT_DIR = r"C:\Users\14288\OneDrive\Desktop\yolov12\aux_heatmap_vis"
IMG_SIZE = 640


def load_yolo_label(image_path):
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
    else:
        label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, x, y, w, h = map(float, parts[:5])
        boxes.append((int(cls), x, y, w, h))
    return boxes


def draw_boxes(image, labels):
    h, w = image.shape[:2]
    out = image.copy()
    for cls, x, y, bw, bh in labels:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(out, str(cls), (x1, max(y1 - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def normalize_map(x):
    x = x.astype(np.float32)
    x = x - x.min()
    denom = x.max() + 1e-6
    return x / denom


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(WEIGHTS).model
    model.eval()
    model.return_aux = True
    model.force_aux = True

    image_path = Path(IMAGE)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    labels = load_yolo_label(image_path)
    boxed = draw_boxes(resized, labels)

    rgb = resized[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        _, aux_outputs = model(tensor)

    maps = aux_outputs["maps"]
    names = [f"heatmap{i}" for i in range(1, len(maps) + 1)]

    cv2.imwrite(str(out_dir / "input_with_gt_boxes.jpg"), boxed)

    for name, aux_map in zip(names, maps):
        hm = aux_map[0, 0].detach().float().cpu().numpy()
        hm = cv2.resize(hm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        hm = normalize_map(hm)
        hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(boxed, 0.55, hm_color, 0.45, 0)

        cv2.imwrite(str(out_dir / f"{name}_raw.jpg"), (hm * 255).astype(np.uint8))
        cv2.imwrite(str(out_dir / f"{name}_overlay.jpg"), overlay)
        print(f"{name}: min={hm.min():.4f}, max={hm.max():.4f}, saved overlay")

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
