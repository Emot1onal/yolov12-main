from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



DATASET_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\Dataset1-NWPU\NWPU_VHR10_YOLO")
WEIGHTS = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\detect\train10\weights\best.pt")

IMAGES_DIR = DATASET_ROOT / "images" / "val"
LABELS_DIR = DATASET_ROOT / "labels" / "val"

CONF = 0.25
IOU_THRESH = 0.5
# ============================


def yolo_to_xyxy(label, img_w, img_h):
    cls, xc, yc, w, h = label
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return int(cls), np.array([x1, y1, x2, y2], dtype=float)


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union


def read_labels(label_path, img_w, img_h):
    gt_boxes = []

    if not label_path.exists():
        return gt_boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        values = list(map(float, line.split()))
        cls, box = yolo_to_xyxy(values, img_w, img_h)

        area = (box[2] - box[0]) * (box[3] - box[1])

        gt_boxes.append({
            "class": cls,
            "box": box,
            "area": area,
            "detected": False
        })

    return gt_boxes


def main():
    model = YOLO(str(WEIGHTS))

    all_gt = []

    image_paths = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
        image_paths.extend(list(IMAGES_DIR.glob(ext)))

    for img_path in image_paths:
        img = Image.open(img_path)
        img_w, img_h = img.size

        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        gt_boxes = read_labels(label_path, img_w, img_h)

        if len(gt_boxes) == 0:
            continue

        results = model.predict(
            source=str(img_path),
            conf=CONF,
            iou=IOU_THRESH,
            device=0,
            verbose=False
        )[0]

        pred_boxes = []

        if results.boxes is not None:
            for b in results.boxes:
                cls = int(b.cls.item())
                xyxy = b.xyxy.cpu().numpy()[0]

                pred_boxes.append({
                    "class": cls,
                    "box": xyxy
                })

        used_preds = set()

        for gt in gt_boxes:
            best_iou = 0
            best_pred_idx = -1

            for i, pred in enumerate(pred_boxes):
                if i in used_preds:
                    continue

                if pred["class"] != gt["class"]:
                    continue

                iou = box_iou(gt["box"], pred["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i

            if best_iou >= IOU_THRESH:
                gt["detected"] = True
                used_preds.add(best_pred_idx)

        all_gt.extend(gt_boxes)

    areas = np.array([g["area"] for g in all_gt])
    detected = np.array([g["detected"] for g in all_gt])


    bins = [0, 32*32, 64*64, 96*96, 128*128, 256*256, np.inf]
    bin_names = [
        "0-32²",
        "32²-64²",
        "64²-96²",
        "96²-128²",
        "128²-256²",
        ">256²"
    ]

    recalls = []
    counts = []

    for i in range(len(bins) - 1):
        mask = (areas >= bins[i]) & (areas < bins[i + 1])
        count = mask.sum()

        if count == 0:
            recall = 0
        else:
            recall = detected[mask].sum() / count

        counts.append(count)
        recalls.append(recall)


    plt.figure(figsize=(9, 5))
    plt.plot(bin_names, recalls, marker="o", linewidth=2)
    plt.xlabel("Object Size Range (pixel area)")
    plt.ylabel("Recall@0.5")
    plt.title("Detection Performance by Object Size")
    plt.grid(True)
    plt.ylim(0, 1.05)

    for i, r in enumerate(recalls):
        plt.text(i, r + 0.03, f"{r:.2f}", ha="center")

    plt.savefig("performance_by_object_size.png", dpi=300, bbox_inches="tight")
    plt.show()


    plt.figure(figsize=(9, 5))
    plt.bar(bin_names, counts)
    plt.xlabel("Object Size Range (pixel area)")
    plt.ylabel("Number of Objects")
    plt.title("Number of Objects by Size")
    plt.grid(axis="y")

    for i, c in enumerate(counts):
        plt.text(i, c + 1, str(c), ha="center")

    plt.savefig("object_count_by_size.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:")
    print("performance_by_object_size.png")
    print("object_count_by_size.png")


if __name__ == "__main__":
    main()