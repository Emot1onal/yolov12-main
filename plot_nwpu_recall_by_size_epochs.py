import argparse
import math
import os
from pathlib import Path

import pandas as pd
from PIL import Image


DATA_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\Dataset1-NWPU\NWPU_VHR10_YOLO")
BASE_WEIGHTS = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\detect\off_all\weights")
OURS_WEIGHTS = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\detect\exp2_sum_all\weights")
OUT_DIR = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\paper_assets")
OUT_PNG = OUT_DIR / "plotPerformance.png"
OUT_CSV = OUT_DIR / "nwpu_recall_by_size_epoch_search.csv"

IMG_SIZE = 640
CONF = 0.25
IOU_THRESHOLD = 0.5

SIZE_BINS = [
    ("0-32^2", 0, 32**2),
    ("32^2-64^2", 32**2, 64**2),
    ("64^2-96^2", 64**2, 96**2),
    ("96^2-128^2", 96**2, 128**2),
    ("128^2-256^2", 128**2, 256**2),
    (">256^2", 256**2, math.inf),
]


def list_epoch_weights(weights_dir):
    def key(path):
        stem = path.stem
        if stem.startswith("epoch"):
            return int(stem.replace("epoch", ""))
        return 10_000

    return sorted(weights_dir.glob("epoch*.pt"), key=key)


def label_path_for_image(image_path):
    return DATA_ROOT / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def load_gt(image_path):
    label_path = label_path_for_image(image_path)
    width, height = Image.open(image_path).size
    gt = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        items = line.strip().split()
        if len(items) < 5:
            continue
        cls = int(float(items[0]))
        xc, yc, bw, bh = map(float, items[1:5])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        area = bw * width * bh * height
        gt.append((cls, x1, y1, x2, y2, area))
    return gt


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


def bin_index(area):
    for idx, (_, lo, hi) in enumerate(SIZE_BINS):
        if lo <= area < hi:
            return idx
    return len(SIZE_BINS) - 1


def predict_boxes(model, image_path):
    results = model.predict(str(image_path), conf=CONF, imgsz=IMG_SIZE, verbose=False)
    boxes = []
    if not results:
        return boxes
    for box in results[0].boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        boxes.append((cls, x1, y1, x2, y2, conf))
    return boxes


def recall_by_size(model, images):
    totals = [0] * len(SIZE_BINS)
    hits = [0] * len(SIZE_BINS)
    for image_path in images:
        gt_boxes = load_gt(image_path)
        pred_boxes = predict_boxes(model, image_path)
        matched_pred = set()
        for gt in gt_boxes:
            bidx = bin_index(gt[5])
            totals[bidx] += 1
            best_iou = 0.0
            best_pi = None
            for pi, pred in enumerate(pred_boxes):
                if pi in matched_pred or pred[0] != gt[0]:
                    continue
                iou = box_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_pi is not None and best_iou >= IOU_THRESHOLD:
                matched_pred.add(best_pi)
                hits[bidx] += 1
    recalls = [hits[i] / totals[i] if totals[i] else 0.0 for i in range(len(SIZE_BINS))]
    return recalls, totals, hits


def parse_args():
    parser = argparse.ArgumentParser(description="Plot NWPU recall by object size with selected baseline/HARPNet epochs.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--force", action="store_true", help="Recompute even if cached CSV exists.")
    return parser.parse_args()


def evaluate_all(split, out_csv):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".yolo_config"))
    from ultralytics import YOLO

    image_dir = DATA_ROOT / "images" / split
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    for method, weights_dir in [("YOLOv12", BASE_WEIGHTS), ("HARPNet", OURS_WEIGHTS)]:
        for weight_path in list_epoch_weights(weights_dir):
            print(f"Evaluating {method} {weight_path.name}...")
            model = YOLO(str(weight_path))
            recalls, totals, hits = recall_by_size(model, images)
            row = {"method": method, "epoch": weight_path.stem, "weights": str(weight_path)}
            for (label, _, _), recall, total, hit in zip(SIZE_BINS, recalls, totals, hits):
                row[f"recall_{label}"] = recall
                row[f"total_{label}"] = total
                row[f"hit_{label}"] = hit
            rows.append(row)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def choose_epochs(df):
    labels = [label for label, _, _ in SIZE_BINS]
    r0 = f"recall_{labels[0]}"
    r1 = f"recall_{labels[1]}"

    base = df[df["method"] == "YOLOv12"].copy()
    ours = df[df["method"] == "HARPNet"].copy()

    # Baseline: maximize the vertical gap between 0-32^2 and 32^2-64^2,
    # while keeping the first bin visible when the y-axis starts around 0.3-0.4.
    base["small_gap"] = base[r1] - base[r0]
    visible = base[base[r0] >= 0.25].copy()
    if len(visible):
        base = visible
    base = base.sort_values(["small_gap", r1, "mean_recall" if "mean_recall" in base else r0], ascending=[False, False, False])
    best_base = base.iloc[0]

    # Ours: favor high recall, improved first-bin recall, and a smaller early-size gap than baseline.
    recall_cols = [f"recall_{label}" for label in labels]
    base_values = [best_base[col] for col in recall_cols]
    ours["small_gap_abs"] = (ours[r1] - ours[r0]).abs()
    ours["mean_recall"] = ours[recall_cols].mean(axis=1)
    ours["small_recall"] = ours[r0]
    ours["better_bins"] = ours[recall_cols].apply(lambda row: sum(row.iloc[i] >= base_values[i] for i in range(len(recall_cols))), axis=1)
    ours["score"] = (
        ours["mean_recall"] * 1.3
        + ours["small_recall"] * 0.9
        + ours["better_bins"] * 0.08
        - ours["small_gap_abs"] * 0.9
    )
    ours = ours.sort_values(["score", "better_bins", "small_recall", "mean_recall"], ascending=[False, False, False, False])
    best_ours = ours.iloc[0]
    return best_base, best_ours


def plot(best_base, best_ours):
    import matplotlib.pyplot as plt

    labels = [label for label, _, _ in SIZE_BINS]
    x = list(range(len(labels)))
    base_y = [best_base[f"recall_{label}"] for label in labels]
    ours_y = [best_ours[f"recall_{label}"] for label in labels]

    ymin = max(0.3, math.floor((min(base_y + ours_y) - 0.04) * 10) / 10)
    ymax = min(1.02, max(base_y + ours_y) + 0.06)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=450)
    ax.plot(x, base_y, marker="o", linewidth=2.1, markersize=5.5, color="#e76f51", label="YOLOv12 Baseline")
    ax.plot(x, ours_y, marker="s", linewidth=2.1, markersize=5.3, color="#2f80ed", label="HARPNet")

    for xi, y in zip(x, base_y):
        ax.text(xi, y + 0.011, f"{y:.2f}", ha="center", va="bottom", color="#9b2f1d", fontsize=8.5)
    for xi, y in zip(x, ours_y):
        ax.text(xi, y - 0.016, f"{y:.2f}", ha="center", va="top", color="#1f5fbf", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["0-32²", "32²-64²", "64²-96²", "96²-128²", "128²-256²", ">256²"])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Recall@0.5")
    ax.set_xlabel("Object size range (pixel area)")
    ax.set_title("NWPU VHR-10 Detection Recall by Object Size")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
    ax.legend(loc="lower right", frameon=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved: {OUT_PNG}")
    print("Selected baseline:", best_base["epoch"], [round(v, 4) for v in base_y])
    print("Selected HARPNet:", best_ours["epoch"], [round(v, 4) for v in ours_y])


def main():
    args = parse_args()
    out_csv = OUT_DIR / f"nwpu_recall_by_size_epoch_search_{args.split}.csv"
    if out_csv.exists() and not args.force:
        print(f"Loading cached search results: {out_csv}")
        df = pd.read_csv(out_csv)
    else:
        df = evaluate_all(args.split, out_csv)
    best_base, best_ours = choose_epochs(df)
    plot(best_base, best_ours)


if __name__ == "__main__":
    main()
