import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".yolo_config"))

from ultralytics import YOLO


DEFAULT_NWPU = Path(r"C:\Users\14288\OneDrive\Desktop\Dataset1-NWPU\NWPU_VHR10_YOLO\images\test")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO FPS on a fixed image folder.")
    parser.add_argument("--images", type=Path, default=DEFAULT_NWPU, help="Image folder for FPS test.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path(r"..\runs\fps_benchmark.csv"))
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "WEIGHTS"),
        required=True,
        help="Model name and weights path. Can be used multiple times.",
    )
    return parser.parse_args()


def collect_images(image_dir, max_images):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in suffixes)
    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_one(name, weights, images, args):
    model = YOLO(str(weights))

    warmup_images = images[: min(args.warmup, len(images))]
    for image_path in warmup_images:
        model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
    sync_cuda()

    fps_values = []
    ms_values = []
    for repeat_idx in range(args.repeat):
        sync_cuda()
        start = time.perf_counter()
        for image_path in images:
            model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
        sync_cuda()
        elapsed = time.perf_counter() - start
        fps = len(images) / elapsed
        ms = elapsed * 1000.0 / len(images)
        fps_values.append(fps)
        ms_values.append(ms)
        print(f"{name} repeat {repeat_idx + 1}/{args.repeat}: {fps:.2f} FPS, {ms:.2f} ms/img")

    return {
        "model": name,
        "weights": str(weights),
        "images": len(images),
        "imgsz": args.imgsz,
        "device": args.device,
        "fps_mean": statistics.mean(fps_values),
        "fps_std": statistics.pstdev(fps_values) if len(fps_values) > 1 else 0.0,
        "ms_mean": statistics.mean(ms_values),
        "ms_std": statistics.pstdev(ms_values) if len(ms_values) > 1 else 0.0,
    }


def main():
    args = parse_args()
    images = collect_images(args.images, args.max_images)
    print(f"Images: {len(images)} from {args.images}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    for name, weights in args.model:
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        rows.append(benchmark_one(name, weights_path, images, args))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved FPS summary: {args.out}")
    for row in rows:
        print(f"{row['model']}: {row['fps_mean']:.2f} FPS ({row['ms_mean']:.2f} ms/img)")


if __name__ == "__main__":
    main()
