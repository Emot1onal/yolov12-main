import argparse
import csv
import re
from pathlib import Path

from ultralytics import YOLO


DEFAULT_DATA = r"C:\Users\14288\OneDrive\Desktop\Dataset1-NWPU\NWPU_VHR10_YOLO\data.yaml"
DEFAULT_RUNS = r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\detect"
DEFAULT_EVAL_PROJECT = r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\eval_NWPU"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate every saved checkpoint for an NWPU ablation run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory, e.g. C:/.../runs/detect/exp2_sum_all",
    )
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA), help="NWPU data.yaml path.")
    parser.add_argument(
        "--split",
        default="auto",
        choices=("auto", "train", "val", "test"),
        help="Dataset split to evaluate. auto uses test if data.yaml has test, otherwise val.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_EVAL_PROJECT), help="Evaluation output root.")
    parser.add_argument("--include-best-last", action="store_true", help="Also evaluate best.pt and last.pt.")
    parser.add_argument("--no-plots", action="store_true", help="Disable per-checkpoint plots to save time and files.")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing eval folders with the same names.")
    return parser.parse_args()


def yaml_has_key(path, key):
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:")
    return any(pattern.match(line) for line in path.read_text(encoding="utf-8").splitlines())


def resolve_split(data_yaml, requested):
    if requested != "auto":
        return requested
    return "test" if yaml_has_key(data_yaml, "test") else "val"


def checkpoint_sort_key(path):
    match = re.search(r"epoch(\d+)", path.stem)
    if match:
        return (0, int(match.group(1)))
    if path.stem == "best":
        return (1, 0)
    if path.stem == "last":
        return (2, 0)
    return (3, path.stem)


def checkpoint_tag(path):
    match = re.search(r"epoch(\d+)", path.stem)
    if match:
        return f"e{int(match.group(1)):03d}"
    return path.stem


def collect_checkpoints(weights_dir, include_best_last):
    checkpoints = sorted(weights_dir.glob("epoch*.pt"), key=checkpoint_sort_key)
    if include_best_last:
        for name in ("best.pt", "last.pt"):
            path = weights_dir / name
            if path.exists():
                checkpoints.append(path)
    return checkpoints


def metric_value(metrics, name, default=0.0):
    return float(getattr(metrics.box, name, default))


def main():
    args = parse_args()
    run_dir = args.run_dir
    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"weights directory not found: {weights_dir}")
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    split = resolve_split(args.data, args.split)
    checkpoints = collect_checkpoints(weights_dir, args.include_best_last)
    if not checkpoints:
        raise FileNotFoundError(f"No epoch*.pt checkpoints found in {weights_dir}")

    eval_root = args.project / f"{run_dir.name}_{split}"
    eval_root.mkdir(parents=True, exist_ok=True)
    summary_path = eval_root / f"{run_dir.name}_{split}_checkpoint_metrics.csv"

    print(f"Run directory: {run_dir}")
    print(f"Data yaml: {args.data}")
    print(f"Requested split: {args.split}; actual split: {split}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Summary CSV: {summary_path}")

    rows = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        tag = checkpoint_tag(checkpoint)
        eval_name = f"{run_dir.name}_{split}_{tag}"
        print(f"\n[{index}/{len(checkpoints)}] Evaluating {checkpoint.name} -> {eval_name}")

        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(args.data),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(eval_root),
            name=eval_name,
            exist_ok=args.exist_ok,
            plots=not args.no_plots,
        )

        rows.append(
            {
                "run": run_dir.name,
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "eval_dir": str(eval_root / eval_name),
                "split": split,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "precision": metric_value(metrics, "mp"),
                "recall": metric_value(metrics, "mr"),
                "map50": metric_value(metrics, "map50"),
                "map50_95": metric_value(metrics, "map"),
            }
        )

        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    best_map50 = max(rows, key=lambda x: x["map50"])
    best_map = max(rows, key=lambda x: x["map50_95"])
    print("\nDone.")
    print(f"Best mAP50: {best_map50['map50']:.6f} from {best_map50['checkpoint']}")
    print(f"Best mAP50-95: {best_map['map50_95']:.6f} from {best_map['checkpoint']}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
