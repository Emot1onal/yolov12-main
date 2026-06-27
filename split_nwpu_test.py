import argparse
import random
import shutil
from pathlib import Path


DEFAULT_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\Dataset1-NWPU\NWPU_VHR10_YOLO")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a reproducible NWPU test split from the existing val split.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="NWPU YOLO dataset root.")
    parser.add_argument("--test-count", type=int, default=50, help="Number of validation images to copy into test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting.")
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="copy keeps val unchanged; move removes selected samples from val.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing test split.")
    return parser.parse_args()


def image_files(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def copy_or_move(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def update_data_yaml(data_yaml):
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    has_test = any(line.strip().startswith("test:") for line in lines)
    if not has_test:
        out = []
        inserted = False
        for line in lines:
            out.append(line)
            if line.strip().startswith("val:"):
                out.append("test: images/test")
                inserted = True
        if not inserted:
            out.append("test: images/test")
        data_yaml.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = args.root
    val_images = root / "images" / "val"
    val_labels = root / "labels" / "val"
    test_images = root / "images" / "test"
    test_labels = root / "labels" / "test"
    data_yaml = root / "data.yaml"

    if not val_images.exists() or not val_labels.exists():
        raise FileNotFoundError("Expected images/val and labels/val under the dataset root.")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    if (test_images.exists() or test_labels.exists()) and not args.overwrite:
        raise FileExistsError("Test split already exists. Use --overwrite to replace it.")

    if args.overwrite:
        if test_images.exists():
            shutil.rmtree(test_images)
        if test_labels.exists():
            shutil.rmtree(test_labels)

    val_files = image_files(val_images)
    if args.test_count <= 0 or args.test_count >= len(val_files):
        raise ValueError(f"--test-count must be between 1 and {len(val_files) - 1}.")

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(val_files, args.test_count), key=lambda p: p.name)

    missing_labels = []
    for img in selected:
        label = val_labels / f"{img.stem}.txt"
        if not label.exists():
            missing_labels.append(label)
            continue
        copy_or_move(img, test_images / img.name, args.mode)
        copy_or_move(label, test_labels / label.name, args.mode)

    if missing_labels:
        missing = "\n".join(str(p) for p in missing_labels[:10])
        raise FileNotFoundError(f"Missing labels for selected images, first entries:\n{missing}")

    update_data_yaml(data_yaml)

    print(f"Dataset root: {root}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Test images: {len(image_files(test_images))}")
    print(f"Val images remaining: {len(image_files(val_images))}")
    print(f"Updated data.yaml:\n{data_yaml.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
