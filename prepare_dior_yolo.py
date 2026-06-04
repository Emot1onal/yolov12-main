from pathlib import Path
import shutil
import sys



SOURCE_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\DIOR")


OUTPUT_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\DIOR_YOLO")


CLASS_NAMES = [
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
    "windmill"
]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_split_file(split_file: Path):
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def find_image_file(images_dir: Path, stem: str):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for ext in exts:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def validate_yolo_label_file(label_path: Path):
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return True, "empty"

    for line_num, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            return False, f"{label_path.name} line {line_num}: expected 5 fields, got {len(parts)}"

        try:
            class_id = int(parts[0])
        except ValueError:
            return False, f"{label_path.name} line {line_num}: class_id is not int"

        try:
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            return False, f"{label_path.name} line {line_num}: box values are not numeric"

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            return False, f"{label_path.name} line {line_num}: box values not in [0,1]"

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            return False, f"{label_path.name} line {line_num}: class_id {class_id} out of range"

    return True, "ok"


def write_data_yaml(output_root: Path, class_names):
    yaml_path = output_root / "data.yaml"

    lines = [
        f'path: "{output_root.as_posix()}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:"
    ]

    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] data.yaml created: {yaml_path}")


def copy_one_split(split_name: str, source_root: Path, output_root: Path):
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    split_file = source_root / "ImageSets" / f"{split_name}.txt"

    if not split_file.exists():
        print(f"[WARNING] split file not found: {split_file}")
        return

    out_img_dir = output_root / "images" / split_name
    out_lbl_dir = output_root / "labels" / split_name
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    image_ids = read_split_file(split_file)

    missing_images = []
    missing_labels = []
    bad_labels = []
    copied = 0

    for stem in image_ids:
        image_path = find_image_file(images_dir, stem)
        label_path = labels_dir / f"{stem}.txt"

        if image_path is None:
            missing_images.append(stem)
            continue

        if not label_path.exists():
            missing_labels.append(stem)
            continue

        ok, msg = validate_yolo_label_file(label_path)
        if not ok:
            bad_labels.append(msg)
            continue

        shutil.copy2(image_path, out_img_dir / image_path.name)
        shutil.copy2(label_path, out_lbl_dir / label_path.name)
        copied += 1

    print(f"\n[{split_name}]")
    print(f"copied: {copied}")
    print(f"missing images: {len(missing_images)}")
    print(f"missing labels: {len(missing_labels)}")
    print(f"bad labels: {len(bad_labels)}")

    if missing_images:
        print("example missing images:", missing_images[:10])

    if missing_labels:
        print("example missing labels:", missing_labels[:10])

    if bad_labels:
        print("example bad labels:")
        for item in bad_labels[:10]:
            print("  ", item)


def main():
    print("========== PREPARE DIOR FOR YOLO ==========")
    print(f"Source root : {SOURCE_ROOT}")
    print(f"Output root : {OUTPUT_ROOT}")

    if not SOURCE_ROOT.exists():
        print(f"[ERROR] Source dataset not found: {SOURCE_ROOT}")
        sys.exit(1)

    required_dirs = [
        SOURCE_ROOT / "images",
        SOURCE_ROOT / "labels",
        SOURCE_ROOT / "ImageSets",
    ]
    for d in required_dirs:
        if not d.exists():
            print(f"[ERROR] Required folder not found: {d}")
            sys.exit(1)

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(OUTPUT_ROOT / "images")
    ensure_dir(OUTPUT_ROOT / "labels")

    for split in ["train", "val", "test"]:
        copy_one_split(split, SOURCE_ROOT, OUTPUT_ROOT)

    write_data_yaml(OUTPUT_ROOT, CLASS_NAMES)

    print("\n========== DONE ==========")
    print(f"YOLO dataset saved to:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()