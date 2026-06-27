'''
import argparse
import hashlib
import shutil
from pathlib import Path


DEFAULT_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\Dataset1-NWPU\NWPU_VHR10_YOLO")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Make the existing NWPU test split strict by removing the copied test samples from val. "
            "Files are moved into a backup folder instead of being deleted."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="NWPU YOLO dataset root.")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Backup directory. Defaults to <root>/_val_removed_for_strict_test.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be moved without changing files.")
    parser.add_argument(
        "--force-name-match",
        action="store_true",
        help="Move matching val files by filename even if image hashes differ. Not recommended.",
    )
    return parser.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_files(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def move_to_backup(src, dst, dry_run):
    if dry_run:
        print(f"[dry-run] move {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Backup target already exists: {dst}")
    shutil.move(str(src), str(dst))


def remove_cache(root, dry_run):
    for cache in (root / "labels").glob("*.cache"):
        if cache.name in {"val.cache", "test.cache", "train.cache"}:
            if dry_run:
                print(f"[dry-run] remove cache {cache}")
            else:
                cache.unlink(missing_ok=True)


def ensure_data_yaml_has_test(root):
    data_yaml = root / "data.yaml"
    text = data_yaml.read_text(encoding="utf-8")
    if "test:" not in text:
        lines = text.splitlines()
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
    backup = args.backup_dir or (root / "_val_removed_for_strict_test")

    val_images = root / "images" / "val"
    val_labels = root / "labels" / "val"
    test_images = root / "images" / "test"
    test_labels = root / "labels" / "test"

    for path in (val_images, val_labels, test_images, test_labels):
        if not path.exists():
            raise FileNotFoundError(f"Required directory not found: {path}")

    moved = 0
    skipped_missing = 0
    skipped_hash = 0

    for test_img in image_files(test_images):
        val_img = val_images / test_img.name
        val_label = val_labels / f"{test_img.stem}.txt"
        test_label = test_labels / f"{test_img.stem}.txt"

        if not val_img.exists():
            skipped_missing += 1
            continue

        same_image = sha256(test_img) == sha256(val_img)
        if not same_image and not args.force_name_match:
            print(f"[skip] Same filename but different image content: {val_img}")
            skipped_hash += 1
            continue

        move_to_backup(val_img, backup / "images" / "val" / val_img.name, args.dry_run)
        if val_label.exists():
            move_to_backup(val_label, backup / "labels" / "val" / val_label.name, args.dry_run)
        elif test_label.exists():
            print(f"[warn] Test label exists but val label is missing: {val_label}")
        moved += 1

    if not args.dry_run:
        ensure_data_yaml_has_test(root)
    remove_cache(root, args.dry_run)

    train_count = len(image_files(root / "images" / "train"))
    val_count = len(image_files(val_images))
    test_count = len(image_files(test_images))

    print("\nStrict split summary")
    print(f"Dataset root: {root}")
    print(f"Moved val samples to backup: {moved}")
    print(f"Skipped missing in val: {skipped_missing}")
    print(f"Skipped hash mismatch: {skipped_hash}")
    print(f"Backup dir: {backup}")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Test images: {test_count}")
    print("Expected strict counts after default split: train=537, val=107, test=50")


if __name__ == "__main__":
    main()
'''