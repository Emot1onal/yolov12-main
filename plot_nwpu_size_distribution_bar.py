from pathlib import Path

from PIL import Image


NWPU_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\Dataset1-NWPU\NWPU_VHR10_YOLO")
DIOR_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\DIOR_YOLO")
ASSET_DIR = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\paper_assets")


def collect_areas(root):
    areas = []
    for label_path in sorted((root / "labels").glob("*/*.txt")):
        split = label_path.parent.name
        image_path = root / "images" / split / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        width, height = Image.open(image_path).size
        for line in label_path.read_text(encoding="utf-8").splitlines():
            items = line.strip().split()
            if len(items) < 5:
                continue
            box_w = float(items[3]) * width
            box_h = float(items[4]) * height
            areas.append(box_w * box_h)
    return areas


def plot_distribution(root, out, title):
    import matplotlib.pyplot as plt

    areas = collect_areas(root)
    bins = [
        ("Small\n(<$64^2$ px)", lambda a: a < 64**2),
        ("Medium\n($64^2$-$128^2$ px)", lambda a: 64**2 <= a < 128**2),
        ("Large\n($\\geq 128^2$ px)", lambda a: a >= 128**2),
    ]
    counts = [sum(check(a) for a in areas) for _, check in bins]
    total = sum(counts)
    ratios = [c / total * 100 for c in counts]

    labels = [name for name, _ in bins]
    x = [0.0, 0.52, 1.04]
    colors = ["#2f80ed", "#8fc5ff", "#d7e9ff"]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(3.15, 4.4), dpi=450)
    bars = ax.bar(x, ratios, color=colors, edgecolor="#1f2937", linewidth=1.0, width=0.22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.22, 1.26)
    ax.set_ylim(0, max(ratios) + 13)
    ax.set_ylabel("Proportion of objects (%)")
    ax.set_xlabel("Object size range")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)

    for bar, ratio, count in zip(bars, ratios, counts):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(x, y + 1.2, f"{ratio:.1f}%\n({count})", va="bottom", ha="center", fontsize=9)

    ax.text(
        0.98,
        0.94,
        f"Total objects: {total}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#374151",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {out}")
    print("Counts:", dict(zip(labels, counts)))
    print("Ratios:", dict(zip(labels, [round(r, 2) for r in ratios])))


def main():
    plot_distribution(NWPU_ROOT, ASSET_DIR / "Distribution.png", "NWPU VHR-10 Object Size Distribution")
    if DIOR_ROOT.exists():
        plot_distribution(DIOR_ROOT, ASSET_DIR / "DIOR_Distribution.png", "DIOR Object Size Distribution")


if __name__ == "__main__":
    main()
