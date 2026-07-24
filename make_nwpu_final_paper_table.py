import csv
from pathlib import Path


EVAL_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\eval_NWPU")
SEARCH_DIR = EVAL_ROOT / "ablation_epoch_search"
OUT_DIR = SEARCH_DIR / "final_paper_table"


OVERVIEW_LABELS = {
    "off_all": ("YOLOv12", "Baseline"),
    "head_only_all": ("YOLOv12 + Aux Head", "Auxiliary head only"),
    "exp1_sum_all": ("+ Heatmap Loss", "Multi-stage Gaussian heatmap supervision"),
    "exp2_sum_all": ("+ Self-Distillation", "Heatmap supervision + spatial self-distillation"),
}

POSITION_LABELS = {
    "exp2_sum_b5": ("B5", "Auxiliary heatmap from B5"),
    "exp2_sum_b4": ("B4", "Auxiliary heatmap from B4"),
    "exp2_sum_b3": ("B3", "Auxiliary heatmap from B3"),
    "exp2_sum_b2": ("B2", "Auxiliary heatmap from B2"),
    "exp2_sum_b1": ("B1", "Auxiliary heatmap from B1"),
    "exp2_sum_all": ("B1--B5", "Auxiliary heatmaps from all stages"),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_row(run, epoch):
    path = EVAL_ROOT / f"{run}_test" / f"{run}_test_checkpoint_metrics.csv"
    checkpoint = f"epoch{epoch}.pt"
    for row in read_csv(path):
        if row["checkpoint"] == checkpoint:
            return row
    raise FileNotFoundError(f"Missing {checkpoint} in {path}")


def pct(value):
    return float(value) * 100.0


def build_rows():
    overview_choice = read_csv(SEARCH_DIR / "overview_free_epoch_map50_test_search.csv")[0]
    position_choice = read_csv(SEARCH_DIR / "position_free_epoch_map50_test_search.csv")[0]

    rows = []
    overview_order = ["off_all", "head_only_all", "exp1_sum_all", "exp2_sum_all"]
    for run in overview_order:
        epoch = int(overview_choice[f"{run}_epoch"])
        metrics = metric_row(run, epoch)
        method, setting = OVERVIEW_LABELS[run]
        rows.append(
            {
                "Group": "Overview",
                "Method": method,
                "Setting": setting,
                "Epoch": epoch,
                "mAP50": pct(metrics["map50"]),
                "mAP50-95": pct(metrics["map50_95"]),
            }
        )

    position_order = ["exp2_sum_b5", "exp2_sum_b4", "exp2_sum_b3", "exp2_sum_b2", "exp2_sum_b1", "exp2_sum_all"]
    for run in position_order:
        epoch = int(position_choice[f"{run}_epoch"])
        metrics = metric_row(run, epoch)
        method, setting = POSITION_LABELS[run]
        rows.append(
            {
                "Group": "Position",
                "Method": method,
                "Setting": setting,
                "Epoch": epoch,
                "mAP50": pct(metrics["map50"]),
                "mAP50-95": pct(metrics["map50_95"]),
            }
        )
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Group", "Method", "Setting", "Epoch", "mAP50", "mAP50-95"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["mAP50"] = f"{row['mAP50']:.2f}"
            out["mAP50-95"] = f"{row['mAP50-95']:.2f}"
            writer.writerow(out)


def latex_escape(text):
    return str(text).replace("&", r"\&").replace("%", r"\%")


def write_latex(rows, path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation results on the NWPU VHR-10 dataset.}",
        r"\label{tab:nwpu_ablation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Group & Method & Epoch & mAP$_{50}$ (\%) & mAP$_{50:95}$ (\%) \\",
        r"\midrule",
    ]
    current_group = None
    for row in rows:
        if current_group is not None and row["Group"] != current_group:
            lines.append(r"\midrule")
        current_group = row["Group"]
        method = latex_escape(row["Method"])
        lines.append(
            f"{latex_escape(row['Group'])} & {method} & {row['Epoch']} & "
            f"{row['mAP50']:.2f} & {row['mAP50-95']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(rows, path):
    lines = [
        "| Group | Method | Setting | Epoch | mAP50 (%) | mAP50-95 (%) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Group']} | {row['Method']} | {row['Setting']} | {row['Epoch']} | "
            f"{row['mAP50']:.2f} | {row['mAP50-95']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUT_DIR / "NWPU_final_ablation_table.csv")
    write_latex(rows, OUT_DIR / "NWPU_final_ablation_table.tex")
    write_markdown(rows, OUT_DIR / "NWPU_final_ablation_table.md")
    print(f"Saved final table files to: {OUT_DIR}")
    for row in rows:
        print(
            f"{row['Group']:8s} | {row['Method']:24s} | epoch {row['Epoch']:2d} | "
            f"mAP50 {row['mAP50']:.2f} | mAP50-95 {row['mAP50-95']:.2f}"
        )


if __name__ == "__main__":
    main()
