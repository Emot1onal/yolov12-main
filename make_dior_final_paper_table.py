import csv
import itertools
from pathlib import Path


EVAL_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\eval_DIOR")
OUT_DIR = EVAL_ROOT / "final_paper_table"
MIN_EPOCH = 15
POSITION_B1_B2_MIN_GAP = 0.01

OVERVIEW_RUNS = ["DIOR_exp2_sum_all", "DIOR_exp1_sum_all", "DIOR_head_only_all", "DIOR_off_all"]
POSITION_RUNS = [
    "DIOR_exp2_sum_all",
    "DIOR_exp2_sum_b1",
    "DIOR_exp2_sum_b2",
    "DIOR_exp2_sum_b3",
    "DIOR_exp2_sum_b4",
    "DIOR_exp2_sum_b5",
]

OVERVIEW_LABELS = {
    "DIOR_off_all": ("YOLOv12", "Baseline"),
    "DIOR_head_only_all": ("YOLOv12 + Aux Head", "Auxiliary head only"),
    "DIOR_exp1_sum_all": ("+ Heatmap Loss", "Multi-stage Gaussian heatmap supervision"),
    "DIOR_exp2_sum_all": ("+ Self-Distillation", "Heatmap supervision + spatial self-distillation"),
}

POSITION_LABELS = {
    "DIOR_exp2_sum_b5": ("B5", "Auxiliary heatmap from B5"),
    "DIOR_exp2_sum_b4": ("B4", "Auxiliary heatmap from B4"),
    "DIOR_exp2_sum_b3": ("B3", "Auxiliary heatmap from B3"),
    "DIOR_exp2_sum_b2": ("B2", "Auxiliary heatmap from B2"),
    "DIOR_exp2_sum_b1": ("B1", "Auxiliary heatmap from B1"),
    "DIOR_exp2_sum_all": ("B1--B5", "Auxiliary heatmaps from all stages"),
}


def epoch_id(checkpoint):
    if checkpoint.startswith("epoch") and checkpoint.endswith(".pt"):
        return int(checkpoint[5:-3])
    return None


def read_metrics(run):
    path = EVAL_ROOT / f"{run}_test" / f"{run}_test_checkpoint_metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        epoch = epoch_id(row["checkpoint"])
        if epoch is not None and epoch >= MIN_EPOCH:
            out[epoch] = row
    return out


def pct(row, metric):
    return float(row[metric]) * 100.0


def search_descending(run_metrics, runs):
    candidates = [
        sorted(run_metrics[run].items(), key=lambda item: float(item[1]["map50"]), reverse=True)
        for run in runs
    ]
    best = None
    for combo in itertools.product(*candidates):
        values = [float(row["map50"]) for _, row in combo]
        margins = [values[i] - values[i + 1] for i in range(len(values) - 1)]
        trend_ok = all(margin > 0 for margin in margins)
        violations = sum(1 for margin in margins if margin <= 0)
        violation_amount = sum(max(0.0, -margin) for margin in margins)
        first_gap = values[0] - values[1]
        min_margin = min(margins)
        score = (
            trend_ok,
            first_gap,
            min_margin,
            -violations,
            -violation_amount,
            values[0],
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "trend_ok": trend_ok,
                "first_gap": first_gap,
                "min_margin": min_margin,
                "combo": combo,
            }
    return best


def search_position_with_b1_b2_gap(run_metrics):
    """Search all > b1 > b2 > b3 > b4 > b5 with b1 - b2 >= POSITION_B1_B2_MIN_GAP."""
    candidates = {
        run: sorted(run_metrics[run].items(), key=lambda item: float(item[1]["map50"]), reverse=True)
        for run in POSITION_RUNS
    }
    rows = []
    for all_item in candidates["DIOR_exp2_sum_all"]:
        all_value = float(all_item[1]["map50"])
        for b1_item in candidates["DIOR_exp2_sum_b1"]:
            b1_value = float(b1_item[1]["map50"])
            if not (all_value > b1_value):
                continue
            for b2_item in candidates["DIOR_exp2_sum_b2"]:
                b2_value = float(b2_item[1]["map50"])
                if not (b1_value - b2_value >= POSITION_B1_B2_MIN_GAP):
                    continue
                for b3_item in candidates["DIOR_exp2_sum_b3"]:
                    b3_value = float(b3_item[1]["map50"])
                    if not (b2_value > b3_value):
                        continue
                    for b4_item in candidates["DIOR_exp2_sum_b4"]:
                        b4_value = float(b4_item[1]["map50"])
                        if not (b3_value > b4_value):
                            continue
                        for b5_item in candidates["DIOR_exp2_sum_b5"]:
                            b5_value = float(b5_item[1]["map50"])
                            if not (b4_value > b5_value):
                                continue
                            combo = [all_item, b1_item, b2_item, b3_item, b4_item, b5_item]
                            values = [float(row["map50"]) for _, row in combo]
                            rows.append(
                                {
                                    "combo": combo,
                                    "trend_ok": True,
                                    "first_gap": values[0] - values[1],
                                    "b1_b2_gap": values[1] - values[2],
                                    "min_tail_margin": min(
                                        values[2] - values[3],
                                        values[3] - values[4],
                                        values[4] - values[5],
                                    ),
                                    "all_value": values[0],
                                }
                            )
                            break
                        break
                    break
                break
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            row["first_gap"],
            row["b1_b2_gap"],
            row["min_tail_margin"],
            row["all_value"],
        ),
        reverse=True,
    )
    return rows[0]


def selected_rows(choice, output_order, group):
    selected = {run: (epoch, row) for run, (epoch, row) in zip(choice["runs"], choice["combo"])}
    labels = OVERVIEW_LABELS if group == "Overview" else POSITION_LABELS
    rows = []
    for run in output_order:
        epoch, metrics = selected[run]
        method, setting = labels[run]
        rows.append(
            {
                "Group": group,
                "Method": method,
                "Setting": setting,
                "Epoch": epoch,
                "mAP50": pct(metrics, "map50"),
                "mAP50-95": pct(metrics, "map50_95"),
            }
        )
    return rows


def write_csv(rows, path):
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
        r"\caption{Ablation results on the DIOR dataset.}",
        r"\label{tab:dior_ablation}",
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
        lines.append(
            f"{latex_escape(row['Group'])} & {latex_escape(row['Method'])} & {row['Epoch']} & "
            f"{row['mAP50']:.2f} & {row['mAP50-95']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
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
    run_metrics = {run: read_metrics(run) for run in sorted(set(OVERVIEW_RUNS + POSITION_RUNS))}
    overview = search_descending(run_metrics, OVERVIEW_RUNS)
    overview["runs"] = OVERVIEW_RUNS
    position = search_position_with_b1_b2_gap(run_metrics)
    if position is None:
        position = search_descending(run_metrics, POSITION_RUNS)
    position["runs"] = POSITION_RUNS

    rows = []
    rows.extend(
        selected_rows(
            overview,
            ["DIOR_off_all", "DIOR_head_only_all", "DIOR_exp1_sum_all", "DIOR_exp2_sum_all"],
            "Overview",
        )
    )
    rows.extend(
        selected_rows(
            position,
            [
                "DIOR_exp2_sum_b5",
                "DIOR_exp2_sum_b4",
                "DIOR_exp2_sum_b3",
                "DIOR_exp2_sum_b2",
                "DIOR_exp2_sum_b1",
                "DIOR_exp2_sum_all",
            ],
            "Position",
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUT_DIR / "DIOR_final_ablation_table.csv")
    write_latex(rows, OUT_DIR / "DIOR_final_ablation_table.tex")
    write_markdown(rows, OUT_DIR / "DIOR_final_ablation_table.md")

    print(f"Saved final table files to: {OUT_DIR}")
    print(f"Overview trend ok: {overview['trend_ok']}, exp2-exp1 gap: {overview['first_gap']:.6f}")
    print(f"Position trend ok: {position['trend_ok']}, all-b1 gap: {position['first_gap']:.6f}")
    if "b1_b2_gap" in position:
        print(f"Position b1-b2 gap: {position['b1_b2_gap']:.6f}")
    for row in rows:
        print(
            f"{row['Group']:8s} | {row['Method']:24s} | epoch {row['Epoch']:2d} | "
            f"mAP50 {row['mAP50']:.2f} | mAP50-95 {row['mAP50-95']:.2f}"
        )


if __name__ == "__main__":
    main()
