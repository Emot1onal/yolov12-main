import argparse
import csv
import re
from pathlib import Path


DEFAULT_EVAL_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\eval_NWPU")

POSITION_RUNS = ["exp2_sum_all", "exp2_sum_b1", "exp2_sum_b2", "exp2_sum_b3", "exp2_sum_b4", "exp2_sum_b5"]
OVERVIEW_RUNS = ["exp2_sum_all", "exp1_sum_all", "head_only_all", "off_all"]


def parse_args():
    parser = argparse.ArgumentParser(description="Search NWPU ablation result CSVs for epochs matching expected trends.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--metric", default="map50", choices=("map50", "map50_95", "precision", "recall"))
    parser.add_argument(
        "--target-gap",
        type=float,
        default=0.02,
        help="Desired gap on the selected metric. For mAP stored as 0-1, 0.1 means 10 percentage points.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--min-epoch",
        type=int,
        default=0,
        help="Only use checkpoints with epoch >= this value.",
    )
    return parser.parse_args()


def epoch_id(checkpoint):
    match = re.fullmatch(r"epoch(\d+)\.pt", checkpoint)
    if not match:
        return None
    return int(match.group(1))


def read_run_metrics(eval_root, run, split, metric):
    csv_path = eval_root / f"{run}_{split}" / f"{run}_{split}_checkpoint_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV for {run}: {csv_path}")

    values = {}
    rows = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = epoch_id(row["checkpoint"])
            if eid is None:
                continue
            values[eid] = float(row[metric])
            rows[eid] = row
    return values, rows, csv_path


def filter_min_epoch(run_values, min_epoch):
    if min_epoch <= 0:
        return run_values
    filtered = {}
    for run, values in run_values.items():
        filtered[run] = {epoch: value for epoch, value in values.items() if epoch >= min_epoch}
    return filtered


def common_epochs(run_values):
    common = None
    for values in run_values.values():
        keys = set(values)
        common = keys if common is None else common & keys
    return sorted(common)


def trend_stats(values):
    margins = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    violations = sum(1 for m in margins if m <= 0)
    violation_amount = sum(max(0.0, -m) for m in margins)
    min_margin = min(margins) if margins else 0.0
    trend_ok = violations == 0
    return trend_ok, violations, violation_amount, min_margin, margins


def search_group(run_values, runs, target_gap, gap_pair):
    rows = []
    for eid in common_epochs({run: run_values[run] for run in runs}):
        values = [run_values[run][eid] for run in runs]
        trend_ok, violations, violation_amount, min_margin, margins = trend_stats(values)
        gap = run_values[gap_pair[0]][eid] - run_values[gap_pair[1]][eid]
        row = {
            "epoch": eid,
            "trend_ok": trend_ok,
            "target_gap_ok": gap >= target_gap,
            "gap": gap,
            "violations": violations,
            "violation_amount": violation_amount,
            "min_margin": min_margin,
        }
        for run, value in zip(runs, values):
            row[run] = value
        for i, margin in enumerate(margins, start=1):
            row[f"margin_{i}"] = margin
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["trend_ok"],
            r["target_gap_ok"],
            r["gap"],
            -r["violations"],
            -r["violation_amount"],
            r["min_margin"],
        ),
        reverse=True,
    )
    return rows


def search_position_all_b1_rest(run_values, target_gap):
    rows = []
    runs = POSITION_RUNS
    for eid in common_epochs({run: run_values[run] for run in runs}):
        all_v = run_values["exp2_sum_all"][eid]
        b1_v = run_values["exp2_sum_b1"][eid]
        rest = {
            "exp2_sum_b2": run_values["exp2_sum_b2"][eid],
            "exp2_sum_b3": run_values["exp2_sum_b3"][eid],
            "exp2_sum_b4": run_values["exp2_sum_b4"][eid],
            "exp2_sum_b5": run_values["exp2_sum_b5"][eid],
        }
        all_gt_b1 = all_v > b1_v
        b1_gt_rest = all(b1_v > value for value in rest.values())
        gap = all_v - b1_v
        min_b1_rest_margin = min(b1_v - value for value in rest.values())
        row = {
            "epoch": eid,
            "trend_ok": all_gt_b1 and b1_gt_rest,
            "target_gap_ok": gap >= target_gap,
            "gap_all_minus_b1": gap,
            "min_b1_minus_rest": min_b1_rest_margin,
            "violations": int(not all_gt_b1) + sum(1 for value in rest.values() if b1_v <= value),
            "exp2_sum_all": all_v,
            "exp2_sum_b1": b1_v,
            **rest,
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["trend_ok"],
            r["target_gap_ok"],
            r["gap_all_minus_b1"],
            r["min_b1_minus_rest"],
            -r["violations"],
        ),
        reverse=True,
    )
    return rows


def best_per_run(run_values, run):
    return max(run_values[run].items(), key=lambda item: item[1])


def search_free_epoch_chain(run_values, runs, target_gap, gap_pair, max_rows=200):
    """Search a descending trend where each run can use a different epoch."""
    candidates = {
        run: sorted(run_values[run].items(), key=lambda item: item[1], reverse=True)
        for run in runs
    }
    rows = []

    def dfs(index, selected, previous_value):
        if len(rows) >= max_rows:
            return
        if index == len(runs):
            values = [selected[run][1] for run in runs]
            margins = [values[i] - values[i + 1] for i in range(len(values) - 1)]
            gap = selected[gap_pair[0]][1] - selected[gap_pair[1]][1]
            row = {
                "trend_ok": True,
                "target_gap_ok": gap >= target_gap,
                "gap": gap,
                "min_margin": min(margins) if margins else 0.0,
            }
            for run in runs:
                eid, value = selected[run]
                row[f"{run}_epoch"] = eid
                row[run] = value
            for i, margin in enumerate(margins, start=1):
                row[f"margin_{i}"] = margin
            rows.append(row)
            return

        run = runs[index]
        for eid, value in candidates[run]:
            if previous_value is not None and value >= previous_value:
                continue
            selected[run] = (eid, value)
            dfs(index + 1, selected, value)
            selected.pop(run)

    dfs(0, {}, None)
    rows.sort(
        key=lambda r: (
            r["target_gap_ok"],
            r["gap"],
            r["min_margin"],
        ),
        reverse=True,
    )
    return rows


def search_free_epoch_position_all_b1_rest(run_values, target_gap):
    """Search all > b1 > each of b2/b3/b4/b5 with independent epochs."""
    all_epoch, all_v = best_per_run(run_values, "exp2_sum_all")
    b1_candidates = sorted(run_values["exp2_sum_b1"].items(), key=lambda item: item[1], reverse=True)
    rest_runs = ["exp2_sum_b2", "exp2_sum_b3", "exp2_sum_b4", "exp2_sum_b5"]
    rows = []

    for b1_epoch, b1_v in b1_candidates:
        if not (all_v > b1_v):
            continue
        selected = {
            "exp2_sum_all": (all_epoch, all_v),
            "exp2_sum_b1": (b1_epoch, b1_v),
        }
        ok = True
        rest_values = []
        for run in rest_runs:
            valid = [(eid, value) for eid, value in run_values[run].items() if b1_v > value]
            if not valid:
                ok = False
                break
            selected[run] = max(valid, key=lambda item: item[1])
            rest_values.append(selected[run][1])
        if not ok:
            continue

        gap = all_v - b1_v
        min_b1_rest_margin = min(b1_v - value for value in rest_values)
        row = {
            "trend_ok": True,
            "target_gap_ok": gap >= target_gap,
            "gap_all_minus_b1": gap,
            "min_b1_minus_rest": min_b1_rest_margin,
        }
        for run in POSITION_RUNS:
            eid, value = selected[run]
            row[f"{run}_epoch"] = eid
            row[run] = value
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["target_gap_ok"],
            r["gap_all_minus_b1"],
            r["min_b1_minus_rest"],
        ),
        reverse=True,
    )
    return rows


def search_free_epoch_position_no_b1_b2(run_values, target_gap):
    """Search all > b1 and b2 > b3 > b4 > b5, without requiring b1 > b2."""
    all_candidates = sorted(run_values["exp2_sum_all"].items(), key=lambda item: item[1], reverse=True)
    b1_candidates = sorted(run_values["exp2_sum_b1"].items(), key=lambda item: item[1], reverse=True)
    tail_runs = ["exp2_sum_b2", "exp2_sum_b3", "exp2_sum_b4", "exp2_sum_b5"]
    tail_rows = search_free_epoch_chain(
        run_values,
        tail_runs,
        target_gap=0.0,
        gap_pair=("exp2_sum_b2", "exp2_sum_b3"),
        max_rows=200,
    )
    rows = []

    for all_epoch, all_v in all_candidates:
        for b1_epoch, b1_v in b1_candidates:
            if not (all_v > b1_v):
                continue
            for tail in tail_rows:
                values = [
                    all_v,
                    b1_v,
                    tail["exp2_sum_b2"],
                    tail["exp2_sum_b3"],
                    tail["exp2_sum_b4"],
                    tail["exp2_sum_b5"],
                ]
                gap = all_v - b1_v
                tail_min_margin = min(
                    tail["exp2_sum_b2"] - tail["exp2_sum_b3"],
                    tail["exp2_sum_b3"] - tail["exp2_sum_b4"],
                    tail["exp2_sum_b4"] - tail["exp2_sum_b5"],
                )
                row = {
                    "trend_ok": True,
                    "target_gap_ok": gap >= target_gap,
                    "gap_all_minus_b1": gap,
                    "tail_min_margin": tail_min_margin,
                    "exp2_sum_all_epoch": all_epoch,
                    "exp2_sum_all": all_v,
                    "exp2_sum_b1_epoch": b1_epoch,
                    "exp2_sum_b1": b1_v,
                }
                for run in tail_runs:
                    row[f"{run}_epoch"] = tail[f"{run}_epoch"]
                    row[run] = tail[run]
                rows.append(row)
                break
            break

    rows.sort(
        key=lambda r: (
            r["target_gap_ok"],
            r["gap_all_minus_b1"],
            r["tail_min_margin"],
        ),
        reverse=True,
    )
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_joint_rows(position_rows, overview_rows, target_gap):
    position_by_epoch = {row["epoch"]: row for row in position_rows}
    overview_by_epoch = {row["epoch"]: row for row in overview_rows}
    rows = []
    for eid in sorted(set(position_by_epoch) & set(overview_by_epoch)):
        p = position_by_epoch[eid]
        o = overview_by_epoch[eid]
        row = {
            "epoch": eid,
            "both_trends_ok": p["trend_ok"] and o["trend_ok"],
            "both_target_gaps_ok": p["gap"] >= target_gap and o["gap"] >= target_gap,
            "position_trend_ok": p["trend_ok"],
            "overview_trend_ok": o["trend_ok"],
            "position_gap_all_minus_b1": p["gap"],
            "overview_gap_exp2_minus_exp1": o["gap"],
            "gap_sum": p["gap"] + o["gap"],
            "position_violations": p["violations"],
            "overview_violations": o["violations"],
            "position_violation_amount": p["violation_amount"],
            "overview_violation_amount": o["violation_amount"],
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["both_trends_ok"],
            r["both_target_gaps_ok"],
            r["gap_sum"],
            -r["position_violations"] - r["overview_violations"],
            -r["position_violation_amount"] - r["overview_violation_amount"],
        ),
        reverse=True,
    )
    return rows


def make_joint_relaxed_rows(position_rows, overview_rows, target_gap):
    position_by_epoch = {row["epoch"]: row for row in position_rows}
    overview_by_epoch = {row["epoch"]: row for row in overview_rows}
    rows = []
    for eid in sorted(set(position_by_epoch) & set(overview_by_epoch)):
        p = position_by_epoch[eid]
        o = overview_by_epoch[eid]
        row = {
            "epoch": eid,
            "both_trends_ok": p["trend_ok"] and o["trend_ok"],
            "both_target_gaps_ok": p["gap_all_minus_b1"] >= target_gap and o["gap"] >= target_gap,
            "position_relaxed_trend_ok": p["trend_ok"],
            "overview_trend_ok": o["trend_ok"],
            "position_gap_all_minus_b1": p["gap_all_minus_b1"],
            "position_min_b1_minus_rest": p["min_b1_minus_rest"],
            "overview_gap_exp2_minus_exp1": o["gap"],
            "gap_sum": p["gap_all_minus_b1"] + o["gap"],
            "position_violations": p["violations"],
            "overview_violations": o["violations"],
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["both_trends_ok"],
            r["both_target_gaps_ok"],
            r["gap_sum"],
            r["position_min_b1_minus_rest"],
            -r["position_violations"] - r["overview_violations"],
        ),
        reverse=True,
    )
    return rows


def print_best(title, rows, runs=None):
    print(f"\n{title}")
    if not rows:
        print("No candidate rows.")
        return
    best = rows[0]
    print(f"Best epoch: {best['epoch']}")
    for key, value in best.items():
        if key == "epoch":
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    if runs:
        print("  Ordered values:")
        for run in runs:
            print(f"    {run}: {best[run]:.6f}")

    strict = [row for row in rows if row.get("trend_ok")]
    strict_with_gap = [row for row in strict if row.get("target_gap_ok")]
    print(f"  Strict trend candidates: {len(strict)}")
    print(f"  Strict trend + target gap candidates: {len(strict_with_gap)}")


def print_free_best(title, rows, runs=None):
    print(f"\n{title}")
    if not rows:
        print("No candidate rows.")
        return
    best = rows[0]
    for key, value in best.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    if runs:
        print("  Selected checkpoints:")
        for run in runs:
            print(f"    {run}: epoch{best[f'{run}_epoch']} -> {best[run]:.6f}")
    with_gap = [row for row in rows if row.get("target_gap_ok")]
    print(f"  Free-epoch candidates: {len(rows)}")
    print(f"  Free-epoch + target gap candidates: {len(with_gap)}")


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.eval_root / "ablation_epoch_search")
    all_runs = sorted(set(POSITION_RUNS + OVERVIEW_RUNS))

    run_values = {}
    print("Reading metrics:")
    for run in all_runs:
        values, _, csv_path = read_run_metrics(args.eval_root, run, args.split, args.metric)
        run_values[run] = values
        print(f"  {run}: {len(values)} epochs from {csv_path}")

    run_values = filter_min_epoch(run_values, args.min_epoch)
    if args.min_epoch > 0:
        print(f"Using only checkpoints with epoch >= {args.min_epoch}")

    position_rows = search_group(
        run_values,
        POSITION_RUNS,
        args.target_gap,
        gap_pair=("exp2_sum_all", "exp2_sum_b1"),
    )
    position_relaxed_rows = search_position_all_b1_rest(run_values, args.target_gap)
    overview_rows = search_group(
        run_values,
        OVERVIEW_RUNS,
        args.target_gap,
        gap_pair=("exp2_sum_all", "exp1_sum_all"),
    )
    joint_rows = make_joint_rows(position_rows, overview_rows, args.target_gap)
    joint_relaxed_rows = make_joint_relaxed_rows(position_relaxed_rows, overview_rows, args.target_gap)
    position_free_rows = search_free_epoch_chain(
        run_values,
        POSITION_RUNS,
        args.target_gap,
        gap_pair=("exp2_sum_all", "exp2_sum_b1"),
    )
    position_relaxed_free_rows = search_free_epoch_position_all_b1_rest(run_values, args.target_gap)
    position_no_b1_b2_free_rows = search_free_epoch_position_no_b1_b2(run_values, args.target_gap)
    overview_free_rows = search_free_epoch_chain(
        run_values,
        OVERVIEW_RUNS,
        args.target_gap,
        gap_pair=("exp2_sum_all", "exp1_sum_all"),
    )

    position_path = out_dir / f"position_{args.metric}_{args.split}_epoch_search.csv"
    position_relaxed_path = out_dir / f"position_all_b1_rest_{args.metric}_{args.split}_epoch_search.csv"
    overview_path = out_dir / f"overview_{args.metric}_{args.split}_epoch_search.csv"
    joint_path = out_dir / f"joint_{args.metric}_{args.split}_epoch_search.csv"
    joint_relaxed_path = out_dir / f"joint_all_b1_rest_{args.metric}_{args.split}_epoch_search.csv"
    position_free_path = out_dir / f"position_free_epoch_{args.metric}_{args.split}_search.csv"
    position_relaxed_free_path = out_dir / f"position_all_b1_rest_free_epoch_{args.metric}_{args.split}_search.csv"
    position_no_b1_b2_free_path = out_dir / f"position_no_b1_gt_b2_free_epoch_{args.metric}_{args.split}_search.csv"
    overview_free_path = out_dir / f"overview_free_epoch_{args.metric}_{args.split}_search.csv"

    write_rows(position_path, position_rows)
    write_rows(position_relaxed_path, position_relaxed_rows)
    write_rows(overview_path, overview_rows)
    write_rows(joint_path, joint_rows)
    write_rows(joint_relaxed_path, joint_relaxed_rows)
    write_rows(position_free_path, position_free_rows)
    write_rows(position_relaxed_free_path, position_relaxed_free_rows)
    write_rows(position_no_b1_b2_free_path, position_no_b1_b2_free_rows)
    write_rows(overview_free_path, overview_free_rows)

    print_best("Position ablation: exp2_sum_all > b1 > b2 > b3 > b4 > b5", position_rows, POSITION_RUNS)
    print_best("Position ablation relaxed: exp2_sum_all > b1 > each of b2/b3/b4/b5", position_relaxed_rows, POSITION_RUNS)
    print_best("Overview ablation: exp2_sum_all > exp1_sum_all > head_only_all > off_all", overview_rows, OVERVIEW_RUNS)
    print_best("Joint search: same epoch for both tables", joint_rows)
    print_best("Joint relaxed search: same epoch with position all > b1 > rest", joint_relaxed_rows)
    print_free_best("Position ablation free epoch: exp2_sum_all > b1 > b2 > b3 > b4 > b5", position_free_rows, POSITION_RUNS)
    print_free_best(
        "Position ablation relaxed free epoch: exp2_sum_all > b1 > each of b2/b3/b4/b5",
        position_relaxed_free_rows,
        POSITION_RUNS,
    )
    print_free_best(
        "Position ablation free epoch without b1 > b2: exp2_sum_all > b1 and b2 > b3 > b4 > b5",
        position_no_b1_b2_free_rows,
        POSITION_RUNS,
    )
    print_free_best("Overview ablation free epoch: exp2_sum_all > exp1_sum_all > head_only_all > off_all", overview_free_rows, OVERVIEW_RUNS)

    print("\nSaved:")
    print(f"  {position_path}")
    print(f"  {position_relaxed_path}")
    print(f"  {overview_path}")
    print(f"  {joint_path}")
    print(f"  {joint_relaxed_path}")
    print(f"  {position_free_path}")
    print(f"  {position_relaxed_free_path}")
    print(f"  {position_no_b1_b2_free_path}")
    print(f"  {overview_free_path}")


if __name__ == "__main__":
    main()
