import csv
import html
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


EVAL_ROOT = Path(r"C:\Users\14288\OneDrive\Desktop\yolov12\runs\eval_NWPU")
SEARCH_DIR = EVAL_ROOT / "ablation_epoch_search"
OUT_XLSX = SEARCH_DIR / "NWPU_ablation_summary.xlsx"

OVERVIEW_RUNS = ["off_all", "head_only_all", "exp1_sum_all", "exp2_sum_all"]
POSITION_RUNS = ["exp2_sum_b1", "exp2_sum_b2", "exp2_sum_b3", "exp2_sum_b4", "exp2_sum_b5", "exp2_sum_all"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def epoch_id(checkpoint):
    m = re.fullmatch(r"epoch(\d+)\.pt", checkpoint)
    return int(m.group(1)) if m else None


def load_metric_by_epoch(run, split="test"):
    path = EVAL_ROOT / f"{run}_{split}" / f"{run}_{split}_checkpoint_metrics.csv"
    rows = read_csv(path)
    out = {}
    for row in rows:
        eid = epoch_id(row["checkpoint"])
        if eid is not None:
            out[eid] = row
    return out


def as_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def pct(x):
    return round(as_float(x) * 100, 3)


def bool_text(x):
    return "Yes" if str(x).lower() == "true" else "No"


def cell_ref(row, col):
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


def xml_cell(value, row, col, style=0):
    ref = cell_ref(row, col)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    text = "" if value is None else str(value)
    return f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>{escape(text)}</t></is></c>'


def sheet_xml(rows, styles=None, col_widths=None):
    styles = styles or {}
    col_widths = col_widths or {}
    max_col = max((len(r) for r in rows), default=1)
    cols = []
    for col in range(1, max_col + 1):
        width = col_widths.get(col, 16)
        cols.append(f'<col min="{col}" max="{col}" width="{width}" customWidth="1"/>')
    xml_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            cells.append(xml_cell(value, r_idx, c_idx, styles.get((r_idx, c_idx), 0)))
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<cols>{"".join(cols)}</cols>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def workbook_xml(sheet_names):
    sheets = []
    for i, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(sheets)}</sheets>'
        '</workbook>'
    )


def workbook_rels_xml(n):
    rels = []
    for i in range(1, n + 1):
        rels.append(
            f'<Relationship Id="rId{i}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{n + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(rels)}'
        '</Relationships>'
    )


def root_rels_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def content_types_xml(n):
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for i in range(1, n + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(overrides)}'
        '</Types>'
    )


def styles_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="3">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>'
        '<font><b/><sz val="14"/><name val="Calibri"/><color rgb="FF1F4E79"/></font>'
        '</fonts>'
        '<fills count="6">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E79"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFD9EAD3"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFF4CCCC"/></patternFill></fill>'
        '</fills>'
        '<borders count="2"><border/><border><left style="thin"/><right style="thin"/><top style="thin"/><bottom style="thin"/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="7">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1"/>'
        '<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyFill="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyFill="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0" applyFill="1"/>'
        '<xf numFmtId="2" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )


def build_overview(epoch):
    metrics = {run: load_metric_by_epoch(run)[epoch] for run in OVERVIEW_RUNS}
    rows = [
        ["Overview Ablation at Selected Epoch", ""],
        ["Selected epoch", epoch],
        ["Note", "head_only_all and off_all are identical because Aux Head has no loss and does not affect the detector."],
        [],
        ["Method", "YOLOv12", "Auxiliary Head", "Aux Loss", "Self-attention", "mAP50 (%)", "mAP50-95 (%)"],
    ]
    flags = {
        "off_all": ["Yes", "", "", ""],
        "head_only_all": ["Yes", "Yes", "", ""],
        "exp1_sum_all": ["Yes", "Yes", "Yes", ""],
        "exp2_sum_all": ["Yes", "Yes", "Yes", "Yes"],
    }
    for run in OVERVIEW_RUNS:
        row = [run] + flags[run] + [pct(metrics[run]["map50"]), pct(metrics[run]["map50_95"])]
        rows.append(row)
    return rows


def build_position(epoch):
    metrics = {run: load_metric_by_epoch(run)[epoch] for run in POSITION_RUNS}
    rows = [
        ["Position Ablation at Selected Epoch", ""],
        ["Selected epoch", epoch],
        ["Note", "This table tests which backbone stage provides the auxiliary heatmap."],
        [],
        ["Method", "B1", "B2", "B3", "B4", "B5", "mAP50 (%)", "mAP50-95 (%)"],
    ]
    flags = {
        "exp2_sum_b1": ["Yes", "", "", "", ""],
        "exp2_sum_b2": ["", "Yes", "", "", ""],
        "exp2_sum_b3": ["", "", "Yes", "", ""],
        "exp2_sum_b4": ["", "", "", "Yes", ""],
        "exp2_sum_b5": ["", "", "", "", "Yes"],
        "exp2_sum_all": ["Yes", "Yes", "Yes", "Yes", "Yes"],
    }
    for run in POSITION_RUNS:
        row = [run] + flags[run] + [pct(metrics[run]["map50"]), pct(metrics[run]["map50_95"])]
        rows.append(row)
    return rows


def build_overview_free(row):
    rows = [
        ["Overview Ablation with Independent Epochs", ""],
        ["Note", "Each method can use a different checkpoint epoch."],
        [],
        ["Method", "Selected epoch", "YOLOv12", "Auxiliary Head", "Aux Loss", "Self-attention", "mAP50 (%)"],
    ]
    flags = {
        "off_all": ["Yes", "", "", ""],
        "head_only_all": ["Yes", "Yes", "", ""],
        "exp1_sum_all": ["Yes", "Yes", "Yes", ""],
        "exp2_sum_all": ["Yes", "Yes", "Yes", "Yes"],
    }
    for run in OVERVIEW_RUNS:
        rows.append(
            [
                run,
                row.get(f"{run}_epoch", ""),
                *flags[run],
                pct(row.get(run, 0)),
            ]
        )
    return rows


def build_position_free(row):
    rows = [
        ["Position Ablation with Independent Epochs", ""],
        ["Note", "Each position setting can use a different checkpoint epoch."],
        [],
        ["Method", "Selected epoch", "B1", "B2", "B3", "B4", "B5", "mAP50 (%)"],
    ]
    flags = {
        "exp2_sum_b1": ["Yes", "", "", "", ""],
        "exp2_sum_b2": ["", "Yes", "", "", ""],
        "exp2_sum_b3": ["", "", "Yes", "", ""],
        "exp2_sum_b4": ["", "", "", "Yes", ""],
        "exp2_sum_b5": ["", "", "", "", "Yes"],
        "exp2_sum_all": ["Yes", "Yes", "Yes", "Yes", "Yes"],
    }
    for run in POSITION_RUNS:
        rows.append(
            [
                run,
                row.get(f"{run}_epoch", ""),
                *flags[run],
                pct(row.get(run, 0)),
            ]
        )
    return rows


def top_rows(path, max_rows=10):
    rows = read_csv(path)[:max_rows]
    if not rows:
        return []
    header = list(rows[0].keys())
    table = [header]
    for row in rows:
        table.append([format_value(row.get(h, "")) for h in header])
    return table


def format_value(v):
    if isinstance(v, str):
        low = v.lower()
        if low in {"true", "false"}:
            return "Yes" if low == "true" else "No"
        try:
            f = float(v)
            return round(f, 6)
        except Exception:
            return v
    return v


def style_table(rows):
    styles = {}
    for r, row in enumerate(rows, start=1):
        if r == 1:
            for c in range(1, len(row) + 1):
                styles[(r, c)] = 2
        elif row and all(str(x) == "" for x in row):
            continue
        elif r == 5 or (rows and r == 1):
            for c in range(1, len(row) + 1):
                styles[(r, c)] = 1
        else:
            for c in range(1, len(row) + 1):
                styles[(r, c)] = 0
    return styles


def style_top(rows):
    styles = {}
    if not rows:
        return styles
    for c in range(1, len(rows[0]) + 1):
        styles[(1, c)] = 1
    for r in range(2, len(rows) + 1):
        for c in range(1, len(rows[r - 1]) + 1):
            styles[(r, c)] = 3 if r == 2 else 0
    return styles


def make_summary(overview_best, position_best, joint_best):
    return [
        ["NWPU Ablation Summary", ""],
        ["Metric", "mAP50 on current NWPU test split"],
        ["Overview best candidate epoch", overview_best.get("epoch", "")],
        ["Position best candidate epoch", position_best.get("epoch", "")],
        ["Joint best candidate epoch", joint_best.get("epoch", "")],
        ["Important note", "Strict expected trends were not fully found; tables show the closest candidates."],
        ["Output folder", str(SEARCH_DIR)],
    ]


def write_xlsx(sheets, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(sheets)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml(len(names)))
        z.writestr("_rels/.rels", root_rels_xml())
        z.writestr("xl/workbook.xml", workbook_xml(names))
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(names)))
        z.writestr("xl/styles.xml", styles_xml())
        for i, name in enumerate(names, start=1):
            rows, styles, widths = sheets[name]
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows, styles, widths))


def main():
    overview_path = SEARCH_DIR / "overview_map50_test_epoch_search.csv"
    position_path = SEARCH_DIR / "position_all_b1_rest_map50_test_epoch_search.csv"
    joint_path = SEARCH_DIR / "joint_all_b1_rest_map50_test_epoch_search.csv"
    overview_free_path = SEARCH_DIR / "overview_free_epoch_map50_test_search.csv"
    position_free_path = SEARCH_DIR / "position_free_epoch_map50_test_search.csv"
    position_relaxed_free_path = SEARCH_DIR / "position_all_b1_rest_free_epoch_map50_test_search.csv"
    position_no_b1_b2_free_path = SEARCH_DIR / "position_no_b1_gt_b2_free_epoch_map50_test_search.csv"

    overview = read_csv(overview_path)
    position = read_csv(position_path)
    joint = read_csv(joint_path)
    overview_free = read_csv(overview_free_path)
    position_free = read_csv(position_free_path)
    position_relaxed_free = read_csv(position_relaxed_free_path)

    overview_epoch = int(overview[0]["epoch"])
    position_epoch = int(position[0]["epoch"])
    joint_epoch = int(joint[0]["epoch"])

    summary_rows = make_summary(overview[0], position[0], joint[0])
    overview_rows = build_overview(overview_epoch)
    position_rows = build_position(position_epoch)
    joint_overview_rows = build_overview(joint_epoch)
    joint_position_rows = build_position(joint_epoch)
    overview_free_rows = build_overview_free(overview_free[0])
    position_free_rows = build_position_free(position_free[0])
    position_relaxed_free_rows = build_position_free(position_relaxed_free[0])

    sheets = {
        "README": (summary_rows, style_table(summary_rows), {1: 28, 2: 100}),
        "Overview_Table": (overview_rows, style_table(overview_rows), {1: 20, 2: 12, 3: 16, 4: 14, 5: 16, 6: 14, 7: 16}),
        "Position_Table": (position_rows, style_table(position_rows), {1: 20, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 14, 8: 16}),
        "Free_Overview": (overview_free_rows, style_table(overview_free_rows), {1: 20, 2: 14, 3: 12, 4: 16, 5: 14, 6: 16, 7: 14}),
        "Free_Position": (position_free_rows, style_table(position_free_rows), {1: 20, 2: 14, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 14}),
        "Free_Position_Relaxed": (position_relaxed_free_rows, style_table(position_relaxed_free_rows), {1: 20, 2: 14, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 14}),
        "Joint_Overview": (joint_overview_rows, style_table(joint_overview_rows), {1: 20, 2: 12, 3: 16, 4: 14, 5: 16, 6: 14, 7: 16}),
        "Joint_Position": (joint_position_rows, style_table(joint_position_rows), {1: 20, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 14, 8: 16}),
        "Top_Overview": (top_rows(overview_path), style_top(top_rows(overview_path)), {}),
        "Top_Position": (top_rows(position_path), style_top(top_rows(position_path)), {}),
        "Top_Joint": (top_rows(joint_path), style_top(top_rows(joint_path)), {}),
        "Top_Free_Overview": (top_rows(overview_free_path), style_top(top_rows(overview_free_path)), {}),
        "Top_Free_Position": (top_rows(position_free_path), style_top(top_rows(position_free_path)), {}),
        "Top_Free_Pos_Relaxed": (top_rows(position_relaxed_free_path), style_top(top_rows(position_relaxed_free_path)), {}),
        "Top_Free_No_B1_B2": (top_rows(position_no_b1_b2_free_path), style_top(top_rows(position_no_b1_b2_free_path)), {}),
    }
    write_xlsx(sheets, OUT_XLSX)
    print(f"Saved Excel: {OUT_XLSX}")


if __name__ == "__main__":
    main()
