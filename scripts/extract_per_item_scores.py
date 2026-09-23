#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream D: recover the PER-ITEM doctor score printed on the report images.

Why this exists. nailfold-doctor-scale-audit recorded that a per-field score
column is printed on the 978 report images and had never been read. The 465 RTF
files do not contain it: decoding every one of them with scripts/dump_rtf.py's
decoder and searching for 积分 / 分值 / 得分 / 计分 / 总分 / 评分 returns 0 files.
So the images are the only source, and OCR is the only route.

What a report row looks like (archive3/201, x positions in the original 680px):
    x~58 field name        x~238 measured value    x~444 [normal range]
    x~565 PER-ITEM SCORE
and four totals near the bottom: 形态积分 / 流态积分 / 袢周积分 / 总积分, plus 综合判断.

This script reads the score COLUMN, not the value column, and does not predict
anything. The recovered scores are the doctor's own arithmetic, which is why they
are worth having: they say how each field was weighted into the assessment.

Correction discipline (explicit request): an abnormal reading is corrected ONLY
from evidence in the original report - a legible re-read, or the printed totals
failing to add up. Nothing is dropped or clipped because a model disagrees with
it, and nothing is dropped for falling outside a normal range. Every value that
cannot be read confidently is written as unknown, never as normal or abnormal.

IMPORTANT: rep_* report scans are report images. They are read here as TEXT
only. They must never become encoder inputs (standing rule), and this script
writes no image feature.

locked-47: report images belonging to locked cases are NOT opened. The run
records locked_cases_seen and aborts if a locked id slips into the output.

Run (needs rapidocr_onnxruntime; present in the anaconda3 base env):
  PYTHONIOENCODING=utf-8 python scripts/extract_per_item_scores.py --limit 40
  PYTHONIOENCODING=utf-8 python scripts/extract_per_item_scores.py
"""
import argparse
import json
from itertools import combinations, product
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from nailfold_report.data.report_layout import REPORT_FIELDS  # noqa: E402

# Canonical row order and spacing of the printed table. Using the project's own
# layout means the row index -> field mapping is not re-guessed here, and it lets
# a score be placed even when OCR drops that row's label entirely (渗出/出血 are
# routinely missed because they are two short characters).
FIELD_ORDER = [f.name for f in REPORT_FIELDS]
ROW_STRIDE = 19.0

# Group membership, reused verbatim from scripts/derive_score_rules.py so the two
# scripts cannot drift apart. Used only to check the recovered per-item scores
# against the printed subtotals -- never to build a training target.
SCORE_GROUPS = {
    "morphology_score": ["clarity", "capillary_count", "afferent_diameter",
                         "efferent_diameter", "output_input_ratio",
                         "apex_diameter", "loop_length", "crossing_ratio",
                         "malformation_ratio"],
    "flow_score": ["flow_state", "flow_speed_um_s", "vasomotion",
                   "rbc_aggregation", "wbc_count", "microthrombus",
                   "blood_color"],
    "periloop_score": ["exudation", "hemorrhage",
                       "subpapillary_venous_plexus", "papilla", "sweat_duct"],
}

DATA = ROOT / "data"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
OUT_DIR = ROOT / "artifacts" / "experiments" / "per_item_scores_20260922"
RAW = OUT_DIR / "ocr_rows.jsonl"
CELLS = OUT_DIR / "ocr_cells.jsonl"   # raw OCR cells, so --reparse needs no re-OCR
OUT_CSV = OUT_DIR / "per_item_scores.csv"
OUT_JSON = OUT_DIR / "per_item_scores_summary.json"

# Printed Chinese row label -> canonical field. Taken from the report layout as
# it appears on the images; OCR confuses 袢/祥/髮 so the keys are matched loosely.
ROW_FIELDS = {
    "清晰度": "clarity", "管袢数": "capillary_count",
    "输入枝管径": "afferent_diameter", "输出枝管径": "efferent_diameter",
    "输出输入比": "output_input_ratio", "管袢顶宽": "apex_diameter",
    "管袢长": "loop_length", "交叉管袢数": "crossing_ratio",
    "畸形管数": "malformation_ratio", "流态": "flow_state",
    "流速": "flow_speed_um_s", "血管运动性": "vasomotion",
    "红细胞聚集": "rbc_aggregation", "白细胞数": "wbc_count",
    "白微栓": "microthrombus", "血色": "blood_color",
    "渗出": "exudation", "出血": "hemorrhage",
    "乳头下静脉丛": "subpapillary_venous_plexus", "乳头": "papilla",
    "汗腺导管": "sweat_duct",
}
TOTALS = {"形态积分": "morphology_score", "流态积分": "flow_score",
          "袢周积分": "periloop_score", "总积分": "total_score"}
GRADE = "综合判断"

# OCR routinely renders a decimal point as an apostrophe or a space, and these
# scores are always one decimal place, so 0'0 / 0. 0 / 1.6 all mean x.y.
SCORE_RE = re.compile(r"^(\d)\s*['.,`]?\s*(\d)$")
PLAIN_RE = re.compile(r"^(\d(?:\.\d)?)$")
MIN_CONF = 0.55          # below this the cell is written unknown, not guessed
# Ceiling for treating a cell as an ambiguous glyph eligible for re-reading. On
# this corpus confirmed misreads sit at 0.555-0.643 and clean cells at >=0.70.
AMBIG_CONF_MAX = 0.68


def norm_label(s: str) -> str:
    """Fold the OCR's systematic character confusions before matching."""
    for a, b in (("祥", "袢"), ("壁", "袢"), ("髮", "袢"), ("璧", "袢"),
                 ("管祥", "管袢"), (" ", "")):
        s = s.replace(a, b)
    return s.strip()


def parse_score(txt: str) -> float | None:
    t = str(txt).strip().replace(" ", "")
    m = PLAIN_RE.match(t)
    if m:
        return float(m.group(1))
    m = SCORE_RE.match(str(txt).strip())
    if m:
        return float("%s.%s" % (m.group(1), m.group(2)))
    return None


def ocr_image(engine, path: Path) -> list[dict]:
    res, _ = engine(str(path))
    out = []
    for b in (res or []):
        box, txt, conf = b[0], str(b[1]), float(b[2])
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        out.append(dict(text=txt, conf=round(conf, 3),
                        x=min(xs), y=min(ys), x2=max(xs), y2=max(ys)))
    return out


def score_column_x(cells: list[dict]) -> float | None:
    """Locate the score column from the totals row, not from a fixed constant.

    The per-item scores sit under 袢周积分's number, so the column is derived
    per image and a differently laid out report cannot be silently mis-read.
    """
    cand = [c for c in cells if parse_score(c["text"]) is not None]
    if not cand:
        return None
    xs = sorted(c["x"] for c in cand)
    # the score column is the rightmost dense cluster of score-shaped cells
    best, bestn = None, 0
    for x in xs:
        n = sum(1 for v in xs if abs(v - x) <= 18)
        if n >= bestn:
            best, bestn = x, n
    return best if bestn >= 4 else None


def read_report(cells: list[dict]) -> dict:
    col = score_column_x(cells)
    rows, totals, grade, unknown = {}, {}, None, []
    total_raws: dict[str, str] = {}

    for c in cells:
        t = norm_label(c["text"])
        for zh, canon in TOTALS.items():
            if zh in t:
                v = parse_score(t.split("：")[-1].split(":")[-1])
                if v is None:                       # number is a separate cell
                    near = [d for d in cells
                            if abs(d["y"] - c["y"]) <= 8 and d["x"] > c["x"]
                            and parse_score(d["text"]) is not None]
                    if near:
                        nearest = min(near, key=lambda d: d["x"])
                        v, conf = parse_score(nearest["text"]), nearest["conf"]
                    else:
                        conf = c["conf"]
                else:
                    conf = c["conf"]
                if v is not None and conf >= MIN_CONF:
                    # keep the raw glyphs: the subtotal cell suffers the same
                    # 0.8 -> "8'0" misread as the per-item cells, and the raw
                    # text is what lets that be re-read from report evidence
                    totals[canon] = v
                    total_raws[canon] = (t.split("：")[-1].split(":")[-1]
                                         if parse_score(
                                             t.split("：")[-1].split(":")[-1])
                                         is not None else nearest["text"])
                elif v is not None:
                    unknown.append(dict(item=canon, reason="low_confidence",
                                        conf=conf))
        if GRADE in t:
            g = t.split("：")[-1].split(":")[-1].strip()
            grade = g or None

    if col is None:
        return dict(per_item={}, totals=totals, total_raws=total_raws, grade=grade,
                    unknown=unknown + [dict(item="score_column",
                                            reason="column_not_located")])

    # Anchor the table geometrically. Label matching alone loses rows whenever
    # OCR drops a short label, and substring collisions (管袢数 inside 交叉管袢数)
    # mis-assign others, so the row labels that ARE read are used only to fit
    # row_index -> y, and every score cell is then placed by that fit.
    anchors = []
    for c in cells:
        lab = norm_label(c["text"])
        if c["x"] > 200 or "积分" in lab:      # label column only, never a total
            continue
        # A row label occupies the whole cell. The conclusion paragraph under the
        # table also starts at x<200 and contains field words ("血管清晰度尚可..."),
        # so a cell is only an anchor if a label sits at its start and the cell
        # is not much longer than that label.
        hits = [(len(k), v) for k, v in ROW_FIELDS.items()
                if lab.startswith(k) and len(lab) <= len(k) + 3]
        if not hits:
            continue
        canon = max(hits)[1]                  # longest printed label wins
        anchors.append((FIELD_ORDER.index(canon), c["y"]))

    anchors = _drop_outliers(anchors)
    if len(anchors) < 4:
        return dict(per_item={}, totals=totals, total_raws=total_raws, grade=grade,
                    unknown=unknown + [dict(item="table_rows",
                                            reason="too_few_row_anchors",
                                            anchors=len(anchors))])

    idx = [a for a, _ in anchors]
    ys = [y for _, y in anchors]
    if len(set(idx)) >= 3:
        stride, top = _fit_rows(idx, ys)
    else:
        stride, top = ROW_STRIDE, ys[0] - idx[0] * ROW_STRIDE
    resid = max(abs(y - (top + a * stride)) for a, y in anchors)
    if not (0.7 * ROW_STRIDE <= stride <= 1.4 * ROW_STRIDE) or resid > 8:
        return dict(per_item={}, totals=totals, total_raws=total_raws, grade=grade,
                    unknown=unknown + [dict(item="table_rows",
                                            reason="row_fit_rejected",
                                            stride=round(stride, 2),
                                            max_residual=round(resid, 2))])

    last_row_y = top + (len(FIELD_ORDER) - 1) * stride
    for c in cells:
        if abs(c["x"] - col) > 20 or c["y"] > last_row_y + 0.6 * stride:
            continue                          # other column, or the totals block
        v = parse_score(c["text"])
        if v is None:
            continue
        k = int(round((c["y"] - top) / stride))
        if not 0 <= k < len(FIELD_ORDER):
            continue
        if abs(c["y"] - (top + k * stride)) > 0.45 * stride:
            unknown.append(dict(item=FIELD_ORDER[k], reason="row_off_grid"))
            continue
        canon = FIELD_ORDER[k]
        if c["conf"] < MIN_CONF:
            unknown.append(dict(item=canon, reason="low_confidence",
                                conf=c["conf"]))
            continue
        rows[canon] = dict(score=v, conf=c["conf"], raw=c["text"])

    for f in FIELD_ORDER:
        if f not in rows and not any(u["item"] == f for u in unknown):
            unknown.append(dict(item=f, reason="no_score_cell_on_row"))
    return dict(per_item=rows, totals=totals, total_raws=total_raws, grade=grade, unknown=unknown,
                row_fit=dict(stride=round(stride, 2), top=round(top, 2),
                             anchors=len(anchors),
                             max_residual=round(resid, 2)))


def reread_cell(raw, current: float, target: float):
    """Re-read ONE cell's glyphs to a different decimal placement.

    Returns the alternative reading only if the same digits can be read as the
    target. Returns None otherwise, so a cell that simply disagrees is left
    alone and reported rather than nudged toward the target.
    """
    txt = str(raw)
    if not re.search(r"\d\s*['`,]\s*\d", txt):
        return None          # glyphs are unambiguous: do not re-read to fit
    digits = re.sub(r"\D", "", txt)
    if len(digits) != 2:
        return None
    for v in (float("%s.%s" % (digits[0], digits[1])),
              float("%s.%s" % (digits[1], digits[0]))):
        if v != current and abs(v - target) <= 0.15:
            return v
    return None


def reread_against_subtotal(got: dict, raws: dict, printed: float,
                            confs: dict | None = None):
    """Re-read a cell when the group's own printed subtotal says it is wrong.

    This is the ONLY correction mechanism in this script, and it is evidence from
    the original report: the subtotal is printed separately from the per-item
    column, so a mismatch localises an OCR misread. Three hard constraints keep
    it from becoming curve-fitting:

      1. only cells whose raw text is ambiguous BY OCR are candidates, and the
         replacement must be a reading of those same glyphs (8'0 -> 0.8, i.e. the
         digits are kept and only the decimal position changes);
      2. exactly one candidate assignment may reconcile the subtotal. If none or
         several do, nothing is changed and the group is reported as disagreeing;
      3. nothing is deleted, clipped, or judged against a normal range, and no
         model prediction is involved.
    """
    # Only cells whose glyphs are genuinely ambiguous may be re-read. Measured on
    # this corpus: every confirmed misread renders the decimal point as an
    # apostrophe or backtick ("8'0") and reads at confidence 0.555-0.643, while
    # cleanly separated cells ("0.4", "0. 4") read at >=0.70. Without this gate
    # the subtotal search happily rewrites a correct 0.4 into 4.0 to balance a
    # group, which is curve-fitting, not reading the report.
    confs = confs or {}
    cands = {}
    for fld, raw in raws.items():
        txt = str(raw)
        ambiguous = bool(re.search(r"\d\s*['`,]\s*\d", txt))
        if not ambiguous or confs.get(fld, 1.0) > AMBIG_CONF_MAX:
            continue
        digits = re.sub(r"\D", "", txt)
        if len(digits) != 2:
            continue
        alts = {float("%s.%s" % (digits[0], digits[1])),
                float("%s.%s" % (digits[1], digits[0]))}
        alts.discard(got[fld])
        if alts:
            cands[fld] = sorted(alts)

    base = sum(got.values())
    # A page can carry the same misread in more than one cell (two 8'0 in one
    # group is common), so subsets are searched, smallest first: the fewest
    # re-reads that reconcile the subtotal wins, and it must be the ONLY subset
    # of that size that does. Ties leave the group untouched and flagged.
    flds = sorted(cands)
    for size in (1, 2, 3):
        if size > len(flds):
            break
        sols = []
        for combo in combinations(flds, size):
            for vals in product(*(cands[f] for f in combo)):
                delta = sum(v - got[f] for f, v in zip(combo, vals))
                if abs(base + delta - printed) <= 0.15:
                    sols.append((combo, vals))
        if len(sols) == 1:
            combo, vals = sols[0]
            total = round(base + sum(v - got[f] for f, v in zip(combo, vals)), 2)
            return total, [
                dict(field=f, raw=str(raws[f]), **{"from": got[f], "to": v},
                     basis=("group subtotal %s printed on the same report; %d "
                            "cell(s) re-read, uniquely determined"
                            % (printed, size)))
                for f, v in zip(combo, vals)]
        if sols:
            break            # ambiguous at this size: do not guess
    return round(base, 2), []


def _drop_outliers(anchors: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Keep the largest set of anchors consistent with one straight row grid.

    A single mis-assigned label (e.g. 输出/输入枝管径 read as 输入枝管径) would drag
    the least-squares fit and silently shift every row. So the grid is voted on
    by anchor pairs and only the agreeing majority is fitted.
    """
    if len(anchors) < 4:
        return anchors
    best: list[tuple[int, float]] = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            (a1, y1), (a2, y2) = anchors[i], anchors[j]
            if a1 == a2:
                continue
            s = (y2 - y1) / (a2 - a1)
            if not (0.7 * ROW_STRIDE <= s <= 1.4 * ROW_STRIDE):
                continue
            t = y1 - s * a1
            keep = [(a, y) for a, y in anchors if abs(y - (t + s * a)) <= 4]
            if len(keep) > len(best):
                best = keep
    return best if len(best) >= 4 else []


def _fit_rows(idx: list[int], ys: list[float]) -> tuple[float, float]:
    """Least-squares row stride and table top, no numpy dependency needed."""
    n = len(idx)
    mx = sum(idx) / n
    my = sum(ys) / n
    den = sum((a - mx) ** 2 for a in idx)
    stride = sum((a - mx) * (y - my) for a, y in zip(idx, ys)) / den
    return stride, my - stride * mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reparse", action="store_true",
                    help=("re-run only the table parse and reconciliation from "
                          "the cached raw OCR cells; the OCR pass over 784 "
                          "images takes ~28 min and its output does not change"))
    a = ap.parse_args()

    lab = pd.read_csv(LABELS)
    locked = set(lab[lab.development_fold.isna()].exam_case_id.astype(str))
    dev = set(lab[lab.development_fold.notna()].exam_case_id.astype(str))

    imgs = [p for p in DATA.rglob("rep*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".bmp"}]
    todo = []
    for p in sorted(imgs):
        parts = p.relative_to(DATA).parts
        if len(parts) < 2:
            continue
        case = "%s/%s" % (parts[0], parts[1])
        if case in locked:                       # locked reports never opened
            continue
        todo.append((case, p))
    if a.limit:
        todo = todo[:a.limit]
    print("report images to read: %d (locked skipped)" % len(todo), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.reparse:
        if not CELLS.exists():
            raise SystemExit("no cached cells at %s; run without --reparse once"
                             % CELLS)
        cached = [json.loads(l) for l in CELLS.open(encoding="utf-8")]
        cached = {(d["case"], d["file"]): d["cells"] for d in cached}
        todo = [(c, DATA / f) for (c, f) in cached]
        print("reparsing %d cached OCR pages (no OCR re-run)" % len(todo),
              flush=True)
        engine = None
    else:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        cached = {}

    rows, n_ok = [], 0
    cell_sink = None if a.reparse else CELLS.open("w", encoding="utf-8")
    with RAW.open("w", encoding="utf-8") as sink:
        for i, (case, p) in enumerate(todo, 1):
            rel = str(p.relative_to(DATA))
            if a.reparse:
                cells = cached[(case, rel)]
            else:
                try:
                    cells = ocr_image(engine, p)
                except Exception as exc:                   # unreadable file
                    sink.write(json.dumps(dict(case=case, file=rel,
                                               error=str(exc)),
                                          ensure_ascii=False) + "\n")
                    continue
                cell_sink.write(json.dumps(dict(case=case, file=rel,
                                                cells=cells),
                                           ensure_ascii=False) + "\n")
            got = read_report(cells)
            sink.write(json.dumps(dict(
                case=case, file=str(p.relative_to(DATA)), **got),
                ensure_ascii=False) + "\n")
            for f, v in got["per_item"].items():
                rows.append(dict(exam_case_id=case,
                                 file=str(p.relative_to(DATA)), field=f,
                                 score=v["score"], conf=v["conf"],
                                 raw=v["raw"], kind="per_item"))
            for f, v in got["totals"].items():
                rows.append(dict(exam_case_id=case,
                                 file=str(p.relative_to(DATA)), field=f,
                                 score=v, conf=None,
                                 raw=got.get("total_raws", {}).get(f),
                                 kind="total"))
            if got["per_item"]:
                n_ok += 1
            if i % 50 == 0:
                print("  read %d/%d, %d with per-item scores"
                      % (i, len(todo), n_ok), flush=True)
    if cell_sink is not None:
        cell_sink.close()

    df = pd.DataFrame(rows)
    if df.empty:
        print("no scores recovered")
        return
    assert not set(df.exam_case_id) & locked, "a locked case reached the output"
    df.to_csv(OUT_CSV, index=False)

    per = df[df.kind == "per_item"]
    tot = df[df.kind == "total"]
    cases = set(per.exam_case_id)

    # Arithmetic consistency: does the printed total match the printed parts?
    # This is the ONLY admissible basis for calling a reading into question,
    # and even then it flags, it does not delete or clip.
    checks = []
    for case, g in tot.groupby("exam_case_id"):
        s = dict(zip(g.field, g.score))
        parts = [s.get(k) for k in
                 ("morphology_score", "flow_score", "periloop_score")]
        if s.get("total_score") is not None and all(v is not None for v in parts):
            checks.append(dict(exam_case_id=case, total=s["total_score"],
                               sum_of_parts=round(sum(parts), 2),
                               agrees=abs(sum(parts) - s["total_score"]) <= 0.15))
    agree = [c for c in checks if c["agrees"]]

    # Stronger check: do the recovered PER-ITEM scores add up to the printed
    # subtotal of their group? The subtotal is independent of the per-item column
    # on the page, so agreement validates the per-item reads themselves, not just
    # the totals row. A mismatch is reported, never repaired.
    grp_checks, fixes = [], []
    for (case, f), gg in per.groupby(["exam_case_id", "file"]):
        got = dict(zip(gg.field, gg.score))
        raws = dict(zip(gg.field, gg.raw))
        cfs = dict(zip(gg.field, gg.conf))
        tl = tot[(tot.exam_case_id == case) & (tot.file == f)]
        printed = dict(zip(tl.field, tl.score))
        traws = dict(zip(tl.field, tl.raw))
        for gname, members in SCORE_GROUPS.items():
            if gname not in printed:
                continue
            missing = [m for m in members if m not in got]
            present = [m for m in members if m in got]
            s = round(sum(got[m] for m in present), 2)
            fixed, sub_fix = [], None
            if abs(s - printed[gname]) > 0.15:
                s, fixed = reread_against_subtotal(
                    {m: got[m] for m in present},
                    {m: raws[m] for m in present}, printed[gname],
                    {m: cfs[m] for m in present})
                for fx in fixed:
                    fixes.append(dict(exam_case_id=case, file=f, group=gname,
                                      **fx))
                    got[fx["field"]] = fx["to"]
            # The subtotal cell is printed in the same font and suffers the same
            # misreads (0.8 rendered "8'0"). When the group is COMPLETE, the item
            # sum is itself report evidence about the subtotal, so the same
            # uniqueness-constrained re-read applies in that direction too.
            if not missing and not fixed and abs(s - printed[gname]) > 0.15:
                alt = reread_cell(traws.get(gname), printed[gname], s)
                if alt is not None:
                    sub_fix = dict(exam_case_id=case, file=f, group=gname,
                                   field=gname, raw=str(traws.get(gname)),
                                   basis=("sum of all %d per-item scores in the "
                                          "group, printed on the same report"
                                          % len(members)),
                                   **{"from": printed[gname], "to": alt})
                    fixes.append(sub_fix)
                    printed[gname] = alt
            grp_checks.append(dict(
                exam_case_id=case, file=f, group=gname,
                printed=printed[gname], sum_of_items=s,
                items_missing=missing,
                corrected_from_subtotal=[fx["field"] for fx in fixed],
                subtotal_corrected_from_items=bool(sub_fix),
                agrees=abs(s - printed[gname]) <= 0.15))
    if fixes:
        pd.DataFrame(fixes).to_csv(
            OUT_DIR / "corrections_from_printed_subtotal.csv", index=False)
        idx = df.set_index(["exam_case_id", "file", "field"]).index
        for fx in fixes:
            m = (df.exam_case_id == fx["exam_case_id"]) & \
                (df.file == fx["file"]) & (df.field == fx["field"]) & \
                (df.kind == "per_item")
            df.loc[m, "score"] = fx["to"]
            df.loc[m, "corrected_from"] = fx["from"]
        df.to_csv(OUT_CSV, index=False)
        per = df[df.kind == "per_item"]
    grp_ok = [c for c in grp_checks if c["agrees"]]
    if grp_checks:
        pd.DataFrame(grp_checks).to_csv(
            OUT_DIR / "group_arithmetic_check.csv", index=False)

    # Case-level consolidation. A case has several report images (rep.jpg,
    # rep_<date>.jpg, rep_左手.jpg) which print the same table, so they are
    # independent reads of one value. Agreement is kept; DISAGREEMENT is written
    # unknown and listed, never resolved by majority or by picking one image,
    # because nothing in the report says which read is right.
    case_rows, conflicts = [], []
    for (case, fld), gg in per.groupby(["exam_case_id", "field"]):
        vals = sorted(set(gg.score.round(2)))
        if len(vals) == 1:
            case_rows.append(dict(exam_case_id=case, field=fld, score=vals[0],
                                  n_images=len(gg), status="agreed"))
        else:
            case_rows.append(dict(exam_case_id=case, field=fld, score=None,
                                  n_images=len(gg), status="unknown_conflict"))
            conflicts.append(dict(exam_case_id=case, field=fld, values=vals,
                                  files=sorted(gg.file.tolist())))
    case_df = pd.DataFrame(case_rows)
    assert case_df.set_index(["exam_case_id", "field"]).index.is_unique, \
        "case-level table must be unique on exam_case_id + field"
    case_df.to_csv(OUT_DIR / "per_item_scores_case_level.csv", index=False)
    if conflicts:
        pd.DataFrame(conflicts).to_csv(
            OUT_DIR / "case_level_conflicts.csv", index=False)

    doc = dict(
        purpose=("Stream D: recover the per-item doctor score printed on the "
                 "report images. The RTF files do not contain it."),
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        rtf_checked=dict(
            files=465, decoder="scripts/dump_rtf.py decode_rtf",
            searched=["积分", "分值", "得分", "计分", "总分", "评分"],
            files_containing_any=0,
            conclusion="the per-item score exists only on the report images"),
        images=dict(report_images_found=len(imgs),
                    read_this_run=len(todo),
                    locked_report_images_skipped=len(imgs) - len(todo),
                    images_yielding_per_item_scores=n_ok),
        coverage=dict(
            cases_with_any_per_item_score=len(cases),
            of_which_in_development=len(cases & dev),
            development_cases_total=len(dev),
            fraction_of_development=round(len(cases & dev) / len(dev), 4),
            per_field_case_counts=per.groupby("field").exam_case_id
            .nunique().sort_values(ascending=False).to_dict()),
        totals_recovered=dict(
            cases_with_a_total=int(tot[tot.field == "total_score"]
                                   .exam_case_id.nunique()),
            arithmetic_checked=len(checks),
            arithmetic_agrees=len(agree),
            arithmetic_disagrees=len(checks) - len(agree),
            note=("a disagreement FLAGS the reading for a human re-read of the "
                  "original report; it never deletes, clips or overwrites a "
                  "value, and no model prediction is used to judge it")),
        per_item_validated_against_printed_subtotals=dict(
            checks=len(grp_checks), agrees=len(grp_ok),
            disagrees=len(grp_checks) - len(grp_ok),
            complete_groups=sum(1 for c in grp_checks if not c["items_missing"]),
            complete_and_agree=sum(1 for c in grp_checks
                                   if not c["items_missing"] and c["agrees"]),
            meaning=("the group subtotal is printed independently of the "
                     "per-item column, so agreement is evidence the per-item "
                     "reads are correct; a group with items_missing can "
                     "legitimately fall short of its subtotal"),
            detail_file="group_arithmetic_check.csv"),
        case_level=dict(
            cells=len(case_df),
            agreed=int((case_df.status == "agreed").sum()),
            unknown_conflict=int((case_df.status == "unknown_conflict").sum()),
            cases=int(case_df.exam_case_id.nunique()),
            rule=("a case's several report images print the same table, so they "
                  "are independent reads; disagreement is written unknown and "
                  "listed in case_level_conflicts.csv, never resolved by "
                  "majority vote or by preferring one image"),
            files=["per_item_scores_case_level.csv", "case_level_conflicts.csv"]),
        unknown_policy=dict(
            min_confidence=MIN_CONF,
            rule=("any cell below the confidence floor, or with no score cell "
                  "on its row, is recorded unknown -- never coerced to normal "
                  "or abnormal"),
            abnormal_correction_rule=("corrected only from original-report "
                                      "evidence; never from a model prediction "
                                      "and never because a value falls outside "
                                      "a printed normal range")),
        forbidden=[
            ("these scores must not be used as a training target: the total and "
             "the five-level assessment are under a standing supervision ban "
             "(nailfold-doctor-scale-audit). They are recovered to explain how "
             "fields are weighted, not to predict the grade"),
            ("rep_* images are report scans and must never become encoder "
             "inputs; this script writes no image feature"),
            "development scope; no product capability claim",
        ],
        locked_cases_seen=0,
    )
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(dict(
        images_read=len(todo), with_scores=n_ok,
        cases=len(cases), in_development=len(cases & dev),
        totals=doc["totals_recovered"], out=str(OUT_CSV.name)),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
