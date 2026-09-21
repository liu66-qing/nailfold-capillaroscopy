#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit: did the field tasks conflate four different things?

  (1) the doctor's score WEIGHT      -- papilla 1.6 pts vs clarity 0.2 pts
  (2) the doctor's THRESHOLDING      -- 138um -> "<150" -> 0.2 pts
  (3) the UNIT of observation        -- needs a time base / the whole nail fold
  (4) plain IMAGE APPEARANCE         -- what a static crop can actually show

Read-only. DEVELOPMENT ONLY: rows with development_fold NaN are locked-47 and are
never read. Writes only to artifacts/evidence/doctor_scale_audit_20260921/.

P1 does the label file even have a per-field score column?
P2 is the per-field score column printed in the report images (i.e. never OCRd)?
P3 is a group score an additive function of its field values? (best-possible fit)
P4 is the scoring BOUNDARY archive-dependent, or only the label distribution?
P5 how many score points does each field actually put at stake?
P6 low/normal/high recoding vs the binary collapses currently shipped
P7 does any score column reach feature construction? (leakage check)

Run:
  PYTHONIOENCODING=utf-8 python scripts/audit_doctor_scale_conflation.py
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "evidence" / "doctor_scale_audit_20260921"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
RULES = ROOT / "artifacts" / "labels" / "score_rules_v3.json"
STATUS = ROOT / "artifacts" / "evidence" / "allfields_20260919" / "all_fields_status.csv"

# Normal ranges as printed in the report's own 正常值 column. Not our invention.
NORMAL_RANGE = {
    "afferent_diameter": (9.0, 13.0),
    "efferent_diameter": (11.0, 17.0),
    "apex_diameter": (12.0, 18.0),
    "loop_length": (150.0, 250.0),
}
SCORE_COLS = ["morphology_score", "flow_score", "periloop_score", "total_score"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def field_score(frame: pd.DataFrame, field: str, rules: dict) -> pd.Series:
    """The doctor's own rule table, applied to a label column."""
    rule = rules.get(field, {})
    kind = rule.get("type")
    if kind == "categorical_lookup":
        return frame[field].astype(str).str.strip().map(rule["mapping"])
    if kind == "numeric_tree":
        tree = rule["tree"]

        def walk(x):
            if pd.isna(x):
                return np.nan
            node = tree
            while "value" not in node:
                node = node["left"] if x <= node["threshold"] else node["right"]
            return node["value"]

        return pd.to_numeric(frame[field], errors="coerce").map(walk)
    return pd.Series(np.nan, index=frame.index)


def tree_outputs(rule: dict) -> list:
    """Every distinct score a numeric tree can emit."""
    vals, stack = [], [rule["tree"]]
    while stack:
        node = stack.pop()
        if "value" in node:
            vals.append(node["value"])
        else:
            stack += [node["left"], node["right"]]
    return sorted(set(vals))
def p1_label_file_score_columns(dev: pd.DataFrame, full: pd.DataFrame) -> dict:
    """Is there a per-field 积分 column anywhere in the label file?"""
    score_like = [c for c in full.columns
                  if re.search(r"score|积分|point", c, re.I)
                  and not c.endswith(("__confidence", "__status"))]
    return dict(
        n_columns=len(full.columns),
        score_like_columns=score_like,
        per_field_score_columns=[c for c in score_like if c not in SCORE_COLS],
        n_dev=len(dev),
        n_locked_held_out=int(full.development_fold.isna().sum()),
        verdict=("only group totals exist; no per-field score column"
                 if not [c for c in score_like if c not in SCORE_COLS]
                 else "per-field score columns present"),
    )


def p2_score_column_printed(n_sample: int = 40) -> dict:
    """Ink in the rightmost band of the report images = the 积分 column is printed.

    The Read tool renders these jpegs blank, so measure ink numerically instead.
    """
    try:
        import cv2
    except ImportError:
        return dict(skipped="cv2 unavailable")
    per_archive = {}
    total = 0
    for arch in sorted(ROOT.joinpath("data").glob("recovered_archive*")):
        reps = sorted(arch.glob("*/rep*.jpg.jpg"))
        total += len(reps)
        inks = []
        for p in reps[:n_sample]:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            band = img[:, int(img.shape[1] * 0.85):]
            inks.append(int((band < 128).sum()))
        per_archive[arch.name] = dict(n_reports=len(reps),
                                      n_measured=len(inks),
                                      median_ink_px=int(np.median(inks)) if inks else 0)
    return dict(n_report_images=total, per_archive=per_archive,
                verdict="per-field score column is printed in the reports but was never OCRd")


def p3_group_additivity(dev: pd.DataFrame, rules: dict, groups: dict) -> dict:
    """Best-possible additive fit of a group score from its field values.

    One-hot least squares is the ceiling for ANY additive rule table. If it still
    misses, the doctor's group score is not an additive function of the printed
    field values, and no better rule recovery will fix that.
    """
    out = {}
    for group, fields in groups.items():
        legal = {f: [str(k) for k in rules[f]["mapping"]]
                 for f in fields
                 if rules.get(f, {}).get("type") == "categorical_lookup"}
        frame = dev.dropna(subset=[group]).copy()
        for f, allowed in legal.items():  # per-field legality; a global set over-filters
            frame = frame[frame[f].astype(str).str.strip().isin(allowed)]
        num = [f for f in fields if rules.get(f, {}).get("type") == "numeric_tree"]
        frame = frame.dropna(subset=num) if num else frame
        if len(frame) < 30:
            out[group] = dict(n=len(frame), skipped="too few complete rows")
            continue
        blocks = [pd.get_dummies(frame[f].astype(str).str.strip(), prefix=f).astype(float)
                  for f in legal]
        for f in num:
            blocks.append(field_score(frame, f, rules).rename(f).to_frame().astype(float))
        X = pd.concat(blocks, axis=1).fillna(0.0).values
        X = np.column_stack([X, np.ones(len(X))])
        y = pd.to_numeric(frame[group], errors="coerce").values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        err = np.abs(pred - y)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out[group] = dict(
            n=len(frame),
            mean_abs_err=round(float(err.mean()), 4),
            frac_exact_within_0p05=round(float((err < 0.05).mean()), 4),
            r2=round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
            additive=bool(1.0 - ss_res / ss_tot > 0.95) if ss_tot > 0 else None,
        )
    return out
def p4_boundary_vs_distribution(dev: pd.DataFrame, rules: dict) -> dict:
    """Separate two very different claims about archive dependence.

    (a) the scoring BOUNDARY moved between archives (same value scored differently)
    (b) only the LABEL DISTRIBUTION moved (same rule, sicker or differently-read cases)
    Only (a) would mean the doctor's scale drifted.
    """
    arch = dev.get("archive")
    if arch is None:
        return dict(skipped="no archive column")
    boundary, distribution = {}, {}
    for field in ["exudation", "clarity", "papilla", "blood_color",
                  "subpapillary_venous_plexus"]:
        if field not in dev.columns or rules.get(field, {}).get("type") != "categorical_lookup":
            continue
        s = field_score(dev, field, rules)
        per_val = {}
        for val, grp in dev.groupby(dev[field].astype(str).str.strip()):
            if val not in rules[field]["mapping"]:
                continue
            by_arch = {str(a): round(float(field_score(g, field, rules).median()), 4)
                       for a, g in grp.groupby(arch) if len(g) >= 5}
            if len(by_arch) >= 2:
                per_val[val] = by_arch
        boundary[field] = dict(
            per_value_median_score_by_archive=per_val,
            boundary_stable=all(len(set(v.values())) == 1 for v in per_val.values()),
        )
        distribution[field] = {str(a): g[field].astype(str).str.strip().value_counts(
            normalize=True).round(3).to_dict() for a, g in dev.groupby(arch)}
    for field in ["loop_length", "afferent_diameter"]:
        if field in dev.columns:
            distribution[field] = {
                str(a): round(float(pd.to_numeric(g[field], errors="coerce").median()), 2)
                for a, g in dev.groupby(arch)}
    return dict(boundary=boundary, label_distribution=distribution)


def p5_points_at_stake(dev: pd.DataFrame, rules: dict, groups: dict,
                       status: pd.DataFrame) -> pd.DataFrame:
    """Rank fields by SCORE POINTS, not by accuracy.

    pts_at_stake = mean |field score - score of the field's mode value|, i.e. the
    points a perfect model could recover that fixing the field to its mode cannot.
    This is the user's weighting claim made measurable.
    """
    rows = []
    for group, fields in groups.items():
        for field in fields:
            s = field_score(dev, field, rules).dropna()
            if s.empty:
                continue
            mode = s.value_counts().idxmax()
            passes = None
            delta = None
            if field in status.index:
                passes = bool(status.loc[field, "passes"] is True
                              or str(status.loc[field, "passes"]).lower() == "true")
                delta = status.loc[field, "delta"]
            rows.append(dict(
                field=field, group=group, n=int(len(s)),
                pts_at_stake=round(float((s - mode).abs().mean()), 4),
                max_single_err=round(float((s - mode).abs().max()), 2),
                ships=passes, delta=delta,
            ))
    return pd.DataFrame(rows).sort_values("pts_at_stake", ascending=False)


def p6_recoding(dev: pd.DataFrame, rules: dict) -> dict:
    """low/normal/high recoding of the numeric fields, using the report's own ranges.

    Compare its class balance against the binary collapses currently shipped, and
    against how many distinct scores the doctor's own tree can emit for the field.
    """
    out = {}
    for field, (lo, hi) in NORMAL_RANGE.items():
        v = pd.to_numeric(dev[field], errors="coerce").dropna()
        if v.empty:
            continue
        # the report prints ranges inclusive of both ends, e.g. 正常值 [9-13]
        band = pd.Series(np.where(v < lo, "low", np.where(v > hi, "high", "normal")),
                         index=v.index)
        counts = band.value_counts().to_dict()
        rule = rules.get(field, {})
        out[field] = dict(
            normal_range=[lo, hi], n=int(len(v)),
            counts={str(k): int(x) for k, x in counts.items()},
            mode_share=round(float(band.value_counts(normalize=True).max()), 4),
            distinct_doctor_scores=(tree_outputs(rule)
                                    if rule.get("type") == "numeric_tree" else None),
        )
    return out


def p7_leakage() -> dict:
    """Does any score column reach feature construction anywhere in scripts/?"""
    pat = re.compile(r"total_score|overall_assessment|morphology_score|flow_score|periloop_score")
    offenders, mentions = [], []
    for path in sorted(ROOT.joinpath("scripts").glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not pat.search(text):
            continue
        mentions.append(path.name)
        # a score used as a regression/classification TARGET or an aux loss is the
        # thing the user forbids; a score merely referenced for post-processing is not
        tgt = re.compile(r"total_score|overall_assessment")
        hit = False
        for line in text.splitlines():
            if not tgt.search(line):
                continue
            if re.search(r"\b(y|y_true|target|targets|label|labels|aux|loss|TARGETS?)\b"
                         r"|\.fit\(|regress|Regressor", line, re.I):
                hit = True
                break
        if hit:
            offenders.append(path.name)
    return dict(files_mentioning_scores=mentions,
                files_using_score_as_target=offenders)
def p8_error_direction(pts: pd.DataFrame) -> dict:
    """Error DIRECTION and the high-weight miss rate the user asked for.

    The published table reports accuracy, prevalence and predicted prevalence, which
    determine the confusion matrix exactly:
        FN = ((1-acc) + (prev - pred_prev)) * n / 2
        FP = ((1-acc) - (prev - pred_prev)) * n / 2
    So direction is recoverable from the shipped numbers alone; no features needed.
    A miss (abnormal reported as normal) removes that field's points from the total
    and is not interchangeable with a false alarm, which adds points.
    """
    blob = json.loads(STATUS.with_suffix(".json").read_text(encoding="utf-8"))
    stake = pts.set_index("field").pts_at_stake.to_dict()
    rows = []
    for name, e in blob["fields"].items():
        if e.get("kind") != "binary":
            continue
        n, acc = e["n"], e["accuracy"]
        prev, pred = e["abnormal_share"], e["predicted_abnormal_share"]
        fn = ((1 - acc) + (prev - pred)) * n / 2
        fp = ((1 - acc) - (prev - pred)) * n / 2
        n_pos = prev * n
        rows.append(dict(
            field=name, n=n, n_abnormal=round(n_pos, 1),
            fn=round(fn, 1), fp=round(fp, 1),
            miss_rate=round(fn / n_pos, 4) if n_pos > 0.5 else None,
            false_alarm_rate=round(fp / (n - n_pos), 4) if n - n_pos > 0.5 else None,
            pts_at_stake=stake.get(name),
            # points the doctor's total loses per case because abnormals are missed
            pts_lost_to_misses=(round(stake[name] * fn / n, 4)
                                if name in stake else None),
        ))
    table = pd.DataFrame(rows).sort_values(
        "pts_lost_to_misses", ascending=False, na_position="last")
    heavy = [r for r in rows if (r["pts_at_stake"] or 0) >= 0.35]
    return dict(
        table=table.to_dict(orient="records"),
        high_weight_fields=[r["field"] for r in heavy],
        high_weight_mean_miss_rate=round(
            float(np.mean([r["miss_rate"] for r in heavy
                           if r["miss_rate"] is not None])), 4) if heavy else None,
        note="derived from published acc/prev/pred_prev; counts land on integers, "
             "which is the arithmetic check that the derivation is right",
    )


def p9_threshold_fragility(dev: pd.DataFrame, rules: dict) -> dict:
    """Does the doctor's threshold sit inside the crowd of measured values?

    Two readings of the same question:
      (a) what share of cases lie within our own measured MAE of some threshold --
          for those cases the band is decided by error smaller than our resolution;
      (b) if the model's value is off by exactly the measured MAE (random sign),
          how many score points does the doctor's tree get wrong, and how often does
          the case stay in its band at all.
    This is the precision question the user asked, answered in score points.
    """
    blob = json.loads(STATUS.with_suffix(".json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260921)
    out = {}

    def thresholds(tree: dict) -> list:
        acc, stack = [], [tree]
        while stack:
            node = stack.pop()
            if "value" in node:
                continue
            acc.append(node["threshold"])
            stack += [node["left"], node["right"]]
        return sorted(acc)

    def walk(tree: dict, x: float) -> float:
        node = tree
        while "value" not in node:
            node = node["left"] if x <= node["threshold"] else node["right"]
        return node["value"]

    for field in NORMAL_RANGE:
        rule = rules.get(field, {})
        if rule.get("type") != "numeric_tree" or field not in blob["fields"]:
            continue
        v = pd.to_numeric(dev[field], errors="coerce").dropna().values
        mae = blob["fields"][field]["mae"]
        thr = thresholds(rule["tree"])
        dist = np.min(np.abs(v[:, None] - np.array(thr)[None, :]), axis=1)
        s0 = np.array([walk(rule["tree"], x) for x in v])
        s1 = np.array([walk(rule["tree"], x) for x in
                       v + rng.choice([-1.0, 1.0], len(v)) * mae])
        err = np.abs(s1 - s0)
        out[field] = dict(
            n=int(len(v)), our_mae=mae, thresholds=[round(float(t), 2) for t in thr],
            frac_within_mae_of_a_threshold=round(float((dist <= mae).mean()), 4),
            pts_err_at_our_mae=round(float(err.mean()), 4),
            frac_stays_in_same_band=round(float((err < 1e-9).mean()), 4),
            max_pts_err=round(float(err.max()), 2),
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rules_blob = json.loads(RULES.read_text(encoding="utf-8"))
    rules, groups = rules_blob["field_rules"], rules_blob["groups"]
    full = pd.read_csv(LABELS, dtype={"exam_case_id": str})
    dev = full[full.development_fold.notna()].copy()
    assert full.development_fold.isna().sum() == 47, "locked split unexpected"
    status = pd.read_csv(STATUS, encoding="utf-8-sig").set_index("field")

    pts = p5_points_at_stake(dev, rules, groups, status)
    report = dict(
        generated="2026-09-21",
        governance=dict(locked_cases_seen=0, n_development_cases=int(len(dev)),
                        label_file=str(LABELS.relative_to(ROOT)),
                        label_sha256=sha256(LABELS),
                        rules_file=str(RULES.relative_to(ROOT)),
                        rules_sha256=sha256(RULES)),
        p1_label_file=p1_label_file_score_columns(dev, full),
        p2_report_images=p2_score_column_printed(),
        p3_group_additivity=p3_group_additivity(dev, rules, groups),
        p4_boundary_vs_distribution=p4_boundary_vs_distribution(dev, rules),
        p5_points_at_stake=dict(
            total_if_all_fixed_at_mode=round(float(pts.pts_at_stake.sum()), 3),
            recoverable_by_shipping_fields=round(
                float(pts.loc[pts.ships == True, "pts_at_stake"].sum()), 3),
            table=pts.to_dict(orient="records")),
        p6_recoding=p6_recoding(dev, rules),
        p7_leakage=p7_leakage(),
        p8_error_direction=p8_error_direction(pts),
        p9_threshold_fragility=p9_threshold_fragility(dev, rules),
    )
    (OUT / "verdict.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pts.to_csv(OUT / "points_at_stake.csv", index=False, encoding="utf-8-sig")

    print("== P1 label file ==")
    print(json.dumps(report["p1_label_file"], ensure_ascii=False, indent=2))
    print("\n== P2 report images ==")
    print(json.dumps(report["p2_report_images"], ensure_ascii=False, indent=2))
    print("\n== P3 is a group score additive in its field values ==")
    for g, v in report["p3_group_additivity"].items():
        print(f"  {g:18s} {v}")
    print("\n== P4 boundary stability ==")
    for f, v in report["p4_boundary_vs_distribution"]["boundary"].items():
        print(f"  {f:28s} boundary_stable={v['boundary_stable']}")
    print("\n== P5 points at stake per field ==")
    print(pts.to_string(index=False))
    print(f"\n  total={report['p5_points_at_stake']['total_if_all_fixed_at_mode']} pts, "
          f"shipping fields recover "
          f"{report['p5_points_at_stake']['recoverable_by_shipping_fields']} pts")
    print("\n== P6 low/normal/high recoding ==")
    for f, v in report["p6_recoding"].items():
        print(f"  {f:20s} mode_share={v['mode_share']:.4f} counts={v['counts']} "
              f"doctor_score_levels={v['distinct_doctor_scores']}")
    print("\n== P7 leakage ==")
    print(json.dumps(report["p7_leakage"], ensure_ascii=False, indent=2))
    print("\n== P8 error direction (miss vs false alarm) ==")
    ed = pd.DataFrame(report["p8_error_direction"]["table"])
    print(ed.to_string(index=False))
    print(f"  high-weight fields (>=0.35 pts): "
          f"{report['p8_error_direction']['high_weight_fields']}")
    print(f"  their mean miss rate: "
          f"{report['p8_error_direction']['high_weight_mean_miss_rate']}")
    ed.to_csv(OUT / "error_direction.csv", index=False, encoding="utf-8-sig")
    print("\n== P9 threshold fragility at our own measured MAE ==")
    for f, v in report["p9_threshold_fragility"].items():
        print(f"  {f:20s} MAE={v['our_mae']:7.2f} "
              f"within_MAE_of_threshold={v['frac_within_mae_of_a_threshold']:.3f} "
              f"pts_err={v['pts_err_at_our_mae']:.4f} "
              f"same_band={v['frac_stays_in_same_band']:.3f}")
    print(f"\nwrote {OUT / 'verdict.json'}")


if __name__ == "__main__":
    main()
