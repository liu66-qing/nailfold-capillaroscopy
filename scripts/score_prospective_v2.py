"""Pair the prospective shadow log with report labels and score it ONCE.

Implements artifacts/prospective_eval_v2/preregistration.md. Written and
tested on synthetic data before the first prospective case, so nothing in the
scoring rule can be adjusted after labels are seen.

  python scripts/score_prospective_v2.py status LOG
      label-free operational counts only (allowed at any time, section 3.4)
  python scripts/score_prospective_v2.py score LOG LABELS_CSV
      refuses unless the stopping point is reached (N=150 same-device persons
      or on/after 2027-09-28) and refuses if a result already exists.

LABELS_CSV: one row per exam, column exam_id plus the report columns with the
same names and raw values as server_code_audit/locked_evaluation_v1_reviewed.csv
(extracted later by the existing OCR / RTF pipeline, never read by inference).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

FROZEN_SHA = "a413286055a8d2f09efbd59d2a25eb4bdb049f6efffab9b756b7a66dbd3d8177"
QUEUE_OPENS = "2026-09-29"
STOP_N = 150
STOP_DATE = "2027-09-28"
SAME_DEVICE_IDS = ("default",)
N_BOOT, BOOT_SEED = 2000, 20260928
OUT = ROOT / "artifacts" / "prospective_eval_v2" / "result.json"

# preregistration section 5, copied verbatim; never edited after the freeze
GATES = {
    # field: (min_pos, min_neg, min_cov, ba, recall, ci_low, calib, ci_half)
    "clarity": (30, 30, 0.90, 0.70, 0.60, 0.60, 0.10, 0.10),
    "subpapillary_venous_plexus": (30, 30, 0.90, 0.70, 0.60, 0.60, 0.10, 0.10),
    "exudation": (30, 30, 0.90, 0.65, 0.55, 0.55, 0.10, 0.10),
    "blood_color": (30, 30, 0.45, 0.70, 0.60, 0.60, 0.10, 0.12),
    "malformation_ratio": (30, 30, 0.45, 0.65, 0.50, 0.55, 0.10, 0.12),
    "microthrombus": (30, 30, 0.45, 0.65, 0.50, 0.55, 0.10, 0.12),
    "loop_length": (30, 30, 0.45, 0.75, 0.65, 0.60, 0.10, 0.12),
}
PRIMARY = "clarity"


def read_log(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    df = pd.DataFrame(rows)
    # a failed inference has no device block; ShadowLog stores the requested id
    req = df["device_id_requested"] if "device_id_requested" in df else pd.Series(None, index=df.index)
    df["device_id"] = [d.get("id") if isinstance(d, dict) else r
                       for d, r in zip(df.device, req)]
    return df


def build_queues(log: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Entry rules of section 2, in the order they are stated."""
    audit = dict(lines=int(len(log)))
    bad_sha = log.bundle_sha256 != FROZEN_SHA
    audit["protocol_deviation_bundle_sha"] = int(bad_sha.sum())
    log = log[~bad_sha]
    early = log.exam_date.astype(str) < QUEUE_OPENS
    audit["excluded_exam_before_queue_opens"] = int(early.sum())
    log = log[~early]
    # a failure with no device id at all is counted against the same-device
    # queue (conservative: it can only lower same-device coverage)
    log = log.assign(queue=np.where(log.device_id.isin(SAME_DEVICE_IDS) | log.device_id.isna(),
                                    "same_device", "cross_device"))
    queues = {}
    for q, g in log.groupby("queue"):
        g = g.sort_values(["exam_date", "logged_at"], kind="stable")
        first = ~g.subject_key.duplicated()
        audit["%s_repeat_exams_logged_not_scored" % q] = int((~first).sum())
        g = g[first]
        dup_exam = g.exam_id.duplicated()
        audit["%s_duplicate_exam_id" % q] = int(dup_exam.sum())
        queues[q] = g[~dup_exam].set_index("exam_id")
    return queues, audit


def status(log_path: Path) -> dict:
    """Label-free counts only; never touches a report."""
    queues, audit = build_queues(read_log(log_path))
    out = dict(audit=audit)
    for q, g in queues.items():
        err = g.error.notna()
        ok = g[~err]
        abst = {f: float(np.mean([r.get(f, {}).get("status") != "answered"
                                  for r in ok.shadow_fields])) if len(ok) else None
                for f in GATES}
        shifted = [d.get("shifted") for d in ok.device if isinstance(d, dict)]
        out[q] = dict(persons=int(len(g)), error_rate=float(err.mean()) if len(g) else None,
                      abstain_rate_among_successful=abst,
                      device_shift_flag_rate=float(np.mean(shifted)) if shifted else None)
    return out


def reference_labels(labels: pd.DataFrame) -> dict[str, pd.Series]:
    import fit_rag_heads as F
    lab = labels.astype({"exam_id": str}).set_index("exam_id")
    return {f: F.target(lab, f) for f in GATES}


def _ba(y, yhat):
    r = [np.mean(yhat[y == c] == c) for c in (0, 1)]
    return float(np.mean(r)), r


def field_metrics(field, g, y, labels_text, rng_seed=BOOT_SEED):
    """g: queue rows; y: reference label per exam_id (0/1), missing = no label."""
    ids = [i for i in g.index if i in y.index]
    n = len(ids)
    yv = y.reindex(ids).to_numpy().astype(int)
    status, pred, prob = [], [], []
    for i in ids:
        row = g.loc[i]
        failed = isinstance(row.error, str) or not isinstance(row.shadow_fields, dict)
        rec = None if failed else row.shadow_fields.get(field)
        if rec is None or rec.get("status") != "answered":
            status.append("abstained"); pred.append(-1); prob.append(np.nan)
        else:
            status.append("answered"); pred.append(labels_text.index(rec["value"]))
            prob.append(rec["p"])
    status, pred, prob = np.array(status), np.array(pred), np.array(prob, float)
    ans = status == "answered"
    m = dict(n_with_label=n, n_without_label=int(len(g) - n),
             coverage=float(ans.mean()) if n else None,
             answered_pos=int(((yv == 1) & ans).sum()), answered_neg=int(((yv == 0) & ans).sum()),
             correct=int((ans & (pred == yv)).sum()), wrong=int((ans & (pred != yv)).sum()),
             abstained=int((~ans).sum()),
             abstain_rate_by_true_class={str(c): float(np.mean(~ans[yv == c])) if (yv == c).any() else None
                                         for c in (0, 1)})
    # full coverage: an abstention is scored as wrong
    full_pred = np.where(ans, pred, 1 - yv)
    m["full_coverage_ba"] = _ba(yv, full_pred)[0] if n else None
    mn_pos, mn_neg, mn_cov, t_ba, t_rec, t_low, t_cal, t_half = GATES[field]
    if m["answered_pos"] == 0 or m["answered_neg"] == 0:
        m["verdict"] = "insufficient_evidence"
        return m
    ya, pa, qa = yv[ans], pred[ans], prob[ans]
    ba, rec = _ba(ya, pa)
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(len(ya))
    boots = []
    for _ in range(N_BOOT):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(ya[s])) == 2:
            boots.append(_ba(ya[s], pa[s])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    calib = abs(float(np.mean(qa)) - float(np.mean(ya)))
    m.update(answered_ba=ba, recall_0=rec[0], recall_1=rec[1],
             ba_ci95=[float(lo), float(hi)], ci_half_width=float((hi - lo) / 2),
             calibration_gap=calib)
    if (m["answered_pos"] < mn_pos or m["answered_neg"] < mn_neg or m["coverage"] < mn_cov):
        m["verdict"] = "insufficient_evidence"
    else:
        checks = dict(ba=ba >= t_ba, recall=min(rec) >= t_rec, ci_low=lo > t_low,
                      calibration=calib <= t_cal, ci_half_width=m["ci_half_width"] <= t_half)
        m["checks"] = {k: bool(v) for k, v in checks.items()}
        m["verdict"] = "pass" if all(checks.values()) else "fail"
    return m


def score(log_path: Path, labels_path: Path, today: str | None = None,
          out: Path = OUT, _allow_early: bool = False) -> dict:
    if out.exists():
        raise SystemExit("already scored once: %s" % out)
    queues, audit = build_queues(read_log(log_path))
    today = today or dt.date.today().isoformat()
    n_same = len(queues.get("same_device", []))
    if not _allow_early and n_same < STOP_N and today < STOP_DATE:
        raise SystemExit("stopping point not reached: %d/%d persons, date %s < %s"
                         % (n_same, STOP_N, today, STOP_DATE))
    import fit_rag_heads as F
    y = reference_labels(pd.read_csv(labels_path, dtype=str))
    res = dict(preregistration="artifacts/prospective_eval_v2/preregistration.md",
               frozen_bundle_sha256=FROZEN_SHA, scored_on=today, audit=audit,
               underpowered=n_same < STOP_N, primary_endpoint=PRIMARY, queues={})
    for q, g in queues.items():
        res["queues"][q] = dict(persons=int(len(g)), fields={
            f: field_metrics(f, g, y[f], list(F.LABELS[f])) for f in GATES})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "status":
        print(json.dumps(status(Path(sys.argv[2])), ensure_ascii=False, indent=2))
    elif cmd == "score":
        print(json.dumps(score(Path(sys.argv[2]), Path(sys.argv[3])), ensure_ascii=False, indent=2))
    else:
        raise SystemExit(__doc__)
