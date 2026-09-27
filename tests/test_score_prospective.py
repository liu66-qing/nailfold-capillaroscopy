"""The prospective scorer, exercised on synthetic logs before any real case."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
pytestmark = pytest.mark.skipif(not LABELS.exists(), reason="labels absent")


def _synthetic(tmp_path, n=160, oracle=True, seed=0):
    """Fake log whose predictions are either the true label (oracle) or random,
    with report values borrowed from real rows so target() mapping is real."""
    import fit_rag_heads as F
    import score_prospective_v2 as S
    real = pd.read_csv(LABELS, dtype=str)
    rng = np.random.default_rng(seed)
    rows = real.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    rows["exam_id"] = ["E%04d" % i for i in range(n)]
    lab = rows.drop(columns=["exam_case_id"]).copy()
    y = S.reference_labels(lab)
    lines = []
    for i, r in rows.iterrows():
        eid = r.exam_id
        shadow = {}
        for f in S.GATES:
            if eid not in y[f].index:
                shadow[f] = dict(value=None, status="abstained", p=0.5)
                continue
            t = int(y[f][eid]) if oracle else int(rng.integers(2))
            p = 0.9 if t else 0.1
            shadow[f] = dict(value=F.LABELS[f][t], status="answered", p=p)
        lines.append(dict(exam_id=eid, subject_key="P%04d" % i, exam_date="2026-10-01",
                          logged_at="2026-10-01T00:00:%02d" % (i % 60),
                          bundle_sha256=S.FROZEN_SHA, commit="x",
                          device=dict(id="default", shifted=False),
                          shadow_fields=shadow, error=None))
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    lp = tmp_path / "labels.csv"
    lab.to_csv(lp, index=False)
    return log, lp, lines


def test_entry_rules(tmp_path):
    import score_prospective_v2 as S
    log, _, lines = _synthetic(tmp_path, n=10)
    extra = [dict(lines[0], exam_id="R1", exam_date="2026-11-01"),           # repeat person
             dict(lines[1], exam_id="B1", subject_key="Q1", bundle_sha256="bad"),
             dict(lines[2], exam_id="O1", subject_key="Q2", exam_date="2026-09-20"),
             dict(lines[3], exam_id="X1", subject_key="Q3", device=dict(id="devB")),
             dict(exam_id="F1", subject_key="Q4", exam_date="2026-10-02", logged_at="z",
                  bundle_sha256=S.FROZEN_SHA, commit="x", device=None,
                  shadow_fields=None, error="crash", device_id_requested="default")]
    with log.open("a", encoding="utf-8") as fh:
        for e in extra:
            fh.write("\n" + json.dumps(e, ensure_ascii=False))
    q, audit = S.build_queues(S.read_log(log))
    assert audit["protocol_deviation_bundle_sha"] == 1
    assert audit["excluded_exam_before_queue_opens"] == 1
    assert audit["same_device_repeat_exams_logged_not_scored"] == 1
    assert "R1" not in q["same_device"].index and "F1" in q["same_device"].index
    assert list(q["cross_device"].index) == ["X1"]


def test_refuses_before_stopping_point_and_twice(tmp_path):
    import score_prospective_v2 as S
    log, lp, _ = _synthetic(tmp_path, n=20)
    out = tmp_path / "res.json"
    with pytest.raises(SystemExit, match="stopping point"):
        S.score(log, lp, today="2026-12-01", out=out)
    S.score(log, lp, today="2027-09-28", out=out)
    with pytest.raises(SystemExit, match="already scored"):
        S.score(log, lp, today="2027-09-28", out=out)


def test_oracle_passes_random_fails(tmp_path):
    import score_prospective_v2 as S
    log, lp, _ = _synthetic(tmp_path, n=400, oracle=True)
    r = S.score(log, lp, today="2027-01-01", out=tmp_path / "a.json")
    f = r["queues"]["same_device"]["fields"]
    assert f["clarity"]["answered_ba"] == 1.0
    assert f["clarity"]["verdict"] == "pass"
    (tmp_path / "b").mkdir()
    log2, lp2, _ = _synthetic(tmp_path / "b", n=400, oracle=False, seed=1)
    r2 = S.score(log2, lp2, today="2027-01-01", out=tmp_path / "b.json")
    for name, m in r2["queues"]["same_device"]["fields"].items():
        assert m["verdict"] in {"fail", "insufficient_evidence"}, name


def test_failure_counts_as_abstained(tmp_path):
    import score_prospective_v2 as S
    log, lp, lines = _synthetic(tmp_path, n=200)
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    for r in rows[:100]:
        r.update(shadow_fields=None, error="crash", device=None, device_id_requested="default")
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    r = S.score(log, lp, today="2027-01-01", out=tmp_path / "c.json")
    c = r["queues"]["same_device"]["fields"]["clarity"]
    assert c["coverage"] < 0.6 and c["verdict"] == "insufficient_evidence"
