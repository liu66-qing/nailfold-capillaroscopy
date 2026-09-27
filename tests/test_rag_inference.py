from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "artifacts" / "models" / "rag_heads_v2" / "bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="bundle not fitted")


@pytest.fixture(scope="module")
def bundle():
    import joblib
    return joblib.load(BUNDLE)


def _fake(bundle, shift=0.0, scale=1.0, n=6, seed=0):
    rng = np.random.default_rng(seed)
    return {k: (bundle["reference"][k]["mean"]
                + rng.standard_normal((n, len(bundle["reference"][k]["mean"])))
                * bundle["reference"][k]["std"]) * scale + shift
            for k in bundle["poolings"]}


def test_contract_has_no_numbers_and_marks_unmodelled(bundle):
    from nailfold_report.rag_inference import NOT_MODELLED, RagFieldPredictor
    p = RagFieldPredictor.__new__(RagFieldPredictor)
    p.bundle = bundle
    out = p.predict_features(_fake(bundle))
    for f, v in out.items():
        assert v["status"] in {"answered", "abstained", "not_modelled"}
        assert v["value"] is None or isinstance(v["value"], str)
        if v["status"] != "answered":
            assert v["value"] is None
    for f in NOT_MODELLED:
        assert out[f]["status"] == "not_modelled"
    for f in ("clarity", "subpapillary_venous_plexus", "exudation"):
        assert out[f]["status"] == "answered"   # no abstention on these three


def test_calibrator_waits_then_flags_foreign_device(bundle):
    from nailfold_report.rag_inference import DeviceCalibrator
    c = DeviceCalibrator(bundle)
    for i in range(c.need - 1):
        c.add("new", _fake(bundle, shift=3.0, seed=i))
    assert c.state("new")["recentre"] is False
    c.add("new", _fake(bundle, shift=3.0, seed=99))
    st = c.state("new")
    assert st["shifted"] and st["recentre"]
    feats, _ = c.transform("new", _fake(bundle, shift=3.0, seed=7))
    ref = bundle["reference"]["cls"]
    assert np.abs(feats["cls"].mean(0) - ref["mean"]).mean() < \
        np.abs(3.0 + ref["mean"] - ref["mean"]).mean()


def test_withheld_field_is_not_modelled(bundle):
    from nailfold_report.rag_inference import RagFieldPredictor
    p = RagFieldPredictor.__new__(RagFieldPredictor)
    p.bundle = bundle
    out = p.predict_features(_fake(bundle))
    assert "loop_length" in bundle["withheld"]
    assert out["loop_length"]["status"] == "not_modelled"
    assert out["microthrombus"]["role"] == "static_correlation_only_not_for_advice"
    assert out["clarity"]["role"] == "quality_gate"


def test_withheld_field_is_still_in_shadow_predictions(bundle):
    from nailfold_report.rag_inference import RagFieldPredictor
    p = RagFieldPredictor.__new__(RagFieldPredictor)
    p.bundle = bundle
    raw = p.raw_predictions(_fake(bundle))
    assert raw["loop_length"]["status"] in {"answered", "abstained"}
    assert p.predict_features(_fake(bundle))["loop_length"]["status"] == "not_modelled"


def test_shadow_log_appends_and_keeps_failures(tmp_path):
    import json
    from nailfold_report.rag_inference import ShadowLog
    log = ShadowLog(tmp_path / "shadow.jsonl", BUNDLE, "deadbeef")
    log.record("e1", "s1", "2026-10-01", None, None, error="decode failed")
    log.record("e2", "s2", "2026-10-01", None, None, error="no images")
    rows = [json.loads(l) for l in (tmp_path / "shadow.jsonl").read_text("utf-8").splitlines()]
    assert [r["exam_id"] for r in rows] == ["e1", "e2"]
    assert rows[0]["error"] == "decode failed" and len(rows[0]["bundle_sha256"]) == 64
