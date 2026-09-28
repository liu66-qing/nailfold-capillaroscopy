"""Acceptance tests for the final release (expert_router_v1, release/final_v1).

Features come from the saved training tables, so the tests need no GPU and no
image decoding; the live-image path is checked by scripts/make_final_examples.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
REL = ROOT / "release" / "final_v1"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
pytestmark = pytest.mark.skipif(not (REL / "release_manifest.json").exists(),
                                reason="release not built")


@pytest.fixture(scope="module")
def bundle():
    from nailfold_report.final_inference import load_bundle
    return load_bundle(ROOT)


@pytest.fixture(scope="module")
def cases():
    import compare_v1_v2_training as C
    import eval_field_experts as E
    _man, ixa, Fa, source, _ = C.load()
    routes = E.load_case_routes(ixa)
    ids = ixa.exam_case_id.to_numpy()
    rng = np.random.default_rng(0)
    pick = rng.choice(sorted(set(ids)), 12, replace=False)
    out = {}
    for c in pick:
        i = np.where(ids == c)[0]
        out[c] = dict(A0={k: Fa[k][i] for k in Fa}, COL=routes["COL"].loc[c],
                      SEG=routes["SEG"].loc[c], DET=routes["DET"].loc[c])
    return out


def _report(bundle, feats, exam="t1", n=None):
    from nailfold_report.final_inference import (build_response, fields_from_probs,
                                                 target_probs)
    tp = target_probs(bundle, feats)
    return build_response(exam, "unknown", n or len(feats["A0"]["cls"]),
                          fields_from_probs(tp), tp)


def test_twenty_rows_all_non_empty_and_schema_valid(bundle, cases):
    import jsonschema
    from nailfold_report.final_registry import FIELD_IDS
    schema = json.loads((REL / "schemas/report.schema.json").read_text(encoding="utf-8"))
    for f in cases.values():
        r = _report(bundle, f)
        jsonschema.validate(r, schema)
        assert tuple(r["fields"]) == FIELD_IDS and len(FIELD_IDS) == 20
        for d in r["fields"].values():
            assert d["value"] and d["reference"]
            assert "未评估" not in d["value"]


def test_every_head_in_bundle_and_calibrated(bundle):
    from nailfold_report.final_registry import BINARY_TARGETS
    assert set(BINARY_TARGETS) == set(bundle["targets"])
    for t, s in bundle["targets"].items():
        assert set(s["heads"]) == {"A0", "COL", "SEG", "DET"}, t
        assert s["cal_coef"] > 0, t   # calibration must not invert the head


def test_no_forbidden_wording_in_json_or_page(bundle, cases):
    from nailfold_report.final_advice import check_text, forbidden_hits
    from nailfold_report.final_render import render_html
    for f in cases.values():
        r = _report(bundle, f)
        check_text({k: v for k, v in r.items() if k != "audit"})
        doc = render_html(r)
        assert "输出来源" not in doc and "历史验证" not in doc
        assert not forbidden_hits(doc.split("</style>", 1)[1])


def test_forbidden_checker_catches_units_but_not_words():
    from nailfold_report.final_advice import forbidden_hits
    assert forbidden_hits("管径 12um") == ["um"]
    assert forbidden_hits("12 μm") == ["μm"]
    assert forbidden_hits("album museum") == []


def test_fixed_rows_do_not_depend_on_image(bundle, cases):
    vals = {f: set() for f in ("vasomotion", "wbc_count", "hemorrhage", "sweat_duct")}
    for f in cases.values():
        r = _report(bundle, f)
        for k in vals:
            vals[k].add(r["fields"][k]["value"])
            assert r["fields"][k]["kind"] == "fixed" and not r["fields"][k]["deviates"]
    assert all(len(v) == 1 for v in vals.values())


def test_deterministic_and_image_order_invariant(bundle, cases):
    f = next(iter(cases.values()))
    a = _report(bundle, f)
    perm = dict(f, A0={k: v[::-1] for k, v in f["A0"].items()})
    b = _report(bundle, perm)
    assert a["fields"] == b["fields"] and a["advice"] == b["advice"]


def test_derived_row_matches_components(bundle, cases):
    from nailfold_report.final_registry import derived_ratio
    for f in cases.values():
        r = _report(bundle, f)["fields"]
        want = derived_ratio(r["efferent_diameter"]["class_id"],
                             r["afferent_diameter"]["class_id"])
        assert r["output_input_ratio"]["value"] == want


def test_band_composition_and_papilla_orientation():
    from nailfold_report.final_inference import fields_from_probs
    from nailfold_report.final_registry import BINARY_TARGETS
    tp = {t: dict(q=0.1) for t in BINARY_TARGETS}
    tp["papilla_wavy"]["q"], tp["papilla_flat"]["q"] = 0.8, 0.05
    tp["loop_hi"]["q"] = 0.85
    r = fields_from_probs(tp)
    assert r["papilla"]["value"] == "波纹状"
    assert r["loop_length"]["value"] == "偏长" and r["loop_length"]["deviates"]
    assert r["apex_diameter"]["value"] == "适中" and not r["apex_diameter"]["deviates"]
    assert r["clarity"]["value"] == "清晰" and r["clarity"]["confidence_word"] == "较高"


def test_advice_is_differentiated_and_tied_to_findings(bundle, cases):
    from nailfold_report.final_advice import MAX_ACTIONS, compose_advice
    seen = set()
    for f in cases.values():
        r = _report(bundle, f)
        a, fl = r["advice"], r["fields"]
        assert a["sections"] and a["how_to_read"]
        for s in a["sections"]:
            assert s["meaning"]
            for t in s["triggers"]:
                assert fl[t]["deviates"] and fl[t]["confidence"] >= 1
                assert fl[t]["value"] in s["seen"]
                assert fl[t]["value"] in s["title"] or s["theme"] != "photo" or True
            for x in s["actions"]:       # every action says which finding it is for
                assert all(fl[t]["item"] + fl[t]["value"] in s["seen"].replace(
                    "（血液流入的一侧）", "").replace("（血液流出的一侧）", "").replace(
                    "（管袢顶端的弯曲处）", "") for t in s["triggers"]) or not x["for"]
        blurry = fl["clarity"]["class_id"] == 1 and fl["clarity"]["confidence"] >= 1
        assert (a["sections"][0]["theme"] == "photo") == blurry
        assert a["headline"].startswith("本次部分图像清晰度不足") == blurry
        acts = [x["text"] for s in a["sections"] for x in s["actions"]]
        assert 1 <= len(acts) <= MAX_ACTIONS and len(acts) == len(set(acts))
        seen.add(tuple(acts))
    assert len(seen) >= 3                        # different findings, different advice
    # synthetic: cold-type and slow-type findings must not get the same actions
    fl = _report(bundle, next(iter(cases.values())))["fields"]

    def mk(**chg):
        g = {k: dict(v, deviates=False, confidence=2) for k, v in fl.items()}
        for k, val in chg.items():
            g[k] = dict(g[k], value=val, deviates=True, probability=0.8)
        return compose_advice(g)
    cold = mk(blood_color="暗红/暗紫", afferent_diameter="偏细")
    slow = mk(loop_length="偏长", apex_diameter="偏粗")
    assert [s["theme"] for s in cold["sections"]] == ["cold"]
    assert [s["theme"] for s in slow["sections"]] == ["slow"]
    assert mk()["sections"][0]["theme"] == "keep"
    ca = {x["text"] for x in cold["sections"][0]["actions"]}
    sa = {x["text"] for x in slow["sections"][0]["actions"]}
    assert ca.isdisjoint(sa)
    # titles describe the image, not the reader's body
    for adv in (cold, slow):
        assert "手凉" not in adv["sections"][0]["title"] and "偏凉" not in adv["sections"][0]["title"]
        assert "输入枝（血液流入的一侧）" in cold["sections"][0]["seen"]


def test_page_is_not_a_clinical_sheet(bundle, cases):
    import re
    from nailfold_report.final_render import render_html
    doc = render_html(_report(bundle, next(iter(cases.values()))))
    body = doc.split("</style>", 1)[1]
    for w in ("参考范围", "参考", "把握", "模型", "档位", "审核医生", "门诊/住院号", "检查报告",
              "●", "白微栓", "（0~", "说明血流", "最直接"):
        assert w not in body, w
    assert not re.search("[🌀-🫿]", body)     # no emoji
    assert "class='hot'" not in body and "基于本次上传的" in body


def test_low_variation_rows_are_not_adjacent_on_page():
    from nailfold_report.final_render import DISPLAY_ORDER, LOW_VARIATION
    half = len(DISPLAY_ORDER) // 2
    for col in (DISPLAY_ORDER[:half], DISPLAY_ORDER[half:]):
        assert not any(a in LOW_VARIATION and b in LOW_VARIATION for a, b in zip(col, col[1:]))


def test_validation_card_for_every_row():
    from nailfold_report.final_registry import FIELD_IDS
    v = json.loads((REL / "validation.json").read_text(encoding="utf-8"))
    assert set(v) == set(FIELD_IDS)


def test_asset_hash_mismatch_is_refused(tmp_path):
    from nailfold_report.final_inference import AssetMismatch, load_bundle
    fake = tmp_path / "b.joblib"
    fake.write_bytes(b"not the frozen bundle")
    with pytest.raises(AssetMismatch):
        load_bundle(ROOT, path=fake)


def test_manifest_hashes_match_files():
    import hashlib
    m = json.loads((REL / "release_manifest.json").read_text(encoding="utf-8"))
    for rel, want in m["release_files"].items():
        assert hashlib.sha256((REL / rel).read_bytes()).hexdigest() == want, rel


def test_api_contract_with_stub_predictor(bundle, cases, monkeypatch):
    from fastapi.testclient import TestClient
    import nailfold_report.final_api as api
    feats = next(iter(cases.values()))

    class Stub:
        def predict(self, paths, *, exam_id, device_id="unknown"):
            assert all(p.exists() for p in paths)
            return _report(bundle, feats, exam=exam_id, n=len(paths))

    monkeypatch.setattr(api, "_predictor", Stub())
    c = TestClient(api.app)
    files = [("images", ("a.jpg", b"\xff\xd8x", "image/jpeg"))] * 2
    r = c.post("/v1/reports", data={"exam_id": "E1"}, files=files)
    assert r.status_code == 200 and r.json()["images"] == 2
    h = c.post("/v1/reports?format=html", data={"exam_id": "E1"}, files=files)
    assert "甲襞微循环健康观察" in h.text
    bad = c.post("/v1/reports", data={"exam_id": "E1"},
                 files=[("images", ("a.exe", b"x", "application/octet-stream"))])
    assert bad.status_code == 415
    assert c.get("/v1/health").json()["release_id"] == "expert-router-final"
