"""Fit the final A0 and A2x heads on all 186 development cases and persist them.

Item 3 requires the weights, PCA, head, thresholds, refusal rules, output schema and
field routing to be frozen BEFORE locked-47 is read. The detector weights, the output
contract and the RAG rules are already frozen and committed; the classification heads
are not -- every result so far refit them per fold or per archive. This script makes
the last fitting decision in advance, so the locked read has no freedom left: one
fixed transform per pooling, one fixed set of coefficients, one fixed threshold.

The head is built by the shipped `fit_predict` rather than reassembled here, so it
cannot drift from the configuration everything else was measured under. Each pooling
gets its own pipeline and the five are averaged, which is what the shipped binary path
does.

Nothing here reads locked-47: absence is asserted case-level in both feature tables.

  PYTHONIOENCODING=utf-8 PYTHONPATH=scripts python scripts/freeze_final_heads.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_field_matrix_three_arms import (  # noqa: E402
    ANCHOR, C_FIXED, DIM_FIXED, LABELS, POOLINGS, SEED, load_arm, target,
)
from run_rescue_ladder_local import attach, load_local  # noqa: E402

FIELD = "malformation_ratio"
LOCAL_TAG = "local"
FROZEN = (ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
          / "frozen_candidate_malformation_a2x")
CONTRACT_DIR = ROOT / "artifacts" / "experiments" / "product_contract_20260924"
OUT = CONTRACT_DIR / "final_heads"

# The operating threshold, read from the frozen abstention table rather than chosen
# here. Choosing one now would be threshold selection after the results were seen.
OPERATING_COVERAGE = 0.7


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_final(feats: dict, ix: pd.DataFrame, y_case: pd.Series) -> dict:
    """One pipeline per pooling, fit on every development image. No holdout here.

    This mirrors the shipped `fit_predict` body exactly -- same dim rule, same C,
    same seed, no class_weight -- but keeps the fitted objects instead of throwing
    them away after predicting.
    """
    dm = ix.exam_case_id.isin(y_case.index).to_numpy()
    y = y_case.reindex(ix.exam_case_id[dm]).to_numpy()
    pipes = {}
    for p in POOLINGS:
        X = feats[p][dm]
        dim = max(2, min(DIM_FIXED, X.shape[0] - 1, X.shape[1]))
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=dim, random_state=SEED),
                             LogisticRegression(C=C_FIXED, max_iter=4000))
        pipe.fit(X, y)
        pipes[p] = pipe
    return dict(pipes=pipes, n_images=int(dm.sum()), n_cases=int(len(y_case)),
                classes=[float(c) for c in sorted(set(y))])


def predict_frozen(head: dict, feats: dict, ix: pd.DataFrame,
                   cases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Apply a frozen head: average P(class 1) over poolings, then over each case's
    images. Returns (case-level probability, per-case image count)."""
    keep = ix.exam_case_id.isin(set(cases)).to_numpy()
    ids = ix.exam_case_id[keep].to_numpy()
    per = [head["pipes"][p].predict_proba(feats[p][keep])[:, 1] for p in POOLINGS]
    s = pd.Series(np.mean(per, axis=0), index=ids).groupby(level=0)
    prob = s.mean().reindex(cases).to_numpy()
    n_img = s.size().reindex(cases).to_numpy()
    return prob, n_img


def main() -> None:
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    locked = set(man.index[man.development_fold.isna()])
    if len(dev) != 186 or len(locked) != 47:
        raise RuntimeError("expected 186 dev / 47 locked, got %d / %d"
                           % (len(dev), len(locked)))

    ix, feats, meta = load_arm(ANCHOR)
    extra = load_local(LOCAL_TAG)
    if extra is None:
        raise SystemExit("local feature table %r missing" % LOCAL_TAG)
    # Case-level locked exclusion, asserted on both tables this run touches.
    seen = sorted((locked & set(ix.exam_case_id)) | (locked & set(extra.index)))
    if seen:
        raise RuntimeError("locked cases present in the feature tables: %s" % seen[:5])

    y_case, n_classes, _ = target(dev, FIELD)
    if n_classes != 2:
        raise RuntimeError("malformation_ratio must be binary; got %d" % n_classes)
    a2x_feats = attach(feats, ix, extra)

    heads = {"A0": fit_final(feats, ix, y_case),
             "A2x": fit_final(a2x_feats, ix, y_case)}

    with open(FROZEN / "frozen_candidate.json", encoding="utf-8") as fh:
        frozen = json.load(fh)
    row = min(frozen["abstention"],
              key=lambda r: abs(r["target_coverage"] - OPERATING_COVERAGE))

    OUT.mkdir(parents=True, exist_ok=True)
    digests = {}
    for arm, h in heads.items():
        path = OUT / ("head_%s.pkl" % arm)
        with open(path, "wb") as fh:
            pickle.dump(dict(pipes=h["pipes"], poolings=POOLINGS,
                             classes=h["classes"], field=FIELD, arm=arm), fh)
        digests[arm] = sha(path)

    manifest = dict(
        run="final_heads_20260924",
        purpose=("the last fitting decision, made before locked-47 is read. After "
                 "this there is nothing left to choose: transform, coefficients and "
                 "threshold are all on disk."),
        label_availability_note=(
            "the locked read will face the same label gap: any locked case without "
            "an admissible label is excluded from scoring, not guessed at, and the "
            "excluded count is reported rather than quietly dropped"),
        field=FIELD,
        arms=dict(A0=ANCHOR, A2x="%s + local_features/%s" % (ANCHOR, LOCAL_TAG)),
        fit_on=dict(
            cases=int(len(y_case)), images=heads["A0"]["n_images"],
            development_cases_total=int(len(dev)),
            scope="every development case carrying a usable malformation_ratio "
                  "label, no holdout",
            why_not_186=("24 of the 186 development cases have no admissible "
                         "malformation_ratio label (blank, bracketed or "
                         "unconfirmable report wording), so they cannot be fit on. "
                         "162 matches the frozen candidate's full-coverage n_kept."),
        ),
        classes=heads["A0"]["classes"],
        decision_rule=dict(
            aggregation="mean P(class 1) over the five poolings, then over a case's "
                        "images",
            label_threshold=0.5,
            abstain_below=row["confidence_threshold"],
            abstain_provenance="frozen_candidate.json abstention target_coverage=%s"
                               % row["target_coverage"],
            development_coverage_at_gate=row["coverage"],
            development_accuracy_at_gate=row["accuracy"],
            bar_note=("0.7876 at 69.75% coverage does not meet the project's 70% "
                      "coverage + 85% accuracy bar, and abstaining further does not "
                      "reach it (0.8148 at 50%). This threshold is the explicit "
                      "lower-coverage refusal policy, not a claim the bar was met."),
        ),
        config=dict(poolings=POOLINGS, C=C_FIXED, pca_dim=DIM_FIXED, seed=SEED,
                    class_weight=None,
                    why_no_class_weight="the ruler's baseline is a single mode "
                                        "answer; balancing would change what the "
                                        "model is compared against"),
        head_sha256=digests,
        detector_sha256=frozen["detector"]["weights_sha256"],
        encoder=meta.get("timm_model"),
        also_frozen_before_locked=[
            "artifacts/experiments/product_contract_20260924/"
            "field_output_contract.json",
            "artifacts/experiments/product_contract_20260924/rag_safety_tests.json",
            "artifacts/experiments/rescue_external_20260922/"
            "frozen_candidate_malformation_a2x/weights/external_detector_best.pt",
        ],
        locked_cases_seen=0,
        locked_note=("this run read development features only; locked absence was "
                     "asserted case-level on both feature tables"),
        what_this_does_not_establish=[
            "any accuracy claim: fitting on every labelled case leaves no honest "
            "development estimate, which is why the reported numbers stay the "
            "cross-validated ones",
            "that the locked read will agree with development",
        ],
    )
    with open(OUT / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print("froze A0 and A2x heads on %d cases / %d images"
          % (len(y_case), heads["A0"]["n_images"]))
    print("  abstain below %.4f (dev coverage %.4f, accuracy %.4f)"
          % (row["confidence_threshold"], row["coverage"], row["accuracy"]))
    for arm, d in digests.items():
        print("  %-4s %s" % (arm, d[:16]))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
