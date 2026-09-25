"""Matched-count shuffle control for the local-feature rungs.

Why this is needed. malformation_ratio A2x gains +0.0911 BA (CI [0.023, 0.162])
and +0.1605 accuracy over the mode (CI [0.062, 0.259]) -- the first accuracy
delta in this round whose CI excludes zero. But its L2 rung, the local columns
ALONE, gains +0.0017: nothing. So the columns carry no standalone signal yet the
combination improves. That is either a real interaction, or it is the hazard the
pre-registration named: attaching 64 columns to a 768-column problem changes the
StandardScaler -> PCA(64) -> LogisticRegression geometry, and the improvement
belongs to the reshaped projection rather than to vessel morphology.

L2 cannot settle that. L2 asks "do the columns predict on their own?", not "does
adding this many columns help regardless of what is in them?". The control that
separates the two is to attach the SAME columns with their case-to-row
correspondence destroyed: shuffle the rows of the local table across cases with a
fixed seed. Column count, marginal distribution, scale and NaN pattern are all
preserved exactly; only the link between a case and its own vessel counts is
broken. If the shuffled columns reproduce the gain, the gain is geometry. If the
shuffled columns give nothing, the gain is morphology.

Shuffling is done ONCE per repeat with a declared seed, and the whole OOF is
re-run per repeat, so the reported spread is over shuffles, not over bootstrap.

  PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/control_shuffled_local_columns.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_field_matrix_three_arms import (  # noqa: E402
    ANCHOR, LABELS, SEED, load_arm, oof, paired, score, target)
from run_rescue_ladder_local import (  # noqa: E402
    FIELDS, LOCAL, OUT, RUNGS, attach, load_local)

N_SHUFFLES = 20


def main() -> None:
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))
    fold_map = dev.development_fold.to_dict()

    base_ix, base_feats, _ = load_arm(ANCHOR)
    tags = {k: load_local(t) for k, (t, _) in RUNGS.items()}
    tags = {k: v for k, v in tags.items() if v is not None}

    rng = np.random.default_rng(SEED)
    out = {}
    for field in FIELDS:
        y_case, n_classes, _ = target(dev, field)
        order = sorted(y_case.index)
        yv = y_case.reindex(order).to_numpy()
        classes = sorted(set(float(v) for v in yv))
        mc = n_classes > 2
        p0, _ = oof(base_feats, base_ix, y_case, fold_map, order, mc, classes)
        out[field] = {}
        for k, extra in tags.items():
            real = oof(attach(base_feats, base_ix, extra), base_ix, y_case,
                       fold_map, order, mc, classes)[0]
            real_gain = paired(yv, real, p0, mc, rng)["ba_gain"]
            gains = []
            for s in range(N_SHUFFLES):
                g = np.random.default_rng(SEED + s)
                # Permute which case each row of local features belongs to. The
                # frame of numbers is untouched; only the index is reassigned.
                perm = extra.copy()
                perm.index = extra.index.to_numpy()[g.permutation(len(extra))]
                sp = oof(attach(base_feats, base_ix, perm), base_ix, y_case,
                         fold_map, order, mc, classes)[0]
                gains.append(paired(yv, sp, p0, mc, rng)["ba_gain"])
            gains = np.array(gains, float)
            out[field][k] = dict(
                real_ba_gain=round(float(real_gain), 4),
                shuffled_ba_gain_mean=round(float(gains.mean()), 4),
                shuffled_ba_gain_sd=round(float(gains.std(ddof=1)), 4),
                shuffled_ba_gain_min=round(float(gains.min()), 4),
                shuffled_ba_gain_max=round(float(gains.max()), 4),
                shuffles_at_or_above_real=int((gains >= real_gain).sum()),
                n_shuffles=N_SHUFFLES,
                reading=("real gain inside the shuffled range means the gain is "
                         "column-count geometry, not vessel morphology"))
            print("%-20s %-6s real %+.4f | shuffled mean %+.4f sd %.4f range "
                  "[%+.4f, %+.4f] | >=real in %d/%d"
                  % (field, k, real_gain, gains.mean(), gains.std(ddof=1),
                     gains.min(), gains.max(),
                     out[field][k]["shuffles_at_or_above_real"], N_SHUFFLES),
                  flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shuffled_column_control.json").write_text(json.dumps(dict(
        question=("is a local-feature gain real morphology, or the effect of "
                  "attaching that many columns to the shipped PCA?"),
        method=("the local table's rows are permuted across cases with a declared "
                "seed, preserving column count, marginals, scale and NaN pattern "
                "and destroying only the case correspondence; the full OOF is "
                "re-run per shuffle"),
        n_shuffles=N_SHUFFLES, seed=SEED, locked_cases_seen=0,
        results=out), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
