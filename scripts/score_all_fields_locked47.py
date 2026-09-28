"""Every in-scope field: A0 heads fitted on dev 186, scored once on locked-47.

This asks what each field would show if the report printed a value for every
field. The answer is compared with the dev-mode baseline: always printing the
most common development answer, which is what a "typical value" readout does.
Same A0 recipe as rag_heads_v1. Multiclass fields use the declared class set.
The test set is contaminated (7 prior reads, selection in 3), so the result is descriptive.

    python scripts/score_all_fields_locked47.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402

OUT = os.path.join(ROOT, "artifacts/experiments/test_locked47_all_fields")
N_BOOT = 2000


def main():
    man, ixa, Fa, source, _ = C.load()
    T = C.targets(man)
    img_case = ixa.exam_case_id.to_numpy()
    rng = np.random.default_rng(R.SEED)
    res, rows = {}, []
    for f, y in T.items():
        y = y.astype(int)
        train = sorted(c for c in y.index if source[c] == "dev")
        test = sorted(c for c in y.index if source[c] == "locked")
        classes = sorted(y.reindex(train).unique())
        y_img = np.where(np.isin(img_case, train), y.reindex(img_case).fillna(-1), -1).astype(int)
        P = C.fit_predict(Fa, img_case, y_img, y_img >= 0, np.isin(img_case, test), classes)
        P = P.reindex(test)
        pred = np.array(classes)[P.to_numpy().argmax(1)]
        yt = y.reindex(test).to_numpy()
        mode = int(pd.Series(y.reindex(train)).mode()[0])
        acc, acc0 = (pred == yt).mean(), (yt == mode).mean()
        d = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(yt), len(yt))
            d.append((pred[i] == yt[i]).mean() - (yt[i] == mode).mean())
        res[f] = dict(n_test=len(test), classes=[int(c) for c in classes],
                      test_class_counts={int(k): int(v) for k, v in
                                         pd.Series(yt).value_counts().sort_index().items()},
                      accuracy=round(float(acc), 4),
                      balanced_accuracy=round(float(balanced_accuracy_score(yt, pred)), 4),
                      dev_mode=mode, mode_accuracy=round(float(acc0), 4),
                      acc_minus_mode=round(float(acc - acc0), 4),
                      acc_minus_mode_ci95=[round(float(np.percentile(d, q)), 4)
                                           for q in (2.5, 97.5)],
                      predicted_classes=sorted(int(c) for c in set(pred)))
        for c, p, t in zip(test, pred, yt):
            rows.append(dict(field=f, exam_case_id=c, y_true=int(t), pred=int(p)))
        print(f, json.dumps(res[f]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(fields=res, design="A0 fit on dev 186, one read of locked-47 "
                   "(contaminated, descriptive); baseline = dev mode"),
              open(os.path.join(OUT, "all_fields.json"), "w"), indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_case.csv"), index=False)


if __name__ == "__main__":
    main()
