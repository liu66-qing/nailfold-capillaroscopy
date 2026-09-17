import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path("/root/nailfold")
ART = ROOT / "artifacts"
MANIFEST = ART / "manifest" / "locked_evaluation_v1_reviewed.csv"
FIELDS = ["clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla"]
MAP = {
    "clarity": {"清晰": 0, "不清": 1, "模糊": 1},
    "blood_color": {"暗红": 0, "暗紫": 0, "浅红": 1, "淡红": 1},
    "exudation": {"无": 0, "+": 1, "++": 1, "+++": 1},
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 1, ">2排,扩张": 1},
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 1},
}


def metric(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "ba": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "recall_0": float(np.mean(pred[y == 0] == 0)),
        "recall_1": float(np.mean(pred[y == 1] == 1)),
    }


def load_oof(path):
    raw = json.loads(Path(path).read_text())
    return {f: {str(k): float(v) for k, v in raw[f].items()} for f in FIELDS}


def nested_thresholds(oof, dev):
    thresholds = np.round(np.arange(0.20, 0.801, 0.01), 2)
    result = {}
    for field in FIELDS:
        ymap = dev.set_index("exam_case_id")[field].map(MAP[field])
        calibrated_y, calibrated_p, calibrated_pred = [], [], []
        selections = []
        for test_fold in range(5):
            validation_fold = (test_fold + 1) % 5
            val_ids = dev.loc[dev.development_fold.eq(validation_fold), "exam_case_id"]
            val_ids = [c for c in val_ids if c in oof[field] and pd.notna(ymap.get(c))]
            vy = np.array([ymap[c] for c in val_ids], dtype=int)
            vp = np.array([oof[field][c] for c in val_ids])
            scores = [balanced_accuracy_score(vy, vp >= t) for t in thresholds]
            best_score = max(scores)
            candidates = thresholds[np.isclose(scores, best_score)]
            threshold = float(candidates[np.argmin(np.abs(candidates - 0.5))])
            test_ids = dev.loc[dev.development_fold.eq(test_fold), "exam_case_id"]
            test_ids = [c for c in test_ids if c in oof[field] and pd.notna(ymap.get(c))]
            calibrated_y.extend(int(ymap[c]) for c in test_ids)
            calibrated_p.extend(oof[field][c] for c in test_ids)
            calibrated_pred.extend(int(oof[field][c] >= threshold) for c in test_ids)
            selections.append({"test_fold": test_fold, "validation_fold": validation_fold,
                               "threshold": threshold, "validation_ba": float(best_score)})
        y = np.asarray(calibrated_y)
        p = np.asarray(calibrated_p)
        pred = np.asarray(calibrated_pred)
        base = metric(y, p)
        result[field] = {
            "at_0_5": base,
            "cross_fitted_threshold": {
                "ba": float(balanced_accuracy_score(y, pred)),
                "recall_0": float(np.mean(pred[y == 0] == 0)),
                "recall_1": float(np.mean(pred[y == 1] == 1)),
            },
            "selections": selections,
        }
    return result


def paired_models(a, b, dev):
    out = {}
    for field in FIELDS:
        ymap = dev.set_index("exam_case_id")[field].map(MAP[field])
        ids = [c for c in a[field] if c in b[field] and pd.notna(ymap.get(c))]
        y = np.array([ymap[c] for c in ids], dtype=int)
        pa = np.array([a[field][c] for c in ids])
        pb = np.array([b[field][c] for c in ids])
        ca = (pa >= 0.5) == y
        cb = (pb >= 0.5) == y
        out[field] = {
            "frozen": metric(y, pa), "lora": metric(y, pb),
            "lora_minus_frozen_ba": float(metric(y, pb)["ba"] - metric(y, pa)["ba"]),
            "lora_wins": int(np.sum(cb & ~ca)), "frozen_wins": int(np.sum(ca & ~cb)),
            "ties": int(np.sum(ca == cb)),
        }
    return out


def main():
    labels = pd.read_csv(MANIFEST, dtype=str)
    labels["development_fold"] = pd.to_numeric(labels.development_fold, errors="coerce")
    dev = labels[labels.evaluation_role.eq("development")].copy()
    frame = pd.read_csv(ART / "features" / "image_index.csv", dtype=str)

    duplicate_cross_role = []
    duplicate_cross_fold = []
    for group, rows in labels.groupby("duplicate_group", dropna=False):
        if rows.evaluation_role.nunique() > 1:
            duplicate_cross_role.append(str(group))
    for group, rows in dev.groupby("duplicate_group", dropna=False):
        if rows.development_fold.nunique() > 1:
            duplicate_cross_fold.append(str(group))

    label_diffs = {}
    for name in ["locked_evaluation_v1_backup_pre_dirty_fix.csv", "locked_evaluation_v1.csv"]:
        old = pd.read_csv(ART / "manifest" / name, dtype=str)
        common = labels.merge(old, on="exam_case_id", suffixes=("_new", "_old"))
        label_diffs[name] = {
            f: int((common[f + "_new"].fillna("<NA>") != common[f + "_old"].fillna("<NA>")).sum())
            for f in FIELDS
        }

    frozen = load_oof(ART / "experiments" / "progressive_complexity" / "A_frozen_oof.json")
    lora_r4 = load_oof(ART / "experiments" / "progressive_complexity" / "C_lora_r4_b4_oof.json")
    lora_r8 = load_oof(ART / "experiments" / "progressive_complexity" / "D_lora_r8_b4_oof.json")
    cls_patch = load_oof(ART / "experiments" / "patch_pooling" / "B_cls_patch_oof.json")

    report = {
        "scope": "development-only audit; locked test predictions and labels not used",
        "data_split": {
            "cases_total": int(len(labels)), "development_cases": int(len(dev)),
            "locked_cases": int((labels.evaluation_role == "locked_test").sum()),
            "frames_total": int(len(frame)),
            "patient_id_non_null": int(labels.patient_id.notna().sum()),
            "patient_isolation_verifiable": bool(labels.patient_id.notna().all()),
            "duplicate_groups": int(labels.duplicate_group.nunique(dropna=True)),
            "duplicate_groups_crossing_role": duplicate_cross_role,
            "duplicate_groups_crossing_development_folds": duplicate_cross_fold,
        },
        "label_version_differences": label_diffs,
        "selection_rules": {
            "lora_v2": "single seed; frame-level CE; best epoch by mean validation BA of first four fields; papilla excluded; per-field post-hoc routing across four configurations",
            "progressive": "five seeds; 0.75 case CE + 0.25 frame CE; best epoch by all-five-field mean validation BA; fixed 0.5 threshold",
            "patch_pooling": "same nested folds, five seeds, case/frame loss and all-five-field epoch objective as progressive; fixed 0.5 threshold",
        },
        "fair_same_protocol_comparison": {
            "r4_vs_frozen": paired_models(frozen, lora_r4, dev),
            "r8_vs_frozen": paired_models(frozen, lora_r8, dev),
            "cls_patch_vs_frozen": paired_models(frozen, cls_patch, dev),
        },
        "cross_fitted_threshold_analysis": {
            "frozen": nested_thresholds(frozen, dev),
            "cls_patch": nested_thresholds(cls_patch, dev),
        },
    }
    out = ART / "experiments" / "audit_20260908"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
