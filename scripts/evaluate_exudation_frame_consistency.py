"""Strict development OOF audit for exudation frame-consistency pooling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MAP = {"无": 0, "+": 1, "++": 1, "+++": 1}


def models(seed: int, dim: int):
    return {
        "logistic": make_pipeline(
            SimpleImputer(), SelectKBest(f_classif, k=min(128, dim)),
            StandardScaler(), LogisticRegression(C=0.2, class_weight="balanced", max_iter=4000, random_state=seed),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(), ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=seed),
        ),
    }


def pool(prob: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return prob.mean(axis=0)
    if method == "median":
        return np.median(prob, axis=0)
    if method == "max":
        return prob[np.argmax(prob[:, 1])]
    if method.startswith("top"):
        frac = float(method[3:])
        k = max(1, int(np.ceil(len(prob) * frac)))
        return prob[np.argsort(prob[:, 1])[-k:]].mean(axis=0)
    raise ValueError(method)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dinov2", type=Path, required=True)
    p.add_argument("--dino-index", type=Path, required=True)
    p.add_argument("--hulumed", type=Path, required=True)
    p.add_argument("--hulu-index", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--roles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dino = np.asarray(np.load(a.dinov2, mmap_mode="r"), dtype=np.float32)
    hulu = np.asarray(np.load(a.hulumed, mmap_mode="r"), dtype=np.float32)
    idx = pd.read_csv(a.dino_index); hidx = pd.read_csv(a.hulu_index)
    keys = ["exam_case_id", "image_path"]
    hidx = hidx.reset_index(names="hulu_position")
    aligned = idx[keys].merge(hidx[keys + ["hulu_position"]], on=keys, how="left", validate="one_to_one")
    if aligned.hulu_position.isna().any(): raise ValueError("Hulu feature index coverage mismatch")
    roles = pd.read_csv(a.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"])
    roles.exam_case_id = roles.exam_case_id.astype(str)
    if len(roles[roles.evaluation_role.eq("development")]) != 186: raise ValueError("development boundary")
    dev = roles[roles.evaluation_role.eq("development")].drop(columns="evaluation_role")
    labels = pd.read_csv(a.labels, usecols=["exam_case_id", "exudation"])
    labels.exam_case_id = labels.exam_case_id.astype(str)
    cases = dev.merge(labels, on="exam_case_id", validate="one_to_one")
    cases["target"] = cases.exudation.map(MAP)
    cases = cases[cases.target.notna()].copy()
    target = dict(zip(cases.exam_case_id, cases.target.astype(int)))
    frame = idx[idx.exam_case_id.astype(str).isin(target)].copy().reset_index()
    frame.exam_case_id = frame.exam_case_id.astype(str)
    position = frame["index"].to_numpy()
    hpos = aligned.iloc[position].hulu_position.astype(int).to_numpy()
    route = {"dino": dino[position], "hulu": hulu[hpos], "dual": np.concatenate((dino[position], hulu[hpos]), axis=1)}
    frame_cases = frame.exam_case_id.to_numpy()
    y = np.asarray([target[c] for c in frame_cases])
    case_ids = cases.exam_case_id.to_numpy()
    fold_map = cases.set_index("exam_case_id").development_fold.astype(int).to_dict()
    case_folds = np.asarray([fold_map[c] for c in case_ids])
    methods = ("mean", "median", "max", "top0.25", "top0.50", "top0.75")
    records = {}
    for route_name, values in route.items():
        oof = {}; selections = []
        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5
            train_cases = case_ids[(case_folds != test_fold) & (case_folds != val_fold)]
            val_cases = case_ids[case_folds == val_fold]
            test_cases = case_ids[case_folds == test_fold]
            train_rows = np.isin(frame_cases, train_cases)
            val_rows = np.isin(frame_cases, val_cases)
            test_rows = np.isin(frame_cases, test_cases)
            counts = pd.Series(frame_cases[train_rows]).value_counts()
            weights = np.asarray([1.0 / counts[c] for c in frame_cases[train_rows]], dtype=np.float32)
            scores = {}
            fitted = {}
            for name, model in models(20260828 + test_fold, values.shape[1]).items():
                model.fit(values[train_rows], y[train_rows], **({"logisticregression__sample_weight": weights} if name == "logistic" else {"extratreesclassifier__sample_weight": weights}))
                prob = model.predict_proba(values[val_rows]); classes = np.asarray(model.classes_)
                prob_by_case = {c: prob[frame_cases[val_rows] == c] for c in val_cases}
                truth = np.asarray([target[c] for c in val_cases])
                for method in methods:
                    pred = np.asarray([classes[np.argmax(pool(prob_by_case[c], method))] for c in val_cases])
                    key = f"{name}:{method}"; scores[key] = float(balanced_accuracy_score(truth, pred)); fitted[key] = model
            selected = max(scores, key=lambda k: (scores[k], k))
            name, method = selected.split(":")
            model = models(20260828 + test_fold, values.shape[1])[name]
            dev_rows = np.isin(frame_cases, np.concatenate((train_cases, val_cases)))
            counts = pd.Series(frame_cases[dev_rows]).value_counts()
            weights = np.asarray([1.0 / counts[c] for c in frame_cases[dev_rows]], dtype=np.float32)
            model.fit(values[dev_rows], y[dev_rows], **({"logisticregression__sample_weight": weights} if name == "logistic" else {"extratreesclassifier__sample_weight": weights}))
            prob = model.predict_proba(values[test_rows]); classes = np.asarray(model.classes_)
            for c in test_cases:
                rows = frame_cases[test_rows] == c
                oof[c] = int(classes[np.argmax(pool(prob[rows], method))])
            selections.append({"fold": test_fold, "selected": selected, "validation_ba": scores[selected], "test_n": int(len(test_cases))})
        score = float(balanced_accuracy_score([target[c] for c in case_ids], [oof[c] for c in case_ids]))
        records[route_name] = {"balanced_accuracy": score, "selections": selections}
    best = max((v["balanced_accuracy"], k) for k, v in records.items())
    report = {"schema_version": "exudation-frame-consistency-oof/1.0", "evaluation_role": "development_oof", "locked_cases_seen": 0, "cases": len(case_ids), "routes": records, "best": {"balanced_accuracy": best[0], "route": best[1]}, "formal_binary_baseline": 0.6836264402668284, "gate": "pass" if best[0] >= 0.7036264402668284 else "fail"}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
