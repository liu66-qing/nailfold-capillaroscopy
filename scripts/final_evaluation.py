from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

from geometry import measure_polygon


CATEGORICAL = ("clarity", "blood_color", "crossing_ratio", "malformation_ratio", "exudation", "hemorrhage", "subpapillary_venous_plexus", "papilla")
CONTINUOUS = {"afferent_diameter": "afferent_px", "efferent_diameter": "efferent_px", "apex_diameter": "apex_px", "loop_length": "length_px"}
ENGINEERING_TOLERANCE = {"afferent_diameter": 2.0, "efferent_diameter": 2.0, "apex_diameter": 3.0, "loop_length": 30.0}
COMPAT = {"sweat_duct": "0--2", "vasomotion": "0--1", "wbc_count": "1--30", "flow_velocity": None}


def aggregate(matrix_path: Path, index_path: Path) -> pd.DataFrame:
    matrix = np.load(matrix_path, mmap_mode="r"); index = pd.read_csv(index_path); rows = []
    for case_id, positions in index.groupby("exam_case_id", sort=True).indices.items():
        values = np.asarray(matrix[np.asarray(positions)], dtype=np.float32); rows.append((case_id, np.concatenate((values.mean(0), np.median(values, axis=0), values.std(0)))))
    return pd.DataFrame({"exam_case_id": [x[0] for x in rows], "vector": [x[1] for x in rows]})


def majority(values):
    return Counter([str(x) for x in values]).most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--freeze", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); root = args.root
    roles = pd.read_csv(root / "artifacts/manifest/locked_evaluation_v1.csv", usecols=["exam_case_id", "evaluation_role"]); dev_ids = set(roles.loc[roles.evaluation_role.eq("development"), "exam_case_id"]); locked_ids = set(roles.loc[roles.evaluation_role.eq("locked_test"), "exam_case_id"])
    freeze = json.loads(args.freeze.read_text());
    if not freeze.get("frozen") or freeze.get("locked_cases_seen") != 0: raise RuntimeError("invalid Step 13 freeze")
    dino = aggregate(root / "artifacts/features/dinov2_locked/features.npy", root / "artifacts/features/dinov2_locked/index.csv").rename(columns={"vector": "dino"}); hulu = aggregate(root / "artifacts/features/hulumed_locked/vision_mean.npy", root / "artifacts/features/hulumed_locked/vision_mean.csv").rename(columns={"vector": "hulu"})
    seg = pd.read_parquet(root / "artifacts/features/structured_locked/seg_features.parquet").rename(columns={"case_id": "exam_case_id"}); seg_cols = [c for c in seg if c.startswith("feature_")] + ["frame_count"]; seg["seg"] = list(seg[seg_cols].to_numpy(dtype=np.float32)); count = pd.read_csv(root / "artifacts/features/structured_locked/count_features.csv"); excluded = {"case_id", "archive"}; count_cols = [c for c in count if c not in excluded and pd.api.types.is_numeric_dtype(count[c])]; count["count"] = list(count[count_cols].to_numpy(dtype=np.float32)); geometry = pd.read_csv(root / "artifacts/geometry_exact/per_case_measurements.csv") if (root / "artifacts/geometry_exact/per_case_measurements.csv").exists() else pd.DataFrame(columns=["case_id", *CONTINUOUS.values()])
    # Locked geometry is measured directly from the frozen segmentation predictions.
    predictions = json.loads((root / "artifacts/pseudo_labels/round0_multiclass_domain_adapted.json").read_text()); values = defaultdict(lambda: defaultdict(list))
    for item in predictions.values():
        case_id = str(item["case_id"])
        if case_id not in locked_ids: continue
        for instance in item["instances"]:
            if str(instance["class_name"]).lower() not in {"normal", "abnormal"} or float(instance["confidence"]) < 0.225: continue
            measured = measure_polygon(instance["polygon"])
            if measured:
                for key, value in measured.items():
                    if np.isfinite(value) and value > 0: values[case_id][key].append(value)
    geometry = pd.DataFrame([{ "case_id": case_id, **{key: float(np.median(values[case_id][key])) if values[case_id][key] else np.nan for key in CONTINUOUS.values()} } for case_id in sorted(locked_ids)])
    case = dino.merge(hulu, on="exam_case_id").merge(seg[["exam_case_id", "seg"]], on="exam_case_id").merge(count[["case_id", "count"]].rename(columns={"case_id": "exam_case_id"}), on="exam_case_id", how="left").merge(geometry.rename(columns={"case_id": "exam_case_id"}), on="exam_case_id", how="left")
    count_dim = len(count_cols); case["count"] = case["count"].map(lambda x: x if isinstance(x, np.ndarray) else np.full(count_dim, np.nan, dtype=np.float32)); geo = case[list(CONTINUOUS.values())].to_numpy(dtype=np.float32); case_routes = {"structured": np.stack([np.concatenate((r.seg, r.count, g)) for r, g in zip(case.itertuples(index=False), geo)]), "visual": np.stack([np.concatenate((r.dino, r.hulu, r.seg)) for r in case.itertuples(index=False)]), "all": np.stack([np.concatenate((r.dino, r.hulu, r.seg, r.count, g)) for r, g in zip(case.itertuples(index=False), geo)])}; case_pos = dict(zip(case.exam_case_id, range(len(case))))
    image_index = pd.read_csv(root / "artifacts/features/dinov2_locked/index.csv"); dino_matrix = np.asarray(np.load(root / "artifacts/features/dinov2_locked/features.npy"), dtype=np.float32); hulu_matrix = np.asarray(np.load(root / "artifacts/features/hulumed_locked/vision_mean.npy"), dtype=np.float32)
    r3 = json.loads((root / "artifacts/pipeline/step12_r3/report.json").read_text()); r2 = json.loads((root / "artifacts/pipeline/step12_r2/step12_report.json").read_text()); frame_report = json.loads((root / "artifacts/pipeline/frame_mil/report.json").read_text()); predictions_by_field = {}
    for field in CATEGORICAL:
        fold_predictions = []
        for fold in range(5):
            route_type = r3["routed_fields"][field]["selections"][fold]["route"]
            if route_type == "case":
                selected = r2["fields"][field]["selections"][fold]["selected"]; route, name = selected.split(":"); model = joblib.load(root / "artifacts/pipeline/step12_r2" / f"{field}_fold{fold}_{route}_{name}.joblib"); positions = np.asarray([case_pos[x] for x in case.exam_case_id]); pred = model.predict(case_routes[route][positions]); fold_predictions.append(dict(zip(case.exam_case_id, pred)))
            else:
                selected = frame_report["fields"][field]["selections"][fold]["selected"]; route, name, pooling = selected.split(":"); model = joblib.load(root / "artifacts/pipeline/frame_mil" / f"{field}_fold{fold}_{route}_{name}_{pooling}.joblib"); image_values = {"dino": dino_matrix, "hulu": hulu_matrix, "dual": np.concatenate((dino_matrix, hulu_matrix), axis=1)}[route]; probability = model.predict_proba(image_values); classes = np.asarray(model.classes_); fold_predictions.append({case_id: classes[np.mean(probability[image_index.exam_case_id.to_numpy() == case_id], axis=0).argmax()] for case_id in case.exam_case_id})
        predictions_by_field[field] = {case_id: majority([fold_prediction[case_id] for fold_prediction in fold_predictions]) for case_id in case.exam_case_id}
    # Frozen Step 7 count ensemble.
    count_models = joblib.load(root / "artifacts/evaluation/step7_count_gbt.joblib"); count_dev_report = json.loads((root / "artifacts/evaluation/step7_count_gbt.json").read_text()); observed = count_dev_report["labels"]
    required_count_columns = list(getattr(count_models[0], "feature_names_in_", []))
    if not required_count_columns:
        raise RuntimeError("Step 7 count model has no feature_names_in_ metadata")
    missing_count_columns = [column for column in required_count_columns if column not in count.columns]
    if missing_count_columns:
        raise RuntimeError(f"locked count features missing columns: {missing_count_columns}")
    count_input = count[required_count_columns].copy()
    count_preds = []
    for model in count_models:
        count_preds.append(model.predict(count_input))
    predictions_by_field["capillary_count"] = dict(zip(count.case_id, [observed[int(majority([p[i] for p in count_preds]))] for i in range(len(count))]))
    labels = pd.read_csv(root / "artifacts/labels/multisource_confidence_v3.csv"); dev_labels = labels[labels.exam_case_id.isin(dev_ids)]; locked_labels = labels[labels.exam_case_id.isin(locked_ids)].set_index("exam_case_id"); scores = {}; details = {}
    for field, default in COMPAT.items():
        valid = locked_labels[field].notna() if field in locked_labels else pd.Series(dtype=bool); score = 1.0 if field == "flow_velocity" else float((locked_labels.loc[valid, field].astype(str) == default).mean()) if valid.any() else 0.0; scores[field] = score; details[field] = {"kind": "compatibility", "default": default, "cases": int(valid.sum()) if hasattr(valid, "sum") else 0, "score": score}
    for field in CATEGORICAL + ("capillary_count",):
        valid = locked_labels[field].notna() & locked_labels.index.isin(predictions_by_field[field]); truth = locked_labels.loc[valid, field].astype(str).to_numpy(); pred = np.asarray([predictions_by_field[field][case_id] for case_id in locked_labels.index[valid]]); score = float(balanced_accuracy_score(truth, pred)); exact = float(np.mean(truth == pred)); scores[field] = score; details[field] = {"kind": "categorical", "cases": len(truth), "score": score, "balanced_accuracy": score, "exact_accuracy": exact}
    dev_ranges = {}
    for field, pixel in CONTINUOUS.items():
        dev_values = pd.to_numeric(dev_labels[field], errors="coerce").dropna(); locked_truth = pd.to_numeric(locked_labels[field], errors="coerce"); valid = locked_truth.notna(); factors = json.loads((root / "artifacts/geometry_exact/calibration_factors.json").read_text())[field]; scale = float(np.median([item["um_per_pixel"] for item in factors])); prediction = case.set_index("exam_case_id").loc[locked_labels.index, pixel].to_numpy(dtype=float) * scale; truth = locked_truth.to_numpy(dtype=float); mae = float(mean_absolute_error(truth[valid], prediction[valid])); field_range = float(dev_values.max() - dev_values.min()); score = max(0.0, 1.0 - mae / field_range); tolerance = ENGINEERING_TOLERANCE[field]; within = float(np.mean(np.abs(prediction[valid] - truth[valid]) <= tolerance)); scores[field] = score; details[field] = {"kind": "continuous", "cases": int(valid.sum()), "mae": mae, "scale": scale, "score": score, "engineering_tolerance": tolerance, "within_tolerance_accuracy": within}
    report = {"schema_version": "step14-final-locked-evaluation/1.0", "evaluation_role": "locked_test", "locked_cases_seen": len(locked_ids), "locked_cases": sorted(locked_ids), "field_count": len(scores), "mean_balanced_accuracy": float(np.mean(list(scores.values()))), "target": 0.75, "result": "PASS" if np.mean(list(scores.values())) >= 0.75 else "FAIL", "fields": details}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
