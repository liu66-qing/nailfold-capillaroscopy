from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


FIELDS = ("clarity", "blood_color", "crossing_ratio", "malformation_ratio", "exudation", "hemorrhage", "subpapillary_venous_plexus", "papilla")


def aggregate(matrix_path: Path, index_path: Path) -> pd.DataFrame:
    matrix = np.load(matrix_path, mmap_mode="r"); index = pd.read_csv(index_path); rows = []
    for case_id, positions in index.groupby("exam_case_id", sort=True).indices.items():
        values = np.asarray(matrix[np.asarray(positions)], dtype=np.float32); rows.append((case_id, np.concatenate((values.mean(0), np.median(values, axis=0), values.std(0)))))
    return pd.DataFrame({"exam_case_id": [x[0] for x in rows], "vector": [x[1] for x in rows]})


def pool(probabilities, frame_cases, cases, classes, method):
    output = []
    for case_id in cases:
        values = probabilities[frame_cases == case_id]; values = values.mean(0) if method == "mean" else np.median(values, axis=0); output.append(classes[np.argmax(values)])
    return np.asarray(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("dinov2", "dinov2_index", "hulumed", "hulumed_index", "seg_features", "count_features", "geometry", "labels", "roles", "case_dir", "frame_dir", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    dino_matrix = np.asarray(np.load(args.dinov2, mmap_mode="r"), dtype=np.float32); hulu_matrix = np.asarray(np.load(args.hulumed, mmap_mode="r"), dtype=np.float32); image_index = pd.read_csv(args.dinov2_index); hulu_index = pd.read_csv(args.hulumed_index)
    if not image_index.equals(hulu_index): raise ValueError("image indices differ")
    dino = aggregate(args.dinov2, args.dinov2_index).rename(columns={"vector": "dino"}); hulu = aggregate(args.hulumed, args.hulumed_index).rename(columns={"vector": "hulu"})
    seg = pd.read_parquet(args.seg_features).rename(columns={"case_id": "exam_case_id"}); seg_cols = [c for c in seg if c.startswith("feature_")] + ["frame_count"]; seg["seg"] = list(seg[seg_cols].to_numpy(dtype=np.float32))
    count = pd.read_csv(args.count_features).rename(columns={"case_id": "exam_case_id"}); excluded = {"exam_case_id", "archive", "target", "development_fold", "oof_prediction", "exam_case_id_x", "exam_case_id_y"}; count_cols = [c for c in count if c not in excluded and pd.api.types.is_numeric_dtype(count[c])]; count["count"] = list(count[count_cols].to_numpy(dtype=np.float32))
    geometry = pd.read_csv(args.geometry).rename(columns={"case_id": "exam_case_id"}); frame = dino.merge(hulu, on="exam_case_id").merge(seg[["exam_case_id", "seg"]], on="exam_case_id").merge(count[["exam_case_id", "count"]], on="exam_case_id", how="left").merge(geometry, on="exam_case_id", how="left"); count_dim = len(count_cols); frame["count"] = frame["count"].map(lambda x: x if isinstance(x, np.ndarray) else np.full(count_dim, np.nan, dtype=np.float32))
    roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"]); roles = roles[roles.evaluation_role.eq("development")].drop(columns="evaluation_role"); labels = pd.read_csv(args.labels, usecols=["exam_case_id", *FIELDS]); frame = frame.merge(roles, on="exam_case_id").merge(labels, on="exam_case_id")
    geo = frame[["afferent_px", "efferent_px", "apex_px", "length_px"]].to_numpy(dtype=np.float32); case_routes = {"structured": np.stack([np.concatenate((r.seg, r.count, g)) for r, g in zip(frame.itertuples(index=False), geo)]), "visual": np.stack([np.concatenate((r.dino, r.hulu, r.seg)) for r in frame.itertuples(index=False)]), "all": np.stack([np.concatenate((r.dino, r.hulu, r.seg, r.count, g)) for r, g in zip(frame.itertuples(index=False), geo)])}
    case_report = json.loads((args.case_dir / "step12_report.json").read_text()); frame_report = json.loads((args.frame_dir / "report.json").read_text()); output_fields = {}; rows = []
    for field in FIELDS:
        valid = frame[field].notna(); support = frame.loc[valid, field].astype(str).value_counts(); supported = set(support[support >= 5].index); positions = np.flatnonzero(valid & frame[field].astype(str).isin(supported)); subset = frame.iloc[positions].copy(); truth_all, pred_all = [], [] ; selections = []
        for fold in range(5):
            test_mask = subset.development_fold.astype(int).eq(fold).to_numpy(); test_cases = subset.loc[test_mask, "exam_case_id"].to_numpy(); truth = subset.loc[test_mask, field].astype(str).to_numpy()
            case_selection = case_report["fields"][field]["selections"][fold]; case_route, case_name = case_selection["selected"].split(":"); case_model = joblib.load(args.case_dir / f"{field}_fold{fold}_{case_route}_{case_name}.joblib"); case_prediction = case_model.predict(case_routes[case_route][positions][test_mask])
            frame_selection = frame_report["fields"][field]["selections"][fold]; frame_route, frame_name, pooling = frame_selection["selected"].split(":"); frame_model = joblib.load(args.frame_dir / f"{field}_fold{fold}_{frame_route}_{frame_name}_{pooling}.joblib"); image_mask = image_index.exam_case_id.isin(test_cases).to_numpy(); values = {"dino": dino_matrix, "hulu": hulu_matrix, "dual": np.concatenate((dino_matrix, hulu_matrix), axis=1)}[frame_route]; probabilities = frame_model.predict_proba(values[image_mask]); frame_prediction = pool(probabilities, image_index.loc[image_mask, "exam_case_id"].to_numpy(), test_cases, np.asarray(frame_model.classes_), pooling)
            use_frame = frame_selection["validation_score"] > case_selection["validation_score"]; prediction = frame_prediction if use_frame else case_prediction; truth_all.extend(truth); pred_all.extend(prediction); rows.extend({"field": field, "exam_case_id": c, "fold": fold, "truth": y, "prediction": p, "route": "frame" if use_frame else "case"} for c, y, p in zip(test_cases, truth, prediction)); selections.append({"fold": fold, "route": "frame" if use_frame else "case", "case_validation": case_selection["validation_score"], "frame_validation": frame_selection["validation_score"]})
        output_fields[field] = {"score": float(balanced_accuracy_score(truth_all, pred_all)), "cases": len(truth_all), "selections": selections}
    integrated = dict(case_report["field_scores"]); integrated.update({field: value["score"] for field, value in output_fields.items()}); mean_score = float(np.mean(list(integrated.values()))); report = {"schema_version": "step12-case-frame-nested-routing/1.0", "evaluation_role": "development_oof", "locked_cases_seen": 0, "field_count": len(integrated), "mean_score": mean_score, "gate_threshold": 0.72, "gate": "pass" if mean_score >= 0.72 else "fail", "field_scores": integrated, "routed_fields": output_fields}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
