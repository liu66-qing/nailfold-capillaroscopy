from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from ultralytics import YOLO


def resize_gray(frame: np.ndarray, width: int = 256) -> np.ndarray:
    scale = width / frame.shape[1]
    return cv2.cvtColor(cv2.resize(frame, (width, max(32, round(frame.shape[0] * scale)))), cv2.COLOR_BGR2GRAY)


def read_prefix(path: Path, maximum: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < maximum:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def dense_flow(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.hypot(dx, dy)
    return dx, dy, magnitude


def stable_segment(frames: list[np.ndarray], window: int) -> tuple[list[np.ndarray], float, int]:
    if len(frames) <= window:
        return frames, 0.0, 0
    small = [resize_gray(frame) for frame in frames]
    scores = []
    for first, second in zip(small, small[1:]):
        _, _, magnitude = dense_flow(first, second)
        border = np.concatenate((magnitude[:12].ravel(), magnitude[-12:].ravel(), magnitude[:, :12].ravel(), magnitude[:, -12:].ravel()))
        scores.append(float(np.median(border)))
    rolling = np.convolve(np.asarray(scores), np.ones(window - 1) / (window - 1), mode="valid")
    start = int(np.argmin(rolling))
    return frames[start : start + window], float(rolling[start]), start


def centroid(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(2, dtype=np.float32)
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
    return np.median(centers, axis=0).astype(np.float32)


def summary(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        return [0.0] * 10
    delta = np.abs(np.diff(values)) if len(values) > 1 else np.zeros(1)
    return [float(values.mean()), float(values.std()), *np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(), float(values.max()), float(delta.mean()), float((values < 0.3).mean())]


def case_video_features(path: Path, model: YOLO, max_frames: int, window: int) -> tuple[np.ndarray, dict]:
    frames = read_prefix(path, max_frames)
    if len(frames) < 4:
        raise ValueError(f"only {len(frames)} decoded frames")
    segment, stability, start = stable_segment(frames, min(window, len(frames)))
    anchors = sorted(set((0, len(segment) // 2, len(segment) - 1)))
    results = model.predict([segment[index] for index in anchors], conf=0.225, verbose=False, device=0)
    anchor_centroids = []
    reference_boxes = np.empty((0, 4), dtype=np.float32)
    for anchor, result in zip(anchors, results):
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        if anchor == 0:
            reference_boxes = boxes.astype(np.float32)
        anchor_centroids.append(centroid(boxes))
    shifts = np.zeros((len(segment), 2), dtype=np.float32)
    reference_centroid = anchor_centroids[0]
    for left in range(len(anchors) - 1):
        a, b = anchors[left], anchors[left + 1]
        for position in range(a, b + 1):
            ratio = (position - a) / max(1, b - a)
            shifts[position] = (1 - ratio) * (anchor_centroids[left] - reference_centroid) + ratio * (anchor_centroids[left + 1] - reference_centroid)

    small = [resize_gray(frame) for frame in segment]
    residual_global, background_motion, local_means, local_stds = [], [], [], []
    height_scale = small[0].shape[0] / segment[0].shape[0]
    width_scale = small[0].shape[1] / segment[0].shape[1]
    for position, (first, second) in enumerate(zip(small, small[1:])):
        dx, dy, magnitude = dense_flow(first, second)
        border_dx = np.concatenate((dx[:12].ravel(), dx[-12:].ravel(), dx[:, :12].ravel(), dx[:, -12:].ravel()))
        border_dy = np.concatenate((dy[:12].ravel(), dy[-12:].ravel(), dy[:, :12].ravel(), dy[:, -12:].ravel()))
        tx, ty = float(np.median(border_dx)), float(np.median(border_dy))
        residual = np.hypot(dx - tx, dy - ty)
        residual_global.append(float(np.mean(residual)))
        background_motion.append(float(np.hypot(tx, ty)))
        roi_values = []
        shift = shifts[position] * np.array([width_scale, height_scale], dtype=np.float32)
        for box in reference_boxes:
            x1, y1, x2, y2 = box * np.array([width_scale, height_scale, width_scale, height_scale])
            x1, x2 = int(max(0, x1 + shift[0])), int(min(residual.shape[1], x2 + shift[0]))
            y1, y2 = int(max(0, y1 + shift[1])), int(min(residual.shape[0], y2 + shift[1]))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                roi_values.append(residual[y1:y2, x1:x2].ravel())
        if roi_values:
            joined = np.concatenate(roi_values)
            local_means.append(float(joined.mean()))
            local_stds.append(float(joined.std()))
    hsv = [cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2HSV).mean(axis=(0, 1)) for frame in segment]
    hsv = np.asarray(hsv)
    vector = np.asarray([
        stability, float(len(reference_boxes)), *summary(np.asarray(residual_global)),
        *summary(np.asarray(background_motion)), *summary(np.asarray(local_means)),
        *summary(np.asarray(local_stds)), *hsv.mean(0), *hsv.std(0),
        float(np.abs(np.diff(hsv, axis=0)).mean()) if len(hsv) > 1 else 0.0,
    ], dtype=np.float32)
    return vector, {"decoded": len(frames), "used": len(segment), "start": start, "stability": stability, "anchor_instances": int(len(reference_boxes))}


def aggregate_hulu(matrix_path: Path, index_path: Path) -> dict[str, np.ndarray]:
    matrix = np.load(matrix_path, mmap_mode="r")
    index = pd.read_csv(index_path)
    output = {}
    for case_id, positions in index.groupby("exam_case_id").indices.items():
        values = np.asarray(matrix[np.asarray(positions)], dtype=np.float32)
        output[case_id] = np.concatenate((values.mean(0), values.std(0)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--seg-model", type=Path, required=True)
    parser.add_argument("--hulu", type=Path, required=True)
    parser.add_argument("--hulu-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--window", type=int, default=24)
    args = parser.parse_args()
    roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"])
    development = roles.loc[roles.evaluation_role.eq("development")].copy()
    allowed = set(development.exam_case_id)
    videos = sorted(path for path in args.data.glob("recovered_archive*/*/*") if path.suffix.lower() in {".mp4", ".avi", ".mov"} and "/".join(path.relative_to(args.data).parts[:2]) in allowed)
    model = YOLO(args.seg_model)
    rows, features, errors = [], [], []
    for number, path in enumerate(videos, 1):
        case_id = "/".join(path.relative_to(args.data).parts[:2])
        try:
            vector, audit = case_video_features(path, model, args.max_frames, args.window)
            rows.append({"exam_case_id": case_id, "video_path": path.relative_to(args.data).as_posix(), **audit})
            features.append(vector)
        except Exception as error:
            errors.append({"exam_case_id": case_id, "video_path": str(path), "error": f"{type(error).__name__}: {error}"})
        if number % 10 == 0 or number == len(videos):
            print(f"{number}/{len(videos)} usable={len(features)} errors={len(errors)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video_matrix = np.stack(features)
    per_video_index = pd.DataFrame(rows)
    per_video_index.to_csv(args.output_dir / "per_video_audit.csv", index=False)
    case_rows, case_features = [], []
    for case_id, positions in per_video_index.groupby("exam_case_id", sort=True).indices.items():
        values = per_video_matrix[np.asarray(positions)]
        case_features.append(np.concatenate((values.mean(0), np.median(values, axis=0), values.std(0))))
        case_rows.append({"exam_case_id": case_id, "video_count": len(positions)})
    matrix = np.stack(case_features)
    index = pd.DataFrame(case_rows)
    np.save(args.output_dir / "optical_features.npy", matrix)
    index.to_csv(args.output_dir / "optical_features.csv", index=False)
    hulu = aggregate_hulu(args.hulu, args.hulu_index)
    labels = pd.read_csv(args.labels, usecols=["exam_case_id", "flow_state", "flow_state__confidence"])
    frame = index.merge(development[["exam_case_id", "development_fold"]], on="exam_case_id").merge(labels, on="exam_case_id")
    frame["position"] = np.arange(len(frame))
    frame = frame[frame.flow_state.notna() & frame.flow_state__confidence.fillna(0).ge(0.8)].copy()
    support = frame.flow_state.astype(str).value_counts()
    supported = set(support[support >= 5].index)
    frame = frame[frame.flow_state.astype(str).isin(supported)].reset_index(drop=True)
    y = frame.flow_state.astype(str).to_numpy()
    optical = matrix[frame.position.to_numpy()]
    visual = np.stack([hulu[case_id] for case_id in frame.exam_case_id])
    oof_main, oof_aux, oof_vote, baseline = [np.empty(len(frame), dtype=object) for _ in range(4)]
    models = []
    for fold in range(5):
        test = frame.development_fold.astype(int).eq(fold).to_numpy()
        train = ~test
        classes = sorted(set(y[train]))
        main = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=20260816 + fold).fit(optical[train], y[train])
        aux = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=20260826 + fold).fit(visual[train], y[train])
        main_prob = main.predict_proba(optical[test]); aux_prob = aux.predict_proba(visual[test])
        aligned_main = np.zeros((test.sum(), len(classes))); aligned_aux = np.zeros_like(aligned_main)
        for j, name in enumerate(classes):
            if name in main.classes_: aligned_main[:, j] = main_prob[:, list(main.classes_).index(name)]
            if name in aux.classes_: aligned_aux[:, j] = aux_prob[:, list(aux.classes_).index(name)]
        oof_main[test] = np.asarray(classes)[aligned_main.argmax(1)]
        oof_aux[test] = np.asarray(classes)[aligned_aux.argmax(1)]
        oof_vote[test] = np.asarray(classes)[(0.7 * aligned_main + 0.3 * aligned_aux).argmax(1)]
        baseline[test] = Counter(y[train]).most_common(1)[0][0]
        models.append({"main": main, "aux": aux, "classes": classes})
    scores = {"optical_main": float(balanced_accuracy_score(y, oof_main)), "hulu_aux": float(balanced_accuracy_score(y, oof_aux)), "weighted_vote": float(balanced_accuracy_score(y, oof_vote)), "majority_baseline": float(balanced_accuracy_score(y, baseline))}
    report = {"schema_version": "step11-video-flow/1.0", "evaluation_role": "development_oof", "locked_cases_seen": 0, "videos_found": len(videos), "videos_usable": len(features), "cases_usable": len(index), "labeled_cases": len(frame), "classes": sorted(set(y)), "scores": scores, "balanced_accuracy": scores["weighted_vote"], "gate": "pass" if scores["weighted_vote"] > 0.35 else "fail", "errors": errors, "confusion_matrix": confusion_matrix(y, oof_vote).tolist(), "strategy": "stable_segment_roi_tracking_segmentation_anchor_alignment"}
    joblib.dump(models, args.output_dir / "flow_models.joblib")
    (args.output_dir / "step11_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
