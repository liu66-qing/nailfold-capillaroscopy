"""EXP-4A: Optical flow features for flow_state classification.

Extract dense optical flow from nailfold videos, compute statistical features,
classify flow_state using traditional ML (SVM/RF/LightGBM).
"""
import cv2
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

VIDEO_ROOT = Path("/root/nailfold/data")
LABELS_PATH = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
OUT_DIR = Path("/root/nailfold/artifacts/experiments/exp4_flow_state")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Flow state mapping (merged small classes)
FLOW_MAP = {
    "线流": 0, "线粒流": 1, "粒线流": 2, "粒流": 3,
    "粒缓流": 4, "粒摆流": 4, "全停": 4,
    "[线流、线粒流]": 0,  # dirty label
}
N_FLOW_CLASSES = 5

# Score mapping
FLOW_SCORE = {0: 0.0, 1: 0.0, 2: 0.4, 3: 0.8, 4: 1.6}
# Note: merged class 4 = 粒缓流(1.6) + 粒摆流(4.0) + 全停(6.0)
# weighted avg = (13*1.6 + 2*4.0 + 1*6.0)/16 = 2.175
# But for evaluation, true score uses exact label, pred uses 1.6 (粒缓流 representative)

def find_videos():
    """Find all AVI files and map to case IDs."""
    videos = {}
    for avi in VIDEO_ROOT.rglob("*.avi"):
        parts = avi.parts
        for i, p in enumerate(parts):
            if p.startswith("recovered_") or p.startswith("ANFC"):
                if i + 1 < len(parts):
                    case_id = p + "/" + parts[i + 1]
                    videos.setdefault(case_id, []).append(str(avi))
                break
    return videos

def extract_optical_flow_features(video_path, max_frames=200, sample_step=1):
    """Extract dense optical flow features from a video.

    Returns a feature vector summarizing the flow dynamics:
    - Global flow magnitude stats (mean, std, percentiles)
    - Flow direction coherence (how aligned flow vectors are)
    - Temporal variability (how flow changes over time)
    - Spatial distribution (center vs periphery flow)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None

    # Resize for speed
    h, w = prev_frame.shape[:2]
    scale = min(512 / max(h, w), 1.0)
    if scale < 1.0:
        prev_frame = cv2.resize(prev_frame, None, fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Collect per-frame flow statistics
    frame_mags = []       # mean magnitude per frame
    frame_stds = []       # std magnitude per frame
    frame_coherences = [] # directional coherence per frame
    frame_maxes = []      # max magnitude per frame
    all_mags = []         # all pixel magnitudes (subsampled)

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_step != 0:
            continue

        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Global stats
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        max_mag = float(np.percentile(mag, 99))  # 99th to avoid outliers

        frame_mags.append(mean_mag)
        frame_stds.append(std_mag)
        frame_maxes.append(max_mag)

        # Directional coherence: mean resultant length of unit direction vectors
        # High = all flow in same direction (streaming), Low = chaotic (granular)
        dx = flow[..., 0].flatten()
        dy = flow[..., 1].flatten()
        norms = np.sqrt(dx**2 + dy**2) + 1e-8
        ux, uy = dx / norms, dy / norms
        # Only use pixels with significant flow
        significant = mag.flatten() > np.percentile(mag, 50)
        if significant.sum() > 10:
            coherence = float(np.sqrt(np.mean(ux[significant])**2 + np.mean(uy[significant])**2))
        else:
            coherence = 0.0
        frame_coherences.append(coherence)

        # Subsample magnitudes for distribution features
        all_mags.extend(mag.flatten()[::10].tolist())

        prev_gray = gray

    cap.release()

    if len(frame_mags) < 3:
        return None

    frame_mags = np.array(frame_mags)
    frame_stds = np.array(frame_stds)
    frame_coherences = np.array(frame_coherences)
    frame_maxes = np.array(frame_maxes)
    all_mags = np.array(all_mags)

    # Temporal features
    temporal_cv = float(np.std(frame_mags) / (np.mean(frame_mags) + 1e-8))  # coefficient of variation
    temporal_trend = float(np.polyfit(range(len(frame_mags)), frame_mags, 1)[0])  # linear trend

    # Compile feature vector
    features = {
        # Magnitude statistics
        "mag_mean": float(np.mean(frame_mags)),
        "mag_std": float(np.std(frame_mags)),
        "mag_median": float(np.median(frame_mags)),
        "mag_p25": float(np.percentile(frame_mags, 25)),
        "mag_p75": float(np.percentile(frame_mags, 75)),
        "mag_p95": float(np.percentile(frame_mags, 95)),
        "mag_max_mean": float(np.mean(frame_maxes)),
        "mag_max_std": float(np.std(frame_maxes)),

        # Within-frame variability
        "intra_std_mean": float(np.mean(frame_stds)),
        "intra_std_std": float(np.std(frame_stds)),

        # Directional coherence
        "coherence_mean": float(np.mean(frame_coherences)),
        "coherence_std": float(np.std(frame_coherences)),
        "coherence_min": float(np.min(frame_coherences)),

        # Temporal dynamics
        "temporal_cv": temporal_cv,
        "temporal_trend": temporal_trend,
        "temporal_autocorr": float(np.corrcoef(frame_mags[:-1], frame_mags[1:])[0, 1]) if len(frame_mags) > 2 else 0.0,

        # Distribution features (from all pixels)
        "pixel_mag_p10": float(np.percentile(all_mags, 10)),
        "pixel_mag_p50": float(np.percentile(all_mags, 50)),
        "pixel_mag_p90": float(np.percentile(all_mags, 90)),
        "pixel_mag_p99": float(np.percentile(all_mags, 99)),
        "pixel_mag_skew": float(pd.Series(all_mags).skew()),
        "pixel_mag_kurtosis": float(pd.Series(all_mags).kurtosis()),

        # Proportion of "moving" pixels
        "pct_moving_p50": float(np.mean(all_mags > np.median(all_mags))),
        "pct_moving_1px": float(np.mean(all_mags > 1.0)),
        "pct_moving_2px": float(np.mean(all_mags > 2.0)),

        # Meta
        "n_frames": len(frame_mags),
        "fps": fps,
    }
    return features


def main():
    print("=" * 60)
    print("EXP-4A: Optical Flow Features for Flow State Classification")
    print("=" * 60)

    # Load labels
    labels = pd.read_csv(LABELS_PATH, dtype={"exam_case_id": str})
    dev = labels[labels.evaluation_role == "development"].copy()
    dev_idx = dev.set_index("exam_case_id")

    # Find videos
    videos = find_videos()
    print(f"Found videos for {len(videos)} cases")

    # Extract features for dev cases with flow_state labels
    feature_rows = []
    for case_id in sorted(dev.exam_case_id):
        if case_id not in videos:
            continue
        flow_label = dev_idx.at[case_id, "flow_state"]
        if pd.isna(flow_label):
            continue
        flow_cls = FLOW_MAP.get(str(flow_label), -1)
        if flow_cls < 0:
            continue

        # Extract features from all videos for this case
        case_features = []
        for vpath in videos[case_id]:
            print(f"  Processing {case_id} -> {os.path.basename(vpath)}...", end="", flush=True)
            try:
                import subprocess as sp
                result = sp.run(
                    ["/root/miniconda3/bin/python", "/tmp/extract_flow_single.py", vpath],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0 and result.stdout.strip() != "null":
                    feats = json.loads(result.stdout.strip())
                    case_features.append(feats)
                    print(f" mag={feats['mag_mean']:.2f}, coherence={feats['coherence_mean']:.3f}")
                else:
                    print(f" FAILED (rc={result.returncode})")
                    if result.stderr:
                        print(f"    stderr: {result.stderr[:200]}")
            except Exception as e:
                print(f" ERROR: {e}")

        if not case_features:
            continue

        # Aggregate across videos (take mean of each feature)
        agg = {}
        for key in case_features[0]:
            if key in ("n_frames", "fps"):
                agg[key] = case_features[0][key]
            else:
                vals = [f[key] for f in case_features if not np.isnan(f[key])]
                agg[key] = float(np.mean(vals)) if vals else 0.0

        agg["exam_case_id"] = case_id
        agg["flow_state"] = str(flow_label)
        agg["flow_cls"] = flow_cls
        agg["n_videos"] = len(case_features)
        feature_rows.append(agg)

    df = pd.DataFrame(feature_rows)
    df.to_csv(OUT_DIR / "flow_features.csv", index=False)
    print(f"\nFeatures extracted for {len(df)} cases")
    print(f"Class distribution: {df.flow_cls.value_counts().sort_index().to_dict()}")

    # === Classification with 5-fold CV ===
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    feature_cols = [c for c in df.columns if c not in
                    ("exam_case_id", "flow_state", "flow_cls", "n_videos", "n_frames", "fps")]
    X = df[feature_cols].values.astype(np.float64)
    y = df["flow_cls"].values
    case_ids = df["exam_case_id"].values

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Classes: {np.unique(y, return_counts=True)}")

    # Use development_fold from labels for consistent CV
    dev_folds = dev_idx.loc[case_ids, "development_fold"].values.astype(int)

    classifiers = {
        "RF": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"),
        "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42),
    }

    score_rules = json.load(open("/root/nailfold/artifacts/labels/score_rules_v3.json"))
    flow_score_map = score_rules["field_rules"]["flow_state"]["mapping"]

    results = {}
    for clf_name, clf in classifiers.items():
        oof_pred = np.full(len(y), -1)
        for fold in range(5):
            test_mask = dev_folds == fold
            train_mask = ~test_mask
            if test_mask.sum() == 0:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask])
            X_test = scaler.transform(X[test_mask])

            clf_copy = type(clf)(**clf.get_params())
            clf_copy.fit(X_train, y[train_mask])
            oof_pred[test_mask] = clf_copy.predict(X_test)

        valid = oof_pred >= 0
        ba = balanced_accuracy_score(y[valid], oof_pred[valid])

        # Compute sMAE
        idx_to_label = {0: "线流", 1: "线粒流", 2: "粒线流", 3: "粒流", 4: "粒缓流"}
        true_scores = np.array([flow_score_map.get(df.iloc[i]["flow_state"], 0) for i in range(len(df))])
        pred_scores = np.array([flow_score_map.get(idx_to_label.get(int(p), ""), 0) if p >= 0 else 0 for p in oof_pred])
        smae = float(np.mean(np.abs(true_scores[valid] - pred_scores[valid])))

        results[clf_name] = {"ba": float(ba), "smae": smae, "n": int(valid.sum())}
        print(f"\n{clf_name}: BA={ba:.3f}, sMAE={smae:.3f} (n={valid.sum()})")

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y[valid], oof_pred[valid])
        print(f"  Confusion matrix:\n{cm}")

    # Save results
    results["baseline_smae"] = 0.427
    results["n_cases"] = len(df)
    results["n_features"] = X.shape[1]
    results["feature_names"] = feature_cols

    (OUT_DIR / "exp4a_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"\nResults saved to {OUT_DIR}")

    # Feature importance (from RF)
    if "RF" in classifiers:
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced")
        scaler = StandardScaler()
        rf.fit(scaler.fit_transform(X), y)
        importances = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])
        print("\nTop 10 features:")
        for fname, imp in importances[:10]:
            print(f"  {fname}: {imp:.3f}")

if __name__ == "__main__":
    main()
