"""EXP-4A v2: Re-extract flow features using converted videos + classify."""
import json, os, subprocess, numpy as np, pandas as pd
from pathlib import Path

LABELS_PATH = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
CONVERTED_DIR = Path("/root/nailfold/data/converted_videos")
VIDEO_ROOT = Path("/root/nailfold/data")
OUT_DIR = Path("/root/nailfold/artifacts/experiments/exp4_flow_state")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOW_MAP = {
    "线流": 0, "线粒流": 1, "粒线流": 2, "粒流": 3,
    "粒缓流": 4, "粒摆流": 4, "全停": 4, "[线流、线粒流]": 0,
}

labels = pd.read_csv(LABELS_PATH, dtype={"exam_case_id": str})
dev = labels[labels.evaluation_role == "development"].copy().set_index("exam_case_id")

def find_all_videos():
    videos = {}
    for vid in VIDEO_ROOT.rglob("*.avi"):
        for i, p in enumerate(vid.parts):
            if p.startswith("recovered_") or p.startswith("ANFC"):
                if i + 1 < len(vid.parts):
                    cid = p + "/" + vid.parts[i + 1]
                    videos.setdefault(cid, []).append(str(vid))
                break
    for vid in CONVERTED_DIR.rglob("*.mp4"):
        for i, p in enumerate(vid.parts):
            if p.startswith("recovered_") or p.startswith("ANFC"):
                if i + 1 < len(vid.parts):
                    cid = p + "/" + vid.parts[i + 1]
                    avi_name = vid.stem + ".avi"
                    videos.setdefault(cid, [])
                    videos[cid] = [v for v in videos[cid] if os.path.basename(v) != avi_name]
                    videos[cid].append(str(vid))
                break
    return videos

videos = find_all_videos()
print(f"Found videos for {len(videos)} cases")

existing = {}
feat_csv = OUT_DIR / "flow_features.csv"
if feat_csv.exists():
    df_old = pd.read_csv(feat_csv, dtype={"exam_case_id": str})
    for _, row in df_old.iterrows():
        existing[row["exam_case_id"]] = row.to_dict()
    print(f"Loaded {len(existing)} existing features from v1")

feature_rows = []
for case_id in sorted(dev.index):
    if case_id not in videos:
        continue
    flow_label = dev.at[case_id, "flow_state"]
    if pd.isna(flow_label):
        continue
    flow_cls = FLOW_MAP.get(str(flow_label), -1)
    if flow_cls < 0:
        continue

    if case_id in existing:
        feature_rows.append(existing[case_id])
        continue

    case_features = []
    for vpath in videos[case_id]:
        bn = os.path.basename(vpath)
        print(f"  {case_id} -> {bn}...", end="", flush=True)
        try:
            result = subprocess.run(
                ["/root/miniconda3/bin/python", "/tmp/extract_flow_single.py", vpath],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and result.stdout.strip() != "null":
                feats = json.loads(result.stdout.strip())
                case_features.append(feats)
                mm = feats.get("mag_mean", 0)
                print(f" mag={mm:.2f}")
            else:
                print(f" FAIL(rc={result.returncode})")
        except Exception as e:
            print(f" ERR: {e}")

    if not case_features:
        continue
    agg = {}
    for key in case_features[0]:
        if key in ("n_frames", "fps"):
            agg[key] = case_features[0][key]
        else:
            vals = [f[key] for f in case_features if not np.isnan(f.get(key, 0))]
            agg[key] = float(np.mean(vals)) if vals else 0.0
    agg["exam_case_id"] = case_id
    agg["flow_state"] = str(flow_label)
    agg["flow_cls"] = flow_cls
    agg["n_videos"] = len(case_features)
    feature_rows.append(agg)

df = pd.DataFrame(feature_rows)
df.to_csv(OUT_DIR / "flow_features_v2.csv", index=False)
print(f"\nTotal features: {len(df)} cases")
print(f"Class distribution:")
print(df.flow_cls.value_counts().sort_index())

# Classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

feature_cols = [c for c in df.columns if c not in
                ("exam_case_id", "flow_state", "flow_cls", "n_videos", "n_frames", "fps")]
X = df[feature_cols].values.astype(np.float64)
y = df["flow_cls"].values
case_ids_arr = df["exam_case_id"].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
dev_folds = dev.loc[case_ids_arr, "development_fold"].values.astype(int)

score_rules = json.load(open("/root/nailfold/artifacts/labels/score_rules_v3.json"))
flow_score_map = score_rules["field_rules"]["flow_state"]["mapping"]
idx_to_label = {0: "线流", 1: "线粒流", 2: "粒线流", 3: "粒流", 4: "粒缓流"}

classifiers = {
    "RF": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"),
    "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    "SVM_rbf": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42),
    "SVM_linear": SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=42),
}

print(f"\nFeatures: {X.shape}")
print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

results = {}
for clf_name, clf in classifiers.items():
    oof_pred = np.full(len(y), -1)
    for fold in range(5):
        test_mask = dev_folds == fold
        train_mask = ~test_mask
        if test_mask.sum() == 0:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])
        c = type(clf)(**clf.get_params())
        c.fit(X_tr, y[train_mask])
        oof_pred[test_mask] = c.predict(X_te)
    valid = oof_pred >= 0
    ba = balanced_accuracy_score(y[valid], oof_pred[valid])
    true_scores = np.array([flow_score_map.get(df.iloc[i]["flow_state"], 0) for i in range(len(df))])
    pred_scores = np.array([flow_score_map.get(idx_to_label.get(int(p), ""), 0) if p >= 0 else 0 for p in oof_pred])
    smae = float(np.mean(np.abs(true_scores[valid] - pred_scores[valid])))
    results[clf_name] = {"ba": float(ba), "smae": smae, "n": int(valid.sum())}
    print(f"\n{clf_name}: BA={ba:.3f}, sMAE={smae:.3f} (n={valid.sum()})")
    cm = confusion_matrix(y[valid], oof_pred[valid])
    print(f"  CM:\n{cm}")

# Feature importance
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced")
scaler = StandardScaler()
rf.fit(scaler.fit_transform(X), y)
importances = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])
print("\nTop 10 features:")
for fname, imp in importances[:10]:
    print(f"  {fname}: {imp:.3f}")

results["baseline_smae"] = 0.427
results["n_cases"] = len(df)
(OUT_DIR / "exp4a_v2_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2) + "\n"
)
print(f"\nDone. Saved to {OUT_DIR}")
