import pandas as pd, numpy as np
from scipy.stats import spearmanr

df = pd.read_csv("/root/nailfold/artifacts/experiments/test_set_evaluation/test_predictions.csv")

diff = df.model_ts - df.mode_ts
true_diff = df.true_ts - df.mode_ts.iloc[0]

print("=== MODEL vs MODE DECOMPOSITION ===")
print(f"mode_ts (constant): {df.mode_ts.iloc[0]:.3f}")
print(f"Classification delta: mean={diff.mean():.2f}, std={diff.std():.2f}, range=[{diff.min():.2f}, {diff.max():.2f}]")
print(f"True delta from mode: mean={true_diff.mean():.2f}, std={true_diff.std():.2f}, range=[{true_diff.min():.2f}, {true_diff.max():.2f}]")

sp, p = spearmanr(diff, true_diff)
print(f"Spearman(model_delta, true_delta) = {sp:.3f}, p={p:.4f}")

labels = pd.read_csv("/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv", dtype={"exam_case_id": str})
dev = labels[labels.evaluation_role == "development"]
print(f"\n=== EFFECTIVE TRAINING SIZE PER FIELD ===")
fields = ["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla",
          "capillary_count","crossing_ratio","malformation_ratio",
          "flow_state","rbc_aggregation","microthrombus","hemorrhage"]
for f in fields:
    valid = dev[f].dropna()
    n = len(valid)
    nuniq = valid.nunique()
    print(f"  {f:30s}: n={n:3d}, unique_values={nuniq}")

idx = pd.read_csv("/root/nailfold/artifacts/features/image_index.csv", dtype={"exam_case_id": str})
dev_ids = set(dev.exam_case_id)
dev_frames = idx[idx.exam_case_id.isin(dev_ids)]
print(f"\n  Total dev images: {len(dev_frames)}")
print(f"  Images per case: mean={dev_frames.groupby('exam_case_id').size().mean():.1f}, median={dev_frames.groupby('exam_case_id').size().median():.0f}")

# Check class balance per field
MAP = {
    "clarity": {"清晰": 0, "不清": 1, "模糊": 2},
    "blood_color": {"淡红": 0, "浅红": 1, "暗红": 2, "暗紫": 2},
    "exudation": {"无": 0, "+": 1, "++": 2, "+++": 2},
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 2, ">2排,扩张": 3},
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 2},
    "capillary_count": {"<1": 0, "1--2": 0, "3--4": 1, "5--6": 2, ">=7": 3},
    "crossing_ratio": {"<=30%": 0, "30--60%": 1, "60--80%": 2, ">80%": 2, "[<30%]": 0, "10--30%": 0},
    "malformation_ratio": {"<=10%": 0, "10--30%": 1, "30--60%": 2, ">60%": 3, "[<10%]": 0},
    "flow_state": {"线流": 0, "线粒流": 1, "粒线流": 2, "粒流": 3, "粒缓流": 4, "粒摆流": 4, "全停": 4, "[线流、线粒流]": 0},
    "rbc_aggregation": {"无": 0, "轻度": 1, "中度": 2, "重度": 3, "[无]": 0, "1--30": -1},
    "microthrombus": {"无": 0, "1--2": 1, ">2": 2, "[未见]": 0, "个/min": -1},
    "hemorrhage": {"无": 0, "1--2": 1, "管袢/一指甲襞": -1},
}

print("\n=== CLASS DISTRIBUTION (dev) ===")
for f in fields:
    vals = dev[f].dropna().map(MAP[f])
    vals = vals[vals >= 0]
    counts = vals.value_counts().sort_index()
    total = len(vals)
    dist_str = " | ".join([f"c{i}:{counts.get(i,0):3d}({counts.get(i,0)/total*100:4.1f}%)" for i in range(max(MAP[f].values())+1)])
    minority = counts.min() if len(counts) > 0 else 0
    print(f"  {f:25s} (n={total:3d}): {dist_str}  minority={minority}")
