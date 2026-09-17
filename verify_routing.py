import pandas as pd, numpy as np
from sklearn.metrics import balanced_accuracy_score

ROUTING = {
    "clarity": ("rank8_lr1e4", "/root/nailfold/artifacts/experiments/model-v6-20260901/rank8_lr1e4_oof_predictions.csv"),
    "blood_color": ("v3_detector", "/root/nailfold/artifacts/experiments/model-v3-20260901/formal_oof/oof_predictions.csv"),
    "exudation": ("rank4_lr1e4", "/root/nailfold/artifacts/experiments/model-v6-20260901/rank4_lr1e4/oof_predictions.csv"),
    "subpapillary_venous_plexus": ("rank8", "/root/nailfold/artifacts/experiments/model-v6-20260901/oof_rank8/oof_predictions.csv"),
    "papilla": ("rank16", "/root/nailfold/artifacts/experiments/model-v6-20260901/rank16_oof_predictions.csv"),
}
MAP = {
    "clarity": {"清晰": 0, "不清": 1, "模糊": 1},
    "blood_color": {"暗红": 0, "暗紫": 0, "浅红": 1, "淡红": 1},
    "exudation": {"无": 0, "+": 1, "++": 1, "+++": 1},
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 1, ">2排,扩张": 1},
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 1},
}
routing_claimed = {"clarity": 0.7332, "blood_color": 0.7105, "exudation": 0.7719, "subpapillary_venous_plexus": 0.8076, "papilla": 0.6606}

labels = pd.read_csv("/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv")
dev = labels[labels.evaluation_role == "development"].copy()
dev.exam_case_id = dev.exam_case_id.astype(str)
backup = pd.read_csv("/root/nailfold/artifacts/manifest/locked_evaluation_v1_backup_pre_dirty_fix.csv")
bdev = backup[backup.evaluation_role == "development"].copy()
bdev.exam_case_id = bdev.exam_case_id.astype(str)

total = 0
for fname in ROUTING:
    model, path = ROUTING[fname]
    oof = pd.read_csv(path)
    oof.exam_case_id = oof.exam_case_id.astype(str)
    of = oof[oof.field == fname].copy()
    ba = balanced_accuracy_score(of.truth, of.prediction)
    n = len(of)
    np_ = int(of.truth.sum())
    nn_ = n - np_
    claimed = routing_claimed[fname]

    # Check truth vs original dirty labels
    m = of.merge(bdev[["exam_case_id", fname]], on="exam_case_id")
    m["t"] = m[fname].map(MAP[fname])
    v = m.t.notna()
    mm = int((m.loc[v, "truth"] != m.loc[v, "t"]).sum())

    total += ba
    delta = ba - claimed
    print(f"{fname:28s} | {model:14s} | {n:3d} | {np_:3d}/{nn_:3d} | {ba:.4f} | {claimed:.4f} | {delta:+.4f} | {mm} mismatches vs dirty labels")

vals = list(routing_claimed.values())
print(f"\nMean BA verified: {total/5:.4f}")
print(f"Mean BA claimed:  {sum(vals)/5:.4f}")

# Also check: v3_detector has soft probs, show threshold potential
print("\n=== v3_detector threshold analysis (blood_color) ===")
oof = pd.read_csv(ROUTING["blood_color"][1])
oof.exam_case_id = oof.exam_case_id.astype(str)
bc = oof[oof.field == "blood_color"].copy()
if "positive_probability" in bc.columns:
    y = bc.truth.values
    p = bc.positive_probability.values
    for th in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        ba_th = balanced_accuracy_score(y, (p >= th).astype(int))
        print(f"  threshold={th:.2f} -> BA={ba_th:.4f}")
