import csv, pandas as pd, numpy as np
from sklearn.metrics import balanced_accuracy_score

# Fix CSV: merge split ">2排,扩张" back together
rows = []
with open("/tmp/review_v2_results.csv") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 7:
            parts = parts[:4] + [parts[4]+","+parts[5]] + parts[6:]
        rows.append(parts)

header = rows[0]
with open("/tmp/review_v2_fixed.csv", "w", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    for r in rows:
        w.writerow(r)

review = pd.read_csv("/tmp/review_v2_fixed.csv")
print(f"Review shape: {review.shape}")
review.exam_case_id = review.exam_case_id.astype(str)

# Clean bracket residue
review = review.replace("[不见]", "不见")

FIELDS = ["clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla"]
MAP = {
    "clarity": {"清晰": 0, "不清": 1, "模糊": 1},
    "blood_color": {"暗红": 0, "暗紫": 0, "浅红": 1, "淡红": 1},
    "exudation": {"无": 0, "+": 1, "++": 1, "+++": 1},
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 1, ">2排,扩张": 1},
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 1},
}

print("\n=== Review value counts ===")
for f in FIELDS:
    print(f"\n{f}:")
    print(review[f].value_counts(dropna=False).to_string())

# Load current labels (with dirty fix already applied)
labels = pd.read_csv("/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv")
labels.exam_case_id = labels.exam_case_id.astype(str)
dev = labels[labels.evaluation_role == "development"].copy()

print(f"\n=== DIFF ANALYSIS ===")
total_corrections = 0
total_fills = 0
total_blanks = 0
changes = {}

for f in FIELDS:
    corrections = []
    fills = []
    blanks = []
    for _, row in review.iterrows():
        cid = row.exam_case_id
        new_val = str(row[f]).strip()
        old_row = dev[dev.exam_case_id == cid]
        if len(old_row) == 0:
            continue
        old_val = old_row.iloc[0][f]

        if new_val == "空白":
            blanks.append(cid)
            continue

        old_is_nan = pd.isna(old_val) or str(old_val).strip() == ""
        old_in_map = (not old_is_nan) and (str(old_val) in MAP[f])
        new_in_map = new_val in MAP[f]

        if old_is_nan and new_in_map:
            fills.append((cid, None, new_val))
        elif (not old_is_nan) and new_in_map and str(old_val) != new_val:
            corrections.append((cid, str(old_val), new_val))

    changes[f] = {"corrections": corrections, "fills": fills, "blanks": blanks}
    print(f"\n{f}: {len(corrections)} corrections, {len(fills)} fills, {len(blanks)} blanks")
    for cid, old, new in corrections:
        print(f"  CORRECT: {cid}: {old} -> {new}")
    for cid, old, new in fills:
        print(f"  FILL:    {cid}: NaN -> {new}")
    if blanks:
        print(f"  BLANK:   {blanks}")
    total_corrections += len(corrections)
    total_fills += len(fills)
    total_blanks += len(blanks)

print(f"\n=== TOTAL: {total_corrections} corrections, {total_fills} fills, {total_blanks} blanks ===")

# Apply changes
labels_new = labels.copy()
for f in FIELDS:
    for cid, old, new in changes[f]["corrections"]:
        mask = labels_new.exam_case_id == cid
        labels_new.loc[mask, f] = new
    for cid, old, new in changes[f]["fills"]:
        mask = labels_new.exam_case_id == cid
        labels_new.loc[mask, f] = new
    for cid in changes[f]["blanks"]:
        mask = labels_new.exam_case_id == cid
        labels_new.loc[mask, f] = np.nan

out_path = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
labels_new.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# Count valid samples before/after
print("\n=== Valid sample counts ===")
dev_new = labels_new[labels_new.evaluation_role == "development"].copy()
for f in FIELDS:
    old_valid = dev[f].map(MAP[f]).notna().sum()
    new_valid = dev_new[f].map(MAP[f]).notna().sum()
    print(f"  {f:28s}: {old_valid} -> {new_valid} ({new_valid - old_valid:+d})")

# Re-evaluate routing on new labels
print("\n=== ROUTING BA: old labels vs reviewed labels ===")
ROUTING_PATHS = {
    "clarity": "/root/nailfold/artifacts/experiments/model-v6-20260901/rank8_lr1e4_oof_predictions.csv",
    "blood_color": "/root/nailfold/artifacts/experiments/model-v3-20260901/formal_oof/oof_predictions.csv",
    "exudation": "/root/nailfold/artifacts/experiments/model-v6-20260901/rank4_lr1e4/oof_predictions.csv",
    "subpapillary_venous_plexus": "/root/nailfold/artifacts/experiments/model-v6-20260901/oof_rank8/oof_predictions.csv",
    "papilla": "/root/nailfold/artifacts/experiments/model-v6-20260901/rank16_oof_predictions.csv",
}

total_old = 0
total_new = 0
for f in FIELDS:
    oof = pd.read_csv(ROUTING_PATHS[f])
    oof.exam_case_id = oof.exam_case_id.astype(str)
    of = oof[oof.field == f].copy()

    # Old
    m_old = of.merge(dev[["exam_case_id", f]], on="exam_case_id")
    m_old["y"] = m_old[f].map(MAP[f])
    v_old = m_old.y.notna()
    ba_old = balanced_accuracy_score(m_old.loc[v_old, "y"], m_old.loc[v_old, "prediction"])

    # New
    m_new = of.merge(dev_new[["exam_case_id", f]], on="exam_case_id")
    m_new["y"] = m_new[f].map(MAP[f])
    v_new = m_new.y.notna()
    ba_new = balanced_accuracy_score(m_new.loc[v_new, "y"], m_new.loc[v_new, "prediction"])

    total_old += ba_old
    total_new += ba_new
    delta = ba_new - ba_old
    print(f"  {f:28s} | {ba_old:.4f} -> {ba_new:.4f} ({delta:+.4f})")

print(f"\n  Mean BA: {total_old/5:.4f} -> {total_new/5:.4f} ({(total_new-total_old)/5:+.4f})")
