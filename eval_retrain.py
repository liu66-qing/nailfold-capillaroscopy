import json, os
import pandas as pd, numpy as np
from sklearn.metrics import balanced_accuracy_score

FIELDS = ["clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla"]
MAP = {
    "clarity": {"清晰": 0, "不清": 1, "模糊": 1},
    "blood_color": {"暗红": 0, "暗紫": 0, "浅红": 1, "淡红": 1},
    "exudation": {"无": 0, "+": 1, "++": 1, "+++": 1},
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 1, ">2排,扩张": 1},
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 1},
}

base = "/root/nailfold/artifacts/experiments/retrain_reviewed"
models = ["rank8_lr1e4", "rank4_lr1e4", "rank8", "rank16"]
routing_field = {"rank8_lr1e4": "clarity", "rank4_lr1e4": "exudation", "rank8": "subpapillary_venous_plexus", "rank16": "papilla"}

# Routing baseline (old labels)
routing_ba = {"clarity": 0.7332, "blood_color": 0.7105, "exudation": 0.7719, "subpapillary_venous_plexus": 0.8076, "papilla": 0.6606}
# Routing baseline evaluated on new labels (from earlier analysis)
routing_new = {"clarity": 0.8819, "blood_color": 0.7334, "exudation": 0.8249, "subpapillary_venous_plexus": 0.8294, "papilla": 0.6722}

print("=== RETRAINED MODELS (reviewed labels) — per-field BA ===\n")
for m in models:
    mpath = os.path.join(base, m, "metrics.json")
    if os.path.exists(mpath):
        data = json.load(open(mpath))
        target = routing_field[m]
        print(f"--- {m} (routing target: {target}) ---")
        print(f"  delivery_mean_ba: {data['delivery_mean_ba']:.4f}")
        for f in FIELDS:
            if f in data["fields"]:
                ba = data["fields"][f]["balanced_accuracy"]
                old_r = routing_ba[f]
                new_r = routing_new[f]
                marker = " <-- routing field" if f == target else ""
                print(f"  {f:28s}: {ba:.4f}  (vs old_routing {old_r:.4f} {ba-old_r:+.4f}, vs old_model_new_labels {new_r:.4f} {ba-new_r:+.4f}){marker}")
        print()

# Build new best-per-field routing
print("=== NEW ROUTING TABLE (cherry-pick best per field) ===\n")
best_per_field = {}
for f in FIELDS:
    best_ba = 0
    best_model = ""
    for m in models:
        mpath = os.path.join(base, m, "metrics.json")
        if os.path.exists(mpath):
            data = json.load(open(mpath))
            if f in data["fields"]:
                ba = data["fields"][f]["balanced_accuracy"]
                if ba > best_ba:
                    best_ba = ba
                    best_model = m
    best_per_field[f] = (best_model, best_ba)
    old_r = routing_ba[f]
    new_r = routing_new[f]
    print(f"{f:28s}: {best_ba:.4f} ({best_model:14s}) | old_routing={old_r:.4f} ({best_ba-old_r:+.4f}) | old_model_new_labels={new_r:.4f} ({best_ba-new_r:+.4f})")

vals = [v[1] for v in best_per_field.values()]
old_vals = [routing_ba[f] for f in FIELDS]
new_vals = [routing_new[f] for f in FIELDS]
print(f"\nMean BA: {np.mean(vals):.4f} (was {np.mean(old_vals):.4f} old_routing, {np.mean(new_vals):.4f} old_model_new_labels)")
