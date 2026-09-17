#!/usr/bin/env python3
"""EXP-3 Calibration: find μm/pixel from geometry features + label values.

Uses existing 394-dim geometry features and known label values (μm) to
reverse-engineer the pixel-to-μm conversion factor via linear regression.

Fields: afferent_diameter, efferent_diameter, apex_diameter, loop_length
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Paths
GEOM_DIR = Path("/root/nailfold/artifacts/features/geometry_dev")
LABELS_PATH = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
INSTANCE_DIR = Path("/root/nailfold/artifacts/experiments/model-upgrade-20260831/instance_npz_dev")
OUT_DIR = Path("/root/nailfold/artifacts/experiments/exp3_calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEASURE_FIELDS = ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]

def load_geometry_features():
    """Load all geometry feature files and return DataFrame."""
    rows = []
    for npz_path in sorted(GEOM_DIR.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        # Try common keys
        for key in ["features", "geometry", "arr_0"]:
            if key in data:
                feats = data[key]
                break
        else:
            feats = data[list(data.keys())[0]]

        # Extract case_id from filename
        stem = npz_path.stem
        rows.append({"file": stem, "features": feats})
    return rows

def explore_geometry_structure():
    """Understand what's in the geometry features."""
    sample_files = sorted(GEOM_DIR.glob("*"))[:3]
    print(f"Geometry dir: {GEOM_DIR}")
    print(f"Total files: {len(list(GEOM_DIR.glob('*')))}")

    for f in sample_files:
        print(f"\n--- {f.name} ---")
        if f.suffix == ".npz":
            data = np.load(f, allow_pickle=True)
            print(f"  Keys: {list(data.keys())}")
            for k in data.keys():
                arr = data[k]
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
                if arr.ndim == 2:
                    print(f"    First row: {arr[0][:10]}...")
                elif arr.ndim == 1:
                    print(f"    Values: {arr[:10]}...")
        elif f.suffix == ".csv":
            df = pd.read_csv(f, nrows=3)
            print(f"  Columns: {list(df.columns)[:20]}")
            print(f"  Shape: {df.shape}")
        elif f.suffix == ".npy":
            arr = np.load(f, allow_pickle=True)
            print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")

def explore_instance_masks():
    """Understand instance mask structure."""
    sample_files = sorted(INSTANCE_DIR.glob("*"))[:3]
    print(f"\nInstance dir: {INSTANCE_DIR}")
    print(f"Total files: {len(list(INSTANCE_DIR.glob('*')))}")

    for f in sample_files:
        print(f"\n--- {f.name} ---")
        if f.suffix == ".npz":
            data = np.load(f, allow_pickle=True)
            print(f"  Keys: {list(data.keys())}")
            for k in data.keys():
                arr = data[k]
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
                if arr.ndim >= 2:
                    print(f"    Sample: {arr[0][:5] if arr.ndim == 2 else arr.shape}")

def main():
    print("=" * 60)
    print("EXP-3: Geometry Feature Exploration & Calibration")
    print("=" * 60)

    # Step 1: Explore what we have
    explore_geometry_structure()
    explore_instance_masks()

    # Step 2: Load labels
    labels = pd.read_csv(LABELS_PATH, dtype={"exam_case_id": str})
    dev = labels[labels.evaluation_role == "development"]
    idx = dev.set_index("exam_case_id")

    for mf in MEASURE_FIELDS:
        vals = pd.to_numeric(idx[mf], errors="coerce").dropna()
        print(f"\n{mf}: n={len(vals)}, mean={vals.mean():.1f}, std={vals.std():.1f}, "
              f"range=[{vals.min()}, {vals.max()}]")

    # Step 3: Try to find geometry features that correlate with measurements
    # This requires understanding the 394-dim feature vector structure
    # We'll do systematic correlation analysis

    geom_files = sorted(GEOM_DIR.glob("*"))
    print(f"\nGeometry files: {[f.name for f in geom_files[:10]]}")

    # Try to load and match features to cases
    # The matching depends on file naming convention
    # We'll figure this out from the filenames

    results = {
        "status": "exploration_complete",
        "geometry_dir": str(GEOM_DIR),
        "instance_dir": str(INSTANCE_DIR),
        "n_geometry_files": len(geom_files),
    }

    (OUT_DIR / "exploration_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(f"\nResults saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
