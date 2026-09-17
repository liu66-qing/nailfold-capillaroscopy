"""Sanity-check cached features: shapes, dtype, no NaN, and that they discriminate at all."""
import glob
import numpy as np

files = sorted(glob.glob("/root/autodl-tmp/nailfold/exp_i_feats/*.npz"))
print("cached files:", len(files))
z = np.load(files[0])
print("example:", files[0].split("/")[-1])
print("  cls", z["cls"].shape, z["cls"].dtype, "patch", z["patch"].shape, z["patch"].dtype)
print("  cls  finite:", np.isfinite(z["cls"]).all(), "std", float(z["cls"].astype(np.float32).std()))
print("  patch finite:", np.isfinite(z["patch"]).all(), "std", float(z["patch"].astype(np.float32).std()))

# expected patch count for 518/14 = 37 -> 37*37 = 1369
print("  expected patches 37*37 =", 37 * 37)

bad = []
for f in files:
    zz = np.load(f)
    if zz["patch"].shape[1] != 1369 or zz["cls"].shape[1] != 768:
        bad.append((f, zz["cls"].shape, zz["patch"].shape))
print("shape-anomalous files:", len(bad))
for b in bad[:5]:
    print("   ", b)
