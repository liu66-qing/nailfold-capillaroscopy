"""Download the NON-derived parts of HanaNguyen/Capillary-Dataset to data/.

SKIPPED ON PURPOSE: Classification/ (48027 files, the bulk of the 13.5 GB) is
combinatorial re-pairing of the 8368 classification_original images -- e.g.
concat_100_(0, 1, 16, 17, ...).jpg.jpg. Derived, not new data, so downloading it
would buy nothing but disk.

WHAT WE ACTUALLY WANT, and why:
  Morphology_detection/   1298 img + YOLO labels, nc=4 bushy/crossing/hairpin/
                          tortuous. Our crossing_ratio (-0.011) and
                          malformation_ratio (+0.080, LOAO 0/3) are the two
                          fields where the segmenter's correlation sign is
                          INVERTED (rho=-0.257). These are someone else's boxes
                          on exactly those morphologies.
  classification_original/ 8368 img from 140 PEOPLE (126 diabetic + 14 healthy).
                          The only non-circular per-person disease label we have.
  Video/                  121 .mpg -- the only time-base source in reach. Note
                          EXP-B already closed frame-averaged video for
                          flow_state (0.204 vs random 0.250), so this is for
                          pretraining only, NOT supervision of the C-class
                          (per-minute / per-15s) fields.

The server's huggingface-cli chokes on --include with an empty match list
(tqdm ValueError: min() arg is an empty sequence), and list_repo_files times out
through hf-mirror, so this uses snapshot_download with allow_patterns and a
retry loop, and reports counts from the filesystem rather than from any listing.
"""
import os
import sys
import time
import collections

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download  # noqa: E402

REPO = "HanaNguyen/Capillary-Dataset"
DEST = r"E:\甲劈微循环\data\hf_capillary"

# Smallest first: fail fast on the cheap one before committing to 7 GB of video.
GROUPS = [
    ("morphology", ["Morphology_detection/**", "README.md", ".gitattributes"]),
    ("classification_original", ["classification_original/**"]),
    ("video", ["Video/**"]),
]


def count(root):
    n = b = 0
    ext = collections.Counter()
    for dirpath, _dirs, files in os.walk(root):
        if ".cache" in dirpath:
            continue
        for f in files:
            p = os.path.join(dirpath, f)
            try:
                b += os.path.getsize(p)
            except OSError:
                continue
            n += 1
            ext[os.path.splitext(f)[1].lower()] += 1
    return n, b, ext


def main():
    only = sys.argv[1:] or [g for g, _ in GROUPS]
    os.makedirs(DEST, exist_ok=True)
    for name, pats in GROUPS:
        if name not in only:
            continue
        print("=== %s  %s" % (name, pats[0]), flush=True)
        for attempt in (1, 2, 3, 4):
            try:
                snapshot_download(
                    repo_id=REPO, repo_type="dataset", local_dir=DEST,
                    allow_patterns=pats, max_workers=8)
                break
            except Exception as e:
                print("  attempt %d failed: %s: %s"
                      % (attempt, type(e).__name__, str(e)[:200]), flush=True)
                if attempt == 4:
                    print("  GIVING UP on %s" % name, flush=True)
                time.sleep(15)
        n, b, _ext = count(DEST)
        print("  cumulative: %d files, %.2f GB" % (n, b / 1e9), flush=True)

    n, b, ext = count(DEST)
    print("\nTOTAL %d files %.2f GB in %s" % (n, b / 1e9, DEST))
    for e, c in ext.most_common(10):
        print("  %-8s %d" % (e or "(none)", c))
    for sub in ("Morphology_detection", "classification_original", "Video"):
        p = os.path.join(DEST, sub)
        if os.path.isdir(p):
            sn, sb, _ = count(p)
            print("  %-26s %5d files %7.2f GB" % (sub, sn, sb / 1e9))


if __name__ == "__main__":
    main()
