"""Dump the full decoded text of one RTF to see what these small files hold."""
import sys
sys.path.insert(0, "/tmp")
from probe_rtf import rtf_to_text  # reuse the de-tokenizer

for case in ["recovered_archive1/51", "recovered_archive3/243"]:
    for name in ("rep_rch1.rtf", "prn_rch1.rtf"):
        path = "/root/nailfold/data/%s/%s" % (case, name)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        print("=" * 70)
        print(case, name)
        txt = rtf_to_text(raw)
        print(repr(txt[:900]))
        sys.stdout.flush()
