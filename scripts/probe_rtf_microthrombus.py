"""Extract the microthrombus line from the RTF reports of the ambiguous cases."""
import re
import sys

BS = chr(92)  # literal backslash

CASES = [
    "recovered_archive1/51",
    "recovered_archive3/243",
    "recovered_archive2/137",
    "recovered_archive2/188",
]
ROOT = "/root/nailfold/data"


def rtf_to_text(raw: bytes) -> str:
    """Very small RTF de-tokenizer: resolve \\'xx hex escapes to GBK bytes, drop controls."""
    s = raw.decode("latin-1", errors="replace")
    # drop binary-ish groups that carry pictures
    esc = BS + BS  # regex-escaped literal backslash
    s = re.sub(r"\{" + esc + r"\*?" + esc + r"(?:pict|fonttbl|colortbl|stylesheet|info)[^{}]*\}", " ", s)
    out = bytearray()
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == BS:
            # hex escape \'xx
            if i + 1 < n and s[i + 1] == "'":
                try:
                    out.extend(bytes([int(s[i + 2:i + 4], 16)]))
                    i += 4
                    continue
                except ValueError:
                    i += 2
                    continue
            # control word
            m = re.match(BS + BS + r"([a-zA-Z]+)(-?[0-9]*) ?", s[i:])
            if m:
                word = m.group(1)
                if word in ("par", "line", "cell", "row", "tab"):
                    out.extend(b"\n")
                i += m.end()
                continue
            i += 2
            continue
        if c in "{}":
            i += 1
            continue
        out.extend(c.encode("latin-1", errors="replace"))
        i += 1
    for enc in ("gbk", "gb18030", "utf-8"):
        try:
            return out.decode(enc)
        except UnicodeDecodeError:
            continue
    return out.decode("gbk", errors="replace")


KEYS = ["微血栓", "血栓"]

for case in CASES:
    print("=" * 70)
    print("CASE", case)
    for name in ("rep_rch1.rtf", "prn_rch1.rtf"):
        path = "%s/%s/%s" % (ROOT, case, name)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            print("  [missing]", name, exc)
            continue
        text = rtf_to_text(raw)
        hits = []
        for ln in text.splitlines():
            ln = ln.strip()
            if any(k in ln for k in KEYS):
                hits.append(ln)
        print("  --- %s (%d bytes) hits=%d" % (name, len(raw), len(hits)))
        for h in hits:
            print("      >>", h[:300])
        if not hits:
            # show a compact window around the keyword even if line-splitting failed
            for k in KEYS:
                idx = text.find(k)
                if idx >= 0:
                    print("      ~~ ctx:", repr(text[max(0, idx - 120):idx + 160]))
                    break
            else:
                print("      (no 血栓 keyword found at all)")
    sys.stdout.flush()
