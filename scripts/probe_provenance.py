"""Probe raw archive for label provenance and patient-level dedup signals."""
import re, glob, os, collections

D = '/root/nailfold/data'
BS = chr(92)   # backslash, avoids escaping issues through ssh


def dec(path):
    """Decode a GBK-escaped RTF into plain text."""
    s = open(path, 'rb').read().decode('latin1')
    s = re.sub(r'\{' + BS + r'\\fonttbl.*?\}\}', '', s, flags=re.S)
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == BS and s[i + 1:i + 2] == "'":
            try:
                out += bytes([int(s[i + 2:i + 4], 16)])
            except ValueError:
                pass
            i += 4
        elif c == BS:
            m = re.match(BS + r'[a-z]+-?\d*\s?', s[i:])
            i += m.end() if m else 2
        elif c in '{}':
            i += 1
        else:
            out += c.encode('latin1')
            i += 1
    return out.decode('gbk', errors='replace').strip()


sample = f'{D}/recovered_archive1/1/rep_rch1.rtf'
if os.path.exists(sample):
    print("=== rep_rch1.rtf decoded ===")
    print(dec(sample))

allf = glob.glob(f'{D}/recovered_archive*/*/*')
print(f"\ntotal files under case dirs: {len(allf)}")

pats = collections.Counter()
ids = collections.Counter()
lat = collections.Counter()
for f in allf:
    n = os.path.basename(f)
    pats[re.sub(r'\d+', '#', n)] += 1
    m = re.match(r'rep_(\d{5,})', n)
    if m:
        ids[m.group(1)] += 1
    if '手' in n:
        lat[n] += 1

print("\ntop filename patterns:")
for k, v in pats.most_common(20):
    print(f"  {v:5d}  {k}")

print(f"\ndistinct rep_<numericID> stems: {len(ids)}")
for k, v in ids.most_common(15):
    print(f"  {k}: {v}")

print(f"\nlaterality-marked filenames: {sum(lat.values())} files, {len(lat)} distinct")
for k, v in lat.most_common(10):
    print(f"  {k}: {v}")

# Do distinct RTF conclusion texts repeat across cases? Identical text on two
# different case dirs is a dedup signal (same person or copied report).
texts = collections.defaultdict(list)
for f in glob.glob(f'{D}/recovered_archive*/*/rep_*.rtf'):
    case = '/'.join(f.split('/')[-3:-1])
    try:
        texts[dec(f)].append(case)
    except Exception:
        pass
dupes = {k: v for k, v in texts.items() if len(set(v)) > 1}
print(f"\ndistinct RTF conclusion texts: {len(texts)}")
print(f"texts shared by >1 case:        {len(dupes)}")
for k, v in list(dupes.items())[:5]:
    print(f"  {len(set(v))} cases share: {k[:60]!r}")
