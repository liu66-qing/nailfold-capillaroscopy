import sys, glob, os, re
BS = chr(92)
def decode_rtf(p):
    raw = open(p,'rb').read()
    txt = raw.decode('latin-1')
    out=[]; i=0; n=len(txt)
    while i<n:
        c=txt[i]
        if c==BS:
            if i+1<n and txt[i+1]=="'":
                hexs=txt[i+2:i+4]
                try: out.append(bytes([int(hexs,16)])); i+=4; continue
                except: pass
            m=re.match(r'[a-zA-Z]+(-?[0-9]+)?[ ]?', txt[i+1:])
            if m:
                ctrl=m.group(0).strip()
                if ctrl.startswith('par') or ctrl.startswith('line'): out.append(b'\n')
                i+=1+len(m.group(0)); continue
            i+=2; continue
        if c in '{}': i+=1; continue
        out.append(c.encode('latin-1')); i+=1
    b=b''.join(out)
    for enc in ('gbk','gb18030','utf-8'):
        try: return b.decode(enc)
        except: pass
    return b.decode('latin-1',errors='replace')

for case in sys.argv[1:]:
    d=f"/root/nailfold/data/{case}"
    print("="*72); print("CASE:",case)
    fs=sorted(glob.glob(d+"/*.rtf"))+sorted(glob.glob(d+"/*.RTF"))
    if not fs: print("  no rtf; files:",[os.path.basename(x) for x in os.listdir(d)][:20])
    for f in fs:
        print(f"--- {os.path.basename(f)} ({os.path.getsize(f)} bytes) ---")
        t=decode_rtf(f)
        t=re.sub(r'\n{3,}','\n\n',t)
        print(t.strip()[:3000])
