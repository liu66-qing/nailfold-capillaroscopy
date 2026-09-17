"""EXP-F follow-up 4: adjudicate field-value vs subscore using an INDEPENDENT
source -- the narrative RTF (rep_rch*.rtf), which the score-table OCR did not use.

The narrative describes findings in prose. If it mentions exudation/flat papilla
etc. for a case where the field value says "none" but the subscore implies
"present", the SUBSCORE is right and the FIELD VALUE is the corrupted side
(and vice versa).

Run on the periloop-implicated cases for exudation and papilla, the two fields
that dominate. Read-only: opens report files, writes nothing outside OUT.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

OUT = Path('/root/nailfold/artifacts/experiments/exp_f_label_noise')
DATA = Path('/root/nailfold/data')
A_CSV = OUT / 'periloop_implicated_fields.csv'
BS = chr(92)
QT = chr(39)


def rtf_text(path):
    raw = open(path, 'rb').read().decode('latin-1')
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == BS and i + 1 < len(raw) and raw[i + 1] == QT:
            try:
                out.append(int(raw[i + 2:i + 4], 16))
                i += 4
                continue
            except ValueError:
                pass
        out += raw[i].encode('latin-1')
        i += 1
    txt = out.decode('gbk', errors='ignore')
    txt = re.sub(r'[{}]', ' ', txt)
    txt = re.sub(BS + r'[a-zA-Z]+-?[0-9]* ?', ' ', txt)
    return re.sub(r'\s+', ' ', txt)


# narrative cues: (positive = finding present, negative = explicitly absent)
CUES = {
    'exudation': {
        'present': ['血浆成分渗出', '明显的血浆', '有渗出', '渗出明显', '管袢周围有'],
        'absent': ['无渗出', '未见渗出', '渗出无', '没有渗出'],
    },
    'papilla': {
        'present': ['乳头下', '乳头平坦', '平坦', '乳头变平'],
        'absent': ['乳头波纹', '波纹状'],
    },
}


def main():
    A = pd.read_csv(A_CSV, dtype={'exam_case_id': str})
    res = []
    for _, r in A.iterrows():
        f = r['field']
        if f not in CUES:
            continue
        case_dir = DATA / r['exam_case_id']
        rtfs = sorted(case_dir.glob('rep_rch*.rtf')) + sorted(case_dir.glob('prn_rch*.rtf'))
        if not rtfs:
            res.append({**r.to_dict(), 'narrative': 'NO_RTF', 'verdict': 'unknown'})
            continue
        txt = ' '.join(rtf_text(p) for p in rtfs)
        pos = [c for c in CUES[f]['present'] if c in txt]
        neg = [c for c in CUES[f]['absent'] if c in txt]
        # does the narrative support the extracted value or the implied one?
        def severity(v):
            if f == 'exudation':
                return 0 if str(v).strip() in ('无',) else 1
            return 1 if '平坦' in str(v) else 0
        ex_sev, im_sev = severity(r['extracted_value']), severity(r['implied_value'])
        nar_sev = 1 if (pos and not neg) else (0 if (neg and not pos) else None)
        verdict = 'unknown'
        if nar_sev is not None and ex_sev != im_sev:
            if nar_sev == im_sev:
                verdict = 'subscore_right_field_wrong'
            elif nar_sev == ex_sev:
                verdict = 'field_right_subscore_wrong'
        res.append({**r.to_dict(), 'pos_cues': ';'.join(pos), 'neg_cues': ';'.join(neg),
                    'narrative_severity': nar_sev, 'extracted_sev': ex_sev,
                    'implied_sev': im_sev, 'verdict': verdict})
    D = pd.DataFrame(res)
    summary = {'n_examined': int(len(D)),
               'verdict_counts': {k: int(v) for k, v in Counter(D.verdict).items()},
               'by_field': {f: {k: int(v) for k, v in Counter(g.verdict).items()}
                            for f, g in D.groupby('field')}}
    D.to_csv(OUT / 'adjudication.csv', index=False, encoding='utf-8')
    (OUT / 'adjudication.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(D[['exam_case_id', 'field', 'extracted_value', 'implied_value',
             'narrative_severity', 'verdict']].to_string())


if __name__ == '__main__':
    main()
