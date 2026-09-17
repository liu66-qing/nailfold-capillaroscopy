"""EXP-F follow-up 5: which side is the more reliable anchor, across ALL cases?

Uses the narrative RTF phrase 血浆成分渗出 ("plasma exudation") as an independent
binary indicator of exudation presence, then asks which predictor agrees with it
better on the full dev set:
  (a) the extracted exudation FIELD value  (无 vs +/++/+++)
  (b) the exudation contribution IMPLIED by periloop_score minus the other 4 fields

Whichever agrees better with the narrative is the more trustworthy side. Reports
agreement, and the 2x2 tables. Read-only.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RULES = '/root/nailfold/artifacts/labels/score_rules_v3.json'
MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
OUT = Path('/root/nailfold/artifacts/experiments/exp_f_label_noise')
DATA = Path('/root/nailfold/data')
CONFLICTED = {"recovered_archive2/180", "recovered_archive3/263"}
BS, QT = chr(92), chr(39)
PHRASE = '血浆成分渗出'


def rtf_text(p):
    raw = open(p, 'rb').read().decode('latin-1')
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
    t = out.decode('gbk', errors='ignore')
    t = re.sub(r'[{}]', ' ', t)
    t = re.sub(BS + r'[a-zA-Z]+-?[0-9]* ?', ' ', t)
    return re.sub(r'\s+', ' ', t)


def score(rule, raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == '':
        return None
    s = str(raw).strip()
    if rule.get('type') == 'categorical_lookup':
        return rule['mapping'].get(s)
    if rule.get('type') == 'numeric_tree':
        try:
            x = float(s)
        except ValueError:
            return None
        nd = rule['tree']
        while 'threshold' in nd:
            nd = nd['left'] if x < nd['threshold'] else nd['right']
        return nd.get('value')
    return None


def main():
    R = json.load(open(RULES))
    rules, peri = R['field_rules'], R['groups']['periloop_score']
    L = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
    L['periloop_score'] = pd.to_numeric(L['periloop_score'], errors='coerce')
    L['total_score'] = pd.to_numeric(L['total_score'], errors='coerce')
    d = L[(L.evaluation_role == 'development') & L.total_score.notna()]
    d = d[~d.exam_case_id.isin(CONFLICTED)].reset_index(drop=True)

    others = [f for f in peri if f != 'exudation']
    oth = pd.DataFrame({f: [score(rules[f], v) for v in d[f]] for f in others}).fillna(0.0).sum(axis=1)
    implied = (d.periloop_score - oth).round(4)     # exudation score the subscore implies
    field_sc = pd.Series([score(rules['exudation'], v) for v in d['exudation']])

    nar = []
    for cid in d.exam_case_id:
        ps = sorted((DATA / cid).glob('rep_rch*.rtf')) + sorted((DATA / cid).glob('prn_rch*.rtf'))
        if not ps:
            nar.append(None)
            continue
        txt = ' '.join(rtf_text(p) for p in ps)
        nar.append(PHRASE in txt)
    nar = pd.Series(nar)

    # binarise both predictors: exudation present (>0) vs absent
    field_pos = field_sc > 0.05
    implied_pos = implied > 0.05
    m = nar.notna() & field_sc.notna() & implied.notna()
    res = {'n_with_narrative': int(m.sum()),
           'narrative_positive_rate': float(nar[m].mean())}

    def agree(pred):
        p = pred[m].astype(bool)
        t = nar[m].astype(bool)
        return {'agreement': float((p == t).mean()),
                'n': int(m.sum()),
                'tp': int((p & t).sum()), 'fp': int((p & ~t).sum()),
                'fn': int((~p & t).sum()), 'tn': int((~p & ~t).sum())}

    res['field_value_vs_narrative'] = agree(field_pos)
    res['subscore_implied_vs_narrative'] = agree(implied_pos)

    # restrict to the cases where the two sides DISAGREE -- the informative subset
    dis = m & (field_pos != implied_pos)
    if dis.any():
        t = nar[dis].astype(bool)
        res['on_disagreement_subset'] = {
            'n': int(dis.sum()),
            'narrative_supports_field': int((field_pos[dis].astype(bool) == t).sum()),
            'narrative_supports_subscore': int((implied_pos[dis].astype(bool) == t).sum()),
        }
    (OUT / 'narrative_agreement.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
