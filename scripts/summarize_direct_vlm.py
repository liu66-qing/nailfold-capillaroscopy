import json
from pathlib import Path
import pandas as pd

out = Path('/root/autodl-tmp/vlm_direct_20260827/outputs/qwen3_vl_8b')
rows = []
for method in ('zero_shot', 'few_shot'):
    table = pd.read_csv(out / f'{method}_metrics.csv')
    pooled = table[table['fold'].astype(str) == 'pooled']
    for _, row in pooled.iterrows():
        rows.append({'method': method, 'field': row['field'], 'n': int(row['n']),
                     'macro_f1': float(row['macro_f1']),
                     'balanced_accuracy': float(row['balanced_accuracy']),
                     'exact_accuracy': float(row['exact_accuracy'])})
meta_path = out / 'run_metadata.json'
summary = {'model': 'Qwen3-VL-8B-Instruct', 'methods': rows,
           'run_metadata': json.loads(meta_path.read_text()) if meta_path.exists() else {}}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
for method in ('zero_shot', 'few_shot'):
    print('===', method)
    for row in rows:
        if row['method'] == method:
            print(f"{row['field']}\tBA={row['balanced_accuracy']:.4f}\tEX={row['exact_accuracy']:.4f}\tF1={row['macro_f1']:.4f}\tn={row['n']}")
