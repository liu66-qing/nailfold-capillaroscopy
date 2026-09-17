from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
lab = pd.read_csv(ROOT / 'artifacts/manifest/locked_evaluation_v1.csv')
idx = pd.read_csv(ROOT / 'artifacts/features/image_index.csv')
lab.exam_case_id = lab.exam_case_id.astype(str)
idx.exam_case_id = idx.exam_case_id.astype(str)
dev = lab[lab.evaluation_role.eq('development')].copy()

rows = []
for field in ('crossing_ratio', 'malformation_ratio'):
    for value, count in dev[field].value_counts(dropna=False).items():
        rows.append({'field': field, 'raw_class': value, 'count': int(count)})
pd.DataFrame(rows).to_csv(ROOT / 'artifacts/label_distribution_morphology.csv', index=False)

frame = idx.groupby('exam_case_id').size().rename('frame_count').to_frame()
frame = frame.join(dev.set_index('exam_case_id')[['development_fold', 'hemorrhage']], how='inner')
frame['hemorrhage_positive'] = frame.hemorrhage.ne('无')
frame.groupby('development_fold').agg(cases=('frame_count', 'size'), frame_mean=('frame_count', 'mean'), frame_median=('frame_count', 'median'), hemorrhage_positive_cases=('hemorrhage_positive', 'sum')).reset_index().to_csv(ROOT / 'artifacts/fold_frame_summary.csv', index=False)
frame[frame.hemorrhage_positive].reset_index().to_csv(ROOT / 'artifacts/hemorrhage_frame_analysis.csv', index=False)

routes = [
    ('clarity', .733228, 'rank8_lr1e4', 'development', 0, 'candidate'),
    ('blood_color', .710474, 'v3_detector', 'development', 0, 'baseline'),
    ('exudation', .771922, 'rank4_lr1e4', 'development', 4, 'candidate_locked_drop'),
    ('subpapillary_venous_plexus', .807598, 'rank8', 'development', 4, 'candidate'),
    ('papilla', .660569, 'rank16', 'development', 0, 'candidate'),
    ('hemorrhage', .500000, 'dino_et', 'development', 0, 'fail'),
    ('capillary_count', .528698, 'v1_detector_gbt', 'development', 0, 'fail'),
    ('crossing_ratio', .496575, 'lagging_xgb', 'development', 0, 'fail'),
    ('malformation_ratio', .610606, 'lagging_binary_exploratory', 'development', 0, 'hold'),
]
pd.DataFrame(routes, columns=['field', 'best_ba', 'source', 'dev_or_locked', 'fold_wins', 'gate_status']).to_csv(ROOT / 'artifacts/field_routing_status.csv', index=False)
print('reports_written')
