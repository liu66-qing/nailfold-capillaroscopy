from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
idx = pd.read_csv(ROOT / 'artifacts/features/dinov2/index.csv')
X = np.load(ROOT / 'artifacts/features/dinov2/features.npy')
lab = pd.read_csv(ROOT / 'artifacts/manifest/locked_evaluation_v1.csv')
idx.exam_case_id = idx.exam_case_id.astype(str)
lab.exam_case_id = lab.exam_case_id.astype(str)
frame = pd.DataFrame(X)
frame['exam_case_id'] = idx.exam_case_id
case = frame.groupby('exam_case_id').mean().join(
    lab.set_index('exam_case_id')[['hemorrhage', 'development_fold', 'evaluation_role']], how='inner')
case = case[case.evaluation_role.eq('development') & case.hemorrhage.notna()].copy()
y = case.hemorrhage.map({'无': 0, '1--2': 1, '管袢/一指甲襞': 1}).astype(int).to_numpy()
Z = case.iloc[:, :-3].to_numpy()
pred = np.zeros(len(case), dtype=int)
for fold in range(5):
    train = case.development_fold.to_numpy() != fold
    model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight='balanced', random_state=17, n_jobs=-1)
    model.fit(Z[train], y[train])
    pred[~train] = model.predict(Z[~train])
out = ROOT / 'artifacts/experiments/model-v6-20260901'
pd.DataFrame({'exam_case_id': case.index, 'development_fold': case.development_fold.astype(int),
              'field': 'hemorrhage', 'truth': y, 'prediction': pred}).to_csv(out / 'hemorrhage_dino_et_oof.csv', index=False)
print({'balanced_accuracy': balanced_accuracy_score(y, pred), 'n_cases': len(case),
       'positive_cases': int(y.sum()), 'locked_cases_seen': 0})
