import json, pandas as pd
r=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv')
print(r.evaluation_role.value_counts().to_dict())
print('dev',sum(r.evaluation_role=='development'),'locked',sum(r.evaluation_role=='locked_test'))
print('folds',r.loc[r.evaluation_role=='development','development_fold'].value_counts().sort_index().to_dict())
