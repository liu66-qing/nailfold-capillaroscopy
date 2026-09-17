import json,pandas as pd
m=json.load(open('/root/nailfold/artifacts/features/geometry_dev/metadata.json')); print(m)
i=pd.read_csv('/root/nailfold/artifacts/features/geometry_dev/index.csv'); r=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv'); print(i.shape,list(i.columns)); print('dev overlap',len(set(i.exam_case_id.astype(str))&set(r.loc[r.evaluation_role=='development','exam_case_id'].astype(str))))
