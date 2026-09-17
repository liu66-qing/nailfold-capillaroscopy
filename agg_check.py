import pandas as pd
d=pd.read_parquet('/root/nailfold/artifacts/experiments/v10_A3_instance_mil/instance_features.parquet')
d.exam_case_id=d.exam_case_id.astype(str)
r=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv'); dev=set(r.loc[r.evaluation_role=='development','exam_case_id'].astype(str))
g=d[d.exam_case_id.isin(dev)].groupby('exam_case_id').size()
print('cases',len(g),'min',g.min(),'median',g.median(),'max',g.max(),'missing',len(dev-set(g.index)))
