import pandas as pd, numpy as np
r=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv'); d=r[r.evaluation_role=='development']; s=pd.read_parquet('/root/nailfold/artifacts/experiments/model-v4-20260901/seg_features_det_unet.parquet'); print('dev',len(d),'seg',len(s),'overlap',len(set(d.exam_case_id.astype(str))&set(s.case_id.astype(str))))
