import pandas as pd
r=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv'); d=pd.read_parquet('/root/nailfold/artifacts/features/geometry_dev/features.parquet') if False else pd.DataFrame()
print('manifest',r.evaluation_role.value_counts().to_dict())
