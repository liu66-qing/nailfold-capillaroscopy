import pandas as pd
d=pd.read_parquet('/root/nailfold/artifacts/experiments/v10_A3_instance_mil/instance_features.parquet')
print(d.shape); print(list(d.columns)[:20]); print(d.index.name)
