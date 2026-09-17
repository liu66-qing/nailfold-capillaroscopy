import pandas as pd
d=pd.read_parquet('/root/nailfold/artifacts/experiments/v10_A3_instance_mil/instance_features.parquet')
print(list(d.columns))
