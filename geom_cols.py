import pandas as pd
d=pd.read_csv('/root/nailfold/artifacts/features/geometry_dev/features.csv'); print(d.shape); print(list(d.columns)[:30]); print(d.head(1).to_dict())
