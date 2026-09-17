import json,glob,os
for p in glob.glob('/root/nailfold/artifacts/experiments/retrain_reviewed/**/*',recursive=True):
 if 'svp' in p.lower() or 'subpapillary' in p.lower(): print(p)

