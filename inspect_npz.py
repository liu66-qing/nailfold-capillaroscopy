import numpy as np,glob,os,json
fs=glob.glob('/root/autodl-tmp/nailfold/artifacts/experiments/model-upgrade-20260831/instance_npz_dev/*.npz'); out={'n':len(fs),'samples':[]}
for f in fs[:20]: out['samples'].append({'file':os.path.basename(f),'keys':np.load(f,allow_pickle=True).files})
print(json.dumps(out))
