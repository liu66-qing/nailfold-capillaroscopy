import numpy as np,glob,collections,json,hashlib,os
fs=glob.glob('/root/autodl-tmp/nailfold/artifacts/experiments/model-upgrade-20260831/instance_npz_dev/*.npz')
c=collections.Counter(); bad=[]
for f in fs:
 try:c[tuple(np.load(f,allow_pickle=True).files)]+=1
 except Exception as e:bad.append([os.path.basename(f),str(e)])
x={'n':len(fs),'key_sets':{str(k):v for k,v in c.items()},'bad':bad,'cls_present_files':sum('cls' in k for k in [np.load(f,allow_pickle=True).files for f in fs])}
print(json.dumps(x))
