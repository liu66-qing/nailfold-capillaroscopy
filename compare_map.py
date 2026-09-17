import glob,os,json
npz=glob.glob('/root/autodl-tmp/nailfold/artifacts/experiments/model-upgrade-20260831/instance_npz_dev/*.npz')
js=json.load(open('/root/autodl-tmp/nailfold/artifacts/pseudo_labels/round1_multiclass_domain_adapted.json'))
keys=set(k.replace('/','__').replace('.jpg','.npz') for k in js)
base=set(os.path.basename(x) for x in npz)
print(json.dumps({'npz':len(base),'pseudo':len(keys),'basename_overlap':len(base&keys),'examples':list(base&keys)[:5]}))
