"""Extract geometry image features for development cases only."""
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd
from extract_geometry_features import image_features

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--files',type=Path,required=True); ap.add_argument('--image-root',type=Path,required=True); ap.add_argument('--roles',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    roles=pd.read_csv(args.roles,usecols=['exam_case_id','evaluation_role']); roles.exam_case_id=roles.exam_case_id.astype(str)
    dev=set(roles.loc[roles.evaluation_role.eq('development'),'exam_case_id']); locked=set(roles.loc[roles.evaluation_role.eq('locked_test'),'exam_case_id'])
    if len(dev)!=186 or len(locked)!=47 or dev&locked: raise ValueError('invalid role boundary')
    files=pd.read_csv(args.files); files.exam_case_id=files.exam_case_id.astype(str)
    files=files[files.exam_case_id.isin(dev)].copy()
    if files.exam_case_id.nunique()!=186: raise ValueError(f'development frame coverage={files.exam_case_id.nunique()}')
    vectors=[]; rows=[]; errors=[]
    for n,row in enumerate(files.itertuples(index=False),1):
        try: vectors.append(image_features(args.image_root/row.image_path)); rows.append({'exam_case_id':row.exam_case_id,'image_path':row.image_path})
        except Exception as e: errors.append({'image_path':row.image_path,'error':f'{type(e).__name__}: {e}'})
        if n%100==0 or n==len(files): print(f'{n}/{len(files)} errors={len(errors)}',flush=True)
    if errors: raise RuntimeError(f'{len(errors)} extraction errors')
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True); matrix=np.stack(vectors).astype(np.float32); np.save(out/'features.npy',matrix); pd.DataFrame(rows).to_csv(out/'index.csv',index=False)
    meta={'schema_version':'geometry-image-features-development/1.0','evaluation_role':'development','locked_cases_seen':0,'cases':186,'frames':len(rows),'shape':list(matrix.shape),'source_cases':int(pd.DataFrame(rows).exam_case_id.nunique())}
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__': main()
