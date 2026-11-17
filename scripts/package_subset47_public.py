"""Package the authorised 47-exam public image/label dataset."""
import csv,json,hashlib,shutil,zipfile
from pathlib import Path
from PIL import Image
R=Path(__file__).resolve().parents[1]
D=R/'_assets/subset47_public_v1';D.mkdir(parents=True,exist_ok=True)
idx=list(csv.DictReader((R/'artifacts/experiments/locked_consumed_20260924/features_locked/index.csv').open(encoding='utf-8-sig')))
ids=sorted({r['exam_case_id'] for r in idx});assert len(ids)==47
labels={r['exam_case_id']:r for r in csv.DictReader((R/'server_code_audit/locked_evaluation_v1_reviewed.csv').open(encoding='utf-8-sig'))}
fields=['clarity','capillary_count','afferent_diameter','efferent_diameter','output_input_ratio','apex_diameter','loop_length','crossing_ratio','malformation_ratio','flow_state','flow_speed_um_s','vasomotion','rbc_aggregation','wbc_count','microthrombus','blood_color','exudation','hemorrhage','subpapillary_venous_plexus','papilla','sweat_duct']
lout=[];images=[];metadata_count=0
for k,c in enumerate(ids,1):
    case=f'S47_{k:03d}'; lab={'case_id':case}
    for f in fields:
        for key in (f,f+'__confidence',f+'__status'):lab[key]=labels[c][key]
    lout.append(lab)
    for j,r in enumerate([r for r in idx if r['exam_case_id']==c],1):
        src=R/'data/data'/r['image_path']; rel=f'images/{case}/image_{j:03d}.jpg'; dst=D/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(src) as im:
            assert im.size==(1024,768)
            if len(im.getexif()) or any(key in im.info for key in ('comment','xmp','XML:com.adobe.xmp','exif')):
                raise RuntimeError('Metadata requires review before release')
        shutil.copyfile(src,dst)
        images.append({'case_id':case,'image_path':rel,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'bytes':dst.stat().st_size})
for filename,rr in [('labels.csv',lout),('images.csv',images)]:
    with (D/filename).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(D/'README.md').write_text('# 47例图像与逐病例标签\n\n公开发布已获项目提供方授权。47个检查病例；'+str(len(images))+'张原始甲襞图像。图像保持原始字节、分辨率和编码；文件及病例编号已改为S47编号。标签为既有报告解析值，保留缺失值及解析置信度/status，不表示独立临床诊断。\n\n- labels.csv：每病例一行，21字段及解析状态。\n- images.csv：病例与图像对应、大小和SHA256。\n- images/：原图。\n- 同病例图像应作为一组使用。\n- 对应expert-router-final；评估类型见仓库47例子集评估（内部交叉验证折外结果）。\n\n未包含身份映射、姓名、报告扫描、日期或源目录。\n',encoding='utf-8')
zipout=R/'_assets/subset47_public_v1.zip'
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(D.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(D).as_posix())
with zipfile.ZipFile(zipout) as z:assert z.testzip() is None
meta={'cases':47,'images':len(images),'zip_bytes':zipout.stat().st_size,'zip_sha256':hashlib.sha256(zipout.read_bytes()).hexdigest(),'url':'https://github.com/liu66-qing/nailfold-capillaroscopy/releases/download/expert-router-final/subset47_public_v1.zip'}
(R/'release/final_v1/evidence/subset_47/data_download.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
print(json.dumps(meta))
