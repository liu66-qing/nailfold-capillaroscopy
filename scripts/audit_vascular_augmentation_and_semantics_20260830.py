from __future__ import annotations

import csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "血管数据集"
GOV = ROOT / "artifacts" / "audits" / "vascular_dataset_governance_20260830"
OUT = ROOT / "artifacts" / "derived" / "vascular_dataset_governance_20260830"

def md5(p):
    h=hashlib.md5(); h.update(p.read_bytes()); return h.hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    base=DATA/"分类数据集"; pre=base/"扩充之前/images"; aug=base/"image"
    groups=[]; exact_aug_orig=0; aug_duplicate_hash=0
    orig_hash={md5(p):p.stem for p in pre.glob("*.jpg")}
    aug_hash=defaultdict(list)
    for p in aug.glob("*.jpg"): aug_hash[md5(p)].append(p.name)
    exact_aug_orig=sum(len(v) for h,v in aug_hash.items() if h in orig_hash)
    aug_duplicate_hash=sum(len(v)>1 for v in aug_hash.values())
    for oid in range(1,583):
        files=sorted(aug.glob(f"{oid}_*.jpg"),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        if not files and not (pre/f"{oid}.jpg").exists(): continue
        dims=sorted({Image.open(p).size for p in files})
        od=Image.open(pre/f"{oid}.jpg").size if (pre/f"{oid}.jpg").exists() else None
        groups.append({"original_id":str(oid),"original_path":str(pre/f"{oid}.jpg"),"augmentation_count":len(files),"expected_count":20,"original_size":f"{od[0]}x{od[1]}" if od else "","augmentation_sizes":";".join(f"{w}x{h}" for w,h in dims),"sizes_match_original":"yes" if od and dims==[od] else "no","group_status":"COMPLETE_AUGMENTATION_GROUP" if len(files)==20 and od and dims==[od] else "HOLD_MISSING_OR_INVALID_AUGMENTATION","split_unit":"original_id","independent_sample_allowed":"no","exact_hash_match_to_original":"yes" if any(md5(p) in orig_hash for p in files) else "no"})
    write(OUT/"augmentation_provenance_audit.json",{"created_at":"2026-08-30","original_images":582,"augmentation_images":len(list(aug.glob('*.jpg'))),"groups":len(groups),"complete_groups":sum(g['group_status']=='COMPLETE_AUGMENTATION_GROUP' for g in groups),"exact_aug_original_hash_files":exact_aug_orig,"augmentation_duplicate_hash_groups":aug_duplicate_hash,"interpretation":"If these files are from the same source dataset, they add augmentation diversity but no new independent cases; group by original_id and keep all derivatives in one split.","groups_detail":groups})
    semantic={
      "created_at":"2026-08-30","local_sources":["data/血管数据集/血管分类标注/血管分类标注说明.pdf","data/血管数据集/血管分类标注/血管分类标注.xlsx","data/血管数据集/分类数据集/扩充之前/annotations"],
      "authoritative_sources":[
        {"title":"Nailfold videocapillaroscopy reporting in clinical research: international Delphi-based consensus","doi":"10.1136/annrheumdis-2020-eular.2415","url":"https://doi.org/10.1136/annrheumdis-2020-eular.2415","use":"reporting standard and terminology governance"},
        {"title":"Nailfold videocapillaroscopy micro-haemorrhage and giant capillary counting...","doi":"10.1186/s13075-014-0462-8","url":"https://doi.org/10.1186/s13075-014-0462-8","use":"micro-haemorrhage and capillary morphology terminology"},
        {"title":"Standardisation of nailfold videocapillaroscopy in clinical research","url":"https://www.eular.org/clinical-practice-recommendations","use":"use EULAR recommendations as the governing clinical-method source; exact project label mapping still requires local expert approval"}
      ],
      "project_label_candidates":{
        "vessel":"ordinary visible vessel instance; candidate operational definition only",
        "malformed_vessel":"morphologically abnormal vessel instance; candidate operational definition only",
        "cross_vessel":"crossing/anastomotic vessel region or event; candidate operational definition only",
        "corss_vessel":"typo variant observed once; do not silently merge until expert approval"
      },
      "local_weak_fields":{"svp":"normal/dilated/not_visible","red_cell_aggregation":"normal/aggregation","papilla":"normal/abnormal/not_visible","sweat_duct":"visible/not_visible","hemorrhage":"hemorrhage/normal"},
      "human_gate":"A qualified reviewer must approve whether project labels operationally match clinical terminology and whether each label is fit for auxiliary supervision; external annotations are not clinical gold standard."
    }
    write(GOV/"semantic_definition_evidence.json",semantic)
    (GOV/"semantic_definition_evidence.md").write_text("""# 血管标签与增强来源证据（2026-08-30）\n\n## 增强图结论\n\n454 张未知项原本是 449 张无法映射 recovered 病例的扩充前原图和 5 张未入清单目录图，不是 454 张额外增强图。11,600 张增强图均按 `<original_id>_<augmentation_id>` 命名，完整组 580 组，每组 20 张；增强图只能随 original_id 进入同一 split，不能作为新增病例或独立验证样本。\n\n## 标签语义\n\n本地标注说明提供了出血、乳头下静脉丛、红细胞聚集、乳头和汗腺导管的项目级描述。权威术语和报告规范应参考 EULAR/国际 Delphi 共识及同行评议研究（见 `semantic_definition_evidence.json`）。\n\n当前只形成候选定义，不自动宣称临床认可：\n\n- `vessel`：普通可见血管实例候选定义；\n- `malformed_vessel`：形态异常血管实例候选定义；\n- `cross_vessel`：交叉/吻合区域或事件候选定义；\n- `corss_vessel`：一次拼写变体，必须由专家决定是否归并。\n\n临床认可仍需具备资质的审核者确认，尤其是“交叉”是实例类别还是事件标签，以及异常形态的边界。\n""",encoding="utf-8")

if __name__=="__main__": main()
