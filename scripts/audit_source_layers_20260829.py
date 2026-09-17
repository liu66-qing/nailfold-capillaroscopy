from __future__ import annotations
import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANFC = ROOT / "data" / "ANFC-THU.v1i.coco"
OUT = ROOT / "artifacts" / "audits" / "vascular_dataset_20260829"

def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()

def main():
    source_groups=defaultdict(set); splits={}
    stats={}
    hashes=defaultdict(list)
    for split in ("train","valid"):
        d=json.loads((ANFC/split/"_annotations.coco.json").read_text(encoding="utf-8"))
        imgs=list((ANFC/split).glob("*.jpg"))
        cats={c["id"]:c["name"] for c in d.get("categories",[])}
        ann_counts=Counter(a.get("category_id") for a in d.get("annotations",[]))
        for im in d.get("images",[]):
            base=im["file_name"].split("_jpg.rf")[0]; source_groups[base].add(split)
        stats[split]={"images":len(d.get("images",[])),"files":len(imgs),"annotations":len(d.get("annotations",[])),"categories":cats,"category_counts":{cats.get(k,str(k)):v for k,v in ann_counts.items()},"all_640x640":all((i.get("width"),i.get("height"))==(640,640) for i in d.get("images",[])),"segmentation_annotations":sum(bool(a.get("segmentation")) for a in d.get("annotations",[])),"bbox_invalid":sum(a.get("bbox",[0,0,0,0])[2]<=0 or a.get("bbox",[0,0,0,0])[3]<=0 for a in d.get("annotations",[]))}
        for p in imgs: hashes[md5(p)].append(str(p))
    vascular_dirs=[ROOT/"data"/"血管数据集"/"分类数据集"/"扩充之前"/"images",ROOT/"data"/"血管数据集"/"分类数据集"/"image",ROOT/"data"/"血管数据集"/"分割数据集"/"images"]
    vh=defaultdict(list)
    for b in vascular_dirs:
        for p in b.glob("*"): vh[md5(p)].append(str(p))
    recovered=defaultdict(list)
    for arc in ("recovered_archive1","recovered_archive2","recovered_archive3"):
        for p in (ROOT/"data"/arc).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"}: recovered[md5(p)].append(str(p))
    anfc_vascular=sum(1 for h in hashes if h in vh); anfc_recovered=sum(1 for h in hashes if h in recovered)
    out={"source_layers":{"colleague_annotation_package":"data/血管数据集 (absolute paths in XML are source-machine metadata)","local_raw_case_archives":"data/recovered_archive1,2,3","local_anfc":"data/ANFC-THU.v1i.coco (Roboflow export, CC BY 4.0)"},"anfc":stats,"anfc_source_groups":len(source_groups),"anfc_cross_split_source_groups":sum(len(v)>1 for v in source_groups.values()),"anfc_exact_overlap_file_count_with_vascular_dataset":anfc_vascular,"anfc_exact_overlap_file_count_with_recovered":anfc_recovered}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"source_layers.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__ == "__main__": main()
