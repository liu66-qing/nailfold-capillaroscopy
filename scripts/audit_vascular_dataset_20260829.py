from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
except Exception:
    Image = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "血管数据集"
RECOVERED = ROOT / "data"
MANIFEST = ROOT / "artifacts" / "manifest" / "locked_evaluation_v1.csv"
OUT = ROOT / "artifacts" / "audits" / "vascular_dataset_20260829"
OUT.mkdir(parents=True, exist_ok=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def phash(path: Path) -> str | None:
    if Image is None:
        return None
    try:
        im = Image.open(path).convert("L").resize((16, 16))
        pix = list(im.getdata())
        avg = sum(pix) / len(pix)
        bits = "".join("1" if p >= avg else "0" for p in pix)
        return f"{int(bits, 2):064x}"[:64]
    except Exception:
        return None


def image_info(path: Path):
    if Image is None:
        return None, None, None
    try:
        with Image.open(path) as im:
            return im.width, im.height, im.mode
    except Exception:
        return None, None, None


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_inventory():
    base = DATA / "分类数据集"
    pre_img = base / "扩充之前" / "images"
    pre_xml = base / "扩充之前" / "annotations"
    img = base / "image"
    yolo = base / "labelYOLOs"
    xmls = sorted(pre_xml.glob("*.xml"))
    pre_images = sorted(pre_img.glob("*"))
    aug_images = sorted(img.glob("*"))
    yolo_files = sorted(yolo.glob("*.txt"))
    labels = Counter(); bbox_stats = Counter(); xml_missing_img=[]; xml_dim_mismatch=[]
    xml_objects = 0; invalid_boxes=[]; xml_dims=Counter(); source_paths=Counter()
    for xf in xmls:
        try:
            root = ET.parse(xf).getroot()
            fn = (root.findtext("filename") or "").strip()
            sz = root.find("size")
            w = int(sz.findtext("width")); h = int(sz.findtext("height"))
            xml_dims[(w,h)] += 1; source_paths[(root.findtext("path") or "").strip()] += 1
            im = pre_img / fn
            if not im.exists(): xml_missing_img.append(str(xf))
            iw, ih, _ = image_info(im) if im.exists() else (None,None,None)
            if iw and (iw,ih)!=(w,h): xml_dim_mismatch.append({"xml":str(xf),"declared":[w,h],"actual":[iw,ih]})
            for o in root.findall("object"):
                lab=(o.findtext("name") or "").strip(); labels[lab]+=1; xml_objects+=1
                bb=o.find("bndbox"); vals=[]
                for k in ("xmin","ymin","xmax","ymax"):
                    try: vals.append(float(bb.findtext(k)))
                    except Exception: vals.append(float("nan"))
                if any(math.isnan(v) for v in vals) or vals[0] < 0 or vals[1] < 0 or vals[2] <= vals[0] or vals[3] <= vals[1] or (sz is not None and (vals[2]>w or vals[3]>h)):
                    invalid_boxes.append({"xml":str(xf),"label":lab,"bbox":vals,"size":[w,h]})
                bbox_stats[(lab, (o.findtext("truncated") or "").strip(), (o.findtext("difficult") or "").strip())]+=1
        except Exception as e:
            invalid_boxes.append({"xml":str(xf),"error":repr(e)})
    yolo_classes=Counter(); yolo_bad=[]; yolo_lines=0
    for tf in yolo_files:
        try:
            lines=[x.strip() for x in tf.read_text(encoding="utf-8",errors="replace").splitlines() if x.strip()]
            for ln in lines:
                p=ln.split(); yolo_lines+=1
                if len(p)!=5: yolo_bad.append({"file":str(tf),"line":ln}); continue
                try:
                    c=int(p[0]); vals=list(map(float,p[1:])); yolo_classes[c]+=1
                    if not all(0<=v<=1 for v in vals) or vals[2]<=0 or vals[3]<=0: yolo_bad.append({"file":str(tf),"line":ln})
                except Exception: yolo_bad.append({"file":str(tf),"line":ln})
        except Exception as e: yolo_bad.append({"file":str(tf),"error":repr(e)})
    # augmentation groups and exact duplicates
    groups=defaultdict(list)
    for p in aug_images:
        m=re.match(r"^(\d+)(?:_(\d+))?\.[^.]+$", p.name)
        groups[m.group(1) if m else p.stem].append(p.name)
    dim_counter=Counter(); md5_counter=Counter(); ph_counter=Counter(); bad_img=[]
    for p in pre_images+aug_images:
        w,h,mode=image_info(p); dim_counter[(w,h,mode)]+=1
        try: md5_counter[md5(p)]+=1; ph_counter[phash(p)]+=1
        except Exception: bad_img.append(str(p))
    return {"pre_images":len(pre_images),"pre_xml":len(xmls),"aug_images":len(aug_images),"yolo_files":len(yolo_files),"xml_objects":xml_objects,"xml_labels":dict(labels),"xml_bbox_flags":{str(k):v for k,v in bbox_stats.items()},"xml_dims":{str(k):v for k,v in xml_dims.items()},"xml_missing_img":xml_missing_img,"xml_dim_mismatch":xml_dim_mismatch,"invalid_boxes":invalid_boxes,"yolo_classes":dict(yolo_classes),"yolo_lines":yolo_lines,"yolo_bad":yolo_bad,"augmentation_group_count":len(groups),"augmentation_group_size_counts":dict(Counter(len(v) for v in groups.values())),"augmentation_examples":dict(list(groups.items())[:10]),"image_dimensions":{str(k):v for k,v in dim_counter.items()},"exact_duplicate_image_hash_groups":sum(v>1 for v in md5_counter.values()),"exact_duplicate_extra_files":sum(v-1 for v in md5_counter.values() if v>1),"phash_duplicate_groups":sum(v>1 for v in ph_counter.values()),"bad_images":bad_img,"source_path_examples":source_paths.most_common(10)}


def segmentation_inventory():
    base=DATA/"分割数据集"; imgs=sorted((base/"images").glob("*")); masks=sorted((base/"mask").glob("*")); anns=sorted((base/"annotations").glob("*.json"))
    labels=Counter(); shape_types=Counter(); json_stats=Counter(); missing_img=[]; missing_mask=[]; dim_mismatch=[]; invalid=[]; imagepath_names=[]; mask_values=Counter(); mask_dims=Counter()
    for jf in anns:
        try:
            d=json.loads(jf.read_text(encoding="utf-8")); json_stats["valid"]+=1
            if d.get("imageData"): json_stats["embedded_imageData"]+=1
            else: json_stats["no_imageData"]+=1
            ip=Path(str(d.get("imagePath", "")).replace("\\","/")).name; imagepath_names.append(ip)
            im=base/"images"/ip
            if not im.exists(): missing_img.append({"json":jf.name,"imagePath":d.get("imagePath")})
            iw,ih,_=image_info(im) if im.exists() else (None,None,None)
            if iw and (d.get("imageWidth"),d.get("imageHeight"))!=(iw,ih): dim_mismatch.append({"json":jf.name,"declared":[d.get("imageWidth"),d.get("imageHeight")],"actual":[iw,ih]})
            for s in d.get("shapes",[]):
                lab=str(s.get("label")); labels[lab]+=1; shape_types[str(s.get("shape_type"))]+=1
                pts=s.get("points") or []
                if len(pts)<3 or any(len(p)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in p) for p in pts): invalid.append({"json":jf.name,"label":lab,"points":len(pts)})
        except Exception as e:
            json_stats["invalid"]+=1; invalid.append({"json":jf.name,"error":repr(e)})
    for p in masks:
        w,h,mode=image_info(p); mask_dims[(w,h,mode)]+=1
        if Image:
            try:
                with Image.open(p) as im: mask_values.update(im.convert("L").getdata())
            except Exception: pass
    img_stems={p.stem for p in imgs}; mask_stems={p.stem for p in masks}; ann_stems={p.stem for p in anns}
    stem_dim_mismatch=[]; stem_pairs=0
    for jf in anns:
        im=base/"images"/(jf.stem+".jpg"); mk=base/"mask"/(jf.stem+".png")
        if im.exists() and mk.exists():
            stem_pairs += 1
            d=json.loads(jf.read_text(encoding="utf-8"))
            actual=image_info(im)[0:2]
            if tuple(actual)!=(d.get("imageWidth"),d.get("imageHeight")) or image_info(mk)[0:2]!=actual:
                stem_dim_mismatch.append({"stem":jf.stem,"json_declared":[d.get("imageWidth"),d.get("imageHeight")],"image":list(actual),"mask":list(image_info(mk)[0:2])})
    missing_mask=[s for s in sorted(img_stems-ann_stems)[:100]]
    return {"images":len(imgs),"masks":len(masks),"json_annotations":len(anns),"json_stats":dict(json_stats),"labels":dict(labels),"shape_types":dict(shape_types),"missing_images_for_json":missing_img[:100],"missing_image_count":len(missing_img),"image_dim_mismatch":dim_mismatch[:100],"image_dim_mismatch_count":len(dim_mismatch),"stem_pairs_image_mask_json":stem_pairs,"stem_dimension_mismatch_count":len(stem_dim_mismatch),"stem_dimension_mismatch_examples":stem_dim_mismatch[:100],"invalid_shapes":invalid[:100],"invalid_shape_count":len(invalid),"image_without_json_stem_count":len(img_stems-ann_stems),"json_without_image_stem_count":len(ann_stems-img_stems),"image_without_json_examples":missing_mask,"mask_dimensions":{str(k):v for k,v in mask_dims.items()},"mask_value_top20":mask_values.most_common(20),"imagepath_examples":imagepath_names[:10]}


def overlap_inventory():
    # Exact and perceptual hashes for external/recovered images; no model or locked image selection.
    new_files=[]
    for p in (DATA/"分类数据集"/"扩充之前"/"images").glob("*"): new_files.append(("classification_pre",p))
    for p in (DATA/"分类数据集"/"image").glob("*"): new_files.append(("classification_aug",p))
    for p in (DATA/"分割数据集"/"images").glob("*"): new_files.append(("segmentation",p))
    rec_files=[]
    for arc in ("recovered_archive1","recovered_archive2","recovered_archive3"):
        for p in (RECOVERED/arc).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"}: rec_files.append((arc,p))
    new_md5=defaultdict(list); new_ph=defaultdict(list)
    for typ,p in new_files:
        try: new_md5[md5(p)].append((typ,str(p))); new_ph[phash(p)].append((typ,str(p)))
        except Exception: pass
    exact=[]; perceptual=[]
    for arc,p in rec_files:
        try:
            h=md5(p)
            if h in new_md5: exact.append({"archive":arc,"recovered":str(p),"matches":new_md5[h]})
            ph=phash(p)
            if ph in new_ph: perceptual.append({"archive":arc,"recovered":str(p),"matches":new_ph[ph]})
        except Exception: pass
    return {"new_image_files":len(new_files),"recovered_image_files":len(rec_files),"exact_overlap_count":len(exact),"exact_overlap":exact[:500],"perceptual_hash_overlap_count":len(perceptual),"perceptual_hash_overlap_examples":perceptual[:200]}


def manifest_inventory():
    rows=list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig",newline="")))
    roles=Counter(r.get("evaluation_role") for r in rows); archives=Counter(r.get("archive") for r in rows); splits=Counter(r.get("split") for r in rows)
    ids={r.get("exam_case_id") for r in rows}; locked={r.get("exam_case_id") for r in rows if r.get("evaluation_role")=="locked_test"}; dev={r.get("exam_case_id") for r in rows if r.get("evaluation_role")=="development"}
    return {"rows":len(rows),"unique_exam_case_id":len(ids),"roles":dict(roles),"archives":dict(archives),"splits":dict(splits),"locked_count":len(locked),"development_count":len(dev),"locked_ids_sample":sorted(locked)[:10],"development_ids_sample":sorted(dev)[:10]}


def main():
    report={"audit_date":"2026-08-29","scope":"read-only inventory; no GPU; locked set not used for model selection","classification":classify_inventory(),"segmentation":segmentation_inventory(),"overlap_with_recovered":overlap_inventory(),"existing_manifest":manifest_inventory()}
    write_json("audit.json",report)
    lines=["# 血管数据集只读治理审计（2026-08-29）","","机器可读明细：`audit.json`。","", "## 核心计数"]
    c=report["classification"]; s=report["segmentation"]; o=report["overlap_with_recovered"]; m=report["existing_manifest"]
    lines += [f"- 分类：扩充前图片 {c['pre_images']}、XML {c['pre_xml']}、扩充后图片 {c['aug_images']}、YOLO 文件 {c['yolo_files']}；XML 目标 {c['xml_objects']}。",f"- 分类 XML 类别：{c['xml_labels']}；YOLO 类别 ID：{c['yolo_classes']}。",f"- 分割：图片 {s['images']}、mask {s['masks']}、LabelMe JSON {s['json_annotations']}；形状类别 {s['labels']}；shape_type {s['shape_types']}。",f"- recovered 图像精确哈希重叠 {o['exact_overlap_count']} 条（感知哈希候选 {o['perceptual_hash_overlap_count']} 条）。",f"- 既有清单：{m['unique_exam_case_id']} 个病例，locked_test {m['locked_count']}，development {m['development_count']}。"]
    (OUT/"audit_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
