"""Convert VOC XML annotations to YOLO detection format + fix typos.
Also splits into 5-fold CV based on case_id.
"""
import os, xml.etree.ElementTree as ET, random, shutil
from pathlib import Path
from collections import Counter, defaultdict

ANNO_DIR = Path(r"E:\甲劈微循环\data\血管数据集\分类数据集\annotations")
IMG_DIR = Path(r"E:\甲劈微循环\data\血管数据集\分类数据集\images")
OUT_DIR = Path(r"E:\甲劈微循环\data\yolo_det_3class")

CLASS_MAP = {"vessel": 0, "malformed_vessel": 1, "cross_vessel": 2, "corss_vessel": 2}  # fix typo

def parse_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    filename = root.find("filename").text
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            print(f"WARN: unknown class '{name}' in {xml_path}")
            continue
        cls = CLASS_MAP[name]
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        # YOLO format: cx cy w h (normalized)
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        boxes.append((cls, cx, cy, bw, bh))
    return filename, boxes

def get_case_id(filename):
    """Extract case numeric id from filename like '100_0.xml' -> '100'."""
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[0]

def main():
    xmls = sorted(ANNO_DIR.glob("*.xml"))
    print(f"Found {len(xmls)} XML files")

    # Parse all
    data = []  # (xml_path, img_filename, boxes, case_id)
    class_counts = Counter()
    for xml_path in xmls:
        filename, boxes = parse_voc(xml_path)
        case_id = get_case_id(xml_path.name)
        data.append((xml_path, filename, boxes, case_id))
        for cls, *_ in boxes:
            class_counts[cls] += 1

    print(f"Class counts: {class_counts}")
    print(f"Total boxes: {sum(class_counts.values())}")

    # Group by case_id for case-level fold split
    cases = defaultdict(list)
    for i, (_, _, _, cid) in enumerate(data):
        cases[cid].append(i)
    case_ids = sorted(cases.keys(), key=int)
    print(f"Unique cases: {len(case_ids)}")

    # 5-fold split (deterministic)
    random.seed(42)
    random.shuffle(case_ids)
    folds = {cid: i % 5 for i, cid in enumerate(case_ids)}

    # Check if images exist
    if IMG_DIR.exists():
        sample = data[0][1]
        img_path = IMG_DIR / sample
        print(f"Sample image check: {img_path} exists={img_path.exists()}")
    else:
        print(f"IMG_DIR {IMG_DIR} does not exist — will need to verify on server")

    # Write YOLO format
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold_id in range(5):
        for split in ["train", "val"]:
            (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
            (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # For each fold: val=fold, train=rest
    fold_stats = defaultdict(lambda: {"images": 0, "boxes": 0})
    for fold_id in range(5):
        fold_dir = OUT_DIR / f"fold{fold_id}"
        for split in ["train", "val"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        for xml_path, img_filename, boxes, case_id in data:
            is_val = (folds[case_id] == fold_id)
            split = "val" if is_val else "train"
            label_stem = xml_path.stem
            label_path = fold_dir / split / "labels" / f"{label_stem}.txt"
            with open(label_path, "w") as f:
                for cls, cx, cy, bw, bh in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            # Write image symlink/path reference (actual images will be on server)
            img_list_path = fold_dir / f"{split}.txt"
            with open(img_list_path, "a") as f:
                f.write(f"./images/{img_filename}\n")
            fold_stats[(fold_id, split)]["images"] += 1
            fold_stats[(fold_id, split)]["boxes"] += len(boxes)

    # Print stats
    for fold_id in range(5):
        tr = fold_stats[(fold_id, "train")]
        va = fold_stats[(fold_id, "val")]
        print(f"Fold {fold_id}: train {tr['images']} imgs / {tr['boxes']} boxes, "
              f"val {va['images']} imgs / {va['boxes']} boxes")

    # Write dataset YAML template
    yaml_template = """# YOLO Detection - 3 class nailfold vessels
# Use with: yolo detect train data=dataset.yaml model=yolo11m.pt
path: /root/nailfold/data/yolo_det_3class/fold{fold}
train: train/images
val: val/images

names:
  0: vessel
  1: malformed_vessel
  2: cross_vessel
"""
    for fold_id in range(5):
        yaml_path = OUT_DIR / f"fold{fold_id}" / "dataset.yaml"
        yaml_path.write_text(yaml_template.format(fold=fold_id))

    # Also write a flat labels dir (all labels, no fold split) for quick stats
    flat_dir = OUT_DIR / "all_labels"
    flat_dir.mkdir(exist_ok=True)
    for xml_path, _, boxes, _ in data:
        label_path = flat_dir / f"{xml_path.stem}.txt"
        with open(label_path, "w") as f:
            for cls, cx, cy, bw, bh in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"\nDone. Output: {OUT_DIR}")
    print(f"Fold YAMLs written. Labels converted. Typo fixed (corss_vessel -> cross_vessel).")

if __name__ == "__main__":
    main()
