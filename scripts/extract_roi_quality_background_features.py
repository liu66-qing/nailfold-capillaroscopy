"""Extract compact spatial color/quality features for development frames only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def stats(x: np.ndarray) -> list[float]:
    x = x.astype(np.float32).reshape(-1)
    return [float(x.mean()), float(x.std()), *np.quantile(x, [0.1, 0.5, 0.9]).astype(float)]


def region_features(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out: list[float] = []
    for image in (rgb, lab, hsv):
        for channel in range(3): out.extend(stats(image[..., channel]))
    out.extend(stats(gray))
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    out.extend([float(lap.var()), *stats(np.hypot(gx, gy))])
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [32], [0, 256]).reshape(-1)
    prob = hist / max(float(hist.sum()), 1.0); prob = prob[prob > 0]
    out.append(float(-(prob * np.log2(prob)).sum()))
    chromatic_red = rgb[..., 0] / (rgb.sum(axis=2) + 1.0)
    out.extend(stats(chromatic_red))
    return np.asarray(out, dtype=np.float32)


def features(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None: raise ValueError(f"decode failed: {path}")
    bgr = cv2.resize(bgr, (512, 384), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    regions = [bgr, bgr[: int(.45*h)], bgr[int(.35*h):], bgr[int(.1*h):int(.9*h), int(.15*w):int(.85*w)]]
    vectors = [region_features(x) for x in regions]
    return np.concatenate(vectors + [vectors[1] - vectors[2], vectors[3] - vectors[0]])


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--index",type=Path,required=True); p.add_argument("--image-root",type=Path,required=True); p.add_argument("--roles",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    idx=pd.read_csv(a.index); idx.exam_case_id=idx.exam_case_id.astype(str)
    roles=pd.read_csv(a.roles,usecols=["exam_case_id","evaluation_role"]); roles.exam_case_id=roles.exam_case_id.astype(str)
    dev=set(roles.loc[roles.evaluation_role.eq("development"),"exam_case_id"]); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"])
    if set(idx.exam_case_id)-dev or set(idx.exam_case_id)&locked or idx.exam_case_id.nunique()!=186: raise ValueError("development-only boundary failed")
    vectors=[]
    for n,row in enumerate(idx.itertuples(index=False),1):
        vectors.append(features(a.image_root/row.image_path))
        if n%200==0: print(f"{n}/{len(idx)}",flush=True)
    matrix=np.stack(vectors).astype(np.float32); a.output_dir.mkdir(parents=True,exist_ok=True); np.save(a.output_dir/"features.npy",matrix); idx.to_csv(a.output_dir/"index.csv",index=False)
    meta={"schema_version":"roi-quality-background-development/1.0","evaluation_role":"development","locked_cases_seen":0,"cases":186,"frames":len(idx),"shape":list(matrix.shape),"regions":["full","upper45","lower65","center","upper-minus-lower","center-minus-full"]}
    (a.output_dir/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(meta,ensure_ascii=False))
if __name__=="__main__": main()
