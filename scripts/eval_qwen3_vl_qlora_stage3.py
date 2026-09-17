"""Evaluate one nailfold Qwen3-VL LoRA adapter on its held-out development fold."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import balanced_accuracy_score, f1_score
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


FIELDS = (
    "clarity", "capillary_count", "crossing_ratio", "malformation_ratio",
    "blood_color", "exudation", "hemorrhage", "subpapillary_venous_plexus", "papilla",
)
SCHEMA = {
    "clarity": ["不清", "模糊", "清晰"],
    "capillary_count": ["3--4", "5--6", ">=7"],
    "crossing_ratio": ["<=30%", "30--60%", "60--80%", ">80%"],
    "malformation_ratio": ["<=10%", "10--30%", "30--60%", ">60%"],
    "blood_color": ["淡红", "浅红", "暗红"],
    "exudation": ["无", "+", "++", "+++"],
    "hemorrhage": ["无", "1--2"],
    "subpapillary_venous_plexus": ["不见", "可见1排", "可见2排", ">2排,扩张"],
    "papilla": ["平坦", "浅波纹状", "波纹状"],
}
ALIASES = {
    "crossing_ratio": {"[<30%]": "<=30%", "10--30%": "<=30%"},
    "malformation_ratio": {"[<10%]": "<=10%"},
    "blood_color": {"[淡红色]": "淡红"},
    "exudation": {"[无]": "无"},
    "subpapillary_venous_plexus": {"[不见]": "不见"},
    "papilla": {"[波纹状]": "波纹状"},
}


def canonical(field: str, value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = ALIASES.get(field, {}).get(str(value).strip(), str(value).strip())
    return text if text in SCHEMA[field] else None


def target(row: pd.Series, fields: tuple[str, ...]) -> dict[str, str | None]:
    return {field: canonical(field, row[field]) for field in fields}


def prompt(fields: tuple[str, ...]) -> str:
    choices = "\n".join(f"{field}: {' | '.join(SCHEMA[field])}" for field in fields)
    return (
        "你是甲襞微循环静态字段分类器。只依据同一病例的4张甲襞图像。"
        "不要诊断、解释或猜测未显示的信息。只输出一个JSON对象。"
        "无法判断的字段输出null。禁止Markdown和额外文字。\n" + choices
    )


def images(root: Path, case_id: str) -> list[Image.Image]:
    paths = sorted((root / case_id).glob("CAPorg*.jpg"))
    indexes = [round(i * (len(paths) - 1) / 3) for i in range(4)]
    return [Image.open(paths[i]).convert("RGB") for i in indexes]


def parse(text: str, fields: tuple[str, ...]) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return {field: canonical(field, value.get(field)) for field in fields}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fields", default=",".join(FIELDS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fields = tuple(part.strip() for part in args.fields.split(",") if part.strip())
    if not fields or any(field not in FIELDS for field in fields):
        raise ValueError(f"invalid fields: {fields}")

    table = pd.read_csv(args.manifest)
    test = table[(table.evaluation_role == "development") & (table.development_fold == args.fold)].copy()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, trust_remote_code=True, quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    records = []
    started = time.time()
    for index, (_, row) in enumerate(test.iterrows(), start=1):
        content = [{"type": "image", "image": image} for image in images(args.image_root, str(row.exam_case_id))]
        content.append({"type": "text", "text": prompt(fields)})
        batch = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        batch = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in batch.items()}
        tick = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=300, do_sample=False)
        generated = output[:, batch["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        prediction = parse(raw, fields)
        records.append({"exam_case_id": row.exam_case_id, "fold": args.fold, "target": target(row, fields), "prediction": prediction, "raw": raw, "seconds": time.perf_counter() - tick})
        print(f"{index}/{len(test)} {row.exam_case_id} valid={prediction is not None}", flush=True)

    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    rows = []
    for field in fields:
        eligible = [record for record in records if record["target"][field] is not None]
        y_true = [record["target"][field] for record in eligible]
        y_pred = [(record["prediction"] or {}).get(field) or "__INVALID__" for record in eligible]
        labels = sorted(set(y_true))
        rows.append({
            "fold": args.fold, "field": field, "n": len(y_true), "class_count": len(labels),
            "min_class_support": min(Counter(y_true).values()),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "exact_accuracy": float(np.mean(np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object))),
        })
    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    distributions = []
    for field in fields:
        counts = Counter((record["prediction"] or {}).get(field) for record in records)
        for value, count in counts.items():
            distributions.append({"field": field, "value": value, "count": count})
    pd.DataFrame(distributions).to_csv(args.output / "prediction_distributions.csv", index=False)
    metadata = {"fold": args.fold, "cases": len(records), "json_valid_rate": sum(record["prediction"] is not None for record in records) / len(records), "mean_latency_seconds": float(np.mean([record["seconds"] for record in records])), "elapsed_seconds": time.time() - started, "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
