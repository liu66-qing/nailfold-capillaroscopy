"""Leakage-safe Qwen3-VL static-field baselines for nailfold development data.

This program never opens report images.  Parsed report labels are evaluation
targets; the optional few-shot condition supplies only normalized class JSON
from training-fold demonstration cases, paired with their capillaroscopy
images.  Locked cases are excluded before any schema, prompt, or metric work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from transformers import AutoModelForImageTextToText, AutoProcessor


FIELDS = (
    "clarity",
    "capillary_count",
    "crossing_ratio",
    "malformation_ratio",
    "blood_color",
    "exudation",
    "hemorrhage",
    "subpapillary_venous_plexus",
    "papilla",
    "sweat_duct",
)


ALIASES: dict[str, dict[str, str]] = {
    "capillary_count": {">7": ">=7", "≥7": ">=7", "7以上": ">=7"},
    "crossing_ratio": {"<30%": "<=30%", "≤30%": "<=30%", "10--30%": "<=30%", "30%-60%": "30--60%", "60%-80%": "60--80%"},
    "malformation_ratio": {"<10%": "<=10%", "≤10%": "<=10%", "10%-30%": "10--30%", "30%-60%": "30--60%"},
    "blood_color": {"淡红色": "淡红", "浅红色": "浅红", "暗红色": "暗红"},
    "exudation": {"未见": "无", "不见": "无"},
    "hemorrhage": {"未见": "无", "不见": "无"},
    "subpapillary_venous_plexus": {"未见": "不见", "无": "不见"},
}

# These are the report-template medical categories.  Everything else is either
# an OCR/layout fragment or outside the fixed report vocabulary and is missing,
# never a new training/scoring class.
VALID_VALUES: dict[str, set[str]] = {
    "clarity": {"清晰", "模糊", "不清"},
    "capillary_count": {"1--2", "3--4", "5--6", ">=7"},
    "crossing_ratio": {"<=30%", "30--60%", "60--80%", ">80%"},
    "malformation_ratio": {"<=10%", "10--30%", "30--60%", ">60%"},
    "blood_color": {"淡红", "浅红", "暗红"},
    "exudation": {"无", "+", "++", "+++"},
    "hemorrhage": {"无", "1--2", "3--4", ">=5"},
    "subpapillary_venous_plexus": {"不见", "可见1排", "可见2排", ">2排,扩张"},
    "papilla": {"平坦", "浅波纹状", "波纹状"},
    "sweat_duct": {"0--2"},
}

CANONICAL_MAPPING = {
    "policy": "v2-medical-enum-only",
    "capillary_count": {"keep": ["3--4", "5--6", ">=7"], "missing": ["<1", "条/mm"]},
    "crossing_ratio": {"aliases": {"[<30%]": "<=30%", "10--30%": "<=30%"}},
    "malformation_ratio": {"aliases": {"[<10%]": "<=10%"}},
    "blood_color": {"aliases": {"[淡红色]": "淡红"}, "missing": ["+", "++", "+++", "暗紫"]},
    "exudation": {"aliases": {"[无]": "无"}},
    "hemorrhage": {"keep": ["无", "1--2"], "missing": ["管袢/一指甲襞"]},
    "subpapillary_venous_plexus": {"aliases": {"[不见]": "不见"}},
    "papilla": {"aliases": {"[波纹状]": "波纹状"}},
    "sweat_duct": {"aliases": {"0--2个/-指甲襞": "0--2"}, "missing": ["个/-指甲襞", "3--4"]},
}


def canonical(field: str, value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = text.replace("≥", ">=").replace("≤", "<=").replace("至", "-")
    text = re.sub(r"\s+", "", text)
    if len(text) >= 2 and text[0] in "[【(" and text[-1] in "])】":
        text = text[1:-1]
    if field in {"capillary_count", "crossing_ratio", "malformation_ratio", "sweat_duct"}:
        text = re.sub(r"(?<=\d)-(?=\d)", "--", text)
        text = re.sub(r"-{3,}", "--", text)
    text = ALIASES.get(field, {}).get(text, text)
    if field == "capillary_count" and (text == "<1" or "条/mm" in text):
        return None
    if field == "hemorrhage" and "管袢/" in text:
        return None
    if field == "sweat_duct" and ("个/" in text or "指甲襞" in text):
        match = re.match(r"^(0--2|3--4|>=5)", text)
        text = match.group(1) if match else ""
    if field == "blood_color" and text in {"+", "++", "+++"}:
        return None
    return text if text in VALID_VALUES[field] else None


def parse_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == set(FIELDS):
            return value
    return None


def image_paths(image_root: Path, case_id: str) -> list[Path]:
    paths = sorted((image_root / case_id).glob("CAPorg*.jpg"))
    if not paths:
        return []
    positions = [round(index * (len(paths) - 1) / 3) for index in range(4)]
    return [paths[index] for index in positions]


def labels_from_row(row: pd.Series) -> dict[str, str | None]:
    return {field: canonical(field, row[field]) for field in FIELDS}


def schema_from_development(df: pd.DataFrame) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    for field in FIELDS:
        values = sorted({canonical(field, value) for value in df[field]} - {None})
        if not values:
            raise ValueError(f"no usable development labels for {field}")
        schema[field] = values
    return schema


def base_prompt(schema: dict[str, list[str]]) -> str:
    choices = "\n".join(f"{field}: " + " | ".join(schema[field]) for field in FIELDS)
    keys = ", ".join(FIELDS)
    return (
        "你是甲襞微循环静态字段分类器。只依据同一病例给出的4张甲襞图像。"
        "不要诊断、解释、测量或使用图中不可见信息。"
        "只输出一个合法JSON对象，必须恰好包含以下键：" + keys + "。\n"
        "每个字段只能使用列出的枚举值；无法可靠判断时填null。禁止Markdown、额外键或文字。\n"
        + choices
    )


def stable_order(case_id: str, seed: int = 20260803) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{case_id}".encode()).digest()[:8], "big")


def choose_demos(train: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select a fixed training-only illustrative set without inspecting validation labels."""
    # The demonstrations must have a complete medical label JSON.  This makes
    # prompt context auditable and prevents residual OCR text from being shown.
    clean = train[train.apply(lambda row: all(labels_from_row(row).values()), axis=1)].copy()
    if len(clean) < count:
        raise RuntimeError("not enough completely labelled training cases for few-shot context")
    if len(clean) <= count:
        return clean.sort_values("exam_case_id")
    rarity: dict[str, Counter[str]] = {
        field: Counter(canonical(field, value) for value in clean[field]) for field in FIELDS
    }
    scored: list[tuple[float, int, int]] = []
    for index, row in clean.iterrows():
        # Prefer examples populated by common, well represented labels.  This
        # is chosen wholly within the training fold and never adapts to a query.
        score = sum(math.log1p(rarity[field][canonical(field, row[field])]) for field in FIELDS)
        scored.append((-score, stable_order(str(row.exam_case_id)), index))
    chosen = [item[2] for item in sorted(scored)[:count]]
    return train.loc[chosen].sort_values("exam_case_id")


def metric_rows(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in list(range(5)) + ["pooled"]:
        source = records if fold == "pooled" else [record for record in records if record["fold"] == fold]
        for field in FIELDS:
            eligible = [record for record in source if record["target"][field] is not None]
            if not eligible:
                continue
            y_true = [record["target"][field] for record in eligible]
            y_pred = [record["prediction"].get(field) for record in eligible]
            # Invalid JSON and unsupported values are errors, represented by a
            # deliberately non-target sentinel so they cannot gain credit.
            y_pred = [value if value is not None else "__INVALID__" for value in y_pred]
            labels = sorted(set(y_true))
            rows.append({
                "method": method,
                "fold": fold,
                "field": field,
                "n": len(y_true),
                "class_count": len(labels),
                "min_class_support": min(Counter(y_true).values()),
                "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "exact_accuracy": float(np.mean(np.array(y_true, dtype=object) == np.array(y_pred, dtype=object))),
            })
    return rows


def load_images(paths: list[Path]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def infer(
    processor: Any,
    model: Any,
    query_paths: list[Path],
    prompt: str,
    demos: list[tuple[list[Path], dict[str, str | None]]],
) -> tuple[str, dict[str, Any] | None, float]:
    messages: list[dict[str, Any]] = []
    for paths, labels in demos:
        content = [{"type": "image", "image": image} for image in load_images(paths)]
        content.append({"type": "text", "text": "下面是一个已标注示例。"})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": json.dumps(labels, ensure_ascii=False)})
    content = [{"type": "image", "image": image} for image in load_images(query_paths)]
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=320, do_sample=False)
    elapsed = time.perf_counter() - started
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return raw, parse_json(raw), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--few-shot-demos", type=int, default=2)
    parser.add_argument("--stability-cases", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.folds)
    required = {"evaluation_role", "development_fold", "duplicate_group", "exam_case_id"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"fold file missing {sorted(missing)}")
    table = table[(table["evaluation_role"] == "development") & table["development_fold"].notna()].copy()
    table["development_fold"] = table["development_fold"].astype(int)
    table = table[table["report_label_available"].astype(bool)].copy()
    table = table[table["exam_case_id"].map(lambda value: bool(image_paths(args.image_root, str(value))))].copy()
    if table.empty:
        raise RuntimeError("no development cases with labels and images")
    if table.groupby("duplicate_group")["development_fold"].nunique().max() != 1:
        raise RuntimeError("duplicate group leaks across development folds")
    schema = schema_from_development(table)
    (args.output / "canonical_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "canonical_mapping.json").write_text(json.dumps(CANONICAL_MAPPING, ensure_ascii=False, indent=2), encoding="utf-8")
    locked = pd.read_csv(args.folds)
    locked = locked[locked["evaluation_role"] == "locked_test"]
    development_groups = set(table["duplicate_group"].astype(str))
    locked_groups = set(locked["duplicate_group"].astype(str))
    leakage_audit = {
        "development_cases_with_images": len(table),
        "locked_test_cases_excluded": len(locked),
        "duplicate_groups_in_development": len(development_groups),
        "duplicate_groups_in_locked_test": len(locked_groups),
        "development_locked_group_overlap": sorted(development_groups & locked_groups),
        "max_development_folds_per_duplicate_group": int(table.groupby("duplicate_group")["development_fold"].nunique().max()),
    }
    if leakage_audit["development_locked_group_overlap"]:
        raise RuntimeError("duplicate group overlaps development and locked test")
    (args.output / "leakage_audit.json").write_text(json.dumps(leakage_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    class_rows = []
    for field in FIELDS:
        counts = Counter(canonical(field, value) for value in table[field])
        class_rows.extend({"field": field, "label": label, "count": count} for label, count in sorted(counts.items(), key=lambda item: str(item[0])) if label is not None)
    pd.DataFrame(class_rows).to_csv(args.output / "class_counts.csv", index=False)

    prompt = base_prompt(schema)
    demo_by_fold: dict[int, list[tuple[list[Path], dict[str, str | None]]]] = {}
    demo_manifest: list[dict[str, Any]] = []
    for fold in range(5):
        train = table[table["development_fold"] != fold]
        examples = choose_demos(train, args.few_shot_demos)
        entries: list[tuple[list[Path], dict[str, str | None]]] = []
        for _, row in examples.iterrows():
            paths = image_paths(args.image_root, str(row.exam_case_id))
            entries.append((paths, labels_from_row(row)))
            demo_manifest.append({"fold": fold, "exam_case_id": row.exam_case_id, "duplicate_group": row.duplicate_group, "target": labels_from_row(row)})
        demo_by_fold[fold] = entries
    (args.output / "few_shot_demo_manifest.json").write_text(json.dumps(demo_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    load_seconds = time.perf_counter() - load_started

    all_metrics: list[dict[str, Any]] = []
    run_meta: dict[str, Any] = {
        "model": str(args.model), "model_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "folds": str(args.folds), "image_root": str(args.image_root), "development_cases": len(table),
        "locked_cases_excluded": len(locked),
        "load_seconds": load_seconds, "few_shot_demos": args.few_shot_demos,
        "method_note": "No report image or original report text is input. Few-shot context contains only four nailfold images plus normalized fixed-JSON target labels from training-fold cases.",
    }
    for method in ("zero_shot", "few_shot"):
        records: list[dict[str, Any]] = []
        output_path = args.output / f"{method}_predictions.jsonl"
        with output_path.open("w", encoding="utf-8") as stream:
            for sequence, (_, row) in enumerate(table.sort_values(["development_fold", "exam_case_id"]).iterrows(), start=1):
                fold = int(row.development_fold)
                demos = [] if method == "zero_shot" else demo_by_fold[fold]
                paths = image_paths(args.image_root, str(row.exam_case_id))
                raw, parsed, seconds = infer(processor, model, paths, prompt, demos)
                raw_prediction = {field: canonical(field, parsed.get(field)) if parsed else None for field in FIELDS}
                unsupported = [field for field in FIELDS if raw_prediction[field] is not None and raw_prediction[field] not in schema[field]]
                prediction = {field: raw_prediction[field] if field not in unsupported else None for field in FIELDS}
                record = {
                    "sequence": sequence, "method": method, "exam_case_id": row.exam_case_id,
                    "duplicate_group": row.duplicate_group, "fold": fold, "image_paths": [str(path) for path in paths],
                    "target": labels_from_row(row), "prediction": prediction, "json_valid": parsed is not None,
                    "unsupported_output_fields": unsupported,
                    "seconds": seconds, "raw": raw,
                }
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"[{method} {sequence}/{len(table)}] fold={fold} case={row.exam_case_id} valid={record['json_valid']} {seconds:.2f}s", flush=True)
        pd.DataFrame(metric_rows(records, method)).to_csv(args.output / f"{method}_metrics.csv", index=False)
        all_metrics.extend(metric_rows(records, method))
        stable = sorted(records, key=lambda record: stable_order(record["exam_case_id"]))[: args.stability_cases]
        stability: list[dict[str, Any]] = []
        for record in stable:
            paths = [Path(path) for path in record["image_paths"]]
            demos = [] if method == "zero_shot" else demo_by_fold[record["fold"]]
            raw, parsed, seconds = infer(processor, model, paths, prompt, demos)
            raw_repeated = {field: canonical(field, parsed.get(field)) if parsed else None for field in FIELDS}
            repeated = {field: raw_repeated[field] if raw_repeated[field] in schema[field] else None for field in FIELDS}
            stability.append({"exam_case_id": record["exam_case_id"], "fold": record["fold"], "initial": record["prediction"], "repeat": repeated, "exact_match": repeated == record["prediction"], "json_valid": parsed is not None, "seconds": seconds, "raw": raw})
        (args.output / f"{method}_stability.json").write_text(json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8")
        run_meta[f"{method}_json_valid_rate"] = float(np.mean([record["json_valid"] for record in records]))
        run_meta[f"{method}_mean_latency_seconds"] = float(np.mean([record["seconds"] for record in records]))
        run_meta[f"{method}_stability_exact_rate"] = float(np.mean([row["exact_match"] for row in stability]))
    pd.DataFrame(all_metrics).to_csv(args.output / "all_metrics.csv", index=False)
    run_meta["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    (args.output / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
