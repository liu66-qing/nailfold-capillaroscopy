"""Leakage-safe Qwen3-VL QLoRA for nailfold static fields.

The script trains one development fold at a time. Report-derived labels are
targets only; report images and report text are never opened. Loss is computed
only on the assistant JSON response, not on the prompt or image placeholders.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


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
    text = str(value).strip()
    text = ALIASES.get(field, {}).get(text, text)
    return text if text in SCHEMA[field] else None


def case_images(root: Path, case_id: str) -> list[Image.Image]:
    paths = sorted((root / case_id).glob("CAPorg*.jpg"))
    if not paths:
        raise FileNotFoundError(case_id)
    indexes = [round(i * (len(paths) - 1) / 3) for i in range(4)]
    return [Image.open(paths[i]).convert("RGB") for i in indexes]


def prompt(fields: tuple[str, ...]) -> str:
    choices = "\n".join(f"{field}: {' | '.join(SCHEMA[field])}" for field in fields)
    return (
        "你是甲襞微循环静态字段分类器。只依据同一病例的4张甲襞图像。"
        "不要诊断、解释或猜测未显示的信息。只输出一个JSON对象。"
        "无法判断的字段输出null。禁止Markdown和额外文字。\n" + choices
    )


def target(row: pd.Series, fields: tuple[str, ...]) -> dict[str, str | None]:
    return {field: canonical(field, row[field]) for field in fields}


def encode(processor: Any, images: list[Image.Image], answer: dict[str, Any], fields: tuple[str, ...]) -> dict[str, torch.Tensor]:
    user_content = [{"type": "image", "image": image} for image in images]
    user_content.append({"type": "text", "text": prompt(fields)})
    user = [{"role": "user", "content": user_content}]
    full = user + [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]
    prompt_batch = processor.apply_chat_template(
        user, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    batch = processor.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    labels = batch["input_ids"].clone()
    prompt_tokens = prompt_batch["input_ids"].shape[1]
    labels[:, :prompt_tokens] = -100
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    if int((labels != -100).sum()) < 10:
        raise RuntimeError("assistant target masking failed")
    return batch


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-cases", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--fields", default=",".join(FIELDS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    fields = tuple(part.strip() for part in args.fields.split(",") if part.strip())
    if not fields or any(field not in FIELDS for field in fields):
        raise ValueError(f"invalid fields: {fields}")

    table = pd.read_csv(args.manifest)
    development = table[(table.evaluation_role == "development") & table.development_fold.notna()].copy()
    locked = table[table.evaluation_role == "locked_test"]
    if set(development.duplicate_group) & set(locked.duplicate_group):
        raise RuntimeError("locked/development duplicate-group overlap")
    train = development[development.development_fold != args.fold].copy()
    train["usable"] = train.apply(lambda row: sum(v is not None for v in target(row, fields)), axis=1)
    train = train[train.usable >= max(2, len(fields) - 2)].sort_values("exam_case_id")
    if args.max_train_cases > 0:
        train = train.head(args.max_train_cases)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    for name, parameter in model.named_parameters():
        if "visual" in name or "vision" in name:
            parameter.requires_grad = False
    model.print_trainable_parameters()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr)
    device = next(model.parameters()).device

    losses: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    started = time.time()
    model.train()
    for epoch in range(args.epochs):
        for index, (_, row) in enumerate(train.iterrows()):
            images = case_images(args.image_root, str(row.exam_case_id))
            batch = move(encode(processor, images, target(row, fields), fields), device)
            output = model(**batch)
            loss = output.loss / args.grad_accum
            loss.backward()
            should_step = (index + 1) % args.grad_accum == 0 or index + 1 == len(train)
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            value = float(output.loss.detach().cpu())
            losses.append({"epoch": epoch, "case": row.exam_case_id, "loss": value, "optimizer_step": step})
            print(f"epoch={epoch} case={row.exam_case_id} loss={value:.6f} step={step}", flush=True)

    model.save_pretrained(args.output / "adapter")
    processor.save_pretrained(args.output / "processor")
    metadata = {
        "fold": args.fold,
        "train_cases": len(train),
        "locked_cases_excluded": len(locked),
        "fields": list(fields),
        "sweat_duct_excluded_reason": "one canonical class in development data",
        "vision_encoder_frozen": True,
        "loss_scope": "assistant_json_tokens_only",
        "elapsed_seconds": time.time() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "first_loss": losses[0]["loss"],
        "last_loss": losses[-1]["loss"],
        "optimizer_steps": step,
    }
    (args.output / "losses.json").write_text(json.dumps(losses, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
