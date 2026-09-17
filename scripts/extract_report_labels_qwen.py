from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nailfold_report.labels.parsing import extract_json_object, validate_ocr_payload  # noqa: E402
from nailfold_report.labels.schema import REPORT_OCR_PROMPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def report_paths(data_root: Path) -> list[Path]:
    paths: list[Path] = []
    for archive in sorted(data_root.glob("recovered_archive*")):
        if not archive.is_dir():
            continue
        for case_dir in sorted(
            (item for item in archive.iterdir() if item.is_dir() and item.name.isdigit()),
            key=lambda item: int(item.name),
        ):
            paths.extend(sorted(case_dir.glob("rep*.jpg.jpg")))
    return paths


def report_groups(data_root: Path) -> list[list[Path]]:
    by_hash: dict[str, list[Path]] = {}
    for path in report_paths(data_root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_hash.setdefault(digest, []).append(path)
    return list(by_hash.values())


def load_completed(output: Path) -> set[str]:
    if not output.exists():
        return set()
    completed: set[str] = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok":
            completed.add(str(row.get("report_path")))
    return completed


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed = load_completed(args.output)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    pending = [
        group
        for group in report_groups(args.data_root)
        if any(
            path.relative_to(args.data_root).as_posix() not in completed
            for path in group
        )
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    with args.output.open("a", encoding="utf-8") as stream:
        for index, group in enumerate(pending, start=1):
            path = group[0]
            decoded: str | None = None
            try:
                # The fixed 680x750 template keeps every measurement and score
                # above y=465; content below is clinical imagery/recommendation
                # text and must not be used to infer labels. Upscaling makes the
                # small legacy bitmap font substantially easier to transcribe.
                with Image.open(path) as source:
                    table = source.convert("RGB").crop((0, 0, source.width, 465))
                    report_image = table.resize(
                        (table.width * 2, table.height * 2), Image.Resampling.LANCZOS
                    )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": report_image},
                            {"type": "text", "text": REPORT_OCR_PROMPT},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                prompt_length = inputs["input_ids"].shape[1]
                decoded = processor.batch_decode(
                    generated[:, prompt_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                fields = validate_ocr_payload(extract_json_object(decoded))
                result = {"status": "ok", "fields": fields, "raw_output": decoded}
            except Exception as error:
                result = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                if decoded is not None:
                    result["raw_output"] = decoded
            for alias in group:
                relative = alias.relative_to(args.data_root).as_posix()
                if relative in completed:
                    continue
                row = {
                    **result,
                    "exam_case_id": "/".join(relative.split("/")[:2]),
                    "report_path": relative,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"[{index}/{len(pending)}] {path.name} x{len(group)}: {result['status']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
