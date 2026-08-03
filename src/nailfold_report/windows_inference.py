from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CATEGORICAL = (
    "clarity", "capillary_count", "crossing_ratio", "malformation_ratio",
    "blood_color", "exudation", "hemorrhage", "subpapillary_venous_plexus",
    "papilla", "sweat_duct",
)
CONTINUOUS = (
    "afferent_diameter", "efferent_diameter", "output_input_ratio",
    "apex_diameter", "loop_length",
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.55


def _target_size(height: int, width: int, patch: int = 16, maximum: int = 256):
    minimum_scale, maximum_scale = 1e-6, 100.0
    while maximum_scale - minimum_scale >= 1e-5:
        scale = (minimum_scale + maximum_scale) / 2
        target_height = max(patch, math.ceil(height * scale / patch) * patch)
        target_width = max(patch, math.ceil(width * scale / patch) * patch)
        if target_height * target_width / patch**2 <= maximum:
            minimum_scale = scale
        else:
            maximum_scale = scale
    return (
        max(patch, math.ceil(height * minimum_scale / patch) * patch),
        max(patch, math.ceil(width * minimum_scale / patch) * patch),
    )


def preprocess_uvc_image(image: Image.Image, maximum: int = 256) -> np.ndarray:
    """Exact NumPy equivalent of the fixed SigLIP2 NaFlex image processor."""
    image = image.convert("RGB")
    target_height, target_width = _target_size(
        image.height, image.width, maximum=maximum
    )
    image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32) / 255.0
    values = (values - 0.5) / 0.5
    rows, columns = target_height // 16, target_width // 16
    patches = values.reshape(rows, 16, columns, 16, 3).transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(rows * columns, 16 * 16 * 3)
    if len(patches) > maximum:
        raise ValueError(f"preprocessing generated {len(patches)} patches")
    return np.pad(
        patches, ((0, maximum - len(patches)), (0, 0))
    ).astype(np.float32)[None]


def quality_score(path: Path) -> float:
    try:
        with Image.open(path) as source:
            gray = np.asarray(source.convert("L"), dtype=np.float32)
    except Exception:
        return -math.inf
    mean, standard_deviation = float(gray.mean()), float(gray.std())
    if standard_deviation < 3.0 or mean < 8.0 or mean > 247.0:
        return -math.inf
    interior = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * interior
    )
    sharpness = math.log1p(float(laplacian.var()))
    exposure = max(0.0, 1.0 - abs(mean - 125.0) / 125.0)
    return sharpness + exposure + math.log1p(standard_deviation) * 0.25


def select_views(paths: list[Path], count: int = 8) -> list[Path]:
    ranked = sorted(((quality_score(path), path) for path in paths), reverse=True)
    usable = [path for score, path in ranked if math.isfinite(score)]
    if not usable:
        raise ValueError("no usable CAP images after quality control")
    selected = usable[:count]
    return (selected * math.ceil(count / len(selected)))[:count]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _score_rule(rule: dict[str, Any], value: Any) -> float | None:
    if rule["type"] == "missing":
        return 0.0
    if value is None:
        return None
    if rule["type"] == "categorical_lookup":
        result = rule["mapping"].get(str(value))
        return float(result) if result is not None else None
    if rule["type"] == "numeric_tree":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        node = rule["tree"]
        while "value" not in node:
            node = node["left"] if numeric <= node["threshold"] else node["right"]
        return float(node["value"])
    raise ValueError(f"unknown score rule type: {rule['type']}")


def deterministic_scores(
    fields: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    scores: dict[str, float | None] = {}
    field_contributions = {
        field: _score_rule(rule, fields.get(field, {}).get("value"))
        for field, rule in rules["field_rules"].items()
    }
    for group, members in rules["groups"].items():
        contributions = [field_contributions[field] for field in members]
        scores[group] = (
            round(float(sum(contributions)), 1)
            if all(value is not None for value in contributions) else None
        )
    components = [
        scores["morphology_score"], scores["flow_score"], scores["periloop_score"]
    ]
    total = (
        round(float(sum(components)), 1)
        if all(value is not None for value in components) else None
    )
    scores["total_score"] = total
    assessment = None
    if total is not None:
        assessment_rule = rules["assessment_rule"]
        index = sum(total > threshold for threshold in assessment_rule["thresholds"])
        assessment = assessment_rule["labels"][index]
    scores["overall_assessment"] = assessment
    scores["field_contributions"] = {
        field: (round(float(value), 1) if value is not None else None)
        for field, value in field_contributions.items()
    }
    scores["status"] = "deterministic_from_predicted_fields"
    return scores


def deterministic_recommendations(fields: dict[str, Any], scores: dict[str, Any]) -> list[str]:
    """Generate non-diagnostic, rule-traceable health reminders.

    These are deliberately not generated by a language model and are not
    treatment, medication, or supplement recommendations. They are displayed
    as reminders for professional review and patient education only.
    """
    recommendations: list[str] = []
    total = scores.get("total_score")
    if isinstance(total, (int, float)) and total >= 10:
        recommendations.append("建议携带本报告及原始图像，交由专业医生结合病史复核。")
    if str(fields.get("blood_color", {}).get("value", "")) not in {"淡红色", "浅红色", "正常"}:
        recommendations.append("建议检查期间保持手部温暖，避免寒冷、剧烈运动及吸烟饮酒等短时因素影响观察。")
    if str(fields.get("flow_state", {}).get("value", "")) in {"全停", "粒流", "停滞"}:
        recommendations.append("如出现手指发冷、麻木、疼痛或颜色持续异常，请及时咨询医生；不要仅依据本报告自行用药。")
    if str(fields.get("exudation", {}).get("value", "")) in {"+", "++", "+++", "有"}:
        recommendations.append("渗出相关结果需要结合原图和临床情况复核，建议由专业人员确认后再决定是否进一步检查。")
    if not recommendations:
        recommendations.append("保持规律作息、适度活动和手部保暖；如有持续不适，请咨询专业医生。")
    recommendations.append("以上为基于本次图像指标的通用健康提示，不构成诊断、处方或保健品推荐。")
    return recommendations


class WindowsCasePredictor:
    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort

        self.model_dir = model_dir
        self.metadata = json.loads(
            (model_dir / "model_metadata.json").read_text(encoding="utf-8")
        )
        score_rules_path = model_dir / "score_rules.json"
        self.score_rules = (
            json.loads(score_rules_path.read_text(encoding="utf-8"))
            if score_rules_path.exists() else None
        )
        available = set(ort.get_available_providers())
        providers = [
            provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
        self.vision = ort.InferenceSession(
            str(model_dir / "siglip2_vision.onnx"), providers=providers
        )
        self.head = ort.InferenceSession(
            str(model_dir / "case_head.onnx"), providers=providers
        )

    def encode_image(self, image: Image.Image) -> np.ndarray:
        values = preprocess_uvc_image(image)
        return self.vision.run(None, {"pixel_values": values})[0][0]

    def predict(self, paths: list[Path]) -> dict[str, Any]:
        selected = select_views(paths, 8)
        unique_selected = len(set(selected))
        features = []
        for path in selected:
            with Image.open(path) as image:
                values = preprocess_uvc_image(image)
            features.append(self.vision.run(None, {"pixel_values": values})[0][0])
        case_features = np.stack(features, axis=0)[None].astype(np.float32)
        outputs = self.head.run(None, {"image_features": case_features})
        fields: dict[str, Any] = {}
        low_confidence_fields: list[str] = []
        confidence_threshold = float(
            self.metadata.get("confidence_policy", {}).get(
                "categorical_threshold", DEFAULT_CONFIDENCE_THRESHOLD
            )
        )
        for index, field in enumerate(CATEGORICAL):
            probability = _softmax(outputs[index])[0]
            class_index = int(probability.argmax())
            vocabulary = self.metadata["vocabularies"][field]
            confidence = float(probability[class_index])
            status = "predicted"
            if confidence < confidence_threshold:
                status = "predicted_low_confidence"
                low_confidence_fields.append(field)
            fields[field] = {
                "value": vocabulary[class_index],
                "confidence": confidence,
                "status": status,
            }
        for offset, field in enumerate(CONTINUOUS, start=len(CATEGORICAL)):
            normalized = float(outputs[offset][0])
            mean, standard_deviation = self.metadata["scales"][field]
            fields[field] = {
                "value": normalized * standard_deviation + mean,
                "confidence": None,
                "status": "predicted_uncalibrated",
            }
        input_value = fields["afferent_diameter"]["value"]
        output_value = fields["efferent_diameter"]["value"]
        fields["output_input_ratio"]["value"] = (
            output_value / input_value if input_value > 0 else None
        )
        fields["output_input_ratio"]["status"] = "derived"
        warnings: list[str] = []
        if unique_selected < 4:
            warnings.append(
                f"仅有 {unique_selected} 张不同的合格图像，病例多图证据不足"
            )
        if low_confidence_fields:
            warnings.append(
                "以下分类字段低于未校准置信度阈值："
                + ", ".join(low_confidence_fields)
            )
        result = {
            "schema_version": "nailfold-report/1.0",
            "model_version": self.metadata.get(
                "model_version", "siglip2-case-unknown"
            ),
            "fields": fields,
            "evidence_images": [str(path) for path in selected],
            "quality": {
                "usable_images": unique_selected,
                "needs_review": bool(warnings),
                "warnings": warnings,
            },
        }
        if self.score_rules is not None:
            result["scores"] = deterministic_scores(fields, self.score_rules)
            result["recommendations"] = deterministic_recommendations(
                fields, result["scores"]
            )
        return result
