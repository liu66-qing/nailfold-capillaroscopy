# 甲襞微循环图像检查报告 · expert-router-final

输入同一次检查的若干张静态甲襞图像，输出一页健康观察卡（健康观察卡版式）。20 项观察每项都有值，按主题分组；解读与建议按识别结果组合成主题卡（看到了 / 这意味着 / 可以这样做），不同结果给不同建议。页面不显示参考列、置信标记。不依赖大模型，不需要联网。

## 模型结构

- 字段分组：20 个二分类目标，每个目标各有一组头。
  - 测量类字段（管径、管袢长）由"偏低"和"偏高"两个单侧头组成三档。
  - 乳头由"波纹状"和"平坦"两个头组成三档。
- 四个特征来源，所有目标共享：
  - A0：DINOv2-B 图像级特征，5 种池化，PCA64，LogReg C=0.03，按病例取均值。
  - COL：颜色和清晰度统计。
  - SEG：S0 分割器几何特征。
  - DET：在 Capillary-Dataset 人工框上训练的 YOLO 检测器（bushy/crossing/hairpin/tortuous），输出病例级计数和比例。
- 每个目标先平均A0五个池化头的概率，再与COL、SEG、DET等权平均，再用 5 折 OOF 拟合的 Platt 校准。置信等级（JSON 的 `confidence`，页面不显示）按校准后概率划分。
- 训练使用全部 233 例有标签病例。
- 4 行固定输出（血管运动、白细胞、出血、汗腺导管）：直接输出注册表中的固定措辞，不代表本病例的图像发现。页面上这几行与其他行穿插排列。
- 输入/输出枝比例由两枝档位推出，不做除法。
- 输出不含尺寸、计数、频次、积分或五级分层。

## 验证（evaluation/validation.json）

口径：233 例检查病例级 5 折内部交叉验证，对照"每例都答最常见答案"，给出增益 95% bootstrap 置信区间。这批数据此前做过多轮探索和几十次配置比较，所以数字偏乐观。没有外部验证，也不是临床验证。

- 置信区间排除 0、优于常见答案：
  - 清晰度、渗出、血色、乳头下静脉丛、白微栓、畸形、管袢数。
  - 输出枝、袢顶、管袢长三档。
- 没有优于常见答案：
  - 输入枝三档、乳头三档、交叉。
  - 流态、红细胞聚集：模型塌缩为常见答案。
  - 这些行照常输出，但置信等级一般偏低。建议区只由置信等级≥中等的偏离行触发，流态和红细胞聚集不触发建议。

## 使用

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

scikit-learn 必须同版本，因为头是 pickle 的 sklearn 管线。

```python
from nailfold_report.final_inference import FinalReportPredictor
from pathlib import Path
from nailfold_report.final_render import render_html, thumbnail_bytes
ROOT = Path.cwd()
paths = sorted(Path("某次检查").glob("*.jpg"))
p = FinalReportPredictor(ROOT, device="cpu")   # 加载时校验 bundle/编码器/分割器/检测器四个 sha256
r = p.predict(paths, exam_id="E001", device_id="unknown")
render_html(r, "report.html", patient={"name": "…", "sex": "女", "age": "52"}, images=thumbnail_bytes(paths))
```

在 CPU 上每张图约需 5~10 秒。HTTP 接口没有鉴权，只能绑定在 localhost，或放在平台网关后面：

```bash
uvicorn nailfold_report.final_api:app --host 127.0.0.1 --port 8000
```

- `POST /v1/reports`：multipart 上传 exam_id、device_id 和 images。加 `?format=html` 返回打印页。
- `GET /examples/1`：仅在本地生成对应HTML后可用；仓库JSON示例在 `examples/reports/`。
- JSON 结构见 `schemas/report.schema.json`。

## 文件

- `artifacts/models/expert_router_v1/`：bundle.joblib、seg_s0.pt、det_capillary.pt、metadata.json。二进制从 GitHub Release `expert-router-final` 下载，放置方法见根目录 README §1.0
- `evaluation/validation.json`：逐行准确率、常见答案基线、增益置信区间
- `examples/reports/`：8 例开发集病例的 JSON（HTML 本地生成，不入库）。生成时校验了实时推理与训练特征表的一致性，|Δq| ≤ 0.0028
- `artifacts/models/expert_router_v1/manifest.json`：发布文件的 sha256

## 局限

1. 训练数据只有一家来源、一种设备，换设备后表现未知。
2. 流态和白微栓在原报告中来自动态观察。本系统只从静态图像给出关联预测。
3. 固定行不随图像变化，不能用于排除出血等异常。
4. 置信等级是内部交叉验证上校准的概率，换人群后需要重新校准。

```bash
python -m pytest tests/test_final_release.py
```

## 接手前核验

从仓库根目录运行 `python scripts/verify_handoff.py --check-models`。测试和训练复现前置条件见[首页](../README.md)。`expert_features.py`会调用`scripts/`中的特征函数，因此需要完整克隆并使用editable安装；只复制`src/`或只安装wheel不构成完整推理包。当前接入使用 `FinalReportPredictor`，模型文件按首页下载表放置。
