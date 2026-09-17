# 甲襞微循环 DINOv2 LoRA 分类模型：继续提分

你是本项目的算法工程师。本文件为你提供完整的项目背景、当前进度、服务器环境和你的任务。

---

## 一、项目背景

我们在做甲襞微循环（nailfold capillaroscopy）自动分析系统。输入是每个病例的多帧甲襞图像，输出是 17 个临床字段的预测。当前聚焦 5 个视觉分类字段的性能提升。

### 数据

- 233 个病例：186 development + 47 locked（严格隔离，禁止用于训练/调参/选模型）
- 每个病例有多帧图像（平均 ~9 帧），共 1708 帧
- 5 个分类字段，全部做二分类：
  - clarity：清晰(0) vs 不清+模糊(1)，n=185
  - blood_color：暗红+暗紫(0) vs 浅红+淡红(1)，n=174
  - exudation：无(0) vs +(1)++(1)+++(1)，n=182
  - SVP（甲襞下静脉丛）：不见(0) vs 可见1排+可见2排+>2排扩张(1)，n=181
  - papilla：平坦(0) vs 浅波纹状+波纹状(1)，n=181
- 标签来自病例级 OCR，不是帧级标注
- 5-fold case-level split，折大小：38/32/39/43/34

### 已知标签噪声

- blood_color fold 1 混入 1 个 `+++`（exudation 标签误入）
- blood_color fold 2 混入 1 个 `+`（同上）
- 各 fold 存在 `[淡红色]`、`[不见]`、`[波纹状]`、`[无]` 等带方括号的 OCR 残留
- 这些脏标签目前没有清洗，训练时被映射为 -1（忽略），**浪费了样本**

### 评估协议

- 主指标：Balanced Accuracy（BA），对各类等权
- 训练使用嵌套 5-fold OOF：每折用 3 折训练、1 折验证选 epoch、1 折测试
- locked 47 例仅在最终冻结模型后做一次评测，严禁用于任何调参

---

## 二、方法演进历史

### v1：结构化特征 + ExtraTrees（基线）

- DINOv2 冻结特征均值 + HuluMed 冻结特征均值 + 105 个分割统计特征 + 帧数
- ExtraTrees(300, leaf=2, balanced) 分类器
- 这是表格学习方法，已证明到顶（v4 在此框架内尝试了 YOLO 检测特征、UNet 分割特征、XGBoost、不同 pooling，均未稳定超过 v1）

### v3：YOLO 检测特征 + ExtraTrees

- 训练了 YOLO11m-seg 检测器（vessel/malformed_vessel/cross_vessel），321 张检测标注图
- 提取检测特征替代部分分割统计
- 在 blood_color 上是当前最优（0.710），其他字段被 v6 LoRA 超过

### v5：DINOv2 LoRA 首次尝试

- 脚本：`scripts/finetune_dinov2_lora_binary_cv.py`
- 配置：rank=4, lr=2e-4, epochs=5, 二值标签, 帧级 logit 均值聚合, 无额外增强
- 关键发现：SVP 从 v1 的 0.699 提升到 0.751，证明端到端 LoRA 路线可行
- 问题：只跑了 1 个配置，epoch 太少（5），其他字段反而退化

### v6：LoRA 超参搜索（当前最优来源）

v6 通过复制 v5 脚本并修改 rank/lr 硬编码值，跑了 4 个配置。**这 4 个配置产出了当前 4 个字段的最优成绩。**

#### v6 实际做法

v6 没有命令行参数控制 rank 和 lr，而是为每个配置复制了一份脚本：

| 脚本文件 | rank | lr | epochs |
|----------|------|----|--------|
| `finetune_dinov2_lora_binary_cv.py` | 4 (默认) | 2e-4 (默认) | 30 |
| `finetune_dinov2_lora_rank8_cv.py` | 8 | 2e-4 | 30 |
| `finetune_dinov2_lora_rank8_lr1e4_cv.py` | 8 | 1e-4 | 30 |
| `finetune_dinov2_lora_rank16_cv.py` | 16 | 2e-4 | 30 |
| `finetune_dinov2_lora_rank4_lr1e4_cv.py` | 4 | 1e-4 | 30 |

脚本之间的唯一差异：
- rank 变化：在 `LoRALinear(block.attn.qkv, rank=X)` 处改数字
- lr 变化：在 `opt=torch.optim.AdamW(..., lr=X)` 处改数字
- rank8_lr1e4 脚本还把 LoRALinear 的默认 rank 参数从 4 改为了 8

所有脚本共享完全相同的：模型结构（LoRALinear + 5 字段共享 backbone + 独立二分类 head）、数据加载、训练循环、OOF 评估逻辑。

---

## 三、当前成绩（从服务器实际文件验证）

### 各版本 Development OOF BA 完整对照表

| 字段 | v1 ET | v3 ET | v5 r4lr2e4 | v6 r4lr2e4 30ep | v6 r8lr2e4 | v6 r8lr1e4 | v6 r16lr2e4 | v6 r4lr1e4 | **字段最优** |
|------|-------|-------|------------|-----------------|------------|------------|-------------|------------|-------------|
| clarity | 0.695 | 0.716 | 0.687 | 0.708 | 0.722 | **0.733** | 0.668 | 0.706 | **0.733** (r8lr1e4) |
| blood_color | 0.662 | **0.710** | 0.646 | 0.681 | 0.641 | 0.652 | 0.626 | 0.647 | **0.710** (v3 ET) |
| exudation | 0.684 | 0.711 | 0.727 | 0.715 | 0.715 | 0.707 | 0.715 | **0.772** | **0.772** (r4lr1e4) |
| SVP | 0.699 | 0.695 | 0.751 | 0.762 | **0.808** | 0.771 | 0.787 | 0.761 | **0.808** (r8lr2e4) |
| papilla | 0.564 | 0.577 | 0.647 | 0.632 | 0.602 | 0.606 | **0.661** | 0.602 | **0.661** (r16lr2e4) |

### 字段级最优路由

| 字段 | Dev BA | 来源 | Locked BA | Dev/Locked 落差 |
|------|--------|------|-----------|----------------|
| clarity | 0.733 | v6 rank8 lr1e-4 | 0.713* | 2pp |
| blood_color | 0.710 | v3 ExtraTrees | 0.746* | -3.6pp (locked更好) |
| exudation | 0.772 | v6 rank4 lr1e-4 | 0.618* | **15.4pp 过拟合** |
| SVP | 0.808 | v6 rank8 lr2e-4 | 0.873* | -6.5pp (locked更好) |
| papilla | 0.661 | v6 rank16 lr2e-4 | 0.570* | **9.1pp 过拟合** |

*locked 评测只在 rank4_lr1e4 一个配置上做过，不是各字段最优配置的 locked 结果

### 关键观察

1. **不同字段的最优 rank/lr 完全不同**：clarity 要 rank8+低lr，exudation 要 rank4+低lr，SVP 要 rank8+高lr，papilla 要 rank16
2. **blood_color 的 LoRA 全面低于 ExtraTrees**（最高 0.652 vs v3 的 0.710），说明颜色特征可能更适合统计方法
3. **只跑了 4 个 rank/lr 配置**，网格远未完整——rank16+lr1e-4、rank4+lr5e-5、rank8+lr5e-5 等都没试过
4. **v6 的 epoch 从 5 增到 30 带来了明显提升**（v5 r4lr2e4 0.727 → v6 r4lr2e4 30ep 0.715 for exudation 但其他字段均有提升）
5. **exudation 和 papilla 过拟合严重**，需要正则化手段

---

## 四、服务器环境

```
ssh -p 12956 root@connect.westc.seetacloud.com
```

- 双 RTX 4090（各 24GB），当前空闲
- Conda 环境：`/root/miniconda3/envs/nfc/bin/python`
- PyTorch 2.8.0+cu128, timm 1.0.28
- 项目目录：`/root/autodl-tmp/nailfold`
- DINOv2 权重：`Model/DINOv2-base/dinov2_vitb14_pretrain.pth`（331MB）
- 数据索引：`artifacts/features/dinov2/index.csv`（1708 帧）
- 标签文件：`artifacts/manifest/locked_evaluation_v1.csv`（233 病例）
- 图像目录：`data/`

### 已有脚本（全部在 `scripts/` 下）

| 脚本 | rank | lr | 说明 |
|------|------|----|------|
| `finetune_dinov2_lora_binary_cv.py` | 4 | 2e-4 | 基础版本 |
| `finetune_dinov2_lora_rank8_cv.py` | 8 | 2e-4 | 仅改 rank |
| `finetune_dinov2_lora_rank8_lr1e4_cv.py` | 8 | 1e-4 | 改 rank 和 lr |
| `finetune_dinov2_lora_rank16_cv.py` | 16 | 2e-4 | 仅改 rank |
| `finetune_dinov2_lora_rank4_lr1e4_cv.py` | 4 | 1e-4 | 仅改 lr |

### 已有实验产出（全部在 `artifacts/experiments/model-v6-20260901/` 下）

| 目录/文件 | 内容 |
|-----------|------|
| `oof_30ep/` | rank4 lr2e-4 30ep 的 OOF + metrics |
| `dinov2_lora_rank8/` | rank8 lr2e-4 的 checkpoint |
| `oof_rank8/` | rank8 lr2e-4 的 OOF + metrics |
| `rank4_lr1e4/` | rank4 lr1e-4 的 OOF + metrics + checkpoint |
| `rank4_lr1e4_locked/` | rank4 lr1e-4 的 locked 评测结果 |
| `rank8_lr1e4_oof_predictions.csv` | rank8 lr1e-4 的 OOF |
| `rank8_lr1e4_metrics.json` | rank8 lr1e-4 的 metrics |
| `rank16_oof_predictions.csv` | rank16 lr2e-4 的 OOF |
| `rank16_metrics.json` | rank16 lr2e-4 的 metrics |

---

## 五、你的任务

### 目标

在 v6 已有成绩的基础上继续提分。目标：

| 字段 | 当前最优 | 目标 |
|------|----------|------|
| clarity | 0.733 | ≥ 0.76 |
| exudation | 0.772 | ≥ 0.80（同时缩小 dev/locked 落差） |
| SVP | 0.808 | ≥ 0.83 |
| papilla | 0.661 | ≥ 0.70 |
| blood_color | 0.710 (v3) | LoRA 如果超过就切换，否则保留 v3 |

### 路线 1：补跑 rank/lr 网格空位（最高优先）

v6 只跑了 5 个配置中的 4 个（base rank4lr2e4 算第 5 个，但它是 v5 的 30ep 版本）。以下配置从未试过：

| 需要跑的配置 | 对哪个字段可能有帮助 | 理由 |
|-------------|---------------------|------|
| **rank16 lr1e-4** | papilla, clarity | papilla 在 rank16 最好(0.661)，clarity 在 lr1e-4 时比 lr2e-4 高 0.011-0.065pp |
| **rank4 lr5e-5** | exudation | exudation 过拟合严重，更低 lr 可能缓解 |
| **rank8 lr5e-5** | clarity, SVP | 已知 rank8 对这两个字段最好，更低 lr 可能进一步提升 |
| **rank16 lr5e-5** | papilla | papilla 需要高 rank，低 lr 可能帮助泛化 |
| **rank32 lr1e-4** | papilla | papilla 可能需要更大容量 |
| **rank32 lr2e-4** | papilla | 同上 |

**做法**：复制已有脚本，修改 rank 和 lr 的硬编码值，其他代码完全不动。

例如创建 rank16_lr1e4 配置：
```bash
cp scripts/finetune_dinov2_lora_rank16_cv.py scripts/finetune_dinov2_lora_rank16_lr1e4_cv.py
# 编辑：把 lr=2e-4 改为 lr=1e-4
# 只改这一处，其他不动
```

运行：
```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/nfc/bin/python scripts/finetune_dinov2_lora_rank16_lr1e4_cv.py \
  --index artifacts/features/dinov2/index.csv \
  --roles-labels artifacts/manifest/locked_evaluation_v1.csv \
  --image-root data \
  --weights Model/DINOv2-base/dinov2_vitb14_pretrain.pth \
  --output-dir artifacts/experiments/model-v6-20260901/rank16_lr1e4 \
  --epochs 30 --batch-size 16 \
  2>&1 | tee artifacts/experiments/model-v6-20260901/rank16_lr1e4.log &
```

**双卡并行**：GPU0 和 GPU1 同时跑不同配置。每个配置约 1.5 小时。

### 路线 2：在各字段最优配置上加数据增强

v6 的训练增强非常弱：只有 `hflip=0.5`，没有颜色抖动、垂直翻转或仿射变换。

在找到新的最优 rank/lr 后（或确认当前最优不变时），对每个字段的最优脚本做一处修改：

把：
```python
trtf=timm.data.create_transform(**cfg, is_training=True, color_jitter=0, hflip=.5, vflip=0)
```
改为：
```python
trtf=timm.data.create_transform(**cfg, is_training=True, color_jitter=0.3, hflip=0.5, vflip=0.5)
```

只改这一处。这增加了颜色抖动和垂直翻转，有助于缓解过拟合（特别是 exudation 和 papilla）。

### 路线 3：脏标签清洗

在数据加载的 MAP 字典中补充脏标签映射：

```python
MAP = {
    "clarity": {"清晰":0, "不清":1, "模糊":1},
    "blood_color": {"暗红":0, "暗紫":0, "浅红":1, "淡红":1, "[淡红色]":1},  # 加了 [淡红色]
    "exudation": {"无":0, "[无]":0, "+":1, "++":1, "+++":1},  # 加了 [无]
    "subpapillary_venous_plexus": {"不见":0, "[不见]":0, "可见1排":1, "可见2排":1, ">2排,扩张":1},  # 加了 [不见]
    "papilla": {"平坦":0, "浅波纹状":1, "波纹状":1, "[波纹状]":1},  # 加了 [波纹状]
}
```

同时，blood_color 列中的 `+++` 和 `+` 不应被映射（它们是 exudation 标签误入），保持为 -1 跳过。当前代码已经是这个行为（MAP 中没有 `+++`），所以 blood_color 不需要额外处理。但上面新增的方括号映射可以多回收几个样本。

---

## 六、执行计划

### 时间预算

两张 GPU，每配置约 1.5h（5-fold 30ep），可并行。8 小时 ≈ 跑 10 个配置。

### 优先级排序

1. rank16_lr1e4（GPU0）+ rank8_lr5e-5（GPU1）→ 并行 1.5h
2. rank4_lr5e-5（GPU0）+ rank16_lr5e-5（GPU1）→ 并行 1.5h
3. rank32_lr1e4（GPU0）+ rank32_lr2e-4（GPU1）→ 并行 1.5h（如果 papilla 仍无突破）
4. 在各字段新最优配置上加 color_jitter=0.3 + vflip=0.5 → 并行 1.5h
5. 在各字段新最优配置上加脏标签清洗 → 并行 1.5h
6. 如果仍有时间：尝试 lr=3e-4（比默认 2e-4 稍高）

### 每个实验完成后

立即计算逐字段 BA：

```bash
/root/miniconda3/envs/nfc/bin/python -c "
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
df = pd.read_csv('artifacts/experiments/model-v6-20260901/<CONFIG>/oof_predictions.csv')
for f in sorted(df['field'].unique()):
    s = df[df['field'] == f]
    ba = balanced_accuracy_score(s['truth'], s['prediction'])
    print(f'{f}: {ba:.4f} (n={len(s)})')
"
```

**如果某字段 BA 超过当前最优 → 记录为新最优路由。**

---

## 七、行为规则

1. **不要重写脚本**。所有配置通过复制已有脚本 + 修改 rank/lr 硬编码值实现。修改后 diff 确认只改了该改的地方。
2. **不要写新的模型结构、数据加载或训练循环**。v6 的代码已经产出了当前最优成绩，你的任务是探索超参空间，不是重构代码。
3. **两张 GPU 始终并行**。不要一张跑一张闲。
4. **每个实验必须完整跑完 5-fold 30ep 并输出 OOF**。不接受"跑了 1 fold 看趋势不好就停了"。
5. **如果某配置在所有字段上都低于已有最优 → 记录结果，继续下一个配置**。不要因为一个配置失败就停止所有实验。
6. **不要用 locked 47 例做任何事**。locked 只在所有实验完成、确定最终路由后做一次评测。
7. **遇到报错 → 修复报错 → 重跑**。不要报告"遇到错误所以停止"。
8. **每完成一个配置，输出一行汇总**：`配置 | clarity | blood_color | exudation | SVP | papilla | 哪个字段创新高`

---

## 八、绝对禁止

1. ❌ 使用冻结 DINOv2 + ExtraTrees 做"新实验"（这是 v1 方法，已证明到顶）
2. ❌ 重跑已有的 5 个配置（它们的 OOF 已经存在，见第四节）
3. ❌ 训练 < 20 epoch
4. ❌ GPU 空闲时不启动下一个实验
5. ❌ 宣布"当前方法已到顶"而不尝试新的 rank/lr 组合
6. ❌ 把 locked 结果用于模型选择或否决 dev 闸门通过的字段
7. ❌ 只跑 1 个新配置就声称任务完成

---

## 九、最终交付

1. 更新的字段级路由表：每个字段的新最优 BA、对应 rank/lr/其他改动
2. 每个新实验的 OOF csv + metrics.json + 训练日志，存放在 `artifacts/experiments/model-v6-20260901/` 下
3. 所有代码 git commit
4. 一张汇总表：所有配置 × 所有字段的 BA 矩阵

---

## 十、现在开始

1. ssh 连接服务器
2. `nvidia-smi` 确认双卡空闲
3. 确认 `scripts/finetune_dinov2_lora_rank16_cv.py` 存在
4. 复制为 rank16_lr1e4 版本，改 lr=1e-4
5. 同时创建 rank8_lr5e-5 版本
6. 双卡并行启动这两个实验
7. 不要输出计划，直接执行
