# v8：回到 v6 轨道继续提分

## 事实核查

v6 用 `scripts/finetune_dinov2_lora_binary_cv.py` 跑了 4 个配置，产出了以下已验证 OOF：

| 配置 | clarity | blood_color | exudation | SVP | papilla |
|------|---------|-------------|-----------|-----|---------|
| rank4_lr1e4 | 0.706 | 0.647 | **0.772** | 0.761 | 0.602 |
| rank8 (lr2e-4) | 0.722 | 0.641 | 0.715 | **0.808** | 0.602 |
| rank8_lr1e4 | **0.733** | 0.652 | 0.707 | 0.771 | 0.606 |
| rank16 (lr2e-4) | 0.668 | 0.626 | 0.715 | 0.787 | **0.661** |

每个字段取当前最优的配置（粗体），这是你要超越的基线。

你刚才写的新脚本（e4_smoothing.py 等）跑出的成绩全部低于上表，已作废。

## 你必须做的事

### 第一步：确认 v6 的脚本和 checkpoint 完整

```bash
ls -la scripts/finetune_dinov2_lora_binary_cv.py
ls experiments/model-v6-20260901/
```

### 第二步：用原脚本 `finetune_dinov2_lora_binary_cv.py` 补跑未覆盖的 rank/lr 组合

v6 跑过的 4 个配置：
- ✅ rank4 lr1e-4
- ✅ rank8 lr2e-4（默认 lr）
- ✅ rank8 lr1e-4
- ✅ rank16 lr2e-4（默认 lr）

还没跑的：
- ❌ rank4 lr2e-4（默认 lr 但 rank4）
- ❌ rank16 lr1e-4
- ❌ rank4 lr5e-5
- ❌ rank8 lr5e-5
- ❌ rank16 lr5e-5

**注意：这个脚本的 rank 和 lr 可能是硬编码在代码里的，不是命令行参数。** 你需要先 `grep -n "rank\|lr\|2e-4\|1e-4" scripts/finetune_dinov2_lora_binary_cv.py` 确认怎么改。

如果是硬编码，每次修改 rank 和 lr 值后运行，把 OOF 存到不同目录：
```bash
# 例：rank16 lr1e-4
# 编辑脚本改 rank=16, lr=1e-4
python scripts/finetune_dinov2_lora_binary_cv.py \
  --index artifacts/features/dinov2/index.csv \
  --roles-labels artifacts/manifest/locked_evaluation_v1.csv \
  --image-root data \
  --weights Model/DINOv2-base/dinov2_vitb14_pretrain.pth \
  --output-dir experiments/model-v6-20260901/rank16_lr1e4 \
  --epochs 30 --batch-size 16 \
  2>&1 | tee logs/rank16_lr1e4.log
```

### 第三步：双卡并行

GPU0 和 GPU1 同时跑不同配置：
```bash
# GPU0
CUDA_VISIBLE_DEVICES=0 python scripts/finetune_dinov2_lora_binary_cv.py ... --output-dir .../rank16_lr1e4 &

# GPU1
CUDA_VISIBLE_DEVICES=1 python scripts/finetune_dinov2_lora_binary_cv.py ... --output-dir .../rank4_lr5e5 &
```

### 第四步：每个实验完成后立即计算逐字段 BA

```python
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
df = pd.read_csv("experiments/model-v6-20260901/rank16_lr1e4/oof_predictions.csv")  # 或实际路径
for field in df['field'].unique():
    sub = df[df['field']==field]
    ba = balanced_accuracy_score(sub['truth'], sub['prediction'])
    print(f"{field}: {ba:.4f}")
```

如果某字段 BA 超过上表中该字段当前最优 → 记录为新最优。

### 第五步：在找到的最优配置基础上，尝试增强

对每个字段的最优 rank/lr 配置，在脚本的 `create_transform` 中把 `color_jitter=0` 改为 `color_jitter=0.3`，`vflip=0` 改为 `vflip=0.5`。重新跑一轮，看是否进一步提升。

### 第六步：对 exudation，尝试改 predict() 为 top-k

exudation 的 dev/locked 落差最大（0.772 vs 0.618）。在最优 checkpoint 上，只改推理聚合：
把 `int(np.mean(v,0).argmax())` 改为取帧级阳性概率最高的 k 帧。
对 k=1,2,3 分别推理一次，比较 OOF BA。这不需要重新训练。

## 优先级排序

1. **补跑 rank16_lr1e4** — papilla 在 rank16 最好（0.661），但还没试过 lr1e-4，而 clarity 在 lr1e-4 时明显好于 lr2e-4（+0.011→+0.065 pp），papilla 可能也是
2. **补跑 rank4_lr5e-5 和 rank8_lr5e-5** — 更低的 lr 可能减少过拟合，对 exudation 特别有意义
3. **在各字段最优配置上加增强** — color_jitter + vflip
4. **exudation top-k 推理** — 不需要重新训练

## 验证命令

每个实验结束后执行：
```bash
# 1. 确认 OOF 文件已产出
ls -la experiments/model-v6-20260901/<config>/

# 2. 计算逐字段 BA
python -c "
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
df = pd.read_csv('experiments/model-v6-20260901/<config>/oof_predictions.csv')
for f in sorted(df['field'].unique()):
    s=df[df['field']==f]; print(f'{f}: {balanced_accuracy_score(s.truth,s.prediction):.4f}')
"

# 3. 确认 GPU 在用
nvidia-smi
```

## 绝对禁止

1. ❌ 写新脚本代替 `finetune_dinov2_lora_binary_cv.py`
2. ❌ 重跑已有的 4 个配置（rank4_lr1e4, rank8, rank8_lr1e4, rank16 已有结果，不浪费时间）
3. ❌ 宣布"v6 失败所以不继续" — v6 的 OOF 文件里有明确的提升证据
4. ❌ GPU 空闲时不启动下一个实验

## 目标

| 字段 | 当前最优 | 目标 |
|------|----------|------|
| clarity | 0.733 | ≥ 0.76 |
| exudation | 0.772 | ≥ 0.80 |
| SVP | 0.808 | ≥ 0.83 |
| papilla | 0.661 | ≥ 0.70 |
| blood_color | 0.710 (v3 ET) | LoRA 如果超过就切换，否则保留 v3 |

# 现在开始。确认原脚本存在，确认如何改 rank 和 lr，然后双卡并行启动前两个补跑实验。
