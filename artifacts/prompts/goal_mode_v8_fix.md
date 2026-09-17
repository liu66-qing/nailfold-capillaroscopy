# 紧急纠偏：回到产出最优成绩的脚本和路线

## 你刚才做错了什么

你写了新脚本 `finetune_dinov2_lora_e4_smoothing.py` 和 `finetune_dinov2_lora_p1_rank8.py`。
经检查，它们和原脚本 `finetune_dinov2_lora_binary_cv.py` 的 diff 只有两处：
1. `self.drop=nn.Dropout(0.5)` — 声明了但 forward() 里从未调用，等于没加
2. `label_smoothing=0.1` — 唯一生效的改动

你声称加了 ColorJitter、RandomVerticalFlip、RandomAffine、focal_loss、top-k pooling、LoRA dropout，**全部没有实现**。
nvidia-smi 显示 GPU 已空闲，训练早已结束，但你之前报告"正在训练中"。

## 事实：哪个脚本产出了当前最优成绩

当前 5 个主力字段中 4 个的最优 BA 全部来自同一个脚本：

```
scripts/finetune_dinov2_lora_binary_cv.py
```

各字段最优配置：
| 字段 | Dev BA | rank | lr | 产出目录 |
|------|--------|------|-----|---------|
| clarity | 0.733 | 8 | 1e-4 | experiments/model-v6-*/rank8_lr1e4_oof_predictions.csv |
| exudation | 0.772 | 4 | 1e-4 | experiments/model-v6-*/rank4_lr1e4_oof_predictions.csv |
| SVP | 0.808 | 8 | 2e-4 | experiments/model-v6-*/rank8_oof_predictions.csv |
| papilla | 0.661 | 16 | 2e-4 | experiments/model-v6-*/rank16_oof_predictions.csv |
| blood_color | 0.710 | — | — | v3 ExtraTrees（非 LoRA） |

## 你现在必须做的事

### 第一步：确认原脚本还在

```bash
ls -la scripts/finetune_dinov2_lora_binary_cv.py
head -5 scripts/finetune_dinov2_lora_binary_cv.py
```

如果还在 → 用它。如果不在 → 从 git 恢复。

### 第二步：不要重写脚本，在原脚本基础上加参数

复制原脚本为 v8 版本：
```bash
cp scripts/finetune_dinov2_lora_binary_cv.py scripts/finetune_dinov2_lora_v8.py
```

然后对 `finetune_dinov2_lora_v8.py` 做以下 **精确修改**（用 sed 或直接编辑，每改一处 diff 确认）：

#### 修改 1：加命令行参数控制 rank 和 lr
在 argparse 部分加：
```python
p.add_argument("--rank", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--lora-dropout", type=float, default=0.0)
p.add_argument("--head-dropout", type=float, default=0.0)
p.add_argument("--label-smoothing", type=float, default=0.0)
p.add_argument("--aggregation", choices=["mean","topk"], default="mean")
p.add_argument("--topk", type=int, default=2)
```

#### 修改 2：LoRALinear 构造时使用 args.rank 和 args.lora_dropout
把 `LoRALinear(block.attn.qkv)` 改为 `LoRALinear(block.attn.qkv, rank=a.rank, dropout=a.lora_dropout)`
（同理 proj）

LoRALinear 类加 dropout：
```python
class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8, dropout=0.0):
        super().__init__()
        ...（原有代码）
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        return self.base(x) + self.b(self.drop(self.a(x))) * self.scale
```

#### 修改 3：Model 的 forward 里加 head dropout
```python
class Model(nn.Module):
    def __init__(self, b, head_dropout=0.0):
        super().__init__()
        self.b = b
        self.drop = nn.Dropout(head_dropout)
        self.heads = nn.ModuleDict({f: nn.Linear(768, 2) for f in FIELDS})
    def forward(self, x):
        z = self.drop(self.b(x))   # ← dropout 在这里生效
        return {f: h(z) for f, h in self.heads.items()}
```

#### 修改 4：loss 加 label_smoothing
```python
nn.functional.cross_entropy(..., label_smoothing=a.label_smoothing)
```

#### 修改 5：predict() 支持 top-k
在现有的 predict 函数中，把 `int(np.mean(v,0).argmax())` 替换为：
```python
if agg == "topk":
    probs = [softmax(frame_logit)[1] for frame_logit in v]
    top = sorted(probs, reverse=True)[:k]
    int(np.mean(top) > 0.5)
else:
    int(np.mean(v, 0).argmax())
```

#### 修改 6：训练 transform 加强增强
把 `timm.data.create_transform(**cfg, is_training=True, color_jitter=0, hflip=.5, vflip=0)` 改为：
```python
timm.data.create_transform(**cfg, is_training=True, color_jitter=0.2, hflip=0.5, vflip=0.5, auto_augment=None)
```

### 第三步：diff 确认修改正确

```bash
diff scripts/finetune_dinov2_lora_binary_cv.py scripts/finetune_dinov2_lora_v8.py
```

贴出完整 diff。如果 diff 超过 80 行，说明你改多了，检查是否引入了不必要的变动。

### 第四步：先用原配置复现基线，确认脚本没改坏

```bash
# 用 exudation 最优配置（rank4 lr1e-4）复现 0.772
python scripts/finetune_dinov2_lora_v8.py \
  --index artifacts/features/dinov2/index.csv \
  --roles-labels artifacts/manifest/locked_evaluation_v1.csv \
  --image-root data \
  --weights Model/DINOv2-base/dinov2_vitb14_pretrain.pth \
  --output-dir experiments/v8_exudation_baseline_check \
  --epochs 30 --batch-size 16 \
  --rank 4 --lr 1e-4 \
  --lora-dropout 0 --head-dropout 0 --label-smoothing 0 \
  --aggregation mean \
  2>&1 | tee logs/v8_baseline_check.log
```

**验收条件**：exudation 的 OOF BA 应在 0.76-0.79 之间（允许随机波动），如果 < 0.72 说明你改坏了。

### 第五步：确认基线后，逐项测试正则化

每次只改一个变量，其他和基线相同：

```bash
# E1: 加增强（color_jitter 已在修改 6 中生效）
# 如果基线复现已经包含增强 → 这就是 E1 的结果

# E2: E1 + head dropout 0.5
python scripts/finetune_dinov2_lora_v8.py \
  ... --rank 4 --lr 1e-4 --head-dropout 0.5 \
  --output-dir experiments/v8_exudation_E2_hdrop \
  2>&1 | tee logs/v8_exudation_E2.log

# E3: E1 + LoRA dropout 0.1
python scripts/finetune_dinov2_lora_v8.py \
  ... --rank 4 --lr 1e-4 --lora-dropout 0.1 \
  --output-dir experiments/v8_exudation_E3_ldrop \
  2>&1 | tee logs/v8_exudation_E3.log

# E4: E1 + label smoothing 0.1
python scripts/finetune_dinov2_lora_v8.py \
  ... --rank 4 --lr 1e-4 --label-smoothing 0.1 \
  --output-dir experiments/v8_exudation_E4_smooth \
  2>&1 | tee logs/v8_exudation_E4.log

# E5: rank2（更强正则）
python scripts/finetune_dinov2_lora_v8.py \
  ... --rank 2 --lr 1e-4 \
  --output-dir experiments/v8_exudation_E5_rank2 \
  2>&1 | tee logs/v8_exudation_E5.log
```

papilla 同理，基线是 rank16 lr2e-4：
```bash
# P1: 基线复现
python scripts/finetune_dinov2_lora_v8.py \
  ... --rank 16 --lr 2e-4 \
  --output-dir experiments/v8_papilla_baseline \
  2>&1 | tee logs/v8_papilla_baseline.log

# P2: rank8 降复杂度
# P3: head dropout 0.5
# P4: label smoothing 0.1
```

### 第六步：血色字段 LoRA 首次尝试

blood_color 从未用过 LoRA。用 v8 脚本直接跑：
```bash
python scripts/finetune_dinov2_lora_v8.py \
  ... --rank 8 --lr 1e-4 \
  --output-dir experiments/v8_blood_color_B1 \
  2>&1 | tee logs/v8_blood_color_B1.log
```

### 第七步：每个实验结束后输出一行汇总

格式：
```
字段 | 配置 | OOF BA | vs 基线 | 5-fold std | 少数类召回 | 结论
```

## 绝对禁止

1. ❌ 重新写一个新脚本（必须基于 finetune_dinov2_lora_binary_cv.py 修改）
2. ❌ 声明了 dropout 但 forward 里不调用
3. ❌ diff 只有 2 行就声称"已加入完整正则化"
4. ❌ 再次报告"正在训练"但 nvidia-smi 显示 GPU 空闲
5. ❌ 跳过第四步基线复现直接跑新实验

## 验证命令（我会要求你执行）

```bash
# 验证 1：新脚本和原脚本的 diff
diff scripts/finetune_dinov2_lora_binary_cv.py scripts/finetune_dinov2_lora_v8.py

# 验证 2：确认 dropout 在 forward 中被调用
grep -n "self.drop" scripts/finetune_dinov2_lora_v8.py

# 验证 3：确认 color_jitter 参数不是 0
grep -n "color_jitter" scripts/finetune_dinov2_lora_v8.py

# 验证 4：训练中 GPU 是否被占用
nvidia-smi

# 验证 5：OOF 文件是否产出
ls -la experiments/v8_*/
```

# 现在开始。执行第一步（确认原脚本存在），然后第二步（复制+修改），然后第三步（贴 diff）。
