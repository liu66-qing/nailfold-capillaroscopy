# v9：模型架构升级 — LoRA + Attention MIL 端到端

你是本项目的算法工程师。本文件为你提供完整背景和任务。

---

## 一、项目背景（简版）

甲襞微循环自动分析系统。输入：每病例多帧图像（平均 ~9 帧，共 1708 帧）。输出：5 个二分类字段的预测。

- 233 病例：186 development（5-fold case-level OOF） + 47 locked（严禁用于训练/调参）
- 折大小：38/32/39/43/34
- 主指标：Balanced Accuracy（BA）

### 5 个分类字段

| 字段 | 标签映射 | 有效样本 |
|------|----------|----------|
| clarity | 清晰→0, 不清/模糊→1 | 185 |
| blood_color | 暗红/暗紫→0, 浅红/淡红→1 | 174 |
| exudation | 无→0, +/++/+++→1 | 182 |
| SVP | 不见→0, 可见1排/可见2排/>2排扩张→1 | 181 |
| papilla | 平坦→0, 浅波纹状/波纹状→1 | 181 |

---

## 二、当前最优成绩和方法瓶颈

### 当前最优（从服务器实际文件验证，不可质疑）

| 字段 | Dev BA | 来源 | Locked BA |
|------|--------|------|-----------|
| clarity | 0.733 | DINOv2 LoRA rank8 lr1e-4 | 0.713 |
| blood_color | 0.710 | v3 YOLO检测特征+ExtraTrees | 0.746 |
| exudation | 0.772 | DINOv2 LoRA rank4 lr1e-4 | 0.618 |
| SVP | 0.808 | DINOv2 LoRA rank8 lr2e-4 | 0.873 |
| papilla | 0.661 | DINOv2 LoRA rank16 lr2e-4 | 0.570 |

### 已证明到顶的方法（不要再试）

1. **冻结特征 + ExtraTrees/XGBoost**：v1-v4 在这个框架内尝试了各种特征组合、pooling、检测/分割特征，已到顶
2. **LoRA 超参搜索**：v6-v8 搜索了 rank {4,8,16,32} × lr {5e-5, 1e-4, 2e-4}，加了数据增强和标签清洗，没有稳定超越上表
3. **冻结特征 + attention MIL**（E5 实验）：DINOv2/HuluMed 冻结帧特征 → 注意力聚合，低于 v1

### 瓶颈分析

当前 LoRA 方法的核心限制：**帧级 logit 均值聚合**。

```python
# 当前 predict() 的聚合方式
return {f: {c: int(np.mean(v, 0).argmax()) for c, v in z.items()} ...}
```

这对所有帧等权平均，假设每帧同等重要。但实际上：
- 有些帧质量差（模糊、遮挡），应该被降权
- exudation/hemorrhage 是"存在性"事件 — 一帧有就够了，均值会被大量阴性帧稀释
- 不同字段需要关注不同的帧（血色看整体，渗出看局部）

E5 的冻结特征 MIL 失败了，但那是因为 **冻结特征没有针对下游任务优化**。如果把 LoRA 微调和 attention MIL 端到端结合，特征和注意力权重可以联合学习。

---

## 三、你的任务：三个新架构方向

### 方向 A：LoRA 微调后的特征 + Attention MIL（两阶段，最稳妥）

**思路**：先用当前最优 LoRA checkpoint 提取微调后的帧特征，再在这些特征上训练 attention MIL。

**为什么可行**：E5 的冻结特征 MIL 失败了（BA < v1），但 LoRA 微调后的特征已经包含了下游任务信号。在更好的特征上做 attention 聚合，有理由期待超过简单均值。

**实现步骤**：

1. 对每个字段的最优 LoRA checkpoint，提取所有帧的 CLS embedding（微调后的 768 维向量）
2. 把提取的特征保存为 npy
3. 用已有的 `train_binary_mil_features.py` 脚本（稍作修改）在这些特征上训练 attention MIL
4. 5-fold OOF 评估

**具体代码**：

提取 LoRA 特征的脚本（新建 `scripts/extract_lora_features.py`）：
```python
"""Extract frame-level CLS features from finetuned LoRA checkpoints."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math

class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        for p in base.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale

class SimpleFrames(Dataset):
    def __init__(self, index, root, transform):
        self.index = index.reset_index(drop=True)
        self.root = root
        self.transform = transform
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        r = self.index.iloc[i]
        img = self.transform(Image.open(self.root / r.image_path).convert("RGB"))
        return img, r.exam_case_id, i

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="dir with fold0.pt ... fold4.pt")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True, help="pretrained DINOv2 weights")
    p.add_argument("--output", type=Path, required=True, help="output .npy path")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    index = pd.read_csv(args.index)
    index.exam_case_id = index.exam_case_id.astype(str)

    # Build model
    backbone = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    backbone.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=False)
    for param in backbone.parameters():
        param.requires_grad = False
    for block in backbone.blocks[-4:]:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=args.rank)
        block.attn.proj = LoRALinear(block.attn.proj, rank=args.rank)

    cfg = timm.data.resolve_model_data_config(backbone)
    transform = timm.data.create_transform(**cfg, is_training=False)
    dataset = SimpleFrames(index, args.image_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    all_features = np.zeros((len(index), 768), dtype=np.float16)

    # For each fold, load checkpoint and extract features for that fold's test frames
    # But simpler: just use fold 0's checkpoint for all frames (features won't be used for that fold's OOF anyway)
    # Actually we need per-fold extraction for OOF to be valid.
    # Strategy: extract ALL frames with each fold's checkpoint, save separately, then MIL script handles fold assignment

    # Simpler approach: extract with each fold checkpoint, save per-fold
    for fold in range(5):
        ckpt_path = args.checkpoint_dir / f"fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Load state dict - need to handle the model wrapper
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # The checkpoint has keys like "b.blocks.8.attn.qkv.base.weight" etc
        # We need to load into our model structure
        backbone_model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
        backbone_model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=False)
        for param in backbone_model.parameters():
            param.requires_grad = False
        for block in backbone_model.blocks[-4:]:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=args.rank)
            block.attn.proj = LoRALinear(block.attn.proj, rank=args.rank)
        
        # Create full model to match state dict keys
        class Model(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.b = b
                self.heads = nn.ModuleDict({"clarity": nn.Linear(768,2), "blood_color": nn.Linear(768,2), 
                    "exudation": nn.Linear(768,2), "subpapillary_venous_plexus": nn.Linear(768,2), "papilla": nn.Linear(768,2)})
            def forward(self, x):
                return self.b(x)
        
        model = Model(backbone_model)
        model.load_state_dict(state, strict=True)
        model = model.cuda().eval()

        fold_feats = []
        fold_indices = []
        with torch.inference_mode():
            for imgs, case_ids, orig_indices in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = model.b(imgs.cuda())  # (B, 768)
                fold_feats.append(feats.float().cpu().numpy())
                fold_indices.extend(orig_indices.tolist())

        fold_feats = np.concatenate(fold_feats, axis=0)
        # Save per-fold features
        out_fold = args.output.parent / f"lora_features_fold{fold}.npy"
        np.save(out_fold, fold_feats.astype(np.float16))
        print(f"Fold {fold}: extracted {len(fold_feats)} features, saved to {out_fold}")

        del model, backbone_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
```

然后在 LoRA 特征上跑 attention MIL — 修改已有的 `train_binary_mil_features.py`，改为加载 LoRA 特征而非冻结特征。

### 方向 B：端到端 LoRA + Attention Pooling（单阶段，最优雅）

**思路**：修改当前 LoRA 训练脚本，把 `predict()` 中的帧均值聚合替换为可学习的 attention pooling 层，整个模型端到端训练。

**关键改动**：

```python
class AttentionPooling(nn.Module):
    """Learnable attention pooling over frame features."""
    def __init__(self, dim=768, heads=4):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, heads),
        )
        self.heads = heads
    
    def forward(self, frame_features):
        """
        frame_features: (num_frames, dim)
        returns: (dim,) aggregated feature
        """
        weights = self.attn(frame_features)  # (num_frames, heads)
        weights = torch.softmax(weights, dim=0)  # normalize over frames
        # Multi-head attention pooling + mean + max
        attended = torch.einsum("fh,fd->hd", weights, frame_features)  # (heads, dim)
        pooled = torch.cat([
            attended.flatten(),           # heads * dim
            frame_features.mean(0),       # dim
            frame_features.max(0).values, # dim
        ], dim=0)  # (heads + 2) * dim
        return pooled

class ModelWithAttention(nn.Module):
    def __init__(self, backbone, attention_heads=4, dropout=0.3):
        super().__init__()
        self.backbone = backbone  # DINOv2 with LoRA
        self.pool = AttentionPooling(768, attention_heads)
        pooled_dim = 768 * (attention_heads + 2)
        self.case_encoder = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 768),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({f: nn.Linear(768, 2) for f in FIELDS})
    
    def forward_frames(self, images):
        """Extract frame features: (batch, dim)"""
        return self.backbone(images)
    
    def forward_case(self, frame_features):
        """Aggregate frames and classify: frame_features is (num_frames, dim)"""
        pooled = self.pool(frame_features)
        encoded = self.case_encoder(pooled)
        return {f: head(encoded) for f, head in self.heads.items()}
```

**训练循环改动**：

当前训练循环是 frame-level 的：每个 batch 是独立的帧，loss 按帧计算。要改成 case-level：

```python
# 训练一个 case 的流程：
# 1. 取出该 case 的所有帧
# 2. backbone 提取帧特征
# 3. attention pooling 聚合为 case 特征
# 4. 分类 head 产出 logits
# 5. 和 case 标签算 loss

for case_id in shuffled_case_ids:
    frames = get_case_frames(case_id)  # 该 case 的所有帧图像
    with torch.autocast("cuda", dtype=torch.bfloat16):
        frame_feats = model.forward_frames(frames.cuda())  # (num_frames, 768)
        outputs = model.forward_case(frame_feats)  # {field: (2,)}
    loss = compute_loss(outputs, case_labels[case_id])
    loss.backward()
    # 每 N 个 case 更新一次
```

这个方向更优雅但改动较大。如果实现得当，attention 权重和 LoRA 参数联合优化，理论上上限更高。

### 方向 C：多特征融合（利用已有特征资产）

**思路**：服务器上已经有多种帧级特征，目前只用了 DINOv2。把多种特征拼接后做 attention MIL。

已有帧级特征（全部 1708 帧）：
| 特征 | 文件 | 维度 |
|------|------|------|
| DINOv2 冻结 CLS | `features/dinov2/features.npy` | 768 |
| HuluMed vision | `features/hulumed_visual/vision_mean.npy` | 1152 |
| Geometry | `features/geometry_dev/features.npy` | 394 |
| ROI quality | `features/roi_quality_dev_v1/features.npy` | 372 |

拼接后总维度：768 + 1152 + 394 + 372 = 2686

在拼接特征上做 attention MIL。如果方向 A 成功（LoRA 特征 + MIL 有提升），可以进一步把 LoRA 特征替代冻结 DINOv2 特征做拼接。

---

## 四、执行优先级

**先做方向 A（两阶段：LoRA 特征提取 + MIL），因为**：
- 改动最小：特征提取是一次性 GPU 操作，MIL 训练在 CPU 上跑（已有脚本）
- 风险最低：如果 MIL 没提升，LoRA 特征还可以用于其他下游任务
- 验证最快：能在 1-2 小时内出结果

**然后做方向 B（端到端 LoRA + Attention Pooling）**：
- 改动较大但上限更高
- 需要修改训练循环从 frame-level 到 case-level

**最后做方向 C（多特征融合）**：
- 如果方向 A 有提升，用 LoRA 特征替换冻结特征做融合

---

## 五、服务器环境

```
ssh -p 12956 root@connect.westc.seetacloud.com
```

- 双 RTX 4090（各 24GB），当前空闲
- Python: `/root/miniconda3/envs/nfc/bin/python`
- PyTorch 2.8.0+cu128, timm 1.0.28, einops 0.8.2
- 项目：`/root/autodl-tmp/nailfold`

### 关键路径

| 资源 | 路径 |
|------|------|
| DINOv2 预训练权重 | `Model/DINOv2-base/dinov2_vitb14_pretrain.pth` |
| 数据索引 | `artifacts/features/dinov2/index.csv` |
| 标签/角色文件 | `artifacts/manifest/locked_evaluation_v1.csv` |
| 图像目录 | `data/` |
| 冻结 DINOv2 特征 | `artifacts/features/dinov2/features.npy` (1708, 768) |
| HuluMed 特征 | `artifacts/features/hulumed_visual/vision_mean.npy` (1708, 1152) |
| Geometry 特征 | `artifacts/features/geometry_dev/features.npy` (1708, 394) |
| 已有 MIL 脚本 | `scripts/train_binary_mil_features.py` |

### LoRA checkpoint 路径（各字段最优配置）

| 字段最优配置 | Checkpoint 目录 | rank |
|-------------|----------------|------|
| rank4 lr1e-4 | `artifacts/experiments/model-v6-20260901/rank4_lr1e4/` | 4 |
| rank8 lr2e-4 | `artifacts/experiments/model-v6-20260901/dinov2_lora_rank8/` | 8 |
| rank8 lr1e-4 | `artifacts/experiments/model-v6-20260901/dinov2_lora_rank8_lr1e4/` | 8 |
| rank16 lr2e-4 | `artifacts/experiments/model-v6-20260901/dinov2_lora_rank16_retry/` 或 `rank16/` | 16 |

每个目录下有 `fold0.pt` ... `fold4.pt`。

### 已有训练脚本的结构（你需要理解）

`scripts/finetune_dinov2_lora_binary_cv.py`：
- `LoRALinear`：给 attention 的 qkv 和 proj 加低秩适配器
- `Frames` Dataset：帧级数据集，每帧返回 (image, labels, case_id)
- `Model`：DINOv2 backbone + 5 个独立二分类 head
- `predict()`：帧级 logit 均值聚合为 case 预测
- 训练循环：frame-level batch，所有帧独立算 loss
- 验证：case-level BA（predict 内部做了 case 聚合）
- 嵌套 5-fold：3 折训练、1 折验证选 epoch、1 折测试

`scripts/train_binary_mil_features.py`：
- `BinaryAttentionMIL`：encoder → attention → pooled → case_encoder → 5 个 head
- 输入是预提取的帧特征 npy，不做图像处理
- Case-level 训练：每个 case 的所有帧特征一起前向
- 多 seed 集成（5 个 seed），250 epoch + patience 35
- 在 CPU 上运行
- 需要参数：`--features`(npy), `--feature-index`(csv), `--labels`(csv), `--folds-file`(csv), `--roles`(csv), `--test-fold`(int)

---

## 六、方向 A 详细步骤

### 步骤 1：确认 LoRA checkpoint 完整

```bash
for dir in artifacts/experiments/model-v6-20260901/rank4_lr1e4 \
           artifacts/experiments/model-v6-20260901/dinov2_lora_rank8 \
           artifacts/experiments/model-v6-20260901/dinov2_lora_rank8_lr1e4; do
  echo "=== $dir ==="
  ls "$dir"/fold*.pt 2>/dev/null | wc -l
done
```

如果某个配置缺 checkpoint → 用对应脚本重新训练。

### 步骤 2：编写并运行特征提取脚本

创建 `scripts/extract_lora_features.py`（上面方向 A 给了代码骨架）。
关键：每个 fold 用该 fold 的 checkpoint 提取特征，不能跨 fold 共享 checkpoint。

对于 OOF 来说，fold i 的 test case 的帧特征应该用 fold i 的 checkpoint 提取。这样在后续 MIL 中，这些特征是 "out-of-fold" 的。

更简单的方案：用每个 fold 的 checkpoint 提取所有帧的特征，保存 5 个 npy。然后 MIL 脚本在评估 fold i 时使用 fold i 的特征。

或者更实用的方案：每个字段用其最优配置的 checkpoint，只提取该 fold 的 test case 帧，最终拼成一个完整的 OOF 特征集。

```bash
# 对 rank8 lr2e-4 (SVP 最优) 提取特征
/root/miniconda3/envs/nfc/bin/python scripts/extract_lora_features.py \
  --checkpoint-dir artifacts/experiments/model-v6-20260901/dinov2_lora_rank8 \
  --rank 8 \
  --index artifacts/features/dinov2/index.csv \
  --image-root data \
  --weights Model/DINOv2-base/dinov2_vitb14_pretrain.pth \
  --output artifacts/features/dinov2_lora_rank8/features.npy \
  --batch-size 32
```

### 步骤 3：在 LoRA 特征上运行 MIL

修改 `train_binary_mil_features.py` 的调用参数，指向新特征：

```bash
for fold in 0 1 2 3 4; do
  /root/miniconda3/envs/nfc/bin/python scripts/train_binary_mil_features.py \
    --features artifacts/features/dinov2_lora_rank8/lora_features_fold${fold}.npy \
    --feature-index artifacts/features/dinov2/index.csv \
    --labels artifacts/manifest/locked_evaluation_v1.csv \
    --folds-file <fold_file> \
    --roles artifacts/manifest/locked_evaluation_v1.csv \
    --test-fold $fold \
    --output-dir artifacts/experiments/model-v9/mil_lora_rank8_fold${fold} \
    --device cpu
done
```

注意：可能需要调整 MIL 脚本以接受 per-fold 特征文件。如果改动太大，可以先用单个 fold 的 checkpoint 提取所有帧特征来做概念验证。

### 步骤 4：评估

汇总 5 个 fold 的 test 预测，计算逐字段 BA。和当前最优对比。

---

## 七、方向 B 详细步骤

### 核心：改造 LoRA 训练脚本为 case-level 训练

创建 `scripts/finetune_dinov2_lora_attention_cv.py`，基于原脚本改造：

1. **新增 `AttentionPooling` 模块**（代码见方向 B 描述）

2. **替换 `Model` 类为 `ModelWithAttention`**

3. **改造训练循环**：
   - 原来：DataLoader 以帧为单位，batch_size=16 个帧
   - 现在：以 case 为单位，每个 case 的所有帧一起前向
   - 梯度累积：每 4-8 个 case 更新一次

4. **改造 `predict()` 函数**：
   - 原来：收集所有帧的 logits → mean → argmax
   - 现在：收集所有帧特征 → attention pooling → head → argmax

5. **显存管理**：
   - 每个 case ~9 帧，每帧 518×518，backbone forward 一次约 200MB
   - 9 帧 → ~1.8GB，加上梯度 → ~4GB 每 case
   - 24GB 卡可以处理，但需要梯度累积

**关键实现细节**：

```python
# Case-level training loop
model.train()
optimizer.zero_grad(set_to_none=True)
accumulation_steps = 4
for i, case_id in enumerate(shuffled_train_case_ids):
    case_frames = get_frames_for_case(case_id)  # (N, 3, 518, 518)
    case_labels = get_labels_for_case(case_id)   # dict {field: int}
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        frame_feats = model.forward_frames(case_frames.cuda())  # (N, 768)
        outputs = model.forward_case(frame_feats)                # {field: (2,)}
        
        losses = []
        for fi, field in enumerate(FIELDS):
            target = case_labels[field]
            if target >= 0:
                losses.append(nn.functional.cross_entropy(
                    outputs[field].unsqueeze(0),
                    torch.tensor([target], device="cuda")
                ))
        loss = torch.stack(losses).mean() / accumulation_steps
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### 方向 B 的超参建议

| 参数 | 值 | 理由 |
|------|-----|------|
| LoRA rank | 8 | 在多数字段表现好 |
| lr | 1e-4 | LoRA 和 attention 联合训练需要较低 lr |
| attention heads | 4 | 和已有 MIL 脚本一致 |
| dropout (attention) | 0.3 | 防止 attention 过拟合 |
| dropout (head) | 0.3 | 同上 |
| epochs | 30 | 和 v6 一致 |
| patience | 5 | early stopping |
| accumulation | 4 cases | 约 36 帧一次更新 |

---

## 八、行为规则

1. **方向 A 优先**。先做特征提取 + MIL，快速验证 attention 聚合是否有帮助。
2. **复用已有代码**。特征提取基于 v6 的模型结构，MIL 基于 `train_binary_mil_features.py`。不要从零重写。
3. **每个方向必须产出 5-fold OOF 和逐字段 BA**。
4. **对比基线**：每个字段和当前最优对比，只要有 1 个字段超过就是有效产出。
5. **如果方向 A 失败（所有字段都没超过当前最优）→ 继续方向 B**。不要因为 A 失败就停止。
6. **两张 GPU 可以并行做不同方向的不同部分**。比如 GPU0 提取特征，GPU1 跑方向 B 的训练。
7. **不要用 locked 47 例做任何事**。
8. **遇到代码问题修复后继续，不要报告"遇到错误所以停止"**。

---

## 九、绝对禁止

1. ❌ 重跑 v6 已有的 LoRA 超参搜索（rank/lr 网格已搜完）
2. ❌ 用冻结特征 + ExtraTrees
3. ❌ 只做方向 A 的概念验证（1 个 fold）就声称完成 — 必须跑完整 5-fold OOF
4. ❌ 使用 locked 数据
5. ❌ 写完代码不跑

---

## 十、最终交付

1. 方向 A 的 5-fold OOF 逐字段 BA
2. 方向 B 的 5-fold OOF 逐字段 BA（如果时间够）
3. 和当前最优的对比表
4. 所有代码和产出 git commit
5. 更新路由：如果有字段超过当前最优

---

## 十一、现在开始

1. ssh 连接服务器，确认 GPU 空闲
2. 确认 LoRA checkpoint 完整（rank8 的 fold0-4.pt 都在）
3. 编写 `scripts/extract_lora_features.py`
4. 对 rank8 lr2e-4 配置提取 LoRA 特征
5. 在 LoRA 特征上运行 MIL
6. 输出结果

不要输出计划，直接执行。
