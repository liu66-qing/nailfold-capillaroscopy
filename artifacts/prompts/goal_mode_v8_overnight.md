你是本项目的主算法工程师，在远端双 RTX 4090 服务器上自主运行一整夜。

# 服务器
ssh -p 12956 root@connect.westd.seetacloud.com
两张 RTX 4090。项目在远端已有完整代码和数据。

# 你的唯一目标
把 5 个主力分类字段的 development OOF BA 推到尽可能高。1 个字段提升就是有效交付。

# 当前最优路由（这是你要超越的基线）

| 字段 | Dev BA | Locked BA | 来源 | 问题诊断 |
|------|--------|-----------|------|----------|
| clarity | 0.733 | 0.713 | LoRA rank8 lr1e-4 | 少数类"模糊"仅~8例，几乎无法学习；主要混淆在"清晰"vs"不清" |
| blood_color | 0.710 | 0.746 | v3 YOLO检测特征+ET | 唯一还在用 ExtraTrees 的主力字段，从未做过 LoRA 端到端 |
| exudation | 0.772 | 0.618 | LoRA rank4 lr1e-4 | **dev/locked 落差 15pp，过拟合严重**，需要正则化 |
| SVP | 0.808 | 0.873 | LoRA rank8 | locked > dev，泛化良好，但 dev 0.808 仍有空间 |
| papilla | 0.661 | 0.570 | LoRA rank16 | **最弱字段，dev/locked 落差 9pp**，少数类"平坦"召回仅 0.50 |

# 已确认的数据资产（你必须利用）

1. **582 分类图像**（186 病例 × 多帧），已有 5-fold case-level split
2. **11,600 增强图**（和原图同 fold，比例约 1:20，可按需采样）
3. **2,332 对分割 mask**（语义分割的 image+mask 对，可做辅助预训练）
4. **671 张检测标注图**（vessel/malformed_vessel/cross_vessel XML 标注）
5. 已有 DINOv2 LoRA 训练脚本：`scripts/finetune_dinov2_lora_binary_cv.py`
6. 已有 YOLO11m-seg 检测模型和 v3 特征

# 已知标签噪声（fold_diagnosis.csv 已确认）

- blood_color fold 1 混入 1 个 `+++`（这是 exudation 标签）
- blood_color fold 2 混入 1 个 `+`（同上）
- 各 fold 存在 `[淡红色]`、`[不见]`、`[波纹状]` 等带方括号的标签（OCR 残留）
- **处理方式**：训练前先清洗——方括号标签映射到对应正常标签，错位标签（如 blood_color 中的 `+++`）标记为 NaN 跳过

# 三条攻击路线（按优先级排序）

## 路线 A：修复已有 LoRA 的过拟合问题（最高优先）

针对 exudation（dev 0.772 / locked 0.618 落差 15pp）和 papilla（dev 0.661 / locked 0.570 落差 9pp）：

1. **增强**：RandomHorizontalFlip + RandomVerticalFlip + ColorJitter(0.2,0.2,0.2,0.1) + RandomAffine(±10°, scale 0.9-1.1)
2. **Dropout**：在 LoRA 层之后加 dropout=0.3；分类 head 加 dropout=0.5
3. **Label smoothing**：CrossEntropyLoss 的 label_smoothing=0.1
4. **更小 rank 防过拟合**：exudation 当前 rank4 已经过拟合 → 试 rank2；papilla 当前 rank16 → 降到 rank4/8
5. **Mixup**：帧级 mixup（alpha=0.2），增加训练样本多样性
6. **Stochastic Weight Averaging (SWA)**：最后 5 epoch 做 SWA，平滑模型权重
7. **每个正则化手段逐步叠加，不要一次全加**，先测单项效果

实验矩阵（至少跑完前 4 行）：
```
exudation:
  E1: rank4 + augmentation only          → 对比 baseline 0.772
  E2: rank4 + aug + dropout 0.3          → 对比 E1
  E3: rank2 + aug + dropout 0.3          → 对比 E2（更强正则）
  E4: rank4 + aug + dropout + smoothing  → 对比 E2
  E5: E{best} + mixup                    → 对比 E{best}

papilla:
  P1: rank8 + augmentation               → 对比 baseline 0.661
  P2: rank4 + augmentation               → 对比 P1
  P3: rank8 + aug + dropout + smoothing  → 对比 P1
  P4: rank8 + aug + focal_loss(gamma=2)  → 对比 P1（因为少数类"平坦"召回差）
```

## 路线 B：blood_color LoRA 端到端（从未尝试过）

blood_color 是 5 个主力字段中唯一还在用 ExtraTrees 的。v3 的 0.710 来自 YOLO 检测特征 + ExtraTrees，但 LoRA 在其他 4 个字段都优于 ExtraTrees。这是最可能一击提升的字段。

标签：暗红/暗紫/浅红/淡红 → 二分类（暗色 vs 浅色）或三分类（暗/中/浅）
- 注意清洗 fold 1 的 `+++` 和 fold 2 的 `+` 错位标签
- 注意清洗 `[淡红色]` → `淡红`

实验矩阵：
```
B1: rank4 lr1e-4 30ep 二分类 + aug     → 直接对比 v3 baseline 0.710
B2: rank8 lr1e-4 30ep 二分类 + aug
B3: rank4 lr1e-4 30ep 三分类 + aug     → 序数信息可能帮助
B4: best config + dropout + smoothing   → 防过拟合
```

## 路线 C：聚合方式改进（跨字段通用）

当前所有 LoRA 模型的 `predict()` 用的是帧级 logit 均值。不同字段适合不同聚合：

1. **Top-k pooling**（k=1,2,3）：对"存在性"字段好（exudation、hemorrhage），一帧有就够
2. **Attention MIL pooling**：可学习的帧注意力权重，让模型自己决定哪帧重要
3. **Quality-weighted pooling**：用 clarity 模型预测帧质量，质量加权聚合

实现 attention pooling 的代码骨架：
```python
class AttentionPooling(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, features, mask=None):
        # features: (num_frames, dim)
        a = self.attn(features)  # (num_frames, 1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        a = torch.softmax(a, dim=0)
        return (a * features).sum(dim=0)  # (dim,)
```

对每个字段在其当前最优 LoRA 上试 mean vs top-k vs attention，选 OOF 最好的。

# 脏标签清洗（训练前必须执行）

在加载标签时执行以下映射，拒绝加载未映射的异常标签：

```python
LABEL_FIXES = {
    'blood_color': {
        '[淡红色]': '淡红',
        '+++': None,   # 错位标签，跳过
        '+': None,     # 错位标签，跳过
    },
    'clarity': {},     # 无已知脏标签
    'exudation': {
        '[无]': '无',
    },
    'subpapillary_venous_plexus': {
        '[不见]': '不见',
    },
    'papilla': {
        '[波纹状]': '波纹状',
    },
}
```

# 驱动规则

## 规则 1：字段级闸门
某字段 BA > 当前最优 且 少数类召回未下降 → 该字段通过闸门，立即更新路由。
不需要所有字段同时提升。1 个字段提升 = 有效交付。

## 规则 2：每条路线至少 3 个变体
不要跑 1 个配置就停。每条路线至少跑 3 个变体对比。rank/lr/正则手段各试 2-3 个值。

## 规则 3：先修 exudation 和 papilla（路线 A），再做 blood_color（路线 B），最后做聚合（路线 C）
因为 A 是修复已知 bug（过拟合），收益确定性最高。B 是新尝试。C 是锦上添花。

## 规则 4：过拟合修复的判断标准
如果新配置的 dev BA 略降（≤2pp）但 5-fold 标准差下降 → 认为泛化改善，值得保留。
最终目标是 locked BA 高，不是 dev BA 高。过拟合的 dev BA 没有意义。

## 规则 5：每个实验必须输出
- 训练日志：每 epoch 的 train_loss 和 val_BA
- GPU 显存峰值和训练总耗时
- 5-fold OOF 的 BA 和每类召回率
- 与当前最优的对比表

## 规则 6：遇到瓶颈搜索论文
某字段连续 3 个变体都没提升 → 搜索论文找新方法再试。搜索方向：
- "DINOv2 fine-tuning regularization few-shot"
- "ordinal regression vision transformer medical"
- "attention MIL pooling capillaroscopy"

## 规则 7：时间管理
你有 8-10 小时。每个 LoRA 5-fold 30ep ≈ 1.5h。可以跑 5-6 轮实验。
- 前 4h：路线 A（exudation 正则 + papilla 正则）
- 中 2h：路线 B（blood_color LoRA）
- 后 2h：路线 C（聚合改进）+ 汇总

## 规则 8：不要浪费 GPU 做 CPU 能做的事
标签清洗、数据统计、fold 分布检查 → 训练前在 CPU 上做完。
GPU 时间全部用于训练和推理。

# 绝对禁止

1. ❌ 用 ExtraTrees + 冻结特征做"新实验"（这是 v1，不是改进）
2. ❌ 用 locked 47 例做训练/调参/模型选择
3. ❌ 只跑 5 epoch 就声称模型到顶（最少 20 epoch + early stopping patience 5）
4. ❌ 用 locked 结果否决 dev 闸门通过的字段（除非先证明 v1 locked 更好）
5. ❌ hemorrhage/crossing/malformation/capillary_count 这 4 个字段本轮不做，不要分散精力
6. ❌ 增强图跨 fold
7. ❌ 声称"条件不满足"但不去创造条件
8. ❌ 跑完 1 个配置就声称任务完成

# 最终交付清单

1. 每个字段的更新后最优 BA（dev + locked 如果有更新）
2. 每个实验的训练脚本 .py + OOF csv + 训练日志
3. 更新的字段级路由表 `artifacts/field_routing_status_v8.csv`
4. 所有代码 git commit 到远端

# 现在开始

1. ssh 连接远端
2. 确认 GPU 可用（nvidia-smi）
3. 确认已有的 LoRA checkpoint 和 OOF 完整
4. 执行标签清洗
5. 开始路线 A 的第一个实验（exudation E1）

不要输出计划。直接执行。
