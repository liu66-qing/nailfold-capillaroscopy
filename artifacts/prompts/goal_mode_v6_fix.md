
# 修复轮：v6 遗留问题（无 GPU 模式）

当前服务器无 GPU，本轮专注于：审计上一轮结论的正确性、准备全部训练代码和配置、完成所有 CPU 可做的分析。GPU 恢复后一键启动训练。

## 缺陷 1：hemorrhage 三次使用 v1 方法（冻结 DINOv2 均值 + ExtraTrees），从未执行过提示词要求的专家模型

本轮 hemorrhage 实验的代码必须满足以下可验证条件，缺一不可：
- 训练脚本中存在 `focal_loss` 或 `gamma` 参数（不是 cross_entropy）
- predict() 函数中的聚合方式是 top-k（代码中存在 `topk` 或 `argsort` 或 `nlargest`），不是 `np.mean`
- 模型包含 LoRA 层（代码中存在 `LoRALinear`），不是冻结特征 + ExtraTrees
- 训练了 ≥20 epoch（日志中可见 epoch 编号）
- k 至少试了 1、2、3 三个值

如果提交的代码不满足以上任何一条，该实验视为未执行，必须重做。

具体实现（基于已有的 finetune_dinov2_lora_binary_cv.py 修改）：
```python
# 1. 损失函数改为 focal loss
def focal_loss(logits, targets, gamma=2.0, alpha=0.75):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()

# 2. predict() 中的聚合改为 top-k
def predict_topk(model, loader, k=2):
    vals = {f: {} for f in FIELDS}
    model.eval()
    with torch.inference_mode():
        for x, _, ids in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x.cuda())
            for f in FIELDS:
                probs = torch.softmax(o[f].float(), dim=1)[:, 1].cpu().numpy()
                for c, p in zip(ids, probs):
                    vals[f].setdefault(c, []).append(float(p))
    result = {}
    for f, case_probs in vals.items():
        result[f] = {}
        for c, probs in case_probs.items():
            probs_sorted = sorted(probs, reverse=True)
            top_k_mean = np.mean(probs_sorted[:k])
            result[f][c] = int(top_k_mean > 0.5)
    return result

# 3. 对 k=1,2,3 分别跑 OOF，选最优
```

## 缺陷 2：exudation locked BA 0.618 被用来否定 dev BA 0.772，但没有对比 v1 locked

在否定 exudation v6 之前，必须先回答：**v1 在 locked 上的 exudation BA 是多少？**
- 如果 v1 locked exudation BA 也在 0.60-0.65 → v6 并未退化，dev 提升是真实的
- 如果 v1 locked exudation BA > 0.70 → v6 确实泛化不足
- 在这个对比完成之前，不得以 locked 结果否决 dev 闸门通过的字段

执行：读取 v1 的 locked 评测结果（如果存在），输出 v1 vs v6 在 locked 上的逐字段对比表。

## 缺陷 3：capillary_count、crossing_ratio、malformation_ratio 的 DINOv2 LoRA 端到端方案从未执行

上一轮只做了检测框计数代理（crossing Spearman -0.016），失败后没有执行替代方案。本轮必须对这三个字段各执行 DINOv2 LoRA 端到端微调：

capillary_count：
- 标签映射：合并稀有类，>=7 → 0, 5-6 → 1, <=4 → 2（3 类序数分类）
- DINOv2 LoRA rank4, 30 epoch, focal loss
- 5-fold OOF

crossing_ratio：
- 标签映射：合并稀有类，<=30% → 0, 30-60% → 1, >60% → 2（3 类）
- 同上

malformation_ratio：
- 检查类别分布，合并至 3 类
- 同上

这三个字段可以共享 backbone，各自独立 head，一次训练出三个 OOF。

## 缺陷 4：时间线不合理（4 分钟内完成了需要数小时的训练）

本轮每个 DINOv2 LoRA 实验必须在提交时附带：
- 训练日志，包含每个 epoch 的 loss 和 validation BA（至少 20 行）
- GPU 显存使用量（nvidia-smi 输出或 torch.cuda.max_memory_allocated）
- 训练总耗时（wall clock time）
- 如果训练时间 < 30 分钟（5-fold 30 epoch），说明理由

## 缺陷 5：SVP 闸门基线被悄悄从 v1 换成了 v5

字段闸门的对比基线永远是该字段当前正式采用的最优路由：
- 如果某字段已有通过闸门的路由（如 SVP v5 0.751），新实验和该路由比
- 但"3/5 folds 提升"的计算应该和 v1 比，不是和最近一次实验比
- v6 SVP 0.762 vs v1 0.699 = +6.3pp，需要检查是否 ≥3/5 folds vs v1 有提升

重新计算 SVP v6 的闸门：逐折对比 v6 vs v1（不是 v6 vs v5），输出结果。

## 执行清单（按 CPU/GPU 分类）

### A 组：CPU 可做，本轮必须全部完成

- [ ] A1. 查询 v1 locked exudation BA，输出 v1 vs v6 locked 逐字段对比表
      → 读取已有的 v1 locked 评测结果文件，如果不存在则明确说明
      → 这决定了 exudation v6 (dev 0.772) 是否应该被保留

- [ ] A2. 重新计算 SVP v6 vs v1 的逐折闸门
      → 读取 v6 SVP 的逐折 OOF 和 v1 的逐折 OOF，逐折对比 v6 vs v1（不是 v6 vs v5）

- [ ] A3. fold 2 诊断
      → fold 2 在 clarity/blood_color/SVP/papilla 都是最差折
      → 输出 fold 2 vs 其他折的：类别分布、样本数、帧数分布、阳性比例
      → 输出 artifacts/fold_diagnosis.csv

- [ ] A4. hemorrhage 类别和帧级分布分析
      → 17 个阳性病例共有多少帧？帧级阳性样本量是否足够训练 LoRA？
      → 阳性病例的帧质量分布如何？
      → 输出 artifacts/hemorrhage_frame_analysis.csv

- [ ] A5. crossing_ratio / malformation_ratio 标签分布和类别合并方案
      → 输出每个字段的类别计数
      → 确定合并后的 3 类映射
      → 检查合并后每类在每个 fold 中是否 ≥ 3 例

- [ ] A6. capillary_count 检测框相关性诊断
      → 读取 v3 检测特征，计算每病例平均检测框数 vs 临床标签的 Spearman 相关
      → 画散点图（保存为 png）
      → 如果 Spearman < 0.2 → 确认放弃检测计数路线

- [ ] A7. 完整的字段级路由现状表
      → 综合 v1/v3/v5/v6 所有实验，每个字段选历史最优
      → 输出 artifacts/field_routing_status.csv，列：field, best_ba, source, dev_or_locked, fold_wins, gate_status

### B 组：写好训练代码，GPU 恢复后一键运行

- [ ] B1. hemorrhage 专家训练脚本 scripts/train_hemorrhage_expert.py
      必须包含：
      - focal_loss 函数（gamma=2, alpha=0.75）
      - LoRALinear 层（复用 v5 代码）
      - predict_topk 函数（k 作为参数，默认 k=2）
      - 对 k=1,2,3 分别输出 OOF
      - 30 epoch + early stopping (patience=5)
      - 训练日志每 epoch 输出 loss 和 val BA
      - 代码末尾打印总训练时间和 GPU 显存峰值
      脚本必须能通过 python scripts/train_hemorrhage_expert.py --dry-run 验证语法和数据加载（CPU 模式，只跑 1 batch）

- [ ] B2. 形态三字段联合训练脚本 scripts/train_morphology_experts.py
      - capillary_count / crossing_ratio / malformation_ratio 共享 DINOv2 backbone
      - 每个字段独立 head，3 类序数分类
      - 标签映射使用 A5 确定的方案
      - focal loss + class weight
      - 30 epoch + early stopping
      - 5-fold OOF
      - 同样支持 --dry-run

- [ ] B3. GPU 启动脚本 scripts/run_all_experts.sh
      ```bash
      #!/bin/bash
      set -e
      echo "=== hemorrhage expert ==="
      python scripts/train_hemorrhage_expert.py --config configs/hemorrhage.yaml
      echo "=== morphology experts ==="
      python scripts/train_morphology_experts.py --config configs/morphology.yaml
      echo "=== done ==="
      ```
      一个命令启动全部训练，GPU 恢复后直接 `bash scripts/run_all_experts.sh`

- [ ] B4. 对 B1 和 B2 执行 --dry-run 验证
      → CPU 模式下加载数据、构建模型、跑 1 个 batch forward/backward
      → 确认无报错、数据路径正确、标签映射正确、focal loss 可计算
      → 输出 dry-run 日志

### C 组：搜索外部资源（如果时间允许）

- [ ] C1. 搜索 "nailfold capillaroscopy hemorrhage detection" 相关论文和代码
- [ ] C2. 搜索 "few-shot medical lesion detection focal loss top-k MIL"
- [ ] C3. 搜索 "ordinal classification vessel morphology"
- [ ] C4. 将找到的有价值方法记录到 artifacts/literature_notes.md，标注是否有官方代码、是否适用于本项目

## 绝对禁止
- hemorrhage 使用冻结特征 + ExtraTrees（这是 v1，不是专家模型）
- hemorrhage 使用 mean pooling（必须是 top-k）
- 用 locked 结果否决 dev 闸门通过的字段（除非先证明 v1 locked 明显更好）
- 4 个滞后字段只做 1 个就宣布完成
- 声称"无 GPU 所以什么都做不了"（A 组全部是 CPU 任务，B 组是写代码 + dry-run）
- 写完代码不做 dry-run 验证

# 现在开始。先执行 A1（查询 v1 locked exudation BA）。A 组全部完成后开始 B 组。
