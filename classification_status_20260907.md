# 甲襞微循环分类模型 — 全字段状态总结

> 更新日期: 2026-09-07

---

## 一、任务定义

5 个二分类字段，从甲襞显微镜图像自动判别：

| 字段 | 正类(1) | 负类(0) |
|------|---------|---------|
| clarity（清晰度） | 不清/模糊 | 清晰 |
| blood_color（血色） | 浅红/淡红 | 暗红/暗紫 |
| exudation（渗出） | +/++/+++ | 无 |
| SVP（乳头下静脉丛） | 可见1排/可见2排/>2排扩张 | 不见 |
| papilla（乳头） | 浅波纹状/波纹状 | 平坦 |

---

## 二、数据

| 项目 | 数值 |
|------|------|
| 开发集 | 186 cases, 5-fold CV |
| 锁定测试集 | 47 cases（仅最终评估，禁止调参） |
| 每 case 帧数 | ~9 帧 (1024×768) |
| 总帧数 | ~2207 |
| 标签文件 | `/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv` |
| 标签状态 | 147/186 cases 人工审核, 173 处修正 |

---

## 三、模型架构

- **Backbone**: DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`), input 518×518
- **预训练权重**: `/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth` (facebook official, 331MB)
- **分类头**: 每字段独立 `nn.Linear(768, 2)`
- **帧→case 聚合**: mean logits → softmax → threshold 0.5

---

## 四、各阶段结果对比

### 4.1 Baseline v2（LoRA, cherry-pick per field, 单 seed）

> 脚本: `/root/nailfold/scripts/retrain_reviewed.sh`
> Checkpoints: `/root/nailfold/artifacts/experiments/retrain_reviewed/{rank8, rank4_lr1e4, rank16}/`
> 记录: `/root/nailfold/artifacts/baseline_v2_reviewed_20260905.json`

| 字段 | Dev BA (cherry-pick) | 选用模型 | Locked BA |
|------|---------------------|----------|-----------|
| clarity | **0.837** | rank8 | 0.727 |
| blood_color | **0.739** | rank8 | 0.772 |
| exudation | **0.799** | rank4_lr1e4 | 0.601 |
| SVP | **0.862** | rank4_lr1e4 | 0.732 |
| papilla | **0.627** | rank16 | 0.470 |
| **Mean** | **0.773** | — | **0.660** |

**Dev-Locked gap = 0.113**，确认为纯过拟合（domain shift AUC=0.518，无分布偏移）。

Locked 结果记录: `/root/nailfold/artifacts/locked_test_v2_20260905.json`
路由表: `/root/nailfold/artifacts/field_routing_status_v2.csv`

### 4.2 Progressive Complexity 实验（当前最新，公平对比）

> 脚本: `/tmp/progressive_complexity.py`（本地备份 `E:\甲劈微循环\progressive_complexity.py`）
> 输出: `/root/nailfold/artifacts/experiments/progressive_complexity/`
> 设计: 4 configs × 5 seeds × 5 folds × 20 epochs, case-level loss, 全 5 字段参与 val objective

| Config | clarity | blood_color | exudation | SVP | papilla | **Mean BA** |
|--------|---------|-------------|-----------|-----|---------|-------------|
| **A_frozen** (无LoRA, lr=1e-3) | 0.789 | 0.758 | 0.732 | 0.794 | 0.604 | **0.736** |
| B_lora_r2_b2 (rank2, 2 blocks) | 0.790 | 0.694 | 0.704 | 0.770 | 0.611 | 0.714 |
| C_lora_r4_b4 (rank4, 4 blocks) | 0.774 | 0.709 | 0.720 | 0.781 | 0.597 | 0.716 |
| D_lora_r8_b4 (rank8, 4 blocks) | 0.758 | 0.721 | 0.743 | 0.772 | 0.601 | 0.719 |

以上均为 **5-seed ensemble, dev OOF balanced accuracy**（无 cherry-pick，无信息泄漏）。

---

## 五、核心结论

1. **Frozen DINOv2 + linear heads 是当前最优配置**。LoRA 微调在 186 cases 的数据量下全部过拟合。
2. **公平评估下 dev BA=0.736**（vs cherry-pick 的 0.773），预计 locked gap 大幅缩小。
3. **Papilla 是瓶颈**（~0.60 BA），原因是标签主观模糊（平坦 vs 浅波纹状），非模型能力问题。
4. **Baseline v2 的 locked BA=0.660 可视为模型真实泛化水平的下界**（因为 cherry-pick + 单 seed 有选择偏差）。

---

## 六、已尝试但无效的方向

| 方向 | 结论 |
|------|------|
| LoRA 超参搜索 (rank 2/4/8/16, 2/4 blocks) | 全部过拟合，不如 frozen |
| 分割特征 (YOLO11m-seg, mAP50=0.952) | 几何统计与语义分类无关，BA 0.49-0.67 |
| 表格学习 (GBT stacking) | 无提升 |
| 多 seed 集成 (5 seeds) | 比单 seed 提升 1-2%，已纳入 |
| Case-level loss (0.75 case + 0.25 frame) | 比纯 frame CE 略好，已纳入 |
| Cherry-pick per field | 造成选择偏差，dev-locked gap 膨胀 |

---

## 七、服务器关键路径

```
/root/nailfold/                              # → 符号链接到 /root/autodl-tmp/nailfold/
├── artifacts/
│   ├── manifest/
│   │   └── locked_evaluation_v1_reviewed.csv  # 标签文件（ground truth）
│   ├── baseline_v2_reviewed_20260905.json     # Baseline v2 完整记录
│   ├── locked_test_v2_20260905.json           # Locked test 评估（禁止用于调参）
│   ├── field_routing_status_v2.csv            # 路由表 v2
│   ├── features/
│   │   └── image_index.csv                    # 帧索引
│   └── experiments/
│       ├── retrain_reviewed/                  # Baseline v2 checkpoints
│       │   ├── rank8/fold{0-4}.pt + metrics.json
│       │   ├── rank4_lr1e4/fold{0-4}.pt + metrics.json
│       │   └── rank16/fold{0-4}.pt + metrics.json
│       └── progressive_complexity/            # 最新实验
│           ├── A_frozen_results.json
│           ├── B_lora_r2_b2_results.json
│           ├── C_lora_r4_b4_results.json
│           ├── D_lora_r8_b4_results.json
│           └── summary.json
├── scripts/
│   ├── retrain_reviewed.sh                    # Baseline v2 复现脚本
│   └── finetune_dinov2_lora_rank8_lr1e4_cv.py # 原始训练脚本
├── data/                                      # 原始图像
└── ...

/tmp/progressive_complexity.py                 # Progressive complexity 实验脚本

/root/autodl-tmp/nailfold/Model/DINOv2-base/
└── dinov2_vitb14_pretrain.pth                 # 预训练权重
```

---

## 八、分割模型 (seg_vessel_v2)

### 8.1 模型概况

| 项目 | 值 |
|------|------|
| 架构 | YOLO11m-seg (Ultralytics) |
| 任务 | 甲襞毛细血管实例分割 (单类: vessel) |
| 训练数据 | 2121 张 labelme 标注 + 321 张 Roboflow = 2442 张 |
| 训练方案 | 5-fold CV, case-level fold 隔离 |
| 数据治理 | `artifacts/seg_data_governance_approval.json` |

### 8.2 分割质量 (5-fold CV)

| 指标 | Mean ± Std |
|------|------------|
| mask mAP50 | **0.952 ± 0.014** |
| mask mAP50-95 | 0.954 ± 0.019 |
| mask Recall | 0.939 ± 0.012 |
| box mAP50 | 0.983 ± 0.012 |

对比 v1 (Roboflow 257 张): mask mAP50 0.697 → 0.952, **+0.255**

10 例人眼抽检全部通过（高/中/低密度场景均准确，无幻觉检测）。

### 8.3 分割特征对分类的作用

已验证**分割特征对分类无帮助**：
- 分割特征 (count, conf, area 等几何统计) 单独用于分类: BA 0.49-0.67
- 与 LoRA 概率 stacking: 4/5 字段反而下降
- 原因: 5 个分类字段是语义/视觉属性 (颜色、清晰度、形态)，与血管几何统计无关

### 8.4 关键文件

| 文件 | 说明 |
|------|------|
| 权重 | `/root/nailfold/artifacts/models/seg_vessel_fold{0-4}/weights/best.pt` |
| Dev 特征 | `/root/nailfold/artifacts/features/seg_vessel_v2/case_features.csv` (233 例) |
| Locked 特征 | `/root/nailfold/artifacts/features/seg_vessel_v2/locked_test_features.csv` (47 例) |
| 评估报告 | `/root/nailfold/artifacts/seg_vessel_v2_evaluation_report.json` |
| 推理详情 | `/root/nailfold/artifacts/features/seg_vessel_v2/locked_test_inference.json` |
| 交接文档 | `E:\甲劈微循环\seg_vessel_v2_handoff.md` |

### 8.5 推理方法 (Ensemble)

每张图 5 个 fold 模型各跑一次 (conf=0.25)，取中位检测数对应的 fold 作为代表，再按 case 聚合 mean/median/total。

### 8.6 结论

分割模型本身质量优秀 (mAP50=0.952)，**无需重训**。但其提取的几何特征对当前 5 个语义分类字段无增益。未来如新增几何相关字段 (如血管密度、管径异常等) 可直接复用。

---

## 九、下一步方向

1. **接受 frozen A 为正式 baseline**，mean BA=0.736
2. **Papilla 字段改为二级/可拒答定义**，减少标签噪声
3. 探索: patch-level features (非 CLS), attention MIL pooling, 医学预训练 backbone (UNI/CONCH)
4. 产品层面: 加入 calibration + 不确定性拒答机制
