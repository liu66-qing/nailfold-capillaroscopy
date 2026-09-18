# 分割模型交接文档 — seg_vessel_v2

> 交接日期: 2026-09-05
> 用途: 供分类模型会话使用分割特征，不需要重新训练分割

---

> ## ⚠️ 更正声明（2026-09-18 追加，请先读这一节）
>
> 本文档的分割指标是真实的，但**三条对接结论已被实验否证**。
> 依据：`artifacts/annotations/v2/SEG_INSTANCE_RECOVERY_20260918.md`
> （5 折权重折匹配重推 1708 张 dev 图，27542 实例，209 维实例级几何）。
>
> **① §9.4「分割质量已达上限、无需重训」——不成立。**
> mask mAP50 = 0.952 只在**已标注实例**上评分。分割器学到的是典型发夹形毛细血管，
> 真正畸形的血管被漏检或切碎，而它们**不在标注里**，因此不进入召回的分母。
> 实测漏检偏差：检出实例数随畸形等级 0→1→2→3 为 17.8 / 17.4 / 12.7 / 14.4；
> `frac_low_circ` vs 畸形等级 rho = −0.265 (p=0.0009)，`circularity_mean` rho = +0.305
> ——**畸形越重，检出的血管越圆越规则**，相关符号与标签相反。
> 0.952 度量的是"标了的都找到了"，不是"该找的都找到了"。
>
> **② §9.1「用 `case_features.csv` (233 例) 训练分类模型」——禁止执行。**
> 该文件含 **47 例 locked_test**。照此执行即在最终测试集上训练。
> 正确做法：只用 `evaluation_role == "development"` 的 186 例，
> 且只用 `development_fold` 划分折（NaN 即 locked47），不要用 `split` 列。
>
> **③ §5.3 的 8~13 个标量不足以支撑 `crossing_ratio` / `malformation_ratio`。**
> 前者定义在血管间拓扑上，后者定义在单血管形状上；`area_mean` / `conf_mean`
> 这类病例级均值把二者都抹掉了。已按实例级几何恢复该信息并在原尺子下重测，
> 三个结构字段仍全部不可用：
> crossing_ratio +0.006 CI[−0.062,+0.078]、malformation_ratio −0.006 CI[−0.098,+0.080]、
> capillary_count +0.028 CI[−0.044,+0.104]。
> 原因是上述 ① 的上游漏检，**不能靠下游建模、特征工程或集成方式解决**；
> 唯一出路是补标畸形/交叉血管并重训分割器（数据采集任务）。
>
> **④ §6 的集成方法有泄漏。** 「5 个 fold 模型各跑一次、取检测数中位数的 fold」
> 对某张图而言，5 个权重里有 4 个是**训练过这张图**的。
> 正确规则：病例 `development_fold == k` 只能用 `seg_vessel_fold{k}` 推理
> （该折是它的验证集）。本项目后续产出均按此规则执行。
>
> 另：§7、§8 的路径前缀是旧服务器的 `/root/nailfold/`；新服务器为
> `/root/autodl-tmp/nailfold/`。迁移时 5 个 `seg_vessel_fold*/weights/` 目录为空，
> 已于 2026-09-18 从旧服务器逐字节恢复（sha256 见上述报告与
> `artifacts/features/seg_instance_v1/meta.json`）。

---

## 1. 模型概况

| 项目 | 值 |
|---|---|
| 模型架构 | YOLO11m-seg (Ultralytics) |
| 任务 | 甲襞毛细血管实例分割 (单类: vessel) |
| 训练方案 | 5-fold 交叉验证, case-level fold 隔离 |
| 训练数据 | 2121 张 labelme 标注裁剪图 + 321 张 Roboflow 图 = 2442 张 |
| 数据治理 | 完整链路记录于 `artifacts/seg_data_governance_approval.json` |

## 2. 分割质量 (5-fold 交叉验证)

| 指标 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Mean ± Std** |
|---|---|---|---|---|---|---|
| mask mAP50 | 0.952 | 0.965 | 0.956 | 0.926 | 0.962 | **0.9523 ± 0.014** |
| mask mAP50-95 | 0.959 | 0.966 | 0.955 | 0.916 | 0.971 | **0.9535 ± 0.019** |
| mask Recall | 0.943 | 0.955 | 0.927 | 0.924 | 0.946 | **0.939 ± 0.012** |
| box mAP50 | 0.973 | 0.997 | 0.992 | 0.965 | 0.986 | **0.9825 ± 0.012** |

**对比基线**: v1 (Roboflow 257 张) mask mAP50 = 0.697 → v2 = 0.952, **提升 +0.255**

## 3. 10 例人眼抽检结果 (locked test, seed=42)

全部 10 例视觉检查通过:
- 高密度场景 (30-33 det): mask 贴合血管轮廓, 无明显漏检
- 中密度场景 (12-16 det): 边缘血管也能检出, 形态还原准确
- 低密度场景 (3-8 det): 无幻觉检测, 不会"凑数"
- 可视化图存放于 `artifacts/seg_vessel_v2_visual_check/` (10 张 JPG)

## 4. Locked Test 推理统计 (47 例, 402 张图)

| 指标 | 值 |
|---|---|
| 案例数 | 47 |
| 图片数 | 402 |
| 平均检测数/图 | 18.01 |
| 中位检测数/图 | 16.17 |
| 检测数标准差 | 8.20 |
| 平均置信度 | 0.478 |
| 中位置信度 | 0.480 |
| 平均 mask 面积 (px) | 1105.8 |
| 零检测案例数 | 0 |

**Dev vs Locked Test 一致性**: Dev 集 (233 例) 平均检测数 18.67, 置信度 0.466 — 与 locked test 分布一致, 无退化。

## 5. 特征文件 (分类模型直接使用)

### 5.1 开发集特征 (用于训练分类模型)

- **路径**: `/root/nailfold/artifacts/features/seg_vessel_v2/case_features.csv`
- **行数**: 233 例 (与 `locked_evaluation_v1.csv` 中 evaluation_role="development" 对应)

### 5.2 锁定测试集特征 (用于最终评估)

- **路径**: `/root/nailfold/artifacts/features/seg_vessel_v2/locked_test_features.csv`
- **行数**: 47 例 (与 `locked_evaluation_v1.csv` 中 evaluation_role="locked_test" 对应)

### 5.3 字段定义

| 字段 | 含义 |
|---|---|
| `exam_case_id` | 案例 ID, 与 `locked_evaluation_v1.csv` 关联的 key |
| `n_images` | 该案例图片数 |
| `count_mean` | 每张图平均检测血管数 |
| `count_total` | 该案例总检测血管数 |
| `conf_mean` | 置信度均值 |
| `conf_median` | 置信度中位数 |
| `area_mean` | mask 面积均值 (像素) |
| `area_median` | mask 面积中位数 (像素) |

> ⚠️ 这些是**病例级标量**。实例级 mask 在此处被丢弃，`crossing_ratio`（血管间拓扑）
> 与 `malformation_ratio`（单血管形状）所依赖的量不在其中。
> 实例级几何已另行提取：`artifacts/features/seg_instance_v1/case_geometry.csv`（209 维），
> 但字段仍不可用，根因见开头更正声明 ①。

## 6. 推理方法 (Ensemble)

> ⚠️ **下面这个方法对 dev 集有泄漏，已弃用。** 对某张 dev 图，5 个权重里有 4 个
> 训练过这张图，"取中位数 fold"会选中其中之一。
> 正确规则见开头更正声明 ④：`development_fold == k` 的病例只用 `seg_vessel_fold{k}`。
> （对 locked_test 47 例无此问题——5 折都没训练过它们。）

```
对每张图片:
  1. 5 个 fold 模型各跑一次推理 (conf=0.25)
  2. 取每个 fold 的检测数量
  3. 选中位数检测数量对应的 fold 作为该图代表结果
  4. 按 case 聚合: mean / median / total
```

## 7. 模型权重路径

```
/root/nailfold/artifacts/models/seg_vessel_fold0/weights/best.pt
/root/nailfold/artifacts/models/seg_vessel_fold1/weights/best.pt
/root/nailfold/artifacts/models/seg_vessel_fold2/weights/best.pt
/root/nailfold/artifacts/models/seg_vessel_fold3/weights/best.pt
/root/nailfold/artifacts/models/seg_vessel_fold4/weights/best.pt
```

## 8. 关键关联文件

| 文件 | 用途 |
|---|---|
| `artifacts/manifest/locked_evaluation_v1.csv` | 233 例角色划分 (186 dev + 47 locked_test) |
| `artifacts/manifest/stratified_folds_v3.csv` | 5-fold case-level 分配 |
| `artifacts/seg_vessel_v2_evaluation_report.json` | 完整评估报告 (机器可读) |
| `artifacts/features/seg_vessel_v2/locked_test_inference.json` | 47 例推理详情 |

## 9. 分类模型对接要点

> ⚠️ **本节第 1 条和第 4 条已被否证，第 5 条已试过且无效。** 详见开头更正声明。
> 保留原文以便追溯误导来源。

1. ~~**训练**: 用 `case_features.csv` (233 例) + 分类标签训练分类模型~~
   → **禁止**。该文件含 47 例 locked_test，照做即在最终测试集上训练。
   只用 `evaluation_role == "development"` 的 186 例，只用 `development_fold` 划折。
2. **测试**: 用 `locked_test_features.csv` (47 例) 做最终评估
   → 仍然有效，但 locked-47 只允许做**一次**最终评估，不得用于调参、选模型或反复比较。
3. **关联 key**: `exam_case_id` 字段与 `locked_evaluation_v1.csv` 一一对应
4. ~~**分割无需重训**: mask mAP50 = 0.952, 人眼检查通过, 分割质量已达上限~~
   → **不成立**。mAP 只覆盖已标注实例，对畸形血管的系统性漏检不进入该指标。
   实测偏差见更正声明 ①。要提升 `crossing_ratio` / `malformation_ratio`，
   **必须**补标畸形与交叉血管并重训分割器。
5. ~~**如需更多特征**: 可基于 5-fold 权重提取额外特征 (如 mask 形状特征、血管密度分布等)~~
   → **已于 2026-09-18 执行完毕**：`scripts/extract_instance_geometry.py` 提取了
   27542 个实例的形状与拓扑几何（209 维病例级特征）。三个结构字段在原尺子下
   仍全部不可用。提特征这条路已走完，瓶颈不在特征维度。
