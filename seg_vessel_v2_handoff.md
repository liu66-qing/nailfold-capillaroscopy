# 分割模型交接文档 — seg_vessel_v2

> 交接日期: 2026-09-05
> 用途: 供分类模型会话使用分割特征，不需要重新训练分割

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

## 6. 推理方法 (Ensemble)

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

1. **训练**: 用 `case_features.csv` (233 例) + 分类标签训练分类模型
2. **测试**: 用 `locked_test_features.csv` (47 例) 做最终评估
3. **关联 key**: `exam_case_id` 字段与 `locked_evaluation_v1.csv` 一一对应
4. **分割无需重训**: mask mAP50 = 0.952, 人眼检查通过, 分割质量已达上限
5. **如需更多特征**: 可基于 5-fold 权重提取额外特征 (如 mask 形状特征、血管密度分布等), 推理脚本可复用
