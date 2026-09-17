# 附件 B：评测指标定义

## 分类字段主指标
- Balanced Accuracy (BA)
- macro-F1
- 少数类召回
- PR-AUC
- 概率校准误差 (ECE)

## 连续字段主指标
- MAE
- 中位绝对误差
- 临床容差内比例（字段专用容差）
- 预测区间覆盖率

## 拓扑字段附加指标
- 过数率（预测 > 真值的比例）
- 少数率（预测 < 真值的比例）
- 帧间稳定性（同病例多帧预测标准差）
- 图结构分母置信度
- 拒识比例

## 禁止的指标误用
- 归一化误差分数只能作为辅助摘要，不能称为"准确率"
- 兼容性字段的默认值命中率不能称为"视觉识别准确率"
- 不得把目标值当作已实现结果

## 交付物清单
最终必须生成：
- 新模型 checkpoint + 配置 + 训练命令
- `final_training_manifest.csv` / `final_feature_manifest.csv` / `final_pairing_table.csv`
- 模型和数据 SHA256
- development OOF 逐字段结果
- locked 一次性最终评测结果
- 失败实验报告
- 论文和方法依据
- 标签定义 + 标定声明 + 不确定度/拒识策略
- locked 排除声明
- Git commit + 环境版本
- 最终发布 README

字段状态标签：`PASS` / `PASS_AUXILIARY` / `HOLD` / `REPAIR_REQUIRED` / `EXCLUDE_LOCKED` / `EXCLUDE_DUPLICATE`
