# 弱字段 OOF 残差审计

## 已证据支持

基于固定 `dual_seg` 二级 ExtraTrees 的 development OOF 逐病例预测：

| 字段 | 有效病例 | 普通准确率 |
|---|---:|---:|
| clarity | 185 | 69.73% |
| blood_color | 174 | 66.67% |
| exudation | 182 | 68.68% |
| SVP | 181 | 70.17% |
| papilla（探索性） | 181 | 67.40% |

- clarity、blood_color、SVP 的有效样本几乎全部属于标签置信度 >=0.95；高置信子集仍未达到高准确率。
- 字段错误共现较低：clarity-blood_color 18 例、clarity-SVP 19 例、blood_color-exudation 22 例、exudation-SVP 18 例。
- fold 间存在波动，但没有单一字段表现出可由低置信度筛选解释的系统性错误。
- `locked_cases_seen=0`，`gpu_used=false`。

## 尚不能判断

- 当前审计不能证明错误完全来自病例级标签；它只排除了“低 OCR 置信度是主要原因”。
- 低错误共现不支持直接做跨字段 stacking，但不能证明字段之间不存在更细粒度的条件关系。

## 下一步判定

优先级转为字段专属概率校准和阈值诊断；不继续扩大 backbone、MIL、geometry 或 self-training 搜索。若阈值校准在严格 OOF 下仍不能达到预设门槛，则接受现有弱字段上限。
