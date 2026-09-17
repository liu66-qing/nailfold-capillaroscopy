# Field-Group Experiment Inventory

Generated from development artifacts only. All cited OOF artifacts require `locked_cases_seen=0`; the historical 47-case locked cohort is not used for selection.

## Protocol boundary

- Formal weak-field comparator: `current_dual_seg_binary_v1`, 186 development cases, fixed five case folds, DINOv2 + HuluMed + segmentation features, ExtraTrees (300 trees, leaf=2, balanced), pooled OOF balanced accuracy.
- Historical multi-class and step12 values are diagnostic context only and are not numerically interchangeable with the formal binary comparator.
- Numeric geometry scores are `1 - MAE / field_range`; compatibility scores are default-value hit rates. Neither is classification balanced accuracy.
- A simple frame-after-aggregation pipeline is not called MIL. MIL is reserved for explicit case-bag supervision.

| 组 | 字段 | 当前最佳可比结果 | 已尝试方向 | 判定与下一步 |
|---|---|---|---|---|
| A 整体质量与颜色 | clarity; blood_color | 正式二级 dual_seg OOF BA: clarity 69.5%; blood_color 66.2% | 三等级 SigLIP/GBT; DINOv2/HuluMed/dual/dual_seg; robust mean/std/median; video-Huber与帧选择; case-mean latent; 严格域适应; MIL; geometry 拼接; LoRA; TTA; 阈值/类别权重 | 低分/中间组。二级定义已保留，但未达到稳定 70% BA；停止泛化模型搜索，优先做标签审查、颜色/曝光可测性门控。 |
| B 管袢实例与形态 | capillary_count; crossing_ratio; malformation_ratio | count 专用 GBT 三分类 BA 52.9%; step12_r7 crossing 34.5%, malformation 34.0% | YOLO 多类分割(v1/plus-pseudo); 阈值扫描; no-expand/skeleton; count GBT; mask统计; DINO/Hulu/seg分类; MIL/视频质量/增强; 交叉与畸形比例由结构特征路由 | 真正低分组。计数仍受漏检/碎片和病例级区间标签影响；形态比例缺实例级真值。只允许后处理/分割改动通过全字段回归门。 |
| C 单袢物理几何 | afferent_diameter; efferent_diameter; apex_diameter; loop_length; output_input_ratio | step12_r7 归一化分数: afferent 98.1%; efferent 94.4%; apex 92.6%; length 88.3%; ratio 由管径派生。该分数不是分类准确率。 | geometry-v2 mask/centerline/scale; visual/structured/all ExtraTrees/Ridge; Hulu/SigLIP; video增强路由。geometry_exact MAE: afferent 5.86, efferent 4.69, apex 7.79, length 92.41 | 高分保护组（按综合归一化指标），冻结为回归保护；不能把 88--98% 归一化分数解释成医生一致性或绝对准确率。后续只在有结构金标准和设备标定时改。 |
| D 局部病灶 | exudation; hemorrhage | 正式二级 exudation BA 68.4%; hemorrhage 旧三等级 BA 47.3%，阳性极少。 | 四级/二级合并; SigLIP/Hulu/DINO/dual_seg; MIL与固定帧; Qwen/QLoRA; geometry/增强; 阈值与置信权重 | exudation 为低/中间组，二级可交付但需复核；hemorrhage 低支持探索性/异常触发，不得用 94% 多数类普通准确率宣称学会。 |
| E 背景组织结构 | subpapillary_venous_plexus; papilla; sweat_duct | 正式二级 SVP BA 69.9%; papilla 探索性 BA 56.4%; sweat_duct 默认命中率 98.8%（非模型能力）。 | 多等级与二级合并; DINO/Hulu/dual_seg; robust pooling/MIL/video-Huber; case-mean latent; 域适应; classifier chain; Qwen质量选择; MedSAM/SigLIP2筛选 | SVP 是中间组；papilla 低分且探索性；汗腺导管采用默认+异常触发。不要把背景字段与几何组共享模型或目标。 |
| F 视频动态与细胞事件 | flow_state; vasomotion; rbc_aggregation; wbc_count; microthrombus; flow_speed | step11 flow_state BA 20.2%，未过 0.35 门；fallback flow_state 普通命中 34.1%、RBC 42.5%。vasomotion/WBC 为默认兼容率约 98%，flow_speed 无有效标签。 | 连续视频光流/Hulu特征; stable segment; K8/K12 video-Huber; Qwen候选帧; dynamic fallback | 低分且视频覆盖受限（95/186 开发病例）。动态字段必须保持视频证据和观察时长；当前只可兼容性降级，不能当静态分类问题继续调参。 |

## Freeze / reopen policy

- Freeze Group C geometry routes as protected regression components; any segmentation change must report all four geometry MAEs and stop on predeclared degradation.
- Treat compatibility defaults (sweat duct, vasomotion, WBC, flow speed) as product policy with anomaly/review triggers, not evidence of learned recognition.
- Reopen only Groups B, D, E and F with group-specific evidence: instance masks/centerlines for B/C, lesion ROI evidence for D, background-region/evaluability labels for E, and continuous timestamped clips for F.
- For Groups A/D/E, the next defensible no-new-doctor-label work is calibration/evaluability and error-queue review; another global encoder or pooling sweep is not supported by current evidence.
