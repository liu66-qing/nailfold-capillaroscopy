# 当前架构与模块职责

本仓库呈现当前已经实现的研发原型，不包含尚未实现的 HTTP 服务。

```text
启动甲襞分析系统.bat
        |
scripts/windows_report_app.py        Windows 桌面界面
        |
scripts/windows_uvc_report.py        单病例编排与 UVC 采集
        |
        +-- WindowsCasePredictor     静态多图 ONNX 推理
        +-- GeometryPredictor        管径和几何字段路由
        +-- DynamicPredictor         视频动态字段推理
        +-- deterministic_scores     确定性积分规则
        +-- render_report_html       HTML/A4 打印报告
        |
report.json + report_print.html
```

## 主要目录

| 路径 | 当前职责 |
|---|---|
| `src/nailfold_report/windows_inference.py` | 图像预处理、质量排序、SigLIP2 ONNX、多图病例头、置信状态和积分 |
| `src/nailfold_report/geometry_inference.py` | 血管几何特征及数值字段预测 |
| `src/nailfold_report/video_inference.py` | 视频解码、帧特征、光流和动态字段模型 |
| `src/nailfold_report/print_report.py` | 固定模板的结构化 HTML 报告 |
| `src/nailfold_report/data/` | 病例清单、文件指纹和报告版面定位 |
| `src/nailfold_report/labels/` | 标签字段、归一化和多源共识 |
| `scripts/windows_uvc_report.py` | 当前最完整的命令行推理入口 |
| `scripts/windows_report_app.py` | 当前 Windows 桌面入口 |
| `scripts/prepare_*` | 私有数据的离线预处理工具 |
| `scripts/extract_*` | OCR、视觉、几何和视频特征抽取 |
| `scripts/train_*` | 静态、融合、几何和视频模型训练 |
| `scripts/export_*` | Windows ONNX 导出与预处理导出 |
| `scripts/validate_*` | 部署一致性与端到端验证 |
| `tests/` | 不依赖私有数据的单元测试 |

## 模型资产

权重不进入公开 Git 仓库。真实运行需要受控分发：

```text
artifacts/
├─ windows_model_v3/
├─ geometry_final_v3/
└─ video_v3_final/
```

模型资产应通过公司网盘或对象存储分发，并用 SHA-256 清单校验。

## 当前能力边界

- 已实现：Windows 桌面原型、UVC/病例文件输入、静态/几何/视频推理、结构化 JSON、积分和打印报告。
- 尚未实现：正式本地 HTTP API、Windows Service、浏览器直连协议、云端任务同步和产品安装/升级体系。
- 当前是回顾性研发候选，锁定集未达到医生级门禁，不能用于临床诊断。
