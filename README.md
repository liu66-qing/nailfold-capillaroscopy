# Nailfold Capillaroscopy Research Pipeline

甲襞图像/视频到医生式结构化报告的研发代码。项目以一次检查为监督单位：同一检查目录中的多张静态图、可选视频与一份历史医生结构化报告共同构成训练病例。

> 当前是可运行的回顾性研发候选，不是经临床验证的诊断模型，不能宣称替代医生。

## 当前已经实现

- 医生报告 OCR/RTF 多源标签解析、规范化和字段级置信度；
- 重复/近重复隔离的病例级划分；
- SigLIP2 多图多任务 Attention MIL；
- 管径和几何字段的自动视觉分支；
- 视频帧特征、光流和轻量动态模型；
- Windows ONNX 推理与预处理一致性验证；
- USB/UVC采集、病例图片/视频导入；
- 20项结构化字段、确定性积分、质量警告；
- 医生式 HTML/A4 打印报告和 Windows 桌面原型。

当前尚未实现正式本地 HTTP API、Windows Service、浏览器直连协议和云端任务同步。接口同事可从真实调用链和输出契约开始研究，详见 [接口通信研究指南](docs/INTEGRATION.md)。

## 快速理解代码

```text
启动甲襞分析系统.bat
  -> scripts/windows_report_app.py
  -> scripts/windows_uvc_report.py
  -> static + geometry + video + scoring
  -> report.json + report_print.html
```

- [架构与模块职责](docs/ARCHITECTURE.md)
- [接口通信研究指南](docs/INTEGRATION.md)
- [数据隐私边界](docs/DATA_PRIVACY.md)
- [脱敏输出结构示例](examples/report.schema.example.json)

## 安装

```powershell
python -m pip install -e ".[dev]"
pytest
```

Windows 推理依赖：

```powershell
python -m pip install -e ".[deploy]"
```

模型权重和私有数据不在仓库中。经授权的开发者需单独取得模型资产并放入 `artifacts/windows_model_v3`、`artifacts/geometry_final_v3` 和 `artifacts/video_v3_final`。

## 当前推理入口

```powershell
python scripts/windows_uvc_report.py `
  --images C:\case\frame1.jpg C:\case\frame2.jpg `
  --videos C:\case\clip.avi `
  --patient-id case-001
```

UVC实时采集：

```powershell
python scripts/windows_uvc_report.py --camera 0 --seconds 20
```

桌面原型：

```powershell
.\启动甲襞分析系统.bat
```

## 数据和模型边界

报告图片只用于生成训练标签，绝不能作为视觉模型输入。仓库不包含患者数据、医生报告、派生标签、训练划分、模型权重、日志、服务器信息和内部业务资料。

## 代码来源说明

训练和标签处理主体与服务器 `/root/autodl-tmp/nailfold/code_latest` 的已核对源码一致；Windows几何、视频、桌面和打印模块来自随后同步回本地的最终 v3 工程交付版本。仓库没有为了展示完整度而增加未落地的 HTTP 服务实现。

## License

采用仓库已有的 MIT License。研究原型不构成医疗器械或诊断建议。
