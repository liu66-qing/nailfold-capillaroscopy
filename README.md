# 甲襞微循环结构化报告预测

本工程以“检查目录”为监督单位：一个数字目录中的多张 `CAPorg*.jpg`、零到多个
AVI 视频共同对应一份医生结构化报告。目录不等同于患者；在找回患者 ID 前，所有
划分还必须通过图像、视频和报告指纹隔离疑似同源数据。

## 当前阶段

已完成数据清单、报告自动标签、近重复隔离的病例级 5 折、SigLIP2 多图多任务
训练、视频动态分支、部署预处理适配及 ONNX 导出。当前是可运行的回顾性工程
候选，不是已经通过临床验证的诊断模型。

## 本地运行

```powershell
python scripts/build_manifest.py `
  --data-root "E:\甲劈微循环\data" `
  --output-dir "E:\甲劈微循环\artifacts\manifest"
```

报告图片只用于生成训练标签，绝不能作为模型输入。

## Windows 历史病例推理

安装项目与部署依赖（动态模型必须使用训练时一致的 scikit-learn 1.9）：

```powershell
python -m pip install -e ".[deploy]"
```

使用已有图片和视频：

```powershell
python scripts/windows_uvc_report.py `
  --images "E:\path\CAPorg1.jpg" "E:\path\CAPorg2.jpg" `
  --videos "E:\path\wmv0.avi"
```

默认静态权重是 `artifacts/windows_model_v3`，原图几何模型是
`artifacts/geometry_final_v3`，动态权重是 `artifacts/video_v3_final`。
结果写入 `captures/<时间>/report.json`，并同时生成与医生原报告字段顺序一致的
`report_print.html`，可在浏览器中直接打印或保存 PDF。
低置信分类、图像不足和训练类别支持不足会明确写入 `quality.warnings`；
当前置信度尚未做前瞻校准。

UVC 实时采集：

```powershell
python scripts/windows_uvc_report.py --camera 0 --seconds 20
```

## Windows 桌面界面和打印

双击 [启动甲襞分析系统.bat](启动甲襞分析系统.bat)，可完成：

- 填写姓名、病例号、性别、年龄和检查手指；
- 选择已有病例图片/视频，或从 UVC 摄像头采集；
- 查看20项测量结果、单位、置信度、状态和四组积分；
- 预览与医生原报告一致的“测量项目—测量值—单位—正常值—积分”表格；
- 调用浏览器打印，或保存为 A4 PDF。

也可把已有 `report.json` 单独转成打印页：

```powershell
python scripts/render_print_report.py "captures\病例目录\report.json"
```

当前仅验证了软件链和历史 AVI；最终 USB Type-C/UVC 设备的倍率、颜色、尺度、
帧率与断连恢复仍必须在目标硬件上验收。

## 效果状态

最终候选已完成字段级多源置信标签、共享＋字段独立 attention MIL、几何字段路由、
视频轻量集成、确定性积分和 47 例算法锁定集。锁定集未达到医生级门禁，当前状态
为研发候选 / No-Go，不能用于临床诊断或宣称替代医生。具体指标和数据边界见
`任务定义.md` 的最新执行结论。
