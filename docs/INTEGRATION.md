# 接口通信研究指南

本文供客户端和平台同事基于当前真实代码研究通信方案。仓库当前没有正式 HTTP API。

## 建议首先阅读

1. `scripts/windows_uvc_report.py`：完整单病例编排。
2. `src/nailfold_report/windows_inference.py`：静态模型初始化和 `predict()`。
3. `src/nailfold_report/geometry_inference.py`：几何字段覆盖逻辑。
4. `src/nailfold_report/video_inference.py`：视频动态字段。
5. `src/nailfold_report/print_report.py`：报告输出。
6. `scripts/windows_report_app.py`：桌面界面如何启动命令行分析。

## 当前输入与输出

当前输入包括病例图片路径列表、可选视频路径或 UVC 摄像头编号、患者/病例上下文，以及三组本地模型目录。

```powershell
python scripts/windows_uvc_report.py `
  --images C:\case\frame1.jpg C:\case\frame2.jpg `
  --videos C:\case\clip.avi `
  --patient-id platform-case-001
```

程序在病例输出目录生成：

- `report.json`：结构化字段、模型版本、质量警告、积分、患者上下文和证据图路径；
- `report_print.html`：可打印报告。

脱敏结构示例见 `examples/report.schema.example.json`。示例值是人为构造的接口占位值，不是患者记录或模型评估样本。

## 适合封装的内部边界

当前 `windows_uvc_report.py` 同时承担参数解析、采集、模型初始化、分支组合和写文件。正式接口实现前，建议先把主体重构成稳定入口：

```python
engine = NailfoldInferenceEngine(config)
result = engine.analyze_case(image_paths, video_paths, patient_context)
```

通信层只负责任务、鉴权、状态和序列化，不应复制模型融合与积分逻辑。

## 需要共同决定的通信架构

### 方案 A：浏览器直接调用本机服务

```text
Browser -> 127.0.0.1 local service -> inference engine
```

需要验证 HTTPS 页面访问本地服务、CORS、Private Network Access、来源白名单、本机鉴权和服务启动检测。

### 方案 B：Windows 客户端主动连接云端

```text
Browser -> cloud task API
Windows client -> pull task -> local inference -> upload result
```

该方案避免浏览器直接访问 localhost，但需要云端任务队列、客户端身份认证和任务关联。

## 若采用本地 HTTP，建议的契约

单病例 CPU 推理可能持续几十秒，应使用异步任务，而不是长时间同步请求：

```text
GET  /health
POST /v1/tasks
GET  /v1/tasks/{task_id}
GET  /v1/tasks/{task_id}/result
GET  /v1/tasks/{task_id}/report
```

这只是建议契约，不代表仓库已经实现这些端点。

## 安全要求

- 原始图片和视频默认留在客户端。
- 不允许远程调用方提交任意本地路径。
- 本地服务默认只监听 `127.0.0.1`，不能开放到 `0.0.0.0`。
- 不使用通配 CORS；限制业务平台来源并配置本机认证。
- 云端只接收获批字段、报告及必要证据图。
- HTTP 成功只表示计算成功，不表示临床结果可靠；必须保留 `quality.needs_review` 和字段 `status`。
