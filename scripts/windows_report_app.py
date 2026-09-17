from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


PROJECT = Path(__file__).resolve().parents[1]
REPORT_ROWS = (
    ("clarity", "清晰度", ""), ("capillary_count", "管袢数", "条/mm"),
    ("afferent_diameter", "输入枝管径", "μm"),
    ("efferent_diameter", "输出枝管径", "μm"),
    ("output_input_ratio", "输出/输入枝管径", ""),
    ("apex_diameter", "袢顶直径", "μm"), ("loop_length", "管袢长", "μm"),
    ("crossing_ratio", "交叉管袢数", ""), ("malformation_ratio", "畸形管袢数", ""),
    ("flow_state", "流态", ""), ("vasomotion", "血管运动性", "次/min"),
    ("rbc_aggregation", "红细胞聚集", ""), ("wbc_count", "白细胞数", "个/15s"),
    ("microthrombus", "白微栓", "个/min"), ("blood_color", "血色", ""),
    ("exudation", "渗出", ""), ("hemorrhage", "出血", "管袢/一指甲襞"),
    ("subpapillary_venous_plexus", "乳头下静脉丛", ""),
    ("papilla", "乳头", ""), ("sweat_duct", "汗腺导管", "个/一指甲襞"),
)


def parse_report_path(stdout: str) -> Path:
    """Read the ASCII-safe child-process result protocol, with legacy fallback."""
    for line in reversed(stdout.splitlines()):
        value = line.strip()
        if not value:
            continue
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("report_path"):
            return Path(payload["report_path"])
        if value.lower().endswith("report.json"):
            return Path(value)
    raise RuntimeError("分析完成但没有返回 report.json")


class ReportApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("甲襞微循环智能分析系统（研发版）")
        self.geometry("1120x780")
        self.minsize(980, 680)
        self.report_path: Path | None = None
        self.print_path: Path | None = None
        self._build()

    def _build(self) -> None:
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=27, font=("Microsoft YaHei", 10))
        style.configure("Treeview.Heading", font=("Microsoft YaHei", 10, "bold"))
        header = ttk.Frame(self, padding=10)
        header.pack(fill="x")
        ttk.Label(
            header, text="甲襞微循环智能分析系统", font=("Microsoft YaHei", 19, "bold")
        ).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 8))
        self.entries: dict[str, ttk.Entry] = {}
        for column, (key, label) in enumerate(
            (("name", "姓名"), ("patient_id", "病例号"), ("age", "年龄"), ("finger", "检查手指"))
        ):
            ttk.Label(header, text=f"{label}：").grid(row=1, column=column * 2, sticky="e")
            entry = ttk.Entry(header, width=16)
            entry.grid(row=1, column=column * 2 + 1, padx=(0, 10), sticky="ew")
            self.entries[key] = entry
        ttk.Label(header, text="性别：").grid(row=2, column=0, sticky="e", pady=6)
        self.sex = ttk.Combobox(header, values=("男", "女", "其他"), width=13, state="readonly")
        self.sex.grid(row=2, column=1, sticky="w")
        self.sex.set("")
        actions = ttk.Frame(self, padding=(10, 0, 10, 8))
        actions.pack(fill="x")
        self.folder_button = ttk.Button(actions, text="选择病例图片/视频并分析", command=self.analyze_folder)
        self.folder_button.pack(side="left", padx=(0, 7))
        self.camera_button = ttk.Button(actions, text="从 UVC 摄像头采集并分析", command=self.analyze_camera)
        self.camera_button.pack(side="left", padx=(0, 7))
        self.preview_button = ttk.Button(actions, text="报告预览", command=self.preview, state="disabled")
        self.preview_button.pack(side="left", padx=(0, 7))
        self.print_button = ttk.Button(actions, text="打印 / 保存PDF", command=self.preview, state="disabled")
        self.print_button.pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=220)
        self.progress.pack(side="right")
        self.status = tk.StringVar(value="请选择病例文件夹，或连接 USB Type-C/UVC 摄像头。")
        ttk.Label(self, textvariable=self.status, padding=(10, 3)).pack(fill="x")
        columns = ("item", "value", "unit", "confidence", "status")
        self.table = ttk.Treeview(self, columns=columns, show="headings")
        headings = ("测量项目", "测量值", "单位", "模型置信度", "状态")
        widths = (220, 220, 145, 130, 240)
        for column, heading, width in zip(columns, headings, widths):
            self.table.heading(column, text=heading)
            self.table.column(column, width=width, anchor="center")
        self.table.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.summary = tk.StringVar(value="形态积分：—    流态积分：—    袢周积分：—    总积分：—    综合判断：—")
        ttk.Label(
            self, textvariable=self.summary, font=("Microsoft YaHei", 12, "bold"),
            padding=(10, 8),
        ).pack(fill="x")
        ttk.Label(
            self,
            text="研发辅助结果，不替代医生诊断；低置信度和低样本字段会在报告中明确标记。",
            foreground="#9a4d00", padding=(10, 4),
        ).pack(fill="x")

    def _patient_args(self) -> list[str]:
        values = {key: entry.get().strip() for key, entry in self.entries.items()}
        return [
            "--patient-name", values["name"], "--patient-id", values["patient_id"],
            "--sex", self.sex.get(), "--age", values["age"], "--finger", values["finger"],
        ]

    def analyze_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择包含甲襞图片/视频的病例文件夹")
        if not folder:
            return
        root = Path(folder)
        images = sorted(
            path for path in root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            and not path.name.lower().startswith("rep")
        )
        videos = sorted(
            path for path in root.rglob("*")
            if path.suffix.lower() in {".avi", ".wmv", ".mp4", ".mov", ".mkv"}
        )
        if not images:
            messagebox.showerror(
                "无法分析",
                "所选文件夹中没有可用甲襞图片。请选择包含 CAPorg*.jpg 的单个病例目录。",
            )
            return
        image_parents = {path.parent.resolve() for path in images}
        if len(image_parents) > 1:
            messagebox.showerror(
                "请选择单个病例",
                f"所选目录包含 {len(image_parents)} 个带图片的子目录，不能合并为一个病例。\n\n"
                "请进入数据集下一层，选择一个编号病例目录（例如 recovered_archive1\\1）。",
            )
            return
        if len(images) > 100 or len(videos) > 20:
            messagebox.showerror(
                "病例文件过多",
                f"检测到 {len(images)} 张图片、{len(videos)} 个视频。"
                "请确认选择的是单个病例目录。",
            )
            return
        command = ["--images", *map(str, images)]
        if videos:
            command += ["--videos", *map(str, videos)]
        self._start(command)

    def analyze_camera(self) -> None:
        if not messagebox.askokcancel(
            "开始采集", "将从摄像头0连续采集20秒。请保持倍率、焦距和照明稳定。"
        ):
            return
        self._start(["--camera", "0", "--seconds", "20", "--sample-fps", "2"])

    def _start(self, source_args: list[str]) -> None:
        self.folder_button.configure(state="disabled")
        self.camera_button.configure(state="disabled")
        self.preview_button.configure(state="disabled")
        self.print_button.configure(state="disabled")
        self.progress.start(12)
        self.status.set("正在进行图像质量筛选、静态/几何/视频分析和积分计算……")
        command = [
            sys.executable, str(PROJECT / "scripts" / "windows_uvc_report.py"),
            "--model-dir", str(PROJECT / "artifacts" / "windows_model_v3"),
            "--geometry-model-dir", str(PROJECT / "artifacts" / "geometry_final_v3"),
            "--video-model-dir", str(PROJECT / "artifacts" / "video_v3_final"),
            "--output-root", str(PROJECT / "captures_app"),
            *self._patient_args(), *source_args,
        ]
        threading.Thread(target=self._run, args=(command,), daemon=True).start()

    def _run(self, command: list[str]) -> None:
        environment = os.environ.copy()
        source_root = str(PROJECT / "src")
        environment["PYTHONPATH"] = (
            source_root + os.pathsep + environment.get("PYTHONPATH", "")
        )
        try:
            result = subprocess.run(
                command, cwd=PROJECT, env=environment, capture_output=True,
                text=True, encoding="utf-8", errors="replace", check=True,
            )
            report_path = parse_report_path(result.stdout)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.after(0, self._show_report, report_path, report)
        except Exception as error:
            error_text = str(error)
            log_path = PROJECT / "captures_app" / "last_error.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"time={datetime.now().isoformat()}\ncommand={command!r}\nerror={error_text}\n",
                encoding="utf-8",
            )
            self.after(0, self._failed, error_text, log_path)

    def _show_report(self, report_path: Path, report: dict) -> None:
        self.report_path = report_path
        self.print_path = report_path.with_name("report_print.html")
        self.table.delete(*self.table.get_children())
        fields = report.get("fields", {})
        for field, label, unit in REPORT_ROWS:
            details = fields.get(field, {})
            value = details.get("value", "—")
            if isinstance(value, float):
                value = f"{value:.2f}".rstrip("0").rstrip(".")
            confidence = details.get("confidence")
            confidence_text = "—" if confidence is None else f"{confidence:.1%}"
            status = str(details.get("status", ""))
            status_text = "需复核" if "low" in status or "uncalibrated" in status else "模型预测"
            self.table.insert("", "end", values=(label, value, unit, confidence_text, status_text))
        scores = report.get("scores", {})
        self.summary.set(
            f"形态积分：{scores.get('morphology_score','—')}    "
            f"流态积分：{scores.get('flow_score','—')}    "
            f"袢周积分：{scores.get('periloop_score','—')}    "
            f"总积分：{scores.get('total_score','—')}    "
            f"综合判断：{scores.get('overall_assessment','—')}"
        )
        warnings = report.get("quality", {}).get("warnings", [])
        self.status.set(
            "分析完成。" + (f" 有 {len(warnings)} 项质量/低支持提示，请复核。" if warnings else "")
        )
        self.progress.stop()
        self.folder_button.configure(state="normal")
        self.camera_button.configure(state="normal")
        self.preview_button.configure(state="normal")
        self.print_button.configure(state="normal")

    def _failed(self, error: str, log_path: Path) -> None:
        self.progress.stop()
        self.folder_button.configure(state="normal")
        self.camera_button.configure(state="normal")
        self.status.set("分析失败。")
        detail = error[-1200:]
        if "no usable CAP images" in detail:
            detail = "图片均未通过质量检查（可能为空白、严重过曝/欠曝或不是有效甲襞原图）。"
        messagebox.showerror(
            "分析失败",
            f"{detail}\n\n失败日志：{log_path}",
        )

    def preview(self) -> None:
        if not self.print_path or not self.print_path.exists():
            messagebox.showwarning("没有报告", "请先完成一次分析。")
            return
        webbrowser.open(self.print_path.resolve().as_uri())


if __name__ == "__main__":
    ReportApp().mainloop()
