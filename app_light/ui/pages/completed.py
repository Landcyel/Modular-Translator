"""已完成任务页面 — 查看、预览、导出、清除已完成任务。

布局：
1. 顶栏：标题 + 自动导出/手动导出/导出目录选择/清空按钮
2. 下方：已完成任务卡片列表
"""

import flet as ft
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

try:
    import winsound  # Windows 内嵌播放（无外部窗口）
except ImportError:  # 非 Windows 平台
    winsound = None

try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo_json
except Exception:  # pydub 缺失时退回纯 FFmpeg 解码路径
    AudioSegment = None
    mediainfo_json = None

from app.ffmpeg import FFPLAY_BIN, configure_pydub, run_ffmpeg
from app.log import log
from ui.theme import Layout, Palette, Radius, Typography
from ui.components import (
    _icon, _text, divider, _shadow, toolbar_panel_header, bordered_button,
)
from ui.widgets.task_list import _task_status_icon, _task_progress_text
from core.system_config import load_section
from core.writer import format_lrc_time


def build_completed(page: ft.Page, facade=None):
    """兼容包装 — 创建 CompletedPage 实例并构建 UI。"""
    return CompletedPage(page, facade).build()


class CompletedPage:
    """已完成任务页面实例 — 长期持有状态，避免导航切换时丢失。"""

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── 状态 ──
        self.export_dir: Path = Path("output")
        self.auto_export_enabled = False
        self.completed_tasks: list = []

        # ── 内嵌音频试听状态（pydub 处理 + FFmpeg 解码 → winsound / ffplay 播放）──
        self._audio_playing = False
        self._audio_temp_wav: Path | None = None
        self._audio_proc: subprocess.Popen | None = None
        self._audio_source: Path | None = None
        self._audio_duration_ms: float = 0.0
        self._audio_position_ms: float = 0.0
        self._audio_started_at: float = 0.0
        self._audio_start_offset_ms: float = 0.0
        self._audio_stop_event = threading.Event()
        self._audio_progress_thread: threading.Thread | None = None
        self._audio_dragging = False
        self._audio_ui_refs: dict = {}
        # pydub：把底层 ffmpeg/ffprobe/ffplay 指向项目自带 FFmpeg
        self._pydub_ready = configure_pydub()

        # 打开软件时读取"输出默认配置"（configs/output/default.json），覆盖默认值
        self._load_default_config()

        # ── Refs ──
        self.list_container = ft.Ref[ft.Column]()
        self.export_dir_label = ft.Ref[ft.Text]()
        self.auto_export_switch = ft.Ref[ft.Switch]()
        self.task_count_label = ft.Ref[ft.Text]()

        # ── 回调注册标志 ──
        self._callbacks_registered = False

        # 注册 facade 回调（仅首次）
        self.register_callbacks()

    def _load_default_config(self) -> None:
        """打开软件时读取"输出默认配置"（configs/system/default.ini [output]）。

        设置页"输出默认配置"管理的字段为：输出目录 output_dir、是否自动导出
        auto_export。文件缺失或解析失败时保持默认（output/、False）。
        """
        try:
            data = load_section("output")
        except Exception:
            return
        out = data.get("output_dir")
        if isinstance(out, str) and out.strip():
            self.export_dir = Path(out)
        # ini 中 auto_export 为字符串（"true"/"false"），需显式转换
        auto = data.get("auto_export")
        if isinstance(auto, str):
            self.auto_export_enabled = auto.strip().lower() in ("true", "1", "yes")

    # ════════════════════════════════════════════════════
    #  公开接口
    # ════════════════════════════════════════════════════

    def build(self) -> ft.Control:
        """构建/重建已完成任务页面 UI。"""

        # ── 单行工具条操作（自动导出 + 全部导出 + 导出目录 + 任务数量 + 清空）──
        top_actions = [
            ft.Row([
                ft.Text("自动导出", size=12, color=Palette.SUBTEXT),
                ft.Switch(
                    ref=self.auto_export_switch,
                    value=self.auto_export_enabled,
                    active_color=Palette.PRIMARY,
                    on_change=self._toggle_auto_export,
                ),
            ], spacing=4),
            bordered_button(
                "全部导出", ft.Icons.FILE_DOWNLOAD,
                on_click=self._manual_export_all,
                padding=ft.Padding.symmetric(horizontal=12, vertical=6),
            ),
            ft.TextButton(
                content=ft.Row([
                    ft.Icon(ft.Icons.FOLDER_OPEN, size=16, color=Palette.SUBTEXT),
                    ft.Text(ref=self.export_dir_label, value=str(self.export_dir.resolve()),
                            size=Typography.BODY_SM, color=Palette.SUBTEXT,
                            no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS, width=160),
                ], spacing=6),
                on_click=self._choose_export_dir,
                tooltip="选择导出目录",
                style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=8)),
            ),
            ft.Text(ref=self.task_count_label,
                    value=f"{len(self.completed_tasks)} 项",
                    size=Typography.BODY_SM, color=Palette.TEXT_MUTED),
            ft.TextButton(
                "清空全部",
                icon=ft.Icons.DELETE_SWEEP,
                on_click=self._clear_all,
                style=ft.ButtonStyle(color=Palette.ERROR, padding=ft.Padding.symmetric(horizontal=8)),
            ),
        ]

        top_bar = ft.Container(
            content=toolbar_panel_header("已完成任务", actions=top_actions),
            bgcolor=Palette.SURFACE,
            border_radius=Radius.XL,
            padding=ft.Padding.symmetric(horizontal=20, vertical=12),
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            shadow=_shadow("low"),
        )

        # ── 任务列表 ──
        task_list_col = ft.Column(
            ref=self.list_container,
            controls=[
                ft.Container(
                    content=ft.Column([
                        _icon(ft.Icons.INBOX, 32, Palette.TEXT_MUTED),
                        _text("暂无已完成任务", Typography.BODY, "normal", Palette.SUBTEXT),
                        _text("完成任务后将在此处显示", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                    ], spacing=6, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    alignment=ft.Alignment.CENTER,
                    bgcolor=Palette.SURFACE2,
                    border_radius=Radius.LG,
                    padding=ft.Padding.all(32),
                    border=ft.Border.all(1, Palette.BORDER_SUBTLE),
                )
            ],
            spacing=6,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        list_panel = ft.Container(
            content=ft.Column([
                task_list_col,
            ], spacing=0, expand=True),
            bgcolor=Palette.SURFACE,
            border_radius=Radius.XL,
            padding=ft.Padding.all(16),
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            shadow=_shadow("low"),
            expand=True,
        )

        return ft.Column([
            top_bar,
            ft.Container(height=Layout.SECTION_GAP),
            list_panel,
        ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    def save_ui_state(self) -> None:
        """离开页面前从控件 refs 同步状态到实例属性。"""
        self._stop_audio()  # 切页时停止内嵌试听，避免后台继续播放
        if self.auto_export_switch.current:
            self.auto_export_enabled = self.auto_export_switch.current.value

    def refresh(self) -> None:
        """facade 回调或切换回页面时刷新任务列表。"""
        self.completed_tasks = self._load_completed_tasks()
        self._rebuild_list()

    def register_callbacks(self) -> None:
        """向 facade 注册回调（仅首次调用生效）。"""
        if self._callbacks_registered or self.facade is None:
            return
        self.facade._on_task_change(self._on_task_change)
        self._callbacks_registered = True

    # ════════════════════════════════════════════════════
    #  内部方法 — 数据加载
    # ════════════════════════════════════════════════════

    def _load_completed_tasks(self) -> list:
        if self.facade is None:
            return []
        return self.facade.list_completed_tasks(task_type=None)

    def _build_task_card(self, task) -> ft.Container:
        """单个已完成任务卡片 — 文件名 + 完成时间 + 预览/导出/清除按钮。"""
        if isinstance(task, dict):
            tid = task.get("id", "")
            ttype = task.get("service_type", "") or task.get("type", "")
            status = task.get("status", "")
            summary = task.get("input_summary", "")
            result = task.get("result")
        else:
            tid = getattr(task, "id", "")
            ttype = getattr(task, "service_type", "") or getattr(task, "type", "")
            status = getattr(task, "status", "")
            summary = getattr(task, "input_summary", "")
            result = getattr(task, "result", None)

        type_label = {
            "translate": "翻译", "transcribe": "转写", "moss": "转写(MOSS)",
            "gsv": "语音合成",
        }.get(ttype, ttype)
        type_icon = {
            "translate": ft.Icons.TRANSLATE, "transcribe": ft.Icons.MIC,
            "moss": ft.Icons.GROUPS, "gsv": ft.Icons.RECORD_VOICE_OVER,
        }.get(ttype, ft.Icons.TASK_ALT)
        type_color = {
            "translate": Palette.PRIMARY, "transcribe": Palette.INFO,
            "moss": Palette.INFO, "gsv": Palette.PRIMARY_DARK,
        }.get(ttype, Palette.SUBTEXT)

        # ── 提交时间（created_at 为 Unix 时间戳；core 无 completed_at，用提交时间展示）──
        time_str = ""
        created = task.get("created_at", 0) if isinstance(task, dict) else getattr(task, "created_at", 0)
        if created:
            try:
                time_str = time.strftime("%m-%d %H:%M", time.localtime(float(created)))
            except (ValueError, OSError, TypeError):
                time_str = ""

        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=_icon(type_icon, 16, type_color),
                    bgcolor=f"{type_color}18",
                    border_radius=Radius.SM,
                    padding=ft.Padding.all(6),
                ),
                ft.Column([
                    ft.Text(summary or tid, size=Typography.BODY,
                            weight=ft.FontWeight.W_600, color=Palette.TEXT,
                            no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Row([
                        _text(f"#{tid[:8]}", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                        _text("·", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                        _text(type_label, Typography.CAPTION, "normal", Palette.SUBTEXT),
                        _text("·", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                        _text(status, Typography.CAPTION, "normal",
                              Palette.SUCCESS if status == "completed" else Palette.ERROR),
                        *([
                            _text("·", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                            _text(time_str, Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                        ] if time_str else []),
                    ], spacing=4),
                ], spacing=2, expand=True),
                _task_status_icon(status),
                ft.TextButton(
                    "预览",
                    icon=ft.Icons.VISIBILITY,
                    visible=result is not None,
                    on_click=lambda e, t=task: self._preview(t),
                    style=ft.ButtonStyle(color=Palette.INFO, padding=ft.Padding.symmetric(horizontal=8)),
                ),
                ft.TextButton(
                    "导出",
                    icon=ft.Icons.FILE_DOWNLOAD,
                    on_click=lambda e, tid=tid: self._export_single(tid),
                    style=ft.ButtonStyle(color=Palette.PRIMARY, padding=ft.Padding.symmetric(horizontal=8)),
                ),
                ft.TextButton(
                    "清除",
                    icon=ft.Icons.DELETE_OUTLINE,
                    on_click=lambda e, tid=tid: self._clear_single(tid),
                    style=ft.ButtonStyle(color=Palette.ERROR, padding=ft.Padding.symmetric(horizontal=8)),
                ),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=Palette.SURFACE2,
            border_radius=Radius.LG,
            padding=ft.Padding.symmetric(horizontal=14, vertical=12),
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
        )

    def _rebuild_list(self):
        if self.list_container.current is None:
            return
        tasks = self._load_completed_tasks()
        if self.task_count_label.current:
            self.task_count_label.current.value = f"{len(tasks)} 项"
        if not tasks:
            self.list_container.current.controls = [
                ft.Container(
                    content=ft.Column([
                        _icon(ft.Icons.INBOX, 32, Palette.TEXT_MUTED),
                        _text("暂无已完成任务", Typography.BODY, "normal", Palette.SUBTEXT),
                        _text("完成任务后将在此处显示", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                    ], spacing=6, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    alignment=ft.Alignment.CENTER,
                    bgcolor=Palette.SURFACE2,
                    border_radius=Radius.LG,
                    padding=ft.Padding.all(32),
                    border=ft.Border.all(1, Palette.BORDER_SUBTLE),
                )
            ]
        else:
            self.list_container.current.controls = [self._build_task_card(t) for t in tasks]
        if self.list_container.current.page:
            self.list_container.current.update()

    # ════════════════════════════════════════════════════
    #  内部方法 — 操作
    # ════════════════════════════════════════════════════

    @staticmethod
    def _task_kind(task) -> str:
        """按任务类型归类导出子目录：translation / transcription / tts / ''（根目录）。

        type 为服务名：翻译（llama/api/openai/translate）、转写（moss/
        transcription）、语音合成（gsv/tts）。
        """
        if isinstance(task, dict):
            ttype = str(task.get("service_type", "") or task.get("type", "") or "").lower()
        else:
            ttype = str(getattr(task, "service_type", "") or getattr(task, "type", "") or "").lower()
        if ttype in ("translate", "translation", "llama", "api", "openai"):
            return "translation"
        if ttype in ("transcribe", "transcription", "moss"):
            return "transcription"
        if ttype in ("gsv", "tts", "voice"):
            return "tts"
        return ""

    @staticmethod
    def _task_file_name(task) -> str:
        """取任务原始文件名（dict / snapshot 兼容），回退 input_summary/空。"""
        if isinstance(task, dict):
            return task.get("file_name", "") or task.get("input_summary", "") or ""
        return (getattr(task, "file_name", "") or getattr(task, "input_summary", "") or "")

    # ── 预览 ──────────────────────────────────────────────

    @staticmethod
    def _task_result(task):
        """取任务结果（dict / snapshot 兼容）。"""
        if isinstance(task, dict):
            return task.get("result")
        return getattr(task, "result", None)

    @classmethod
    def _task_audio_path(cls, task) -> Path | None:
        """TTS/GSV 结果 → 音频文件路径；非音频结果返回 None。"""
        result = cls._task_result(task)
        if isinstance(result, dict):
            path = result.get("audio_path")
            if isinstance(path, str) and path.strip():
                return Path(path)
        return None

    @classmethod
    def _task_title(cls, task) -> str:
        """预览面板标题：文件名/摘要 + 短 id。"""
        name = cls._task_file_name(task)
        tid = str(task.get("id", "") if isinstance(task, dict) else getattr(task, "id", ""))
        if name:
            return f"{name}（#{tid[:8]}）"
        return f"任务 #{tid[:8]}" if tid else "任务预览"

    @classmethod
    def _task_preview_text(cls, task) -> str:
        """把任务结果整理为可预览的纯文本。

        - 翻译结果 str → 原样返回
        - 转写结果 dict → 标准 LRC 时间轴/说话人文本（与转写页预览一致）
        - 其它 dict/对象 → 可读 JSON 文本
        """
        result = cls._task_result(task)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            segments = result.get("segments")
            if isinstance(segments, list) and segments:
                lines = []
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    try:
                        start = float(seg.get("start", 0) or 0)
                    except (TypeError, ValueError):
                        start = 0.0
                    text = str(seg.get("text", "") or "").strip()
                    speaker = str(seg.get("speaker", "") or "")
                    prefix = f"<{speaker}>" if speaker else ""
                    lines.append(f"[{format_lrc_time(start)}]{prefix}{text}".rstrip())
                if lines:
                    return "\n".join(lines)
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        try:
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(result)

    def _preview(self, task):
        """点击卡片「预览」：文本结果 → 文本面板；TTS 结果 → 音频试听面板。"""
        result = self._task_result(task)
        if result is None:
            self._show_snack("该任务暂无结果可预览", error=True)
            return
        audio_path = self._task_audio_path(task)
        if audio_path is not None:
            self._show_audio_preview(task, audio_path, result if isinstance(result, dict) else {})
        else:
            self._show_text_preview(task)

    def _show_text_preview(self, task):
        """弹出文本预览面板（只读多行文本框 + 关闭按钮）。"""
        text = self._task_preview_text(task)
        title = self._task_title(task)
        tid = str(task.get("id", "") if isinstance(task, dict) else getattr(task, "id", ""))
        kind_label = {
            "translation": "翻译文本",
            "transcription": "转写文本",
            "tts": "语音合成",
        }.get(self._task_kind(task), "任务结果")
        meta = f"{kind_label} · #{tid[:8]} · {len(text)} 字符"

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Container(
                    content=_icon(ft.Icons.DESCRIPTION, 16, Palette.PRIMARY),
                    bgcolor=f"{Palette.PRIMARY}18",
                    border_radius=Radius.SM,
                    padding=ft.Padding.all(6),
                ),
                ft.Text(title, size=16, weight=ft.FontWeight.W_600, color=Palette.TEXT,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS, width=670),
            ], spacing=10),
            content=ft.Container(
                content=ft.Column([
                    _text(meta, Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                    ft.TextField(
                        value=text,
                        read_only=True,
                        multiline=True,
                        min_lines=1,
                        max_lines=None,
                        expand=True,
                        width=760,
                        text_size=13,
                        bgcolor=Palette.SURFACE2,
                        border_color=Palette.BORDER_SUBTLE,
                        border_width=1,
                        border_radius=Radius.SM,
                        content_padding=ft.Padding.all(10),
                    ),
                ], spacing=8, expand=True),
                width=760,
                height=520,
            ),
            actions=[
                ft.FilledButton(
                    "关闭",
                    icon=ft.Icons.CLOSE,
                    on_click=lambda e: self._close_dialog(dlg),
                    style=ft.ButtonStyle(bgcolor=Palette.PRIMARY, color="#FFFFFF"),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dlg)

    # ── 音频预览（pydub + FFmpeg 内嵌播放，可拖动进度）────────────────

    _PYDUB_LOAD_MAX_BYTES = 64 * 1024 * 1024  # 超过该体积改用 ffmpeg 直接切片解码，避免 pydub 全量载入内存

    @staticmethod
    def _fmt_ms(ms: float) -> str:
        """毫秒 → mm:ss.d（例如 01:23.4）。"""
        try:
            ms = max(0.0, float(ms))
        except (TypeError, ValueError):
            ms = 0.0
        total_sec = ms / 1000.0
        minutes = int(total_sec // 60)
        seconds = total_sec - minutes * 60
        return f"{minutes:02d}:{seconds:04.1f}"

    def _show_audio_preview(self, task, audio_path: Path, result: dict):
        """弹出音频预览面板：文件信息 + 可拖动进度条 + 内嵌播放/暂停。

        - 时长/采样率优先用 pydub（mediainfo_json → 项目自带 ffprobe）读取，
          result 里已有字段作为兜底。
        - 播放时用 pydub 对音频切片并导出 WAV（大文件自动回退 ffmpeg 直接
          切片解码），Windows 走 winsound、其它平台走 ffplay -nodisp。
        """
        title = self._task_title(task)
        path = Path(audio_path)
        self._stop_audio()  # 打开新预览前先停掉可能还在播的音频
        exists = path.is_file()
        info = result.get("info") if isinstance(result, dict) else None
        info = info if isinstance(info, dict) else {}

        size_kb = ""
        file_bytes = 0
        if exists:
            try:
                file_bytes = path.stat().st_size
                size_kb = f"{file_bytes / 1024:.1f} KB"
            except OSError:
                pass

        # pydub 读取音频信息（ffprobe），失败则退回 result 中的字段
        duration_ms = 0.0
        sample_rate = result.get("sample_rate")
        if exists and self._pydub_ready and mediainfo_json is not None:
            try:
                media = mediainfo_json(str(path))
                try:
                    duration_ms = float(media.get("duration", 0) or 0) * 1000.0
                except (TypeError, ValueError):
                    duration_ms = 0.0
                streams = media.get("streams") or []
                for st in streams:
                    if isinstance(st, dict) and st.get("codec_type") == "audio":
                        try:
                            sample_rate = int(st.get("sample_rate") or 0) or sample_rate
                        except (TypeError, ValueError):
                            pass
                        break
            except Exception as ex:
                log.record("warn", f"[已完成] pydub 读取音频信息失败: {ex}")
        if duration_ms <= 0 and result.get("duration") is not None:
            try:
                duration_ms = float(result.get("duration")) * 1000.0
            except (TypeError, ValueError):
                duration_ms = 0.0

        duration_text = self._fmt_ms(duration_ms) if duration_ms > 0 else "--:--.-"

        def info_row(label: str, value: str) -> ft.Row:
            return ft.Row([
                ft.Text(label, size=12, color=Palette.TEXT_MUTED, width=88),
                ft.Text(value, size=13, color=Palette.TEXT, no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS, width=480),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        rows = [
            info_row("文件", str(path)),
            info_row("状态", "✅ 可试听" if exists else "⚠ 文件缺失"),
        ]
        if size_kb:
            rows.append(info_row("大小", size_kb))
        if sample_rate is not None:
            rows.append(info_row("采样率", f"{sample_rate} Hz"))
        rows.append(info_row("时长", duration_text))
        if info.get("ref_mode"):
            rows.append(info_row("复刻模式", str(info.get("ref_mode"))))
        if info.get("fragments") is not None:
            rows.append(info_row("片段数", str(info.get("fragments"))))

        # 播放器控件引用（每次打开重建，关闭时清空）
        slider_ref = ft.Ref[ft.Slider]()
        cur_ref = ft.Ref[ft.Text]()
        total_ref = ft.Ref[ft.Text]()
        play_btn_ref = ft.Ref[ft.FilledButton]()
        self._audio_ui_refs = {
            "slider": slider_ref,
            "current": cur_ref,
            "total": total_ref,
            "play": play_btn_ref,
        }
        self._audio_position_ms = 0.0
        self._audio_duration_ms = duration_ms
        self._audio_source = path if exists else None

        slider = ft.Slider(
            ref=slider_ref,
            min=0,
            max=duration_ms if duration_ms > 0 else 1.0,
            value=0,
            expand=True,
            disabled=not exists or duration_ms <= 0,
            active_color=Palette.PRIMARY,
            inactive_color=Palette.BORDER_SUBTLE,
            on_change_start=lambda e: self._on_slider_change_start(),
            on_change=lambda e: self._on_slider_change(e),
            on_change_end=lambda e: self._on_slider_change_end(e, path),
        )

        player = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("00:00.0", ref=cur_ref, size=12, color=Palette.TEXT_MUTED, width=52),
                    slider,
                    ft.Text(duration_text, ref=total_ref, size=12, color=Palette.TEXT_MUTED,
                            width=52, text_align=ft.TextAlign.RIGHT),
                ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([
                    ft.FilledButton(
                        "试听",
                        ref=play_btn_ref,
                        icon=ft.Icons.PLAY_ARROW,
                        disabled=not exists or duration_ms <= 0,
                        on_click=lambda e: self._toggle_play(path),
                        style=ft.ButtonStyle(bgcolor=Palette.PRIMARY, color="#FFFFFF"),
                    ),
                    ft.Text("拖动进度条可跳转播放位置", size=11, color=Palette.TEXT_MUTED),
                ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=8),
            bgcolor=Palette.SURFACE2,
            border_radius=Radius.SM,
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        )

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Container(
                    content=_icon(ft.Icons.AUDIO_FILE, 16, Palette.PRIMARY_DARK),
                    bgcolor=f"{Palette.PRIMARY_DARK}18",
                    border_radius=Radius.SM,
                    padding=ft.Padding.all(6),
                ),
                ft.Text(title, size=16, weight=ft.FontWeight.W_600, color=Palette.TEXT,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS, width=550),
            ], spacing=10),
            content=ft.Container(
                content=ft.Column(rows + [
                    ft.Container(height=1, bgcolor=Palette.BORDER_SUBTLE),
                    player,
                ], spacing=10),
                width=640,
            ),
            actions=[
                ft.TextButton(
                    "打开所在目录",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=lambda e: self._open_dir(path),
                    style=ft.ButtonStyle(color=Palette.SUBTEXT),
                ),
                ft.TextButton(
                    "关闭",
                    icon=ft.Icons.CLOSE,
                    on_click=lambda e: self._close_dialog(dlg),
                    style=ft.ButtonStyle(color=Palette.SUBTEXT),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dlg)

    def _toggle_play(self, audio_path: Path):
        """播放/暂停按钮。"""
        if self._audio_playing:
            self._pause_audio()
            return
        start_ms = self._audio_position_ms
        if self._audio_duration_ms > 0 and start_ms >= self._audio_duration_ms - 80:
            start_ms = 0.0
        if self._start_audio_playback(audio_path, start_ms):
            self._set_play_button(True)

    def _pause_audio(self):
        """暂停：停止底层播放，但把当前进度保留在滑块上。"""
        pos = self._current_playback_pos_ms()
        self._stop_audio()
        self._audio_position_ms = min(pos, self._audio_duration_ms) if self._audio_duration_ms > 0 else pos
        self._sync_slider_to_position()
        self._set_play_button(False)

    def _start_audio_playback(self, audio_path: Path, start_ms: float = 0.0) -> bool:
        """从 start_ms 开始内嵌播放。

        - Windows：先生成临时 WAV（pydub 切片导出；大文件回退 ffmpeg 直接
          切片解码），再用 winsound 异步播放。
        - 其它平台：项目自带 ffplay -nodisp -ss start 直接播放。
        """
        path = Path(audio_path)
        if not path.is_file():
            self._show_snack(f"音频文件不存在: {path}", error=True)
            return False
        try:
            start_ms = max(0.0, float(start_ms))
        except (TypeError, ValueError):
            start_ms = 0.0

        self._stop_audio()
        self._audio_position_ms = start_ms
        self._audio_source = path
        if self._audio_duration_ms <= 0:
            self._audio_duration_ms = self._probe_duration_ms(path)

        start_sec = start_ms / 1000.0
        tmp = None
        if winsound is not None:
            try:
                fd, tmp_s = tempfile.mkstemp(prefix="mt_preview_", suffix=".wav")
                os.close(fd)
                tmp = Path(tmp_s)
                if not self._build_playable_wav(path, start_ms, tmp):
                    self._show_snack("音频解码失败，无法试听", error=True)
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return False
                winsound.PlaySound(str(tmp), winsound.SND_FILENAME | winsound.SND_ASYNC)
                self._audio_temp_wav = tmp
                self._audio_playing = True
            except Exception as ex:
                log.record("error", f"[已完成] 内嵌播放失败: {ex}")
                self._show_snack(f"内嵌播放失败: {ex}", error=True)
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass
                return False
        else:
            try:
                cmd = [str(FFPLAY_BIN), "-nodisp", "-autoexit", "-loglevel", "quiet"]
                if start_ms > 0:
                    cmd += ["-ss", f"{start_sec:.3f}"]
                cmd += [str(path)]
                self._audio_proc = subprocess.Popen(cmd)
                self._audio_playing = True
            except Exception as ex:
                log.record("error", f"[已完成] ffplay 播放失败: {ex}")
                self._show_snack(f"ffplay 播放失败: {ex}", error=True)
                return False

        self._audio_started_at = time.monotonic()
        self._audio_start_offset_ms = start_ms
        self._start_progress_thread()
        log.record("info", f"[已完成] 内嵌试听: {path} @ {start_ms:.0f}ms")
        return True

    def _build_playable_wav(self, path: Path, start_ms: float, out: Path) -> bool:
        """生成用于 winsound 播放的临时 WAV（PCM16 / 44.1kHz / 立体声）。

        小文件用 pydub：AudioSegment.from_file(start_second=...) 切片并重采样后导出；
        大文件（pydub 需整段读入内存）回退为项目自带 ffmpeg 直接切片解码。
        """
        try:
            file_bytes = path.stat().st_size
        except OSError:
            file_bytes = 0
        use_pydub = (
            self._pydub_ready and AudioSegment is not None
            and file_bytes <= self._PYDUB_LOAD_MAX_BYTES
        )
        if use_pydub:
            try:
                seg = AudioSegment.from_file(str(path), start_second=start_ms / 1000.0)
                seg = seg.set_frame_rate(44100).set_channels(2).set_sample_width(2)
                seg.export(str(out), format="wav")
                log.record("info", "[已完成] pydub 切片导出试听 WAV")
                return True
            except Exception as ex:
                log.record("warn", f"[已完成] pydub 切片导出失败，回退 ffmpeg: {ex}")

        proc = run_ffmpeg(
            ["-y", "-ss", f"{start_ms / 1000.0:.3f}", "-i", str(path),
             "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(out)],
            capture_output=True,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", "replace") if isinstance(proc.stderr, bytes) else str(proc.stderr)
            log.record("error", f"[已完成] FFmpeg 切片解码失败: {stderr[-300:]}")
            return False
        return True

    def _probe_duration_ms(self, path: Path) -> float:
        """用 pydub（ffprobe）快速读取时长（毫秒）；失败返回 0。"""
        if self._pydub_ready and mediainfo_json is not None:
            try:
                media = mediainfo_json(str(path))
                return float(media.get("duration", 0) or 0) * 1000.0
            except Exception:
                return 0.0
        return 0.0

    # ── 播放进度（后台线程刷新，可拖动跳转）──────────────

    def _start_progress_thread(self):
        self._audio_stop_event.clear()
        self._audio_progress_thread = threading.Thread(
            target=self._audio_progress_loop, args=(self._audio_stop_event,), daemon=True
        )
        self._audio_progress_thread.start()

    def _audio_progress_loop(self, stop_event: threading.Event):
        while not stop_event.is_set():
            time.sleep(0.2)
            if stop_event.is_set() or not self._audio_playing:
                return
            try:
                pos = self._current_playback_pos_ms()
                if self._audio_duration_ms > 0 and pos >= self._audio_duration_ms - 40:
                    self._on_playback_finished()
                    return
                self._update_progress_ui(pos)
            except Exception as ex:
                log.record("error", f"[已完成] 进度刷新失败: {ex}")
                return

    def _current_playback_pos_ms(self) -> float:
        """当前播放位置（毫秒）：播放中按墙钟推算，否则返回滑块记录值。"""
        if not self._audio_playing:
            return self._audio_position_ms
        elapsed_ms = (time.monotonic() - self._audio_started_at) * 1000.0
        pos = self._audio_start_offset_ms + elapsed_ms
        if self._audio_duration_ms > 0:
            pos = min(pos, self._audio_duration_ms)
        return max(0.0, pos)

    def _on_playback_finished(self):
        self._stop_audio()
        self._audio_position_ms = self._audio_duration_ms
        self._sync_slider_to_position()
        self._set_play_button(False)

    def _update_progress_ui(self, pos_ms: float):
        refs = self._audio_ui_refs
        if not refs:
            return
        try:
            slider = refs.get("slider").current
            if slider is not None and not self._audio_dragging:
                slider.value = pos_ms
                slider.update()
        except Exception:
            pass
        try:
            cur = refs.get("current").current
            if cur is not None:
                cur.value = self._fmt_ms(pos_ms)
                cur.update()
        except Exception:
            pass

    def _on_slider_change_start(self):
        self._audio_dragging = True

    def _on_slider_change(self, e):
        try:
            pos = float(e.control.value or 0)
        except (TypeError, ValueError):
            pos = 0.0
        self._audio_position_ms = pos
        refs = self._audio_ui_refs
        if refs:
            try:
                cur = refs.get("current").current
                if cur is not None:
                    cur.value = self._fmt_ms(pos)
                    cur.update()
            except Exception:
                pass

    def _on_slider_change_end(self, e, audio_path: Path):
        self._audio_dragging = False
        try:
            pos = float(e.control.value or 0)
        except (TypeError, ValueError):
            pos = 0.0
        self._seek_audio(pos, audio_path)

    def _seek_audio(self, pos_ms: float, audio_path: Path):
        """跳转到指定位置；若正在播放则从该位置重新开始。"""
        try:
            pos_ms = max(0.0, float(pos_ms))
        except (TypeError, ValueError):
            pos_ms = 0.0
        if self._audio_duration_ms > 0:
            pos_ms = min(pos_ms, self._audio_duration_ms)
        self._audio_position_ms = pos_ms
        self._sync_slider_to_position()
        if self._audio_playing:
            self._start_audio_playback(audio_path, pos_ms)
            self._set_play_button(True)

    def _sync_slider_to_position(self):
        refs = self._audio_ui_refs
        if not refs:
            return
        pos = self._audio_position_ms
        try:
            slider = refs.get("slider").current
            if slider is not None:
                slider.value = pos
                slider.update()
        except Exception:
            pass
        try:
            cur = refs.get("current").current
            if cur is not None:
                cur.value = self._fmt_ms(pos)
                cur.update()
        except Exception:
            pass

    def _stop_audio(self):
        """停止当前内嵌播放并清理临时 WAV / ffplay 进程与进度线程。"""
        self._audio_stop_event.set()
        if (self._audio_progress_thread is not None
                and self._audio_progress_thread.is_alive()
                and self._audio_progress_thread is not threading.current_thread()):
            self._audio_progress_thread.join(timeout=0.5)
        self._audio_progress_thread = None
        if self._audio_playing and winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if self._audio_proc is not None:
            try:
                self._audio_proc.terminate()
            except Exception:
                pass
            self._audio_proc = None
        self._audio_playing = False
        if self._audio_temp_wav is not None:
            try:
                self._audio_temp_wav.unlink(missing_ok=True)
            except OSError:
                pass
            self._audio_temp_wav = None

    def _set_play_button(self, playing: bool):
        """刷新试听按钮为「停止 / 试听」双态。"""
        refs = self._audio_ui_refs
        if not refs:
            return
        try:
            btn = refs.get("play").current
        except Exception:
            return
        if btn is None:
            return
        btn.text = "停止" if playing else "试听"
        btn.icon = ft.Icons.STOP if playing else ft.Icons.PLAY_ARROW
        try:
            btn.update()
        except RuntimeError:
            pass

    def _open_dir(self, audio_path: Path):
        """打开音频所在目录（便于查看/复制文件）。"""
        folder = Path(audio_path).parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as ex:
            log.record("error", f"[已完成] 打开目录失败: {ex}")
            self._show_snack(f"打开目录失败: {ex}", error=True)

    def _show_snack(self, msg: str, error: bool = False):
        """主线程安全地弹出 SnackBar 提示。"""
        if self.page is not None:
            self.page.show_dialog(
                ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR if error else Palette.SUCCESS)
            )

    def _close_dialog(self, dlg):
        self._stop_audio()
        self._audio_ui_refs = {}
        self._audio_position_ms = 0.0
        self._audio_duration_ms = 0.0
        self._audio_source = None
        try:
            dlg.open = False
            self.page.update()
        except Exception:
            pass

    def _export_one(self, task) -> bool:
        """单个任务导出（手动/自动共用）：统一路径 + 成功/失败日志。返回是否成功。

        - 转写任务（transcription）→ LRC 歌词文件（b.mp3 → b.lrc）
        - 语音合成任务（tts）→ 复制结果 wav（{name}.wav / {id}.wav）
        - 翻译/未知类型 → 保留源后缀
        """
        if self.facade is None:
            return False
        tid = task.get("id", "") if isinstance(task, dict) else getattr(task, "id", "")
        name = self._task_file_name(task)
        kind = self._task_kind(task)
        base = self.export_dir / kind if kind else self.export_dir
        if kind == "transcription":
            # 转写结果 → 打好轴的 LRC 歌词文件（b.mp3 → b.lrc；无文件名 → {tid}.lrc）
            out = base / (f"{Path(name).stem}.lrc" if name else f"{tid}.lrc")
        elif kind == "tts":
            # 语音合成 → 音频文件（b.txt → b.wav；无文件名 → {tid}.wav）
            out = base / (f"{Path(name).stem}.wav" if name else f"{tid}.wav")
        else:
            out = base / (name or f"{tid}.txt")
        try:
            self.facade.export_task(tid, out)
            log.record("info", f"[已完成] 已导出: {out}")
            return True
        except Exception as ex:
            log.record("error", f"[已完成] 导出失败 ({tid}): {ex}")
            return False

    def _export_single(self, task_id: str):
        """按任务原始文件名导出（后缀判断交由 export_task 三档规则）；找不到任务回退 {id}.txt。"""
        if self.facade is None:
            return
        for t in self._load_completed_tasks():
            tid = t.get("id", "") if isinstance(t, dict) else getattr(t, "id", "")
            if tid == task_id:
                self._export_one(t)
                return
        # 找不到任务：回退 {id}.txt（根目录）
        out = self.export_dir / f"{task_id}.txt"
        try:
            self.facade.export_task(task_id, out)
            log.record("info", f"[已完成] 已导出: {out}")
        except Exception as ex:
            log.record("error", f"[已完成] 导出失败: {ex}")

    def _clear_single(self, task_id: str):
        if self.facade:
            self.facade.clear_completed_task(task_id)
            self._rebuild_list()

    def _clear_all(self, e):
        if self.facade:
            self.facade.clear_all_completed()
            self._rebuild_list()

    async def _choose_export_dir(self, e):
        """选择导出目录（flet 0.86.2：get_directory_path 直接返回路径字符串）。"""
        if self.file_picker is None:
            log.record("warn", "[已完成] 未接文件选择器，无法选择导出目录")
            return
        try:
            result = await self.file_picker.get_directory_path(dialog_title="选择导出目录")
        except Exception as ex:
            log.record("error", f"[已完成] 选择导出目录失败: {ex}")
            return
        if not result:
            return
        self.export_dir = Path(result)
        if self.export_dir_label.current:
            self.export_dir_label.current.value = str(self.export_dir.resolve())
            self.export_dir_label.current.update()
        if self.page:
            self.page.show_dialog(
                ft.SnackBar(ft.Text(f"导出目录已更新: {self.export_dir.resolve()}"), bgcolor=Palette.SUCCESS)
            )

    def _toggle_auto_export(self, e):
        self.auto_export_enabled = self.auto_export_switch.current.value if self.auto_export_switch.current else False

    def _manual_export_all(self, e):
        """全部导出：逐个任务经 _export_one（统一路径与日志）。"""
        if self.facade is None:
            return
        for task in self._load_completed_tasks():
            self._export_one(task)
        log.record("info", "[已完成] 全部导出完成")

    # ════════════════════════════════════════════════════
    #  Facade 回调
    # ════════════════════════════════════════════════════

    def _on_task_change(self, snapshot):
        # TaskSnapshot 无 service_type 字段（type 为服务名：llama/api/transcribe…）
        # 用 _task_kind 归类判断（translation/transcription 之外忽略）
        if not self._task_kind(snapshot):
            return
        status = getattr(snapshot, "status", "")
        if status in ("completed", "failed", "cancelled"):
            self.completed_tasks = self._load_completed_tasks()
            self._rebuild_list()
            # 自动导出：与手动导出共用 _export_one（统一路径 + 成功/失败日志）
            if self.auto_export_enabled and status == "completed":
                self._export_one(snapshot)
