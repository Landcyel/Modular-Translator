"""Completed tasks page — view, preview, export, and clear completed tasks.

Layout:
1. Top bar: title + auto-export / manual export / export-dir picker / clear buttons
2. Below: list of completed task cards
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
    import winsound  # Windows embedded playback (no external window)
except ImportError:  # non-Windows platforms
    winsound = None

try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo_json
except Exception:  # fall back to a pure-FFmpeg decode path when pydub is missing
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
    """Compatibility wrapper — create a CompletedPage instance and build the UI."""
    return CompletedPage(page, facade).build()


class CompletedPage:
    """Completed tasks page instance — holds state long-term so it survives navigation switches."""

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── State ──
        self.export_dir: Path = Path("output")
        self.auto_export_enabled = False
        self.completed_tasks: list = []

        # ── Embedded audio preview state (pydub processing + FFmpeg decode → winsound / ffplay playback) ──
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
        # pydub: point the underlying ffmpeg/ffprobe/ffplay at the project-bundled FFmpeg
        self._pydub_ready = configure_pydub()

        # Read the "output default config" at app startup (configs/output/default.json) to override defaults
        self._load_default_config()

        # ── Refs ──
        self.list_container = ft.Ref[ft.Column]()
        self.export_dir_label = ft.Ref[ft.Text]()
        self.auto_export_switch = ft.Ref[ft.Switch]()
        self.task_count_label = ft.Ref[ft.Text]()

        # ── Callback registration flag ──
        self._callbacks_registered = False

        # Register facade callbacks (first time only)
        self.register_callbacks()

    def _load_default_config(self) -> None:
        """Read the "output default config" at app startup (configs/system/default.ini [output]).

        The fields managed by the Settings "output default config" are: output directory
        output_dir and auto-export auto_export. If the file is missing or fails to parse,
        keep the defaults (output/, False).
        """
        try:
            data = load_section("output")
        except Exception:
            return
        out = data.get("output_dir")
        if isinstance(out, str) and out.strip():
            self.export_dir = Path(out)
        # auto_export in ini is a string ("true"/"false"); convert explicitly
        auto = data.get("auto_export")
        if isinstance(auto, str):
            self.auto_export_enabled = auto.strip().lower() in ("true", "1", "yes")

    # ── Public interface ──

    def build(self) -> ft.Control:
        """Build/rebuild the completed tasks page UI."""

        # ── Single-row toolbar actions (auto-export + export all + export dir + task count + clear) ──
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

        # ── Task list ──
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
        """Sync state from control refs to instance attributes before leaving the page."""
        self._stop_audio()  # stop embedded preview when switching pages to avoid background playback
        if self.auto_export_switch.current:
            self.auto_export_enabled = self.auto_export_switch.current.value

    def refresh(self) -> None:
        """Refresh the task list on facade callbacks or when switching back to the page."""
        self.completed_tasks = self._load_completed_tasks()
        self._rebuild_list()

    def register_callbacks(self) -> None:
        """Register callbacks with the facade (effective only on first call)."""
        if self._callbacks_registered or self.facade is None:
            return
        self.facade._on_task_change(self._on_task_change)
        self._callbacks_registered = True

    # ── Internal methods — data loading ──

    def _load_completed_tasks(self) -> list:
        if self.facade is None:
            return []
        return self.facade.list_completed_tasks(task_type=None)

    def _build_task_card(self, task) -> ft.Container:
        """Single completed task card — file name + completion time + preview/export/clear buttons."""
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

        # ── Submit time (created_at is a Unix timestamp; core has no completed_at, so show submit time) ──
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

    # ── Internal methods — actions ──

    @staticmethod
    def _task_kind(task) -> str:
        """Categorize the export subdirectory by task type: translation / transcription / tts / '' (root).

        type is the service name: translate (llama/api/openai/translate), transcribe (moss/
        transcription), TTS (gsv/tts).
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
        """Get the task's original file name (dict / snapshot compatible), falling back to input_summary / empty."""
        if isinstance(task, dict):
            return task.get("file_name", "") or task.get("input_summary", "") or ""
        return (getattr(task, "file_name", "") or getattr(task, "input_summary", "") or "")

    # ── Preview ──

    @staticmethod
    def _task_result(task):
        """Get the task result (dict / snapshot compatible)."""
        if isinstance(task, dict):
            return task.get("result")
        return getattr(task, "result", None)

    @classmethod
    def _task_audio_path(cls, task) -> Path | None:
        """TTS/GSV result → audio file path; non-audio results return None."""
        result = cls._task_result(task)
        if isinstance(result, dict):
            path = result.get("audio_path")
            if isinstance(path, str) and path.strip():
                return Path(path)
        return None

    @classmethod
    def _task_title(cls, task) -> str:
        """Preview panel title: file name / summary + short id."""
        name = cls._task_file_name(task)
        tid = str(task.get("id", "") if isinstance(task, dict) else getattr(task, "id", ""))
        if name:
            return f"{name}（#{tid[:8]}）"
        return f"任务 #{tid[:8]}" if tid else "任务预览"

    @classmethod
    def _task_preview_text(cls, task) -> str:
        """Flatten a task result into previewable plain text.

        - Translation result str → returned as-is
        - Transcription result dict → standard LRC timeline / speaker text (matches the transcribe page preview)
        - Other dict/object → readable JSON text
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
        """Card "Preview" click: text result → text panel; TTS result → audio preview panel."""
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
        """Show a text preview dialog (read-only multiline text field + close button)."""
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

    # ── Audio preview (pydub + FFmpeg embedded playback, draggable progress) ──

    _PYDUB_LOAD_MAX_BYTES = 64 * 1024 * 1024  # above this size, slice/decode directly with ffmpeg instead of loading the whole file into memory

    @staticmethod
    def _fmt_ms(ms: float) -> str:
        """Milliseconds → mm:ss.d (e.g. 01:23.4)."""
        try:
            ms = max(0.0, float(ms))
        except (TypeError, ValueError):
            ms = 0.0
        total_sec = ms / 1000.0
        minutes = int(total_sec // 60)
        seconds = total_sec - minutes * 60
        return f"{minutes:02d}:{seconds:04.1f}"

    def _show_audio_preview(self, task, audio_path: Path, result: dict):
        """Show an audio preview dialog: file info + draggable progress bar + embedded play/pause.

        - Duration/sample rate are read with pydub (mediainfo_json → project-bundled ffprobe),
          with fields already present in result as a fallback.
        - For playback, pydub slices the audio and exports a WAV (large files automatically fall
          back to direct ffmpeg slice/decode); Windows uses winsound, other platforms use ffplay -nodisp.
        """
        title = self._task_title(task)
        path = Path(audio_path)
        self._stop_audio()  # stop any audio still playing before opening a new preview
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

        # Read audio info with pydub (ffprobe); fall back to fields in result on failure
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

        # Player control refs (rebuilt on each open, cleared on close)
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
        """Play/pause toggle button."""
        if self._audio_playing:
            self._pause_audio()
            return
        start_ms = self._audio_position_ms
        if self._audio_duration_ms > 0 and start_ms >= self._audio_duration_ms - 80:
            start_ms = 0.0
        if self._start_audio_playback(audio_path, start_ms):
            self._set_play_button(True)

    def _pause_audio(self):
        """Pause: stop the underlying playback but keep the current position on the slider."""
        pos = self._current_playback_pos_ms()
        self._stop_audio()
        self._audio_position_ms = min(pos, self._audio_duration_ms) if self._audio_duration_ms > 0 else pos
        self._sync_slider_to_position()
        self._set_play_button(False)

    def _start_audio_playback(self, audio_path: Path, start_ms: float = 0.0) -> bool:
        """Start embedded playback from start_ms.

        - Windows: first generate a temp WAV (pydub slice export; large files fall back to direct
          ffmpeg slice/decode), then play asynchronously with winsound.
        - Other platforms: play directly with the project-bundled ffplay -nodisp -ss start.
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
        """Generate a temp WAV for winsound playback (PCM16 / 44.1kHz / stereo).

        Small files use pydub: AudioSegment.from_file(start_second=...) slices and resamples before
        exporting; large files (pydub would load the whole segment into memory) fall back to the
        project-bundled ffmpeg for direct slice/decode.
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
        """Quickly read duration in ms with pydub (ffprobe); returns 0 on failure."""
        if self._pydub_ready and mediainfo_json is not None:
            try:
                media = mediainfo_json(str(path))
                return float(media.get("duration", 0) or 0) * 1000.0
            except Exception:
                return 0.0
        return 0.0

    # ── Playback progress (refreshed by a background thread; draggable seek) ──

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
        """Current playback position (ms): computed from the wall clock while playing, otherwise the slider-recorded value."""
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
        """Seek to the given position; if playing, restart playback from that position."""
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
        """Stop the current embedded playback and clean up temp WAV / ffplay process and progress thread."""
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
        """Refresh the preview button between its "Stop / Preview" two states."""
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
        """Open the folder containing the audio (for viewing/copying files)."""
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
        """Show a SnackBar safely from the main thread."""
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
        """Export a single task (shared by manual/auto): unified path + success/failure logs. Returns success.

        - Transcription task → LRC lyric file (b.mp3 → b.lrc)
        - TTS task → copy the result wav ({name}.wav / {id}.wav)
        - Translate/unknown type → keep the source extension
        """
        if self.facade is None:
            return False
        tid = task.get("id", "") if isinstance(task, dict) else getattr(task, "id", "")
        name = self._task_file_name(task)
        kind = self._task_kind(task)
        base = self.export_dir / kind if kind else self.export_dir
        if kind == "transcription":
            # transcription result → timed LRC lyric file (b.mp3 → b.lrc; no name → {tid}.lrc)
            out = base / (f"{Path(name).stem}.lrc" if name else f"{tid}.lrc")
        elif kind == "tts":
            # TTS → audio file (b.txt → b.wav; no name → {tid}.wav)
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
        """Export by the task's original file name (extension logic delegated to export_task's three-rule scheme); falls back to {id}.txt when the task is not found."""
        if self.facade is None:
            return
        for t in self._load_completed_tasks():
            tid = t.get("id", "") if isinstance(t, dict) else getattr(t, "id", "")
            if tid == task_id:
                self._export_one(t)
                return
        # Task not found: fall back to {id}.txt (root dir)
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
        """Choose the export directory (flet 0.86.2: get_directory_path returns a path string directly)."""
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
        """Export all: each task goes through _export_one (unified path and logs)."""
        if self.facade is None:
            return
        for task in self._load_completed_tasks():
            self._export_one(task)
        log.record("info", "[已完成] 全部导出完成")

    # ── Facade callbacks ──

    def _on_task_change(self, snapshot):
        # TaskSnapshot has no service_type field (type is the service name: llama/api/transcribe...)
        # Use _task_kind to categorize (ignore anything other than translation/transcription)
        if not self._task_kind(snapshot):
            return
        status = getattr(snapshot, "status", "")
        if status in ("completed", "failed", "cancelled"):
            self.completed_tasks = self._load_completed_tasks()
            self._rebuild_list()
            # Auto-export: shares _export_one with manual export (unified path + success/failure logs)
            if self.auto_export_enabled and status == "completed":
                self._export_one(snapshot)
