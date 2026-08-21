"""Transcription page — integrates AppFacade for the MOSS transcription service and task management.

Layout (three rows):
1. Service management bar (backend fixed: MOSS)
2. Config bar (service config + transcription args)
3. Workspace (left preview + right task queue)
"""

import asyncio
import flet as ft
from pathlib import Path

from app.log import log
from app.facade import PageUiSink
from core.contracts import TranscriptionRequest
from core.system_config import load_section
from ui.theme import Layout, Palette, Radius, Typography
from ui.components import (
    _icon, _text, divider, _shadow, panel_header,
    service_management_bar, bordered_button,
)
from ui.widgets.config_picker import config_picker
from ui.widgets.task_list import task_queue_panel
from core.writer import format_lrc_time


# Decodable extension whitelist for MOSS (ModelRunner): narrowed so container formats don't enqueue and then FAIL
_MOSS_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}


def build_transcribe(page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
    """Compatibility wrapper — create a TranscribePage instance and build the UI."""
    return TranscribePage(page, facade, file_picker).build()


class TranscribePage:
    """Transcription workbench page instance — holds state long-term so it survives navigation switches."""

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── Service state ──
        self.is_online = False
        self.is_loading = False
        self.is_paused = True  # task queue starts paused: only enabled once the service is loaded
        # This branch (main) transcribes with MOSS only: the service key is the task type
        self.current_backend = "moss"
        self.service_device = None  # actual working device (cuda/cpu), colored in the status bar

        # ── Config picker cache (MOSS-specific: service config / transcription args / hotwords) ──
        self.selected_moss_server = None
        self.selected_moss_args = None
        self.selected_hotword = None

        # Read the "transcription default config" at app startup (configs/system/default.ini [transcribe])
        self._default_cfg: dict = {}
        self._load_default_config()

        # ── Preview state ──
        self.preview_value = ""

        # ── Queue state cache ──
        self.current_task = None
        self.waiting_tasks = []

        # ── Refs ──
        self.status_dot = ft.Ref[ft.Container]()
        self.status_label = ft.Ref[ft.Text]()
        self.start_btn = ft.Ref[ft.TextButton]()
        self.stop_btn = ft.Ref[ft.TextButton]()
        self.preview_text = ft.Ref[ft.TextField]()
        self.task_container = ft.Ref[ft.Container]()

        # ── Config picker getters (assigned at build) ──
        self._get_moss_model = None      # service config (configs/models/moss*.json)
        self._get_transcribe = None      # transcription args (moss_args)
        self._get_hotword = None

        # ── Service bar container ──
        self._service_bar_ctrl = None

        # ── UI sink wiring (registration name fixed as 'moss') ──
        self._sink = None
        if self.facade is not None and hasattr(self.facade, "register_ui_sink"):
            self._sink = PageUiSink(page, self)
            self.facade.register_ui_sink(self.current_backend, self._sink)

    def _load_default_config(self) -> None:
        """Read the "transcription default config" at app startup (configs/system/default.ini [transcribe]).

        Initializes the default selections of the MOSS service config / transcription args /
        hotwords dropdowns; keeps None when the file is missing or fails to parse (dropdowns
        fall back to the first option).
        """
        try:
            data = load_section("transcribe")
        except Exception:
            return
        self._default_cfg = data
        self.selected_moss_server = data.get("moss_server") or None
        self.selected_moss_args = data.get("moss_args") or None
        self.selected_hotword = data.get("hotwords") or None

    # ── Public interface ──

    def build(self) -> ft.Control:
        """Build/rebuild the transcription page UI."""

        # ── Service management bar (MOSS-specific) ──
        self._service_bar_ctrl = ft.Container(content=self._build_service_bar())

        # ── Result preview ──
        preview_field = ft.TextField(
            ref=self.preview_text,
            hint_text="转写结果将在此处预览...",
            hint_style=ft.TextStyle(size=13, color=Palette.SUBTEXT),
            text_style=ft.TextStyle(size=13, color=Palette.TEXT,
                                    font_family="Consolas"),
            bgcolor=Palette.BG,
            border_color=Palette.BORDER,
            border_radius=12,
            multiline=True,
            read_only=True,
            expand=True,
            value=self.preview_value,
        )

        preview_section = ft.Container(
            content=ft.Column([
                panel_header("结果预览",
                    # the pick-audio-file button sits at the far right of the result preview row (former export-LRC position)
                    trailing=bordered_button(
                        "选择音频文件", ft.Icons.FOLDER_OPEN,
                        on_click=self._pick_file,
                    )),
                preview_field,
            ], spacing=8, expand=True),
            bgcolor=Palette.SURFACE,
            border_radius=Radius.XL,
            padding=ft.Padding.all(20),
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            shadow=_shadow("low"),
            expand=3,
        )

        # ── Task queue ──
        task_panel = ft.Container(
            ref=self.task_container,
            content=task_queue_panel(
                waiting_tasks=[],
                callbacks={
                    "on_clear": self._on_clear,
                    "on_pause_toggle": self._on_pause_toggle,
                    "is_paused": self.is_paused,
                },
                empty_text="暂无转写任务",
                expand=2,
            ),
            expand=2,
        )

        # ── Workspace (two columns on wide screens; stacked vertically on narrow screens) ──
        # Wide-screen workspace no longer has a fixed height: expand fills the content
        # area's remaining height so the preview/task-queue panel bottoms align with the
        # content-area bottom and display fully.
        is_narrow = self.page.width > 0 and self.page.width < Layout.DESKTOP_MIN_WIDTH
        workspace = (
            # Narrow screens likewise do not scroll: flex children collapse to 0 height
            # inside a scroll container (not shown); panels already scroll internally,
            # so the outer layer uses bounded flex allocation
            ft.Column([
                preview_section,
                ft.Container(height=Layout.COLUMN_SPACING),
                task_panel,
            ], spacing=0, expand=True)
            if is_narrow else
            ft.Row([
                preview_section,
                ft.Container(width=Layout.COLUMN_SPACING),
                task_panel,
            ], expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)
        )

        # The root column does not scroll: in flet 0.86.2 a scroll container (Flutter
        # ListView) has an unbounded main axis, so inner expand children collapse to 0
        # height and are not shown; the window is fixed, so no full-page scrolling is needed.
        return ft.Column([
            self._service_bar_ctrl,
            ft.Container(height=Layout.SECTION_GAP),
            workspace,
        ], spacing=0, expand=True,
           horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    def _build_service_bar(self) -> ft.Control:
        """Build the MOSS-specific service management bar (no backend switch; config dirs fixed to moss/moss_args)."""
        moss_model_picker, self._get_moss_model = config_picker(
            "服务配置", [],
            config_type="moss",
            glob_filter="*.json",
            width=Layout.PICKER_WIDTH_SM,   # same width as the translate page service config
            value=self.selected_moss_server,
        )
        # ── Transcription args (moss_args) ──
        transcribe_picker, self._get_transcribe = config_picker(
            "转写参数", [],
            config_type="moss_args",
            width=200,
            value=self.selected_moss_args,
        )
        # ── Hotwords (configs/transcribe/hotwords/*.json; MOSS official approach: appended to the prompt) ──
        hotwords_picker, self._get_hotword = config_picker(
            "热词", ["无"],
            config_type="hotwords",
            width=200,
            value=self.selected_hotword,
        )

        return service_management_bar(
            service_type="transcribe",
            status_dot_ref=self.status_dot,
            status_label_ref=self.status_label,
            start_btn_ref=self.start_btn,
            stop_btn_ref=self.stop_btn,
            backend_selector=None,          # this branch is fixed to MOSS; no backend switch
            config_dropdown=None,           # service config moved into the start/stop group (just left of the start button)
            extra_items=[transcribe_picker, hotwords_picker],
            pre_start_actions=[moss_model_picker],
            on_start=self._load_model,
            on_stop=self._unload_model,
            merge_backend=False,
            row1_cols=(4, 8),               # status / start-stop group (incl. service config, right-aligned in group)
            row2_cols=(6, 6),               # transcription args / hotwords
        )

    def _re_register_sink(self):
        """Re-register the sink to the current service key (fixed as 'moss' on the main branch)."""
        if self._sink is not None and self.facade is not None \
                and hasattr(self.facade, "register_ui_sink"):
            self.facade.register_ui_sink(self.current_backend, self._sink)

    def save_ui_state(self) -> None:
        """Sync state from control refs to instance attributes before leaving the page (MOSS-specific cache)."""
        if self.preview_text.current:
            self.preview_value = self.preview_text.current.value or ""
        if self._get_moss_model:
            self.selected_moss_server = self._get_moss_model()
        if self._get_transcribe:
            self.selected_moss_args = self._get_transcribe()
        if self._get_hotword:
            self.selected_hotword = self._get_hotword()

    def refresh(self) -> None:
        """Refresh on facade callbacks or when switching back to the page."""
        if self.facade:
            s = self.facade.get_service_status(self.current_backend)
            self.is_online = s.get("status") == "online"
        self.service_device = self._current_device()
        self._update_service_status()
        self._refresh_tasks()

    def update_service_status(self, online: bool, loading: bool = False, device: str | None = None):
        """Backend push: update the service status (reassign + refresh status dot/label/start-stop buttons).

        Queue-enable linkage (only when the service status changes): stopped/loading → paused;
        loaded successfully → enabled; same-status pushes (e.g. refresh) do not override a user's manual pause.
        """
        was_online = self.is_online
        self.is_online = bool(online)
        self.is_loading = bool(loading)
        if device is not None:
            self.service_device = device
        elif not self.is_online:
            self.service_device = None
        if was_online != self.is_online:
            self.is_paused = not self.is_online
        self._update_service_status()
        if was_online != self.is_online:
            self._refresh_tasks()

    def update_tasks(self, current, waiting):
        """Backend push: reassign the task cache and rebuild the queue panel (dict projection, matching the translate page).

        When the current task is running and the payload carries transcription segments →
        live preview (scrolled/refreshed as transcription progresses).
        """
        self.current_task = current
        self.waiting_tasks = waiting or []
        # MOSS lazy loading: _device only resolves from auto to cuda:0/cpu after the first
        # task completes; task pushes are the best time to refresh the status-bar device.
        self.service_device = self._current_device()
        self._update_service_status()
        self._render_panel(self._build_callbacks(self.waiting_tasks))
        # Live preview: while transcribing, update the preview area with confirmed segments
        # (segments must be a list). MOSS (StreamingModelRunner) pushes payload.segments;
        # the final result is still fully refreshed via update_finished_tasks when done.
        if isinstance(current, dict) and current.get("status") == "running":
            payload = current.get("payload")
            segs = payload.get("segments", []) if isinstance(payload, dict) else []
            if isinstance(segs, list) and segs:
                self._render_segments_preview(segs)

    def update_finished_tasks(self, tasks):
        """Backend push: update the preview with the latest completed results (transcribe-page feature)."""
        if tasks:
            self._update_preview()

    def register_callbacks(self) -> None:
        """The old callback-collection path is deprecated — replaced by UI sink pushes (registered in __init__)."""
        pass

    # ── Internal methods — service state and tasks ──

    @staticmethod
    def _safe_update(ctrl):
        """Silently skip update when the control is not mounted (flet 0.86.2 first-frame/rebuild timing)."""
        if ctrl is None:
            return
        try:
            ctrl.update()
        except RuntimeError:
            pass

    def _update_service_status(self):
        c = Palette.SUCCESS if self.is_online else (Palette.WARNING if self.is_loading else Palette.WARNING)
        label = ("MOSS服务已加载" if self.is_online else
                 "MOSS服务加载中…" if self.is_loading else
                 "MOSS服务未加载")
        if self.status_dot.current:
            self.status_dot.current.bgcolor = c
            self._safe_update(self.status_dot.current)
        if self.status_label.current:
            self.status_label.current.value = label
            self._apply_device_span(self.status_label.current)
            self._safe_update(self.status_label.current)
        if self.start_btn.current:
            self.start_btn.current.visible = not self.is_online and not self.is_loading
            self.start_btn.current.disabled = self.is_loading
            self._safe_update(self.start_btn.current)
        if self.stop_btn.current:
            self.stop_btn.current.visible = self.is_online or self.is_loading
            self.stop_btn.current.disabled = self.is_loading
            self._safe_update(self.stop_btn.current)

    @staticmethod
    def _device_display(device: str | None) -> tuple[str | None, str | None]:
        """Return (display text, color); CUDA green / CPU orange; unknown returns None."""
        if not device:
            return None, None
        dv = str(device).lower()
        if dv.startswith("cuda"):
            return "CUDA", Palette.SUCCESS
        if dv.startswith("cpu"):
            return "CPU", Palette.WARNING
        return dv.upper(), Palette.PRIMARY

    def _apply_device_span(self, label: ft.Text) -> None:
        text, color = self._device_display(self.service_device)
        if text and color and self.is_online:
            label.spans = [
                ft.TextSpan(
                    f" · {text}",
                    style=ft.TextStyle(color=color, size=12, weight=ft.FontWeight.W_700),
                )
            ]
        else:
            label.spans = []

    def _current_device(self) -> str | None:
        if self.facade is not None and hasattr(self.facade, "get_service_device"):
            try:
                return self.facade.get_service_device(self.current_backend)
            except Exception:
                return None
        return None

    def _on_clear(self):
        """Clear waiting tasks (keep the currently running task; completed tasks are managed by the completed page)."""
        if self.facade is not None:
            try:
                self.facade.clear_queue(self.current_backend)
            except Exception as ex:
                log.record("error", f"[转写] 清空队列失败: {ex}")
        self.waiting_tasks = []
        self._refresh_tasks()
        if self.page:
            self.page.show_dialog(
                ft.SnackBar(ft.Text("等待任务已清空"), bgcolor=Palette.SUCCESS)
            )

    def _on_pause_toggle(self):
        """Pause/resume the queue: enable only after the service loads; pausing is always allowed (it is already paused when the service is not loaded)."""
        if self.is_paused and not self.is_online:
            msg = "需先加载服务才能开启队列"
            log.record("warn", f"[转写] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        self.is_paused = not self.is_paused
        if self.facade is not None:
            try:
                if self.is_paused:
                    self.facade.pause_queue(self.current_backend)
                else:
                    self.facade.resume_queue(self.current_backend)
            except Exception as ex:
                log.record("error", f"[转写] 队列切换失败: {ex}")
        self._refresh_tasks()

    def _build_callbacks(self, waiting):
        """Build the queue panel callbacks (shared by _refresh_tasks and update_tasks)."""
        def _on_cancel(tid):
            if self.facade:
                try:
                    self.facade.cancel_task(tid)
                except Exception as ex:
                    log.record("error", f"[转写] 取消失败: {ex}")
            # sync local caches (both current and waiting are possible)
            if self.current_task and self.current_task.get("id") == tid:
                self.current_task = None
            self.waiting_tasks = [t for t in self.waiting_tasks if t.get("id") != tid]
            self._render_panel(self._build_callbacks(self.waiting_tasks))

        def _on_move_up(tid):
            for i, t in enumerate(waiting):
                if t.get("id") == tid and i > 0:
                    self.facade.reorder_task(tid, i - 1)
                    break

        def _on_move_down(tid):
            for i, t in enumerate(waiting):
                if t.get("id") == tid:
                    # boundary: the last item cannot move down (avoid index exceeding queue length-1)
                    if i < len(waiting) - 1:
                        self.facade.reorder_task(tid, i + 1)
                    break

        return {
            "on_cancel": _on_cancel,
            "on_move_up": _on_move_up,
            "on_move_down": _on_move_down,
            "on_clear": self._on_clear,
            "on_pause_toggle": self._on_pause_toggle,
            "is_paused": self.is_paused,
        }

    def _refresh_tasks(self):
        if self.task_container.current is None:
            return
        if self.facade is None:
            self.current_task = None
            self.waiting_tasks = []
        else:
            cur = self.facade.list_current_task(task_type=self.current_backend)
            waiting = self.facade.list_waiting_tasks(task_type=self.current_backend)
            self.current_task = self._snap_to_dict(cur) if cur is not None else None
            self.waiting_tasks = [self._snap_to_dict(t) for t in waiting]
        self._render_panel(self._build_callbacks(self.waiting_tasks))

    def _render_panel(self, callbacks=None):
        """Rebuild the queue panel (standalone method shared by push/pull/local-cache operations)."""
        if self.task_container.current is None:
            return
        if callbacks is None:
            callbacks = self._build_callbacks(self.waiting_tasks)
        self.task_container.current.content = task_queue_panel(
            current_task=self.current_task,
            waiting_tasks=self.waiting_tasks,
            callbacks=callbacks,
            empty_text="暂无转写任务",
            expand=2,
        )
        self._safe_update(self.task_container.current)

    @staticmethod
    def _snap_to_dict(snap) -> dict:
        """Project a TaskSnapshot to a dict (fields match AppFacade._project; used for local-cache operations)."""
        name = snap.file_name or ""
        return {
            "id": snap.id,
            "type": snap.type,
            "status": snap.status,
            "progress": float(snap.progress),
            "file_name": name,
            "input_summary": snap.input_summary or name,
            "result": snap.result,
            "error": snap.error,
            "payload": getattr(snap, "payload", None) or {},
        }

    def _render_segments_preview(self, segments: list):
        """Render transcription segments (dict or Segment compatible) as standard LRC: ``[mm:ss.xx]<speaker>text``."""
        if self.preview_text.current is None:
            return
        lines = []
        for seg in segments:
            start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            speaker = seg.get("speaker", "") if isinstance(seg, dict) else getattr(seg, "speaker", "")
            prefix = f"<{speaker}>" if speaker else ""
            lines.append(f"[{format_lrc_time(start)}]{prefix}{text.strip()}")
        self.preview_value = "\n".join(lines)
        self.preview_text.current.value = self.preview_value
        self._safe_update(self.preview_text.current)

    def _update_preview(self):
        """Fully update the preview area with the latest completed task result (no segment truncation).

        Do not overwrite the live preview while a task is transcribing (prevents the previous full
        result and the current live content from flickering alternately; the moment the task
        finishes, current empties and naturally switches to the new result).
        """
        if self.current_task and self.current_task.get("status") == "running":
            return
        if self.preview_text.current is None or self.facade is None:
            return
        completed = self.facade.list_completed_tasks(task_type=self.current_backend)
        for snap in reversed(completed):
            if getattr(snap.status, "value", snap.status) == "completed" and snap.result:
                segments = snap.result.get("segments", []) if isinstance(snap.result, dict) else []
                if segments:
                    self._render_segments_preview(segments)
                    return
        # show a default hint when there is no result
        if self.preview_text.current:
            self.preview_value = "暂无转写结果"
            self.preview_text.current.value = self.preview_value
            self._safe_update(self.preview_text.current)

    # ── Internal methods — service operations ──

    async def _load_model(self, e):
        if self.facade is None:
            return
        try:
            # First set "loading": the status bar shows the loading message and the start
            # button is hidden to prevent repeated clicks.
            self.update_service_status(False, True)
            # core loading is synchronous/blocking (model load) → thread pool avoids
            # blocking the event loop; config_path comes from the service config picker
            # (configs/models/moss*.json)
            config_path = self._get_moss_model() if self._get_moss_model else None
            await asyncio.to_thread(self.facade.start_service, self.current_backend, None, config_path)
            # proactively refresh the status (push fallback: update immediately even if the backend doesn't callback)
            self.service_device = self._current_device()
            self.update_service_status(True, False, self.service_device)
        except Exception as ex:
            log.record("error", f"[转写] 加载失败: {ex}")
            self.service_device = None
            self.update_service_status(False, False)
            if self.page:
                self.page.show_dialog(
                    ft.SnackBar(ft.Text(f"MOSS 加载失败: {ex}"), bgcolor=Palette.ERROR)
                )

    async def _unload_model(self, e):
        if self.facade is None:
            return
        try:
            # First set "unloading": disable the stop button (prevent repeated clicks) and update the status bar
            self.update_service_status(False, True)
            await asyncio.to_thread(self.facade.stop_service, self.current_backend)
            # proactively refresh the status (push fallback)
            self.service_device = None
            self.update_service_status(False, False)
        except Exception as ex:
            log.record("error", f"[转写] 停止失败: {ex}")
            self.update_service_status(False, False)

    # ── Internal methods — file selection and submission ──

    async def _pick_file(self, e):
        """Choose the import method — AlertDialog branches: pick files (multi-select, audio validated) / pick a folder."""
        if self.file_picker is None or self.page is None:
            return
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("选择导入方式"),
            content=ft.Column([
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.FILE_OPEN),
                    title=ft.Text("选择文件"),
                    subtitle=ft.Text("选择一个或多个音频文件（批量不预览直接入队）"),
                    on_click=self._on_choose_files,
                ),
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.FOLDER_OPEN),
                    title=ft.Text("选择文件夹"),
                    subtitle=ft.Text("导入文件夹顶层的所有音频文件（不递归子目录）"),
                    on_click=self._on_choose_directory,
                ),
            ], spacing=0, tight=True),
            actions=[ft.TextButton("取消", on_click=self._close_import_dialog)],
        )
        self._import_dialog = dlg
        self.page.show_dialog(dlg)

    def _close_import_dialog(self, e=None):
        dlg = getattr(self, "_import_dialog", None)
        if dlg is not None:
            dlg.open = False
            self.page.update()

    async def _on_choose_files(self, e):
        self._close_import_dialog()
        if self.file_picker is None:
            return
        try:
            files = await self.file_picker.pick_files(
                allow_multiple=True,
                dialog_title="选择音频文件",
                file_type=ft.FilePickerFileType.ANY,
            )
        except Exception as ex:
            log.record("error", f"[转写] 选择文件失败: {ex}")
            return
        if not files:
            return
        await self._process_selected([Path(f.path) for f in files])

    async def _on_choose_directory(self, e):
        self._close_import_dialog()
        if self.file_picker is None:
            return
        try:
            # flet 0.86.2: the folder-picking API is get_directory_path (returns a path string directly)
            result = await self.file_picker.get_directory_path(dialog_title="选择待导入文件夹")
        except Exception as ex:
            log.record("error", f"[转写] 选择文件夹失败: {ex}")
            return
        if not result:
            return
        paths = self._import_directory(Path(result))
        if not paths:
            msg = "所选文件夹中没有可导入的文件"
            log.record("warn", f"[转写] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        await self._process_selected(paths)

    def _import_directory(self, path: Path) -> list:
        """Scan the folder's top level: only is_file, no recursion into subdirectories, sorted by file name."""
        try:
            entries = [p for p in path.iterdir() if p.is_file()]
        except OSError as ex:
            log.record("error", f"[转写] 扫描文件夹失败: {ex}")
            return []
        return sorted(entries, key=lambda p: p.name.lower())

    async def _process_selected(self, paths: list):
        """Select-and-submit: single/multiple files and folders are validated uniformly and enqueued directly."""
        self._enqueue_transcriptions(paths)

    def _enqueue_transcriptions(self, paths: list):
        """Validate in batch and enqueue: suffix whitelist check; non-audio files are rejected without interrupting the rest.

        MOSS uses the decodable whitelist (ModelRunner's decodable set).
        """
        allowed = _MOSS_EXTENSIONS
        ok_paths, rejected = [], []
        for p in paths:
            try:
                abs_p = p.resolve()
            except OSError:
                abs_p = p.absolute()
            if abs_p.suffix.lower() not in allowed:
                rejected.append(p.name)
                continue
            ok_paths.append(abs_p)
        if rejected:
            names = ", ".join(rejected[:5]) + ("…" if len(rejected) > 5 else "")
            msg = f"{len(rejected)} 个文件不是支持的音频格式: {names}"
            log.record("warn", f"[转写] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
        if not ok_paths:
            return
        try:
            reqs = self.build_transcription_requests(ok_paths)
        except Exception as ex:
            log.record("error", f"[转写] 任务打包失败: {ex}")
            return
        self._enqueue_requests(reqs)

    def build_transcription_requests(self, paths: list) -> list:
        """Package transcription requests: TranscriptionRequest(task_type=current backend, file_path, configs).

        paths=[...] → one request per file (single-file/batch unified).
        configs keys:
        - args: transcription args template (configs/transcribe/args/*.json)
        - hotwords: hotwords file (MOSS appends to the prompt inside the executor per the official recipe)
        """
        configs = {}
        if self._get_transcribe:
            configs["args"] = self._get_transcribe()
        if self._get_hotword:
            hotwords_path = self._get_hotword()  # "None" selection → None → not set
            if hotwords_path:
                configs["hotwords"] = hotwords_path
        return [TranscriptionRequest(
            task_type=self.current_backend,
            file_path=p,
            file_name=p.name,
            configs=configs,
        ) for p in paths]

    def _enqueue_requests(self, reqs: list) -> int:
        """Unified enqueue: submit one by one + local cache + panel refresh + snackbar; returns the number successfully submitted.

        Once the backend's update_tasks push is wired, the push takes over display.
        """
        submitted = 0
        if self.facade is not None:
            for r in reqs:
                try:
                    self.facade.submit_transcription(r)
                    submitted += 1
                except Exception as ex:
                    log.record("error", f"[转写] 提交失败: {ex}")
        else:
            submitted = len(reqs)  # facade not wired: local-cache display only
        for r in reqs:
            path = getattr(r, "file_path", None)
            name = getattr(r, "file_name", "") or (Path(path).name if path else "")
            self.waiting_tasks.append({
                "id": Path(path).stem if path else "pending",
                "type": getattr(r, "task_type", self.current_backend),
                "status": "pending",
                "progress": 0,
                "file_name": name,
                "input_summary": name,
            })
        # Render the current cache (no re-fetch — a fetch could overwrite the just-appended
        # local cache; the real backend pushes update_tasks via the sink with real id/status)
        self._render_panel(self._build_callbacks(self.waiting_tasks))
        if self.page:
            self.page.show_dialog(
                ft.SnackBar(ft.Text(f"已加入队列 {submitted} 个任务"), bgcolor=Palette.SUCCESS)
            )
        return submitted

