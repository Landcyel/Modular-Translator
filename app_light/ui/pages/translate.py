"""Translate workbench page — UI building kept; functional implementation removed (rebuild pending).

Kept:
- build()'s three-row layout and all controls (visual structure unchanged)
- save_ui_state() / refresh() (layout.py navigation contract)
- build_translate() compatibility wrapper

Removed: register_callbacks() and the remaining service/task/submit/queue/callback
implementations (_start_service/_stop_service/_submit_translation/_update_service_status/
_on_clear/_on_pause_toggle/_refresh_tasks/_on_service_change/_on_task_change are all
placeholders; only their signatures remain for build()'s control callback bindings,
method bodies are empty — button clicks doing nothing is expected behavior).
Restored: backend selection (_on_backend_switch/_set_backend_style, an Llama/API
two-way Switch, styled like the completed page's auto-export switch) and file selection
(_pick_file, including text-format validation).
Original implementation in ui/pages/translate.py.bak.
"""

import flet as ft
from pathlib import Path
import time
import asyncio

from app.log import log
from core.contracts import TranslationRequest
from app.facade import PageUiSink
from core.system_config import load_section
from ui.theme import Layout, Palette, Radius, Typography
from ui.components import (
    _icon, _text, divider, _shadow, panel_header,
    service_management_bar, bordered_button,
)
from ui.widgets.config_picker import config_picker
from ui.widgets.task_list import task_queue_panel


# Encoding order tried when decoding text files (covers common Chinese and Japanese text encodings)
_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp932", "shift_jis", "gbk")


def _read_text_file(path: Path) -> str | None:
    """Read a file as text; returns None for non-text formats (binary / undecodable).

    The returned text uses ``\\n`` newlines uniformly (``\\r\\n``/``\\r`` → ``\\n``): both the
    preview into the TextField and the later temp-file write pass plain ``\\n``, avoiding
    Windows ``write_text``'s default newline=None turning ``\\r\\n`` into a double ``\\r\\r\\n``
    (an extra blank line per row when the executor reads it back).
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:          # NUL byte → binary signature, reject
        return None
    for enc in _TEXT_ENCODINGS:
        try:
            return data.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def build_translate(page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
    """Compatibility wrapper — create a TranslatePage instance and build the UI."""
    return TranslatePage(page, facade, file_picker).build()


class TranslatePage:
    """Translate workbench page instance — holds state long-term so it survives navigation switches."""

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── Service state ──
        self.is_online = False
        self.is_loading = False
        self.is_paused = True  # task queue starts paused: only enabled once the service is loaded
        self.current_backend = "llama"  # "llama" | "api"
        self.service_device = None  # actual working device (cuda/cpu/api), colored in the status bar

        # ── Config picker cache ──
        self.selected_server = None
        self.selected_prompts = None
        self.selected_args = None
        self.selected_rule = None
        self.selected_glossary = None

        # ── Translate default config (configs/defaults/translate/default.json) ──
        self._default_cfg: dict = {}
        self._load_default_config()

        # ── Input state ──
        self.input_path = None
        self.input_value = ""

        # ── Queue state cache ──
        self.current_task = None
        self.waiting_tasks = []
        # Absolute paths of already-enqueued files (dedup for batch import; case-insensitive on Windows)
        self._queued_paths = set()  # legacy dedup compat (deprecated; field kept to avoid external-reference errors)

        # ── Refs ──
        self.status_dot = ft.Ref[ft.Container]()
        self.status_label = ft.Ref[ft.Text]()
        self.start_btn = ft.Ref[ft.TextButton]()
        self.stop_btn = ft.Ref[ft.TextButton]()
        self.input_text = ft.Ref[ft.TextField]()
        self.task_container = ft.Ref[ft.Container]()
        self.backend_switch = ft.Ref[ft.Switch]()      # backend select switch (off=Llama / on=API)
        self.backend_name_label = ft.Ref[ft.Text]()    # dynamic backend name beside the switch (Llama/API)

        # ── Config picker getters (assigned at build) ──
        self._get_server = None
        self._get_prompts = None
        self._get_args = None
        self._get_rule = None
        self._get_glossary = None

        # ── Service-config dropdown config_type switcher (assigned at build; called on backend switch) ──
        self._set_server_ctype = None
        # ── Translation-args dropdown config_type switcher (assigned at build; called on backend switch) ──
        self._set_args_ctype = None

        # ── UI sink wiring (backend push → update_service_status/update_tasks) ──
        # Registered under name = current_backend (service key: 'llama'/'api'); re-registered on backend switch
        self._sink = None
        if self.facade is not None and hasattr(self.facade, "register_ui_sink"):
            self._sink = PageUiSink(page, self)
            self.facade.register_ui_sink(self.current_backend, self._sink)

    # ── Public interface ──

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
            self._rebuild_task_panel()

    def update_tasks(self, current, waiting):
        """Backend push: reassign the task cache and rebuild the queue panel (completed tasks are managed by CompletedPage)."""
        self.current_task = current
        self.waiting_tasks = waiting or []
        self._rebuild_task_panel()

    def build(self) -> ft.Control:
        """Build/rebuild the translate page UI (visual structure kept as-is)."""

        # ── Config pickers (using cached initial values) ──
        prompts_picker, self._get_prompts = config_picker(
            "提示词", [], config_type="prompts", width=Layout.PICKER_WIDTH_SM,
            value=self.selected_prompts,
        )
        args_picker, self._get_args = config_picker(
            "翻译参数", [], config_type="translate_args", width=Layout.PICKER_WIDTH_SM,
            value=self.selected_args,
        )
        self._set_args_ctype = self._get_args.set_config_type
        rule_picker, self._get_rule = config_picker(
            "规则", ["无"], config_type="rules", width=Layout.PICKER_WIDTH_SM,
            value=self.selected_rule,
        )
        glossary_picker, self._get_glossary = config_picker(
            "术语表", ["无"], config_type="glossary", width=Layout.PICKER_WIDTH_SM,
            value=self.selected_glossary,
        )

        # ── Backend select switch (Switch on top, backend name text below; no "Backend" label) ──
        # Off = Llama (local default backend), On = API (remote); switching goes through _on_backend_switch.
        backend_btns = ft.Column([
            ft.Switch(
                ref=self.backend_switch,
                value=(self.current_backend == "api"),
                active_color=Palette.PRIMARY,
                on_change=self._on_backend_switch,
            ),
            ft.Text(ref=self.backend_name_label, value="Llama", size=12,
                    weight=ft.FontWeight.W_600, color=Palette.PRIMARY),
        ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        # ── Service config Dropdown (config_type defaults to llama, matching current_backend;
        #    set_config_type switches the target directory on backend switch)
        #    width matches the other config pickers (PICKER_WIDTH_SM) ──
        server_picker, self._get_server = config_picker(
            "服务配置", [], config_type="llama", width=Layout.PICKER_WIDTH_SM,
            value=self.selected_server,
        )
        self._set_server_ctype = self._get_server.set_config_type

        # ── Service management bar: both rows share the 12-col grid (ResponsiveRow), controls at
        #    the same index are left-aligned horizontally — the 2nd control "Service Config" in
        #    row 1 aligns with the 2nd "Translation Args" in row 2; merge_backend=True: the
        #    backend selector (Switch + backend name) and start/stop buttons merge into one
        #    trailing composite; row 2 extra_items: prompts/translation args/rules/glossary ──
        service_bar = service_management_bar(
            service_type="translate",
            status_dot_ref=self.status_dot,
            status_label_ref=self.status_label,
            start_btn_ref=self.start_btn,
            stop_btn_ref=self.stop_btn,
            backend_selector=backend_btns,
            config_dropdown=server_picker,
            extra_items=[prompts_picker, args_picker, rule_picker, glossary_picker],
            on_start=self._start_service,
            on_stop=self._stop_service,
            merge_backend=True,           # backend selector + start/stop group merged into one trailing composite
            row1_cols=(3, 3, 6),          # status / service config / backend+start-stop
            row2_cols=(3, 3, 3, 3),       # prompts / translation args / rules / glossary
        )

        # ── Input area ──
        input_area = ft.TextField(
            ref=self.input_text,
            hint_text="输入待翻译的日文文本，或点击下方按钮加载文件...",
            hint_style=ft.TextStyle(size=Typography.BODY, color=Palette.SUBTEXT),
            text_style=ft.TextStyle(size=Typography.BODY_LG, color=Palette.TEXT),
            bgcolor=Palette.BG,
            border_color=Palette.BORDER,
            border_radius=Radius.LG,
            multiline=True,
            expand=True,
            value=self.input_value,
        )

        input_actions = ft.Row([
            ft.TextButton(
                "清空",
                icon=ft.Icons.CLEAR,
                on_click=lambda e: (
                    setattr(self.input_text.current, "value", ""),
                    self.input_text.current.update(),
                    setattr(self, "input_value", ""),
                    setattr(self, "input_path", None),
                ) if self.input_text.current else None,
            ),
            ft.FilledButton(
                "提交至队列",
                icon=ft.Icons.SEND,
                on_click=self._submit_translation,
                style=ft.ButtonStyle(
                    bgcolor=Palette.PRIMARY,
                    color="#FFFFFF",
                    shape=ft.RoundedRectangleBorder(radius=Radius.MD),
                ),
            ),
        ], spacing=8, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        input_panel = ft.Container(
            content=ft.Column([
                # the pick-file button is right-aligned at the end of the "Source Text" title row
                panel_header("源文本",
                    trailing=bordered_button(
                        "选择文件", ft.Icons.FILE_OPEN,
                        on_click=self._pick_file,
                    )),
                divider(),
                input_area,
                input_actions,
            ], spacing=10, expand=True),
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
                # render from the instance cache on page-switch rebuild (consistent with _rebuild_task_panel), avoiding task loss
                current_task=self.current_task,
                waiting_tasks=self.waiting_tasks,
                callbacks={
                    "on_cancel": self._on_cancel_task,
                    "on_move_up": self._on_move_up,
                    "on_move_down": self._on_move_down,
                    "on_clear": self._on_clear,
                    "on_pause_toggle": self._on_pause_toggle,
                    "is_paused": self.is_paused,
                },
                empty_text="暂无翻译任务",
                expand=2,
            ),
            expand=2,
        )

        # ── Workspace (two columns on wide screens; stacked vertically on narrow screens) ──
        # Wide-screen workspace no longer has a fixed height: expand fills the content
        # area's remaining height so the source-text/task-queue panel bottoms align with
        # the content-area bottom and display fully.
        is_narrow = self.page.width > 0 and self.page.width < Layout.DESKTOP_MIN_WIDTH
        workspace = (
            # Narrow screens likewise do not scroll: flex children collapse to 0 height
            # inside a scroll container (not shown); panels already scroll internally
            # (task_rows/input field), so the outer layer uses bounded flex allocation
            ft.Column([
                input_panel,
                ft.Container(height=Layout.COLUMN_SPACING),
                task_panel,
            ], spacing=0, expand=True)
            if is_narrow else
            ft.Row([
                input_panel,
                ft.Container(width=Layout.COLUMN_SPACING),
                task_panel,
            ], expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)
        )

        # The root column does not scroll: in flet 0.86.2 a scroll container (Flutter
        # ListView) has an unbounded main axis, so inner expand children collapse to 0
        # height and are not shown; the window is fixed, so no full-page scrolling is needed.
        return ft.Column([
            service_bar,
            ft.Container(height=Layout.SECTION_GAP),
            workspace,
        ], spacing=0, expand=True,
           horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    def save_ui_state(self) -> None:
        """Sync state from control refs to instance attributes before leaving the page."""
        if self.input_text.current:
            self.input_value = self.input_text.current.value or ""
        if self._get_server:
            self.selected_server = self._get_server()
        if self._get_prompts:
            self.selected_prompts = self._get_prompts()
        if self._get_args:
            self.selected_args = self._get_args()
        if self._get_rule:
            self.selected_rule = self._get_rule()
        if self._get_glossary:
            self.selected_glossary = self._get_glossary()

    def refresh(self) -> None:
        """Fallback after nav-back/submit/start-stop: pull the latest snapshots from the backend → render via update_*."""
        if self.facade:
            try:
                s = self.facade.get_service_status(self.current_backend)
                self.is_online = s.get("status") == "online"
            except Exception:
                pass
        self.service_device = self._current_device()
        self._update_service_status()
        self._refresh_tasks()

    def build_translation_requests(self, source) -> list:
        """Package the input into a list of TranslationRequests (always returns a list).

        - str (text) → written to temp/input_{YYYYMMDDHHMMSS}.txt as the file_path
        - Path → used directly as the file_path
        - list[Path] → generated one per item

        task_type is self.current_backend ('llama'/'api' — the frontend uses the backend value as
        the service unit name); configs is the path dict read from the config pickers
        (translate_config / prompts / glossary / rule; glossary keeps a None key when unselected).
        """
        configs = {
            "translate_config": self._get_args() if self._get_args else None,
            "prompts":          self._get_prompts() if self._get_prompts else None,
            "glossary":         self._get_glossary() if self._get_glossary else None,
            "rule":             self._get_rule() if self._get_rule else None,
        }

        def _make(file_path: Path, file_name: str) -> TranslationRequest:
            return TranslationRequest(
                task_type=self.current_backend,
                file_path=file_path,
                file_name=file_name,
                configs=dict(configs),
            )

        if isinstance(source, (list, tuple)):
            return [_make(Path(p), Path(p).name) for p in source]
        if isinstance(source, Path):
            return [_make(source, source.name)]
        # str text → temp file temp/input_{YYYYMMDDHHMMSS}.txt
        # Must use newline="\n": on Windows write_text's default newline=None would turn \n
        # into \r\n; if source itself is \r\n (e.g. from a file previewed into the TextField),
        # it becomes a double \r\r\n, which universal-newline parsing reads back as 2 newlines
        # → an extra blank line per row (see _read_text_file's newline normalization). Here we
        # write \n uniformly and let the reader (read_text universal newline) normalize.
        tmp_dir = Path("temp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"input_{time.strftime('%Y%m%d%H%M%S')}.txt"
        tmp.write_text(str(source), encoding="utf-8", newline="\n")
        return [_make(tmp, tmp.name)]

    # ── Internal methods — UI styles (functionality removed; placeholder signatures kept) ──

    def _set_backend_style(self):
        """Sync the backend select switch and dynamic label from current_backend."""
        switch = self.backend_switch.current
        if switch is not None and switch.value != (self.current_backend == "api"):
            switch.value = (self.current_backend == "api")
            try:
                switch.update()
            except RuntimeError:
                pass
        label = self.backend_name_label.current
        if label is not None:
            label.value = "API" if self.current_backend == "api" else "Llama"
            try:
                label.update()
            except RuntimeError:
                pass

    def _load_default_config(self) -> None:
        """Read the "translate default config" at app startup (configs/system/default.ini [translate]).

        Initializes the default selections of each config picker (llama backend); the api backend
        defaults are applied via config_picker.set_value when switching backends
        (api_server / translate_args_api). Keeps None when the file is missing or fails to parse
        (dropdowns fall back to the first option).
        """
        try:
            data = load_section("translate")
        except Exception:
            return
        self._default_cfg = data
        self.selected_server = data.get("llama_server") or None
        self.selected_prompts = data.get("prompt") or None
        self.selected_args = data.get("translate_args") or None
        self.selected_rule = data.get("rule") or None
        self.selected_glossary = data.get("glossary") or None

    def _on_backend_switch(self, e):
        """Backend select switch toggle: off=Llama (local default), on=API (remote).

        Equivalent to the original _on_llama_click/_on_api_click: links the service config dropdown
        (llama → configs/models/llama; api → configs/models/API), the translation args dropdown
        (translate_args / translate_args_api), style refresh, and sink re-registration.
        """
        self.current_backend = "api" if e.control.value else "llama"
        if self.current_backend == "llama":
            if self._set_server_ctype:
                self._set_server_ctype("llama")   # switch the service config dropdown to configs/models/llama
                if self._get_server and self._default_cfg.get("llama_server"):
                    self._get_server.set_value(self._default_cfg["llama_server"])
            if self._set_args_ctype:
                self._set_args_ctype("translate_args")   # switch the translation args dropdown to the Llama args dir
                if self._get_args and self._default_cfg.get("translate_args"):
                    self._get_args.set_value(self._default_cfg["translate_args"])
        else:
            if self._set_server_ctype:
                self._set_server_ctype("api")     # switch the service config dropdown to configs/models/API
                if self._get_server and self._default_cfg.get("api_server"):
                    self._get_server.set_value(self._default_cfg["api_server"])
            if self._set_args_ctype:
                self._set_args_ctype("translate_args_api")   # switch the translation args dropdown to the API args dir
                if self._get_args and self._default_cfg.get("translate_args_api"):
                    self._get_args.set_value(self._default_cfg["translate_args_api"])
        self._set_backend_style()
        self.service_device = self._current_device()
        self._update_service_status()   # sync status text to the current backend (Llama/API × not-loaded/loaded/loading)
        self._re_register_sink()

    def _re_register_sink(self):
        """After a backend switch: re-register the sink to the current service's key ('llama'/'api')."""
        if self._sink is not None and self.facade is not None \
                and hasattr(self.facade, "register_ui_sink"):
            self.facade.register_ui_sink(self.current_backend, self._sink)

    # ── Internal methods — service state and tasks (functionality removed; placeholder signatures kept) ──

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
        """Refresh the status dot/label/start-stop buttons."""
        dot = self.status_dot.current
        if dot is not None:
            dot.bgcolor = (
                Palette.WARNING if self.is_loading else
                Palette.SUCCESS if self.is_online else Palette.ERROR
            )
            self._safe_update(dot)
        label = self.status_label.current
        if label is not None:
            backend_name = "Llama" if self.current_backend == "llama" else "API"
            label.value = (
                f"正在加载 {backend_name} 服务" if self.is_loading else
                f"{backend_name} 服务已加载" if self.is_online else
                f"{backend_name} 服务未加载"
            )
            self._apply_device_span(label)
            self._safe_update(label)
        if self.start_btn.current is not None:
            self.start_btn.current.visible = not (self.is_online or self.is_loading)
            self._safe_update(self.start_btn.current)
        if self.stop_btn.current is not None:
            self.stop_btn.current.visible = self.is_online
            self._safe_update(self.stop_btn.current)

    @staticmethod
    def _device_display(device: str | None) -> tuple[str | None, str | None]:
        """Return (display text, color); CUDA green / CPU orange / API blue; unknown returns None."""
        if not device:
            return None, None
        dv = str(device).lower()
        if dv.startswith("cuda"):
            return "CUDA", Palette.SUCCESS
        if dv.startswith("cpu"):
            return "CPU", Palette.WARNING
        return dv.upper(), Palette.PRIMARY

    def _apply_device_span(self, label: ft.Text) -> None:
        """Append the actual working device to the status text as a colored TextSpan."""
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
        """Query the current backend's actual working device from the facade (None when no facade)."""
        if self.facade is not None and hasattr(self.facade, "get_service_device"):
            try:
                return self.facade.get_service_device(self.current_backend)
            except Exception:
                return None
        return None

    def _on_move_up(self, task_id):
        """Move a waiting task up."""
        self._move_waiting(task_id, -1)

    def _on_move_down(self, task_id):
        """Move a waiting task down."""
        self._move_waiting(task_id, 1)

    def _move_waiting(self, task_id, delta):
        """Swap positions in the local cache + sync with the facade (the core queue is the source of truth)."""
        idx = next((i for i, t in enumerate(self.waiting_tasks)
                    if isinstance(t, dict) and t.get("id") == task_id), None)
        if idx is None:
            return
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= len(self.waiting_tasks):
            return
        self.waiting_tasks[idx], self.waiting_tasks[new_idx] = \
            self.waiting_tasks[new_idx], self.waiting_tasks[idx]
        if self.facade is not None and hasattr(self.facade, "reorder_task"):
            try:
                self.facade.reorder_task(task_id, new_idx)
            except Exception as ex:
                log.record("error", f"[翻译] 同步队列顺序失败: {ex}")
        self._rebuild_task_panel()

    def _on_cancel_task(self, task_id):
        """Cancel a task (remove from the local cache + sync with the facade).

        Both the current task (current_task) and waiting tasks (waiting_tasks) must be synced:
        facade.cancel_task must always be called (core sends a cancel signal to a running task
        and removes a pending one) — previously it was only called when waiting_tasks changed,
        so the current task could never be cancelled.
        """
        self.waiting_tasks = [t for t in self.waiting_tasks
                              if not (isinstance(t, dict) and t.get("id") == task_id)]
        if isinstance(self.current_task, dict) and self.current_task.get("id") == task_id:
            self.current_task = None
        if self.facade is not None and hasattr(self.facade, "cancel_task"):
            try:
                self.facade.cancel_task(task_id)
            except Exception as ex:
                log.record("error", f"[翻译] 同步取消失败: {ex}")
        self._rebuild_task_panel()

    def _on_clear(self):
        """Clear waiting tasks (keep the currently running and completed tasks)."""
        name = self.current_backend
        if self.facade is not None:
            try:
                self.facade.clear_queue(name)
            except Exception as ex:
                log.record("error", f"[翻译] 清空队列失败: {ex}")
        self.waiting_tasks = []
        self._rebuild_task_panel()
        if self.page:
            self.page.show_dialog(
                ft.SnackBar(ft.Text("等待任务已清空"), bgcolor=Palette.SUCCESS)
            )

    def _on_pause_toggle(self):
        """Pause/resume the queue: enable only after the service loads; pausing is always allowed (it is already paused when the service is not loaded)."""
        if self.is_paused and not self.is_online:
            msg = "需先加载服务才能开启队列"
            log.record("warn", f"[翻译] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        self.is_paused = not self.is_paused
        name = self.current_backend
        if self.facade is not None:
            try:
                if self.is_paused:
                    self.facade.pause_queue(name)
                else:
                    self.facade.resume_queue(name)
            except Exception as ex:
                log.record("error", f"[翻译] 队列切换失败: {ex}")
        self._rebuild_task_panel()

    def _refresh_tasks(self):
        """Pull current/waiting tasks from the backend and render (fallback when navigating back; kept even after the push is wired)."""
        if self.facade is None:
            self.current_task = None
            self.waiting_tasks = []
            self._rebuild_task_panel()
            return
        try:
            cur = self.facade.list_current_task(self.current_backend)
            wait = self.facade.list_waiting_tasks(self.current_backend)
        except Exception:
            return
        self.update_tasks(
            self._snap_to_dict(cur) if cur is not None else None,
            [self._snap_to_dict(t) for t in wait],
        )

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
        }

    # ── Internal methods — service operations (functionality removed; placeholder signatures kept) ──

    async def _start_service(self, e):
        """Start the translation service (current_backend is the registered service key; config_path uses the service config picker)."""
        if self.facade is None:
            log.record("warn", "[翻译] facade 未接线")
            return
        name = self.current_backend
        config_path = self._get_server() if self._get_server else None
        try:
            # core startup is synchronous/blocking (llama launch / model load) → thread pool avoids blocking the event loop
            await asyncio.to_thread(self.facade.start_service, name, None, config_path)
            # proactively refresh the button state (push fallback: update immediately even if the backend doesn't callback)
            self.service_device = self._current_device()
            self.update_service_status(True, False, self.service_device)
        except Exception as ex:
            log.record("error", f"[翻译] 启动失败: {ex}")
            self.service_device = None
            self.update_service_status(False, False)

    async def _stop_service(self, e):
        """Stop the translation service (core cancels the current task first; proactively refresh button state when done)."""
        if self.facade is None:
            return
        try:
            await asyncio.to_thread(self.facade.stop_service, self.current_backend)
            # proactively refresh the button state (push fallback)
            self.service_device = None
            self.update_service_status(False, False)
        except Exception as ex:
            log.record("error", f"[翻译] 停止失败: {ex}")

    # ── Internal methods — file selection and submission (functionality removed; placeholder signatures kept) ──

    async def _pick_file(self, e):
        """Choose the import method — AlertDialog branches: pick files (multi-select, any suffix) / pick a folder."""
        if self.file_picker is None or self.page is None:
            return
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("选择导入方式"),
            content=ft.Column([
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.FILE_OPEN),
                    title=ft.Text("选择文件"),
                    subtitle=ft.Text("选择一个或多个文本文件（不限制后缀，批量不预览直接入队）"),
                    on_click=self._on_choose_files,
                ),
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.FOLDER_OPEN),
                    title=ft.Text("选择文件夹"),
                    subtitle=ft.Text("导入文件夹顶层的所有可读文本文件（不递归子目录）"),
                    on_click=self._on_choose_directory,
                ),
            ], spacing=0, tight=True),
            actions=[ft.TextButton("取消", on_click=self._close_import_dialog)],
        )
        self._import_dialog = dlg
        # flet 0.86.2: Page has no dialog attribute; must use show_dialog() (pushes onto the dialog stack and renders)
        self.page.show_dialog(dlg)

    def _close_import_dialog(self, e=None):
        dlg = getattr(self, "_import_dialog", None)
        if dlg is not None:
            dlg.open = False
            self.page.update()  # consistent with the settings page's _close_dlg pattern (closing inside the show_dialog stack)

    async def _on_choose_files(self, e):
        self._close_import_dialog()
        if self.file_picker is None:
            return
        try:
            files = await self.file_picker.pick_files(
                allow_multiple=True,
                dialog_title="选择待翻译文件",
                file_type=ft.FilePickerFileType.ANY,
            )
        except Exception as ex:
            log.record("error", f"[翻译] 选择文件失败: {ex}")
            return
        if not files:
            return
        await self._process_selected([Path(f.path) for f in files])

    async def _on_choose_directory(self, e):
        self._close_import_dialog()
        if self.file_picker is None:
            return
        try:
            # flet 0.86.2: the folder-picking API is get_directory_path (returns a path
            # string directly); pick_directory was removed
            result = await self.file_picker.get_directory_path(dialog_title="选择待导入文件夹")
        except Exception as ex:
            log.record("error", f"[翻译] 选择文件夹失败: {ex}")
            return
        if not result:
            return
        paths = self._import_directory(Path(result))
        if not paths:
            msg = "所选文件夹中没有可导入的文件"
            log.record("warn", f"[翻译] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        await self._process_selected(paths)

    def _import_directory(self, path: Path) -> list:
        """Scan the folder's top level: only is_file, no recursion into subdirectories, sorted by file name."""
        try:
            entries = [p for p in path.iterdir() if p.is_file()]
        except OSError as ex:
            log.record("error", f"[翻译] 扫描文件夹失败: {ex}")
            return []
        return sorted(entries, key=lambda p: p.name.lower())

    async def _process_selected(self, paths: list):
        """Dispatch: single file → validate then fill the input area for preview; multiple files/folder → batch enqueue."""
        paths = [p for p in paths if p]
        if len(paths) == 1:
            text = _read_text_file(paths[0])
            if text is None:
                msg = f"文件不是可读的文本格式: {paths[0].name}"
                log.record("warn", f"[翻译] {msg}")
                if self.page:
                    self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
                return
            if self.input_text.current:
                self.input_text.current.value = text
                self.input_text.current.update()
                # remember the source file (so the text submission reuses its name; reset after successful submit)
                try:
                    self.input_path = paths[0].resolve()
                except OSError:
                    self.input_path = paths[0].absolute()
            return
        self._enqueue_translations(paths)

    def _enqueue_translations(self, paths: list):
        """Validate in batch and enqueue: no suffix restriction; NUL/decode failures are rejected without interrupting the rest.

        The same file may be added multiple times (each import is an independent task);
        text submissions (_submit_translation) do not go through this path.
        """
        ok_paths, rejected = [], []
        for p in paths:
            try:
                abs_p = p.resolve()
            except OSError:
                abs_p = p.absolute()
            if _read_text_file(abs_p) is None:
                rejected.append(p.name)
                continue
            ok_paths.append(abs_p)
        if rejected:
            names = ", ".join(rejected[:5]) + ("…" if len(rejected) > 5 else "")
            msg = f"{len(rejected)} 个文件不是可读的文本格式: {names}"
            log.record("warn", f"[翻译] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
        if not ok_paths:
            return
        try:
            reqs = self.build_translation_requests(ok_paths)
        except Exception as ex:
            log.record("error", f"[翻译] 任务打包失败: {ex}")
            return
        self._enqueue_requests(reqs)

    def _enqueue_requests(self, reqs: list):
        """Unified enqueue: facade submit (when present) + local cache + snackbar.

        Shared by file batch import and text submission; text tasks skip the _queued_paths dedup
        (each submission is an independent new task).
        - facade wired: queue display is taken over by the core _emit → update_tasks push (no manual rebuild)
        - facade not wired: local cache + manual rebuild fallback
        """
        has_facade = self.facade is not None and hasattr(self.facade, "submit_task")
        submitted = 0
        if has_facade:
            for r in reqs:
                try:
                    self.facade.submit_task(r)
                    submitted += 1
                except Exception as ex:
                    log.record("error", f"[翻译] 提交失败: {ex}")
        else:
            submitted = len(reqs)  # facade not wired: local-cache display only
            for r in reqs:
                name = getattr(r, "file_name", None) or ""
                self.waiting_tasks.append({
                    "id": Path(name).stem,
                    "type": getattr(r, "task_type", "translate"),
                    "status": "pending",
                    "progress": 0,
                    "file_name": name,
                    "input_summary": name,
                })
            self._rebuild_task_panel()
        if self.page:
            # flet 0.86.2: SnackBar is a DialogControl, shown via show_dialog (show_snack_bar was removed)
            self.page.show_dialog(
                ft.SnackBar(ft.Text(f"已加入队列 {submitted} 个任务"), bgcolor=Palette.SUCCESS)
            )
        return submitted

    def _rebuild_task_panel(self):
        """Rebuild the task queue panel (driven by the local cache; the update_tasks push takes over once wired)."""
        if not self.task_container.current:
            return
        self.task_container.current.content = task_queue_panel(
            current_task=self.current_task,
            waiting_tasks=self.waiting_tasks,
            callbacks={
                "on_cancel": self._on_cancel_task,
                "on_move_up": self._on_move_up,
                "on_move_down": self._on_move_down,
                "on_clear": self._on_clear,
                "on_pause_toggle": self._on_pause_toggle,
                "is_paused": self.is_paused,
            },
            empty_text="暂无翻译任务",
            expand=2,
        )
        self._safe_update(self.task_container.current)

    def _submit_translation(self, e):
        """Submit the input-area text as a translation task (written to temp/input_{timestamp}.txt then enqueued uniformly)."""
        text = self.input_text.current.value if self.input_text.current else self.input_value
        if not text.strip():
            msg = "请输入待翻译文本"
            log.record("warn", f"[翻译] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        try:
            reqs = self.build_translation_requests(text)
        except Exception as ex:
            log.record("error", f"[翻译] 任务打包失败: {ex}")
            return
        if not reqs:
            return
        # text read for preview: the task file name reuses the source file name (content is still the input area's current text)
        if self.input_path is not None:
            reqs[0].file_name = self.input_path.name
        submitted = self._enqueue_requests(reqs)
        if submitted > 0 and self.input_text.current:
            self.input_text.current.value = ""
            self.input_text.current.update()
            self.input_value = ""
            self.input_path = None  # reset the preview flag, preventing accidental reuse

    # ── Facade callbacks (functionality removed; placeholder signatures kept) ──

    def _on_service_change(self, status: dict):
        """Service status change callback (functionality removed)."""
        pass

    def _on_task_change(self, snapshot):
        """Task status change callback (functionality removed)."""
        pass
