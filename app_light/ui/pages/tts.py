"""语音合成页面（GSV / GPT-SoVITS）— 对接 AppFacade 实现合成服务与任务管理。

布局（三行 + 工作区）：
1. 服务管理栏（角色服务配置 / 启停 + 角色切换【应用角色配置：重载模型】）
2. 工作区左列：合成输入面板（文本 + 复刻参考 + 角色预设/合成参数 + 操作按钮）
3. 工作区右列：任务队列（左列:右列 = 3:2）

任务契约（与 core/executor.py::GsvTTSExecutor 一致）::

    TaskRequest(task_type="gsv", file_path=<目标文本>, configs={"args": {...}})

返回结果::

    {"audio_path", "sample_rate", "duration",
     "info": {"version", "ref_mode", "fragments", "seed", "elapsed_sec"}}

设计见 PLANS/gsv-moss/app-integration-design.md §6。
"""

import asyncio
import json
import time
from pathlib import Path

import flet as ft

from app.log import log
from app.facade import PageUiSink
from app.paths import project_root
from core.contracts import TaskRequest
from core.system_config import load_section
from ui.theme import Layout, Palette, Radius, Typography
from ui.components import (
    _text, divider, _shadow, panel_header,
    service_management_bar, bordered_button,
)
from ui.widgets.config_picker import config_picker
from ui.widgets.task_list import task_queue_panel


# 参考音频扩展名白名单（GSV 参考硬校验 3~10s；mp3 等由引擎侧兜底读取）
_REF_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp932", "shift_jis", "gbk")

# 输入面板控件统一宽度：参考语种/目标语种/复刻模式 与 角色预设/合成参数
# 及 应用角色预设/提交至队列 保持一致
_CONTROL_WIDTH = 170


def build_tts(page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
    """兼容包装 — 创建 TtsPage 实例并构建 UI。"""
    return TtsPage(page, facade, file_picker).build()


def _read_text_file(path: Path) -> str | None:
    """读取文件为文本；非文本格式（二进制 / 无法解码）返回 None。"""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    for enc in _TEXT_ENCODINGS:
        try:
            return data.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _check_ref_duration(path: str) -> str | None:
    """参考音频 3~10s 客户端预检；失败/不可读返回 None（引擎侧硬校验兜底）。"""
    try:
        import soundfile
        info = soundfile.info(path)
        dur = info.frames / info.samplerate
    except Exception:
        return None
    if not (3.0 <= dur <= 10.0):
        return f"参考音频时长 {dur:.2f}s 超出 3~10s 范围"
    return None


class TtsPage:
    """语音合成工作台页面实例 — 长期持有状态，避免导航切换时丢失。"""

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── 服务状态 ──
        self.is_online = False
        self.is_loading = False
        self.is_paused = True  # 任务队列初始暂停：仅服务加载后才能开启
        self.service_device = None  # 实际工作设备（cuda/cpu），状态栏着色显示

        # ── 配置选择缓存 ──
        self.selected_role_config = None
        self.selected_service_config = None

        # ── 输入状态缓存（切页重建恢复用）──
        self.selected_text = ""
        self.selected_text_lang = "zh"
        self.selected_ref_mode = "default"
        self.selected_emotion_ref = ""
        self.selected_prompt_text = ""
        self.selected_prompt_lang = "ja"
        self.selected_role_ref = ""
        self.selected_adv_args = None      # 合成参数（configs/tts/args/*.json，Path 或 None）

        # ── 队列状态缓存 ──
        self.current_task = None
        self.waiting_tasks = []

        # ── 语音合成默认配置（configs/system/default.ini [gsv]）──
        self._default_cfg: dict = {}
        self._load_default_config()

        # ── Refs ──
        self.status_dot = ft.Ref[ft.Container]()
        self.status_label = ft.Ref[ft.Text]()
        self.start_btn = ft.Ref[ft.TextButton]()
        self.stop_btn = ft.Ref[ft.TextButton]()
        self.target_text = ft.Ref[ft.TextField]()
        self.emotion_ref_field = ft.Ref[ft.TextField]()
        self.prompt_text_field = ft.Ref[ft.TextField]()
        self.role_ref_field = ft.Ref[ft.TextField]()
        # 模式联动引用（build 时填充，on_change 更新用）
        self._emotion_label = None
        self._emotion_field = None
        self._emotion_row = None
        self._role_row = None
        self.task_container = ft.Ref[ft.Container]()

        # ── 控件直持引用（build 时赋值；save_ui_state 前有效）──
        self._text_lang_dd = None
        self._ref_mode_dd = None
        self._prompt_lang_dd = None

        # ── 配置选择器 getter（build 时赋值）──
        self._get_role_config = None
        self._get_adv_args = None

        # ── UI sink 接线（name="gsv"，服务注册 key）──
        self._sink = None
        if self.facade is not None and hasattr(self.facade, "register_ui_sink"):
            self._sink = PageUiSink(page, self)
            self.facade.register_ui_sink("gsv", self._sink)

    # ════════════════════════════════════════════════════
    #  公开接口
    # ════════════════════════════════════════════════════

    def update_service_status(self, online: bool, loading: bool = False, device: str | None = None):
        """后端推送：更新服务状态（重赋值 + 刷新状态点/标签/启停按钮）。"""
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
        """后端推送：重赋值任务缓存并重建队列面板。"""
        self.current_task = current
        self.waiting_tasks = waiting or []
        self._render_panel()

    def build(self) -> ft.Control:
        """构建/重建语音合成页面 UI。"""

        # ── 服务配置选择器（服务管理栏内，启动服务按钮左侧）──
        service_picker, self._get_service_config = config_picker(
            "服务配置", [], config_type="gsv", glob_filter="*.json",
            width=Layout.PICKER_WIDTH_SM, value=self.selected_service_config,
        )

        # ── 服务管理栏 ──
        service_bar = service_management_bar(
            service_type="gsv",
            status_dot_ref=self.status_dot,
            status_label_ref=self.status_label,
            start_btn_ref=self.start_btn,
            stop_btn_ref=self.stop_btn,
            backend_selector=None,
            config_dropdown=None,           # 服务配置移入启停按钮组（紧邻启动按钮左侧）
            pre_start_actions=[service_picker],
            on_start=self._load_model,
            on_stop=self._unload_model,
            row1_cols=(4, 8),   # 状态 / 启停组（含服务配置，组内右对齐）
        )

        # ── 目标文本 ──
        target_field = ft.TextField(
            ref=self.target_text,
            hint_text="输入待合成文本，或点击右上角选择目标文本文件…",
            hint_style=ft.TextStyle(size=Typography.BODY, color=Palette.SUBTEXT),
            text_style=ft.TextStyle(size=Typography.BODY_LG, color=Palette.TEXT),
            bgcolor=Palette.BG,
            border_color=Palette.BORDER,
            border_radius=Radius.LG,
            multiline=True,
            min_lines=3,
            max_lines=6,
            expand=True,
            value=self.selected_text,
        )

        # ── 参考语种 / 目标语种 / 复刻模式（同一行，等宽；左/中/右对齐）──
        self._prompt_lang_dd = ft.Dropdown(
            label="参考语种", dense=True, width=_CONTROL_WIDTH,
            text_style=ft.TextStyle(size=13, color=Palette.TEXT),
            bgcolor=Palette.SURFACE2, border_color=Palette.BORDER,
            border_radius=Radius.MD, filled=True,
            options=[ft.dropdown.Option(k) for k in ("ja", "zh", "en")],
            value=self.selected_prompt_lang,
        )
        self._text_lang_dd = ft.Dropdown(
            label="目标语种", dense=True, width=_CONTROL_WIDTH,
            text_style=ft.TextStyle(size=13, color=Palette.TEXT),
            bgcolor=Palette.SURFACE2, border_color=Palette.BORDER,
            border_radius=Radius.MD, filled=True,
            options=[ft.dropdown.Option(k) for k in ("zh", "ja", "en")],
            value=self.selected_text_lang,
        )
        self._ref_mode_dd = ft.Dropdown(
            label="复刻模式", dense=True, width=_CONTROL_WIDTH,
            text_style=ft.TextStyle(size=13, color=Palette.TEXT),
            bgcolor=Palette.SURFACE2, border_color=Palette.BORDER,
            border_radius=Radius.MD, filled=True,
            tooltip="default=单参考(参考音频+参考文本) · aux=音色/情绪折中 · dual=音色锚定优先",
            options=[
                ft.dropdown.Option("default", "default 默认"),
                ft.dropdown.Option("aux", "aux 折中"),
                ft.dropdown.Option("dual", "dual 音色优先"),
            ],
            value=self.selected_ref_mode,
            on_select=self._on_ref_mode_change,
        )
        control_row = ft.Row([
            self._prompt_lang_dd,
            self._text_lang_dd,
            self._ref_mode_dd,
        ], expand=True, spacing=8,
           alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
           vertical_alignment=ft.CrossAxisAlignment.CENTER)

        # ── 参考音频（default）/ 情绪参考音频（aux/dual）——同一栏动态改名 ──
        _is_default = (self.selected_ref_mode or "default") == "default"
        emotion_field = ft.TextField(
            ref=self.emotion_ref_field,
            hint_text=("参考音频（3~10s，default 必填）" if _is_default
                       else "情绪参考音频（3~10s，必填）"),
            hint_style=ft.TextStyle(size=12, color=Palette.SUBTEXT),
            text_style=ft.TextStyle(size=12, color=Palette.TEXT),
            bgcolor=Palette.BG, border_color=Palette.BORDER,
            border_radius=Radius.MD, dense=True, expand=True,
            value=self.selected_emotion_ref,
        )
        self._emotion_field = emotion_field
        pick_emotion_btn = ft.TextButton(
            "选择音频", icon=ft.Icons.AUDIO_FILE,
            on_click=self._pick_emotion_ref,
            style=ft.ButtonStyle(color=Palette.PRIMARY),
        )
        self._emotion_label = _text(
            "参考音频" if _is_default else "情绪参考音频",
            Typography.SMALL, "w600", Palette.SUBTEXT,
        )
        emotion_row = ft.Row([
            ft.Column([
                self._emotion_label,
                emotion_field,
            ], spacing=4, expand=True),
            pick_emotion_btn,
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.END)
        self._emotion_row = emotion_row

        # ── 参考文本（prompt_text）──
        prompt_field = ft.TextField(
            ref=self.prompt_text_field,
            hint_text="参考音频逐字文本（建议填写，与音频一致效果最佳）",
            hint_style=ft.TextStyle(size=12, color=Palette.SUBTEXT),
            text_style=ft.TextStyle(size=12, color=Palette.TEXT),
            bgcolor=Palette.BG, border_color=Palette.BORDER,
            border_radius=Radius.MD, dense=True, expand=True,
            value=self.selected_prompt_text,
        )
        pick_prompt_text_btn = ft.TextButton(
            "选择文本", icon=ft.Icons.DESCRIPTION,
            on_click=self._pick_prompt_text_file,
            style=ft.ButtonStyle(color=Palette.PRIMARY),
        )
        prompt_row = ft.Row([
            ft.Column([
                _text("参考文本 prompt_text", Typography.SMALL, "w600", Palette.SUBTEXT),
                prompt_field,
            ], spacing=4, expand=True),
            pick_prompt_text_btn,
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.END)

        # ── 角色参考 ──
        role_field = ft.TextField(
            ref=self.role_ref_field,
            hint_text="角色参考音频（dual/aux 必填；选择角色配置后自动预填）",
            hint_style=ft.TextStyle(size=12, color=Palette.SUBTEXT),
            text_style=ft.TextStyle(size=12, color=Palette.TEXT),
            bgcolor=Palette.BG, border_color=Palette.BORDER,
            border_radius=Radius.MD, dense=True, expand=True,
            value=self.selected_role_ref,
        )
        pick_role_btn = ft.TextButton(
            "选择音频", icon=ft.Icons.AUDIO_FILE,
            on_click=self._pick_role_ref,
            style=ft.ButtonStyle(color=Palette.PRIMARY),
        )
        role_row = ft.Row([
            ft.Column([
                _text("角色参考音频", Typography.SMALL, "w600", Palette.SUBTEXT),
                role_field,
            ], spacing=4, expand=True),
            pick_role_btn,
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.END)
        # default 模式隐藏角色参考栏（仅参考音频+参考文本两栏）
        role_row.visible = not _is_default
        self._role_row = role_row

        # ── 角色配置 / 合成参数（角色配置左对齐）──
        role_picker, self._get_role_config = config_picker(
            "角色配置", [], config_type="gsv_role", glob_filter="*.json",
            width=_CONTROL_WIDTH, value=self.selected_role_config,
        )
        args_picker, self._get_adv_args = config_picker(
            "合成参数", [], config_type="gsv_args",
            width=_CONTROL_WIDTH, value=self.selected_adv_args,
        )
        picker_row = ft.Row([
            role_picker,
            ft.Container(expand=True),
            args_picker,
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        # 第二行：应用角色配置（左对齐） + 提交至队列（右对齐）
        apply_role_btn = bordered_button(
            "应用角色配置", ft.Icons.SYNC,
            on_click=self._apply_role,
            tooltip="在线切换角色：热切换 S1/S2 权重（基础模型常驻，秒级）",
            width=_CONTROL_WIDTH,
        )
        submit_btn = ft.FilledButton(
            "提交至队列",
            icon=ft.Icons.SEND,
            on_click=self._submit,
            width=_CONTROL_WIDTH,
            style=ft.ButtonStyle(
                bgcolor=Palette.PRIMARY, color="#FFFFFF",
                shape=ft.RoundedRectangleBorder(radius=Radius.MD),
            ),
        )
        action_row = ft.Row([
            apply_role_btn,
            ft.Container(expand=True),
            submit_btn,
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        # ── 合成输入面板 ──
        input_panel = ft.Container(
            content=ft.Column([
                panel_header("合成文本",
                    trailing=bordered_button(
                        "选择目标文本", ft.Icons.FILE_OPEN,
                        on_click=self._pick_text_file,
                    )),
                divider(),
                target_field,
                control_row,
                divider(),
                emotion_row,
                prompt_row,
                role_row,
                divider(),
                picker_row,
                action_row,
            ], spacing=8),
            bgcolor=Palette.SURFACE,
            border_radius=Radius.XL,
            padding=ft.Padding.all(20),
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            shadow=_shadow("low"),
            expand=3,
        )

        # ── 任务队列 ──
        task_panel = ft.Container(
            ref=self.task_container,
            content=task_queue_panel(
                current_task=self.current_task,
                waiting_tasks=self.waiting_tasks,
                callbacks=self._build_callbacks(),
                empty_text="暂无合成任务",
                expand=2,
            ),
            expand=2,
        )

        # ── 工作区（宽屏：左 输入 / 右 队列；窄屏纵向排列）──
        # 左列（合成文本）与任务队列宽度比 3:2
        is_narrow = self.page.width > 0 and self.page.width < Layout.DESKTOP_MIN_WIDTH
        left_col = ft.Column([
            input_panel,
        ], spacing=0, expand=3)
        if is_narrow:
            workspace = ft.Column([
                left_col,
                ft.Container(height=Layout.COLUMN_SPACING),
                task_panel,
            ], spacing=0, expand=True)
        else:
            workspace = ft.Row([
                left_col,
                ft.Container(width=Layout.COLUMN_SPACING),
                task_panel,
            ], expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)

        return ft.Column([
            service_bar,
            ft.Container(height=Layout.SECTION_GAP),
            workspace,
        ], spacing=0, expand=True,
           horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    def save_ui_state(self) -> None:
        """离开页面前从控件 refs 同步状态到实例属性。"""
        if self.target_text.current:
            self.selected_text = self.target_text.current.value or ""
        if self._text_lang_dd is not None:
            self.selected_text_lang = self._text_lang_dd.value or "zh"
        if self._ref_mode_dd is not None:
            self.selected_ref_mode = self._ref_mode_dd.value or "default"
        if self.emotion_ref_field.current:
            self.selected_emotion_ref = self.emotion_ref_field.current.value or ""
        if self.prompt_text_field.current:
            self.selected_prompt_text = self.prompt_text_field.current.value or ""
        if self._prompt_lang_dd is not None:
            self.selected_prompt_lang = self._prompt_lang_dd.value or "ja"
        if self.role_ref_field.current:
            self.selected_role_ref = self.role_ref_field.current.value or ""
        if self._get_role_config:
            self.selected_role_config = self._get_role_config()
        if self._get_service_config:
            self.selected_service_config = self._get_service_config()
        if self._get_adv_args:
            self.selected_adv_args = self._get_adv_args()

    def refresh(self) -> None:
        """facade 回调或切换回页面时刷新。"""
        if self.facade:
            s = self.facade.get_service_status("gsv")
            self.is_online = s.get("status") == "online"
        self.service_device = self._current_device()
        self._update_service_status()
        self._refresh_tasks()

    # ════════════════════════════════════════════════════
    #  内部方法 — 服务状态与任务
    # ════════════════════════════════════════════════════

    def _load_default_config(self) -> None:
        """打开软件时读取 [gsv] 默认配置（configs/system/default.ini）。"""
        try:
            self._default_cfg = load_section("gsv")
        except Exception:
            self._default_cfg = {}
        self.selected_role_config = self._default_cfg.get("gsv_server") or None
        self.selected_service_config = self._default_cfg.get("gsv_service") or None
        mode = self._default_cfg.get("ref_mode")
        if mode in ("dual", "aux", "single"):
            self.selected_ref_mode = mode
        lang = self._default_cfg.get("text_lang")
        if lang in ("zh", "ja", "en"):
            self.selected_text_lang = lang
        adv = self._default_cfg.get("gsv_args")
        if adv and adv != "无":
            self.selected_adv_args = adv

    @staticmethod
    def _safe_update(ctrl):
        """控件未挂载（flet 0.86.2 首帧/重建时序）时静默跳过 update。"""
        if ctrl is None:
            return
        try:
            ctrl.update()
        except RuntimeError:
            pass

    def _update_service_status(self):
        dot = self.status_dot.current
        if dot is not None:
            dot.bgcolor = (
                Palette.WARNING if self.is_loading else
                Palette.SUCCESS if self.is_online else Palette.ERROR
            )
            self._safe_update(dot)
        label = self.status_label.current
        if label is not None:
            label.value = (
                "GSV 服务加载中…" if self.is_loading else
                "GSV 服务已加载" if self.is_online else
                "GSV 服务未加载"
            )
            self._apply_device_span(label)
            self._apply_role_span(label)
            self._safe_update(label)
        if self.start_btn.current is not None:
            self.start_btn.current.visible = not (self.is_online or self.is_loading)
            self.start_btn.current.disabled = self.is_loading
            self._safe_update(self.start_btn.current)
        if self.stop_btn.current is not None:
            self.stop_btn.current.visible = self.is_online or self.is_loading
            self.stop_btn.current.disabled = self.is_loading
            self._safe_update(self.stop_btn.current)

    @staticmethod
    def _device_display(device: str | None) -> tuple[str | None, str | None]:
        """返回 (显示文本, 颜色)；CUDA 绿 / CPU 橙，未知返回 None。"""
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

    def _apply_role_span(self, label: ft.Text) -> None:
        """状态文本右侧追加当前角色配置名（如「 · 角色: role-ookura-lumine」）。"""
        if not self.is_online:
            return
        role_path = self._get_role_config() if self._get_role_config else None
        name = Path(role_path).stem if role_path is not None else None
        if not name:
            return
        spans = list(label.spans or [])
        spans.append(ft.TextSpan(
            f" · 角色: {name}",
            style=ft.TextStyle(color=Palette.SUBTEXT, size=12),
        ))
        label.spans = spans

    def _current_device(self) -> str | None:
        if self.facade is not None and hasattr(self.facade, "get_service_device"):
            try:
                return self.facade.get_service_device("gsv")
            except Exception:
                return None
        return None

    def _build_callbacks(self):
        def _on_cancel(tid):
            if self.facade:
                try:
                    self.facade.cancel_task(tid)
                except Exception as ex:
                    log.record("error", f"[语音合成] 取消失败: {ex}")
            if self.current_task and self.current_task.get("id") == tid:
                self.current_task = None
            self.waiting_tasks = [t for t in self.waiting_tasks
                                  if not (isinstance(t, dict) and t.get("id") == tid)]
            self._render_panel()

        def _on_move_up(tid):
            for i, t in enumerate(self.waiting_tasks):
                if t.get("id") == tid and i > 0:
                    if self.facade:
                        self.facade.reorder_task(tid, i - 1)
                    break

        def _on_move_down(tid):
            for i, t in enumerate(self.waiting_tasks):
                if t.get("id") == tid and i < len(self.waiting_tasks) - 1:
                    if self.facade:
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

    def _render_panel(self):
        if self.task_container.current is None:
            return
        self.task_container.current.content = task_queue_panel(
            current_task=self.current_task,
            waiting_tasks=self.waiting_tasks,
            callbacks=self._build_callbacks(),
            empty_text="暂无合成任务",
            expand=2,
        )
        self._safe_update(self.task_container.current)

    def _refresh_tasks(self):
        if self.task_container.current is None:
            return
        if self.facade is None:
            self.current_task = None
            self.waiting_tasks = []
        else:
            cur = self.facade.list_current_task(task_type="gsv")
            wait = self.facade.list_waiting_tasks(task_type="gsv")
            self.current_task = self._snap_to_dict(cur) if cur is not None else None
            self.waiting_tasks = [self._snap_to_dict(t) for t in wait]
        self._render_panel()

    @staticmethod
    def _snap_to_dict(snap) -> dict:
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

    def _on_clear(self):
        if self.facade is not None:
            try:
                self.facade.clear_queue("gsv")
            except Exception as ex:
                log.record("error", f"[语音合成] 清空队列失败: {ex}")
        self.waiting_tasks = []
        self._refresh_tasks()
        if self.page:
            self.page.show_dialog(
                ft.SnackBar(ft.Text("等待任务已清空"), bgcolor=Palette.SUCCESS)
            )

    def _on_pause_toggle(self):
        if self.is_paused and not self.is_online:
            msg = "需先加载服务才能开启队列"
            log.record("warn", f"[语音合成] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        self.is_paused = not self.is_paused
        if self.facade is not None:
            try:
                if self.is_paused:
                    self.facade.pause_queue("gsv")
                else:
                    self.facade.resume_queue("gsv")
            except Exception as ex:
                log.record("error", f"[语音合成] 队列切换失败: {ex}")
        self._render_panel()

    # ════════════════════════════════════════════════════
    #  内部方法 — 服务操作
    # ════════════════════════════════════════════════════

    def _merged_gsv_config(self) -> dict:
        """合并所选 GSV 服务配置与所选角色配置 → 提交 dict。

        角色配置（tts/roles/role-*.json）提供 S1/S2 权重与参考音频/文本，
        服务配置（models/gsv/*.json，默认 default.json）提供 device/BERT/
        CNHuBERT/SV；未选角色时仅服务配置（默认权重）。
        """
        from core.gsv.paths import merge_service_role

        role_path = self._get_role_config() if self._get_role_config else None
        role_cfg = None
        if role_path is not None:
            try:
                role_cfg = json.loads(Path(role_path).read_text(encoding="utf-8"))
            except Exception as ex:
                log.record("warn", f"[语音合成] 角色配置读取失败 {role_path}: {ex}")
        svc_path = self._get_service_config() if self._get_service_config else None
        if svc_path is None:
            svc_path = project_root / "configs/models/gsv/default.json"
        service_cfg = {}
        try:
            service_cfg = json.loads(Path(svc_path).read_text(encoding="utf-8"))
        except Exception as ex:
            log.record("warn", f"[语音合成] 服务配置读取失败 {svc_path}: {ex}")
        return merge_service_role(service_cfg, role_cfg)

    async def _load_model(self, e):
        if self.facade is None:
            return
        try:
            self.update_service_status(False, True)
            config = self._merged_gsv_config()
            await asyncio.to_thread(self.facade.start_service, "gsv", None, config)
            self._prefill_role_defaults()
            self.service_device = self._current_device()
            self.update_service_status(True, False, self.service_device)
        except Exception as ex:
            log.record("error", f"[语音合成] 加载失败: {ex}")
            self.service_device = None
            self.update_service_status(False, False)
            if self.page:
                self.page.show_dialog(
                    ft.SnackBar(ft.Text(f"GSV 加载失败: {ex}"), bgcolor=Palette.ERROR)
                )

    async def _unload_model(self, e):
        if self.facade is None:
            return
        try:
            # 先置"卸载中"：停止按钮禁用（防重复点击），状态栏更新
            self.update_service_status(False, True)
            # GSV 引擎 stop 即释放，挂起任务不可续跑 → 先取消当前任务
            await asyncio.to_thread(self.facade.stop_service, "gsv", True)
            self.service_device = None
            self.update_service_status(False, False)
        except Exception as ex:
            log.record("error", f"[语音合成] 停止失败: {ex}")
            self.update_service_status(False, False)

    async def _apply_role(self, e):
        """在线切换角色：取消当前 → 停止 → 新配置启动（重载 10~20s）。"""
        if self.facade is None:
            return
        role_path = self._get_role_config() if self._get_role_config else None
        if role_path is None:
            msg = "请先选择角色配置"
            log.record("warn", f"[语音合成] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        if self.current_task is not None:
            msg = "切换角色将取消当前合成任务，是否继续？"
            if self.page is not None:
                await self._confirm_switch_role(role_path)
                return
        await self._do_switch_role(role_path)

    async def _confirm_switch_role(self, role_path):
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("切换角色"),
            content=ft.Text("切换角色需要重新加载引擎（约 10~20s），当前运行中的合成任务将被取消。"),
            actions=[
                ft.TextButton("取消", on_click=lambda e: self._close_dialog(dlg)),
                ft.FilledButton(
                    "继续切换", on_click=lambda e: (self._close_dialog(dlg),
                                                     self.page.run_task(self._do_switch_role, role_path)),
                    style=ft.ButtonStyle(bgcolor=Palette.PRIMARY, color="#FFFFFF"),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dlg)

    def _close_dialog(self, dlg):
        try:
            dlg.open = False
            self.page.update()
        except Exception:
            pass

    async def _do_switch_role(self, role_path):
        try:
            self.update_service_status(False, True)
            config = self._merged_gsv_config()
            await asyncio.to_thread(self.facade.switch_service_config, "gsv", config)
            self._prefill_role_defaults()
            self.service_device = self._current_device()
            self.update_service_status(True, False, self.service_device)
            if self.page:
                self.page.show_dialog(
                    ft.SnackBar(ft.Text(f"已切换角色: {Path(role_path).stem}"),
                                bgcolor=Palette.SUCCESS)
                )
        except Exception as ex:
            log.record("error", f"[语音合成] 切换角色失败: {ex}")
            self.service_device = None
            self.update_service_status(False, False)
            if self.page:
                self.page.show_dialog(
                    ft.SnackBar(ft.Text(f"切换角色失败: {ex}"), bgcolor=Palette.ERROR)
                )

    # ════════════════════════════════════════════════════
    #  内部方法 — 角色默认值 / 参数模板 / 校验 / 提交
    # ════════════════════════════════════════════════════

    def _prefill_role_defaults(self):
        """读取当前角色配置的顶层 role_ref_audio/prompt_text，自动预填参考栏。

        default 模式：角色参考音频填入"参考音频"栏；aux/dual 填入"角色参考音频"栏。
        角色 JSON 带 mode 键时同步模式下拉（可选联动）。
        """
        cfg_path = self._get_role_config() if self._get_role_config else None
        if cfg_path is None:
            return
        try:
            data = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        # 角色 mode 联动模式下拉（若角色 JSON 声明了模式）
        role_mode = str(data.get("mode") or "").strip()
        if role_mode in ("default", "aux", "dual") and self._ref_mode_dd is not None:
            self._ref_mode_dd.value = role_mode
            self.selected_ref_mode = role_mode
            self._on_ref_mode_change(None)
        is_default = self.selected_ref_mode == "default"
        target_field = (self.emotion_ref_field if is_default else self.role_ref_field)
        target_attr = "selected_emotion_ref" if is_default else "selected_role_ref"
        if target_field.current is not None and not (target_field.current.value or "").strip():
            role_ref = data.get("role_ref_audio") or ""
            if role_ref:
                target_field.current.value = role_ref
                self._safe_update(target_field.current)
                setattr(self, target_attr, role_ref)
        if self.prompt_text_field.current is not None and not (self.prompt_text_field.current.value or "").strip():
            prompt_text = data.get("prompt_text") or ""
            if prompt_text:
                self.prompt_text_field.current.value = prompt_text
                self._safe_update(self.prompt_text_field.current)
                self.selected_prompt_text = prompt_text

    def _template_args(self) -> dict:
        """读取合成参数 JSON（configs/tts/args/*.json）；"无"或读取失败返回 {}。"""
        path = self._get_adv_args() if self._get_adv_args else None
        if path is None:
            return {}
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        # 防御：模板不得覆盖目标文本（text 由页面输入决定）
        return {k: v for k, v in data.items() if k != "text"}

    def _on_ref_mode_change(self, e):
        """模式切换联动：参考栏动态改名 + 角色参考栏显隐。"""
        mode = (self._ref_mode_dd.value if self._ref_mode_dd else None) or "default"
        self.selected_ref_mode = mode
        is_default = mode == "default"
        if self._emotion_label is not None:
            self._emotion_label.value = "参考音频" if is_default else "情绪参考音频"
            self._safe_update(self._emotion_label)
        if self._emotion_field is not None:
            self._emotion_field.hint_text = (
                "参考音频（3~10s，default 必填）" if is_default
                else "情绪参考音频（3~10s，必填）")
            self._safe_update(self._emotion_field)
        if self._role_row is not None:
            self._role_row.visible = not is_default
            self._safe_update(self._role_row)

    def _validate_inputs(self) -> tuple[dict | None, str | None]:
        """输入校验 → (args, error)。args 含全部合成参数；error 非空则中止提交。

        合并顺序（低→高）：合成参数模板（configs/tts/args）→ 页面控件值
        （覆盖模板的 ref_mode/prompt_lang/text_lang 等）。
        """
        text = (self.target_text.current.value if self.target_text.current
                else self.selected_text) or ""
        text = text.strip()
        if not text:
            return None, "目标文本为空"
        ref_mode = self._ref_mode_dd.value if self._ref_mode_dd else "default"
        emotion_ref = (self.emotion_ref_field.current.value if self.emotion_ref_field.current
                       else self.selected_emotion_ref) or ""
        role_ref = (self.role_ref_field.current.value if self.role_ref_field.current
                    else self.selected_role_ref) or ""
        prompt_text = (self.prompt_text_field.current.value if self.prompt_text_field.current
                       else self.selected_prompt_text) or ""
        prompt_lang = (self._prompt_lang_dd.value if self._prompt_lang_dd else "ja") or "ja"
        text_lang = (self._text_lang_dd.value if self._text_lang_dd else "zh") or "zh"

        ref_label = "参考音频" if ref_mode == "default" else "情绪参考音频"
        # default：参考音频必填（官方 v2Pro 族需参考供 S2 音色锚定）；参考文本可选
        if not emotion_ref:
            return None, f"请选择{ref_label}（3~10s）"
        if not Path(emotion_ref).is_file():
            return None, f"{ref_label}不存在: {emotion_ref}"
        if ref_mode in ("aux", "dual") and not role_ref:
            return None, f"ref_mode={ref_mode} 需要角色参考音频"
        if role_ref:
            if not Path(role_ref).is_file():
                return None, f"角色参考音频不存在: {role_ref}"

        for label, path in ((ref_label, emotion_ref), ("角色参考", role_ref)):
            if not path:
                continue
            err = _check_ref_duration(path)
            if err:
                return None, f"{label}: {err}"

        args = {}
        args.update(self._template_args())
        if ref_mode == "default":
            # default：单参考 + 参考文本（空则 ref_free）；不发 role_ref_audio
            args.update({
                "ref_mode": "default",
                "ref_audio_path": emotion_ref,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "text_lang": text_lang,
            })
        else:
            args.update({
                "ref_mode": ref_mode,
                "ref_audio_path": emotion_ref,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "role_ref_audio": role_ref,
                "text_lang": text_lang,
            })
        return args, None

    def _build_request(self, args: dict) -> TaskRequest:
        text = (self.target_text.current.value if self.target_text.current
                else self.selected_text).strip()
        return TaskRequest(
            task_type="gsv",
            file_path=text,                 # Executor._resolve_value：非文件字符串原样返回
            file_name=f"tts_{time.strftime('%Y%m%d%H%M%S')}.txt",
            configs={"args": args},
        )

    def _submit(self, e):
        args, err = self._validate_inputs()
        if err:
            log.record("warn", f"[语音合成] 提交失败: {err}")
            if self.page:
                self.page.show_dialog(
                    ft.SnackBar(ft.Text(err), bgcolor=Palette.ERROR)
                )
            return
        try:
            req = self._build_request(args)
        except Exception as ex:
            log.record("error", f"[语音合成] 任务打包失败: {ex}")
            return
        if self.facade is not None:
            try:
                tid = self.facade.submit_task(req)
            except Exception as ex:
                log.record("error", f"[语音合成] 提交失败: {ex}")
                if self.page:
                    self.page.show_dialog(
                        ft.SnackBar(ft.Text(f"提交失败: {ex}"), bgcolor=Palette.ERROR)
                    )
                return
        else:
            tid = "pending"
            self.waiting_tasks.append({
                "id": "pending", "type": "gsv", "status": "pending",
                "progress": 0, "file_name": req.file_name,
                "input_summary": req.file_name,
            })
            self._render_panel()
        if self.page:
            self.page.show_dialog(
                ft.SnackBar(ft.Text(f"已提交至队列 (id={tid})"), bgcolor=Palette.SUCCESS)
            )

    # ════════════════════════════════════════════════════
    #  内部方法 — 文件选择
    # ════════════════════════════════════════════════════

    async def _pick_text_file(self, e):
        if self.file_picker is None:
            return
        try:
            files = await self.file_picker.pick_files(
                allow_multiple=False,
                dialog_title="选择文本文件",
                file_type=ft.FilePickerFileType.ANY,
            )
        except Exception as ex:
            log.record("error", f"[语音合成] 选择文件失败: {ex}")
            return
        if not files:
            return
        text = _read_text_file(Path(files[0].path))
        if text is None:
            msg = "文件不是可读的文本格式"
            log.record("warn", f"[语音合成] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        if self.target_text.current:
            self.target_text.current.value = text
            self.target_text.current.update()
            self.selected_text = text

    async def _pick_prompt_text_file(self, e):
        """选择 .txt 文件 → 填入参考文本（prompt_text）。"""
        if self.file_picker is None:
            return
        try:
            files = await self.file_picker.pick_files(
                allow_multiple=False,
                dialog_title="选择参考文本文件",
                file_type=ft.FilePickerFileType.ANY,
            )
        except Exception as ex:
            log.record("error", f"[语音合成] 选择文件失败: {ex}")
            return
        if not files:
            return
        text = _read_text_file(Path(files[0].path))
        if text is None:
            msg = "文件不是可读的文本格式"
            log.record("warn", f"[语音合成] {msg}")
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        if self.prompt_text_field.current:
            self.prompt_text_field.current.value = text
            self.prompt_text_field.current.update()
            self.selected_prompt_text = text

    async def _pick_ref_audio(self, target_field: ft.Ref, attr: str):
        if self.file_picker is None:
            return
        try:
            files = await self.file_picker.pick_files(
                allow_multiple=False,
                dialog_title="选择参考音频（3~10s）",
                file_type=ft.FilePickerFileType.ANY,
                allowed_extensions=["wav", "mp3", "flac", "m4a", "ogg", "opus"],
            )
        except Exception as ex:
            log.record("error", f"[语音合成] 选择音频失败: {ex}")
            return
        if not files:
            return
        path = files[0].path
        if Path(path).suffix.lower() not in _REF_EXTENSIONS:
            msg = f"不支持的参考音频格式: {Path(path).suffix}"
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(msg), bgcolor=Palette.ERROR))
            return
        err = _check_ref_duration(path)
        if err:
            if self.page:
                self.page.show_dialog(ft.SnackBar(ft.Text(err), bgcolor=Palette.ERROR))
            return
        if target_field.current:
            target_field.current.value = path
            target_field.current.update()
        setattr(self, attr, path)

    async def _pick_emotion_ref(self, e):
        await self._pick_ref_audio(self.emotion_ref_field, "selected_emotion_ref")

    async def _pick_role_ref(self, e):
        await self._pick_ref_audio(self.role_ref_field, "selected_role_ref")
