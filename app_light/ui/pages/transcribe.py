"""语音转写页面 — 对接 AppFacade 实现 MOSS 转写服务与任务管理。

布局（三行）：
1. 服务管理栏（后端固定：MOSS）
2. 配置栏（服务配置 + 转写参数）
3. 工作区（左预览 + 右任务队列）
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


# MOSS（ModelRunner）可解码扩展名白名单：收窄白名单，避免容器格式直接入队后 FAILED
_MOSS_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}


def build_transcribe(page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
    """兼容包装 — 创建 TranscribePage 实例并构建 UI。"""
    return TranscribePage(page, facade, file_picker).build()


class TranscribePage:
    """语音转写工作台页面实例 — 长期持有状态，避免导航切换时丢失。"""

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── 服务状态 ──
        self.is_online = False
        self.is_loading = False
        self.is_paused = True  # 任务队列初始暂停：仅服务加载后才能开启
        # 本分支（main）转写仅使用 MOSS：服务 key 即任务 type
        self.current_backend = "moss"
        self.service_device = None  # 实际工作设备（cuda/cpu），状态栏着色显示

        # ── 配置选择缓存（MOSS 专用：服务配置 / 转写参数 / 热词）──
        self.selected_moss_server = None
        self.selected_moss_args = None
        self.selected_hotword = None

        # 打开软件时读取"转写默认配置"（configs/system/default.ini [transcribe]）
        self._default_cfg: dict = {}
        self._load_default_config()

        # ── 预览状态 ──
        self.preview_value = ""

        # ── 队列状态缓存 ──
        self.current_task = None
        self.waiting_tasks = []

        # ── Refs ──
        self.status_dot = ft.Ref[ft.Container]()
        self.status_label = ft.Ref[ft.Text]()
        self.start_btn = ft.Ref[ft.TextButton]()
        self.stop_btn = ft.Ref[ft.TextButton]()
        self.preview_text = ft.Ref[ft.TextField]()
        self.task_container = ft.Ref[ft.Container]()

        # ── 配置选择器 getter（build 时赋值）──
        self._get_moss_model = None      # 服务配置（configs/models/moss*.json）
        self._get_transcribe = None      # 转写参数（moss_args）
        self._get_hotword = None

        # ── 服务栏容器 ──
        self._service_bar_ctrl = None

        # ── UI sink 接线（注册名固定 'moss'）──
        self._sink = None
        if self.facade is not None and hasattr(self.facade, "register_ui_sink"):
            self._sink = PageUiSink(page, self)
            self.facade.register_ui_sink(self.current_backend, self._sink)

    def _load_default_config(self) -> None:
        """打开软件时读取"转写默认配置"（configs/system/default.ini [transcribe]）。

        初始化 MOSS 服务配置/转写参数/热词各下拉的默认选中项；
        文件缺失或解析失败时保持 None（下拉回退首项）。
        """
        try:
            data = load_section("transcribe")
        except Exception:
            return
        self._default_cfg = data
        self.selected_moss_server = data.get("moss_server") or None
        self.selected_moss_args = data.get("moss_args") or None
        self.selected_hotword = data.get("hotwords") or None

    # ════════════════════════════════════════════════════
    #  公开接口
    # ════════════════════════════════════════════════════

    def build(self) -> ft.Control:
        """构建/重建转写页面 UI。"""

        # ── 服务管理栏（MOSS 专用）──
        self._service_bar_ctrl = ft.Container(content=self._build_service_bar())

        # ── 结果预览 ──
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
                    # 选择音频文件按钮位于结果预览一行最右侧（原导出 LRC 位置）
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

        # ── 任务队列 ──
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

        # ── 工作区（宽屏双栏；窄屏纵向排列，保留页面滚动）──
        # 宽屏 workspace 不再固定高度：expand 填满内容区剩余高度，
        # 保证结果预览/任务队列面板底边与内容区底边对齐、完整显示。
        is_narrow = self.page.width > 0 and self.page.width < Layout.DESKTOP_MIN_WIDTH
        workspace = (
            # 窄屏同样不滚动：scroll 容器内 flex 子项高度塌缩(不显示)，
            # 面板内部(预览/任务区)已有滚动，外层用有界 flex 分配
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

        # 根列不滚动：flet 0.86.2 的 scroll 容器(Flutter ListView)主轴无界，
        # 内部 expand 子项高度塌缩为 0 导致组件不显示；窗口已固定，无需整页滚动。
        return ft.Column([
            self._service_bar_ctrl,
            ft.Container(height=Layout.SECTION_GAP),
            workspace,
        ], spacing=0, expand=True,
           horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    def _build_service_bar(self) -> ft.Control:
        """构建 MOSS 专用服务管理栏（无后端切换，配置目录固定 moss/moss_args）。"""
        moss_model_picker, self._get_moss_model = config_picker(
            "服务配置", [],
            config_type="moss",
            glob_filter="*.json",
            width=Layout.PICKER_WIDTH_SM,   # 与翻译页服务配置同宽
            value=self.selected_moss_server,
        )
        # ── 转写参数（moss_args）──
        transcribe_picker, self._get_transcribe = config_picker(
            "转写参数", [],
            config_type="moss_args",
            width=200,
            value=self.selected_moss_args,
        )
        # ── 热词（configs/transcribe/hotwords/*.json；MOSS 官方方案：附加到 prompt）──
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
            backend_selector=None,          # 本分支固定 MOSS，不提供后端切换
            config_dropdown=None,           # 服务配置移入启停按钮组（紧邻启动按钮左侧）
            extra_items=[transcribe_picker, hotwords_picker],
            pre_start_actions=[moss_model_picker],
            on_start=self._load_model,
            on_stop=self._unload_model,
            merge_backend=False,
            row1_cols=(4, 8),               # 状态 / 启停组（含服务配置，组内右对齐）
            row2_cols=(6, 6),               # 转写参数 / 热词
        )

    def _re_register_sink(self):
        """sink 注册到当前服务 key（main 分支固定 'moss'）。"""
        if self._sink is not None and self.facade is not None \
                and hasattr(self.facade, "register_ui_sink"):
            self.facade.register_ui_sink(self.current_backend, self._sink)

    def save_ui_state(self) -> None:
        """离开页面前从控件 refs 同步状态到实例属性（MOSS 专用缓存）。"""
        if self.preview_text.current:
            self.preview_value = self.preview_text.current.value or ""
        if self._get_moss_model:
            self.selected_moss_server = self._get_moss_model()
        if self._get_transcribe:
            self.selected_moss_args = self._get_transcribe()
        if self._get_hotword:
            self.selected_hotword = self._get_hotword()

    def refresh(self) -> None:
        """facade 回调或切换回页面时刷新。"""
        if self.facade:
            s = self.facade.get_service_status(self.current_backend)
            self.is_online = s.get("status") == "online"
        self.service_device = self._current_device()
        self._update_service_status()
        self._refresh_tasks()

    def update_service_status(self, online: bool, loading: bool = False, device: str | None = None):
        """后端推送：更新服务状态（重赋值 + 刷新状态点/标签/启停按钮）。

        队列开启联动（仅服务状态变化时）：停止/加载中 → 暂停；加载成功 → 开启；
        同状态推送（如 refresh）不覆盖用户手动暂停。
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
        """后端推送：重赋值任务缓存并重建队列面板（dict 投影，对齐翻译页）。

        当前任务 running 且 payload 携带转写段 → 实时预览（随转写推进滚动刷新）。
        """
        self.current_task = current
        self.waiting_tasks = waiting or []
        # MOSS 懒加载：首次任务完成后 _device 才从 auto 解析为 cuda:0/cpu，
        # 任务推送是刷新状态栏设备的最佳时机。
        self.service_device = self._current_device()
        self._update_service_status()
        self._render_panel(self._build_callbacks(self.waiting_tasks))
        # 实时预览：转写进行中，用已确认转写段更新预览区（segments 必须是 list）。
        # MOSS（StreamingModelRunner）推送 payload.segments，
        # 完成后仍经 update_finished_tasks 全量刷新最终结果。
        if isinstance(current, dict) and current.get("status") == "running":
            payload = current.get("payload")
            segs = payload.get("segments", []) if isinstance(payload, dict) else []
            if isinstance(segs, list) and segs:
                self._render_segments_preview(segs)

    def update_finished_tasks(self, tasks):
        """后端推送：最新完成结果更新预览（转写页特色）。"""
        if tasks:
            self._update_preview()

    def register_callbacks(self) -> None:
        """旧回调收集路径已废弃——改为 UI sink 推送（__init__ 注册）。"""
        pass

    # ════════════════════════════════════════════════════
    #  内部方法 — 服务状态与任务
    # ════════════════════════════════════════════════════

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

    def _current_device(self) -> str | None:
        if self.facade is not None and hasattr(self.facade, "get_service_device"):
            try:
                return self.facade.get_service_device(self.current_backend)
            except Exception:
                return None
        return None

    def _on_clear(self):
        """清空等待任务（保留当前运行任务；已完成任务由已完成页面管理）。"""
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
        """暂停/恢复队列：仅服务加载后可开启；暂停始终允许（服务未加载时本就暂停）。"""
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
        """构造队列面板 callbacks（_refresh_tasks 与 update_tasks 共用）。"""
        def _on_cancel(tid):
            if self.facade:
                try:
                    self.facade.cancel_task(tid)
                except Exception as ex:
                    log.record("error", f"[转写] 取消失败: {ex}")
            # 本地缓存同步（current 与 waiting 均可能）
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
                    # 边界：最后一项不可下移（避免 index 超出队列长度-1）
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
        """重建队列面板（独立方法，供推送/拉取/本地缓存操作共用）。"""
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
        """将 TaskSnapshot 投影为 dict（与 AppFacade._project 字段一致，供本地缓存操作）。"""
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
        """把转写段（dict 或 Segment 兼容）渲染为标准 LRC：``[mm:ss.xx]<说话人>正文``。"""
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
        """用最新已完成任务结果全量更新预览区（不截断段数）。

        有任务正在转写时不覆盖实时预览（防止上一次完整结果与当前
        实时内容交替闪烁；任务完成瞬间 current 变空自然切到新结果）。
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
        # 无结果时显示默认提示
        if self.preview_text.current:
            self.preview_value = "暂无转写结果"
            self.preview_text.current.value = self.preview_value
            self._safe_update(self.preview_text.current)

    # ════════════════════════════════════════════════════
    #  内部方法 — 服务操作
    # ════════════════════════════════════════════════════

    async def _load_model(self, e):
        if self.facade is None:
            return
        try:
            # 先置"加载中"：状态栏显示加载中文案，并隐藏启动按钮防重复点击。
            self.update_service_status(False, True)
            # core 加载为同步阻塞（模型装载）→ 线程池避免阻塞事件循环；
            # config_path 用服务配置选择器（configs/models/moss*.json）
            config_path = self._get_moss_model() if self._get_moss_model else None
            await asyncio.to_thread(self.facade.start_service, self.current_backend, None, config_path)
            # 主动刷新状态（推送兜底：即使后端未回调也即时更新）
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
            # 先置"卸载中"：停止按钮禁用（防重复点击），状态栏更新
            self.update_service_status(False, True)
            await asyncio.to_thread(self.facade.stop_service, self.current_backend)
            # 主动刷新状态（推送兜底）
            self.service_device = None
            self.update_service_status(False, False)
        except Exception as ex:
            log.record("error", f"[转写] 停止失败: {ex}")
            self.update_service_status(False, False)

    # ════════════════════════════════════════════════════
    #  内部方法 — 文件选择与提交
    # ════════════════════════════════════════════════════

    async def _pick_file(self, e):
        """选择导入方式 — AlertDialog 分流：选择文件（可多选、音频校验）/ 选择文件夹。"""
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
            # flet 0.86.2：文件夹选择 API 为 get_directory_path（直接返回路径字符串）
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
        """扫描文件夹顶层：仅 is_file、不递归子目录、按文件名排序。"""
        try:
            entries = [p for p in path.iterdir() if p.is_file()]
        except OSError as ex:
            log.record("error", f"[转写] 扫描文件夹失败: {ex}")
            return []
        return sorted(entries, key=lambda p: p.name.lower())

    async def _process_selected(self, paths: list):
        """选择即提交：单文件/多文件/文件夹统一校验后直接入队。"""
        self._enqueue_transcriptions(paths)

    def _enqueue_transcriptions(self, paths: list):
        """批量校验并入队：后缀白名单校验、非音频拒绝且不中断。

        MOSS 使用可解码白名单（ModelRunner 可解码集合）。
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
        """打包转写请求：TranscriptionRequest(task_type=当前后端, file_path, configs)。

        paths=[...] → 每个文件一个请求（单文件/批量统一）。
        configs 键：
        - args：转写参数模板（configs/transcribe/args/*.json）
        - hotwords：热词文件（MOSS 在 executor 内按官方配方附加到 prompt）
        """
        configs = {}
        if self._get_transcribe:
            configs["args"] = self._get_transcribe()
        if self._get_hotword:
            hotwords_path = self._get_hotword()  # 选「无」→ None → 不设置
            if hotwords_path:
                configs["hotwords"] = hotwords_path
        return [TranscriptionRequest(
            task_type=self.current_backend,
            file_path=p,
            file_name=p.name,
            configs=configs,
        ) for p in paths]

    def _enqueue_requests(self, reqs: list) -> int:
        """统一入队：逐个提交 + 本地缓存 + 面板刷新 + 提示；返回成功提交数。

        后端 update_tasks 推送接线后由推送接管展示。
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
            submitted = len(reqs)  # facade 未接线：仅本地缓存展示
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
        # 渲染当前缓存（不重拉——拉取可能覆盖刚 append 的本地缓存；
        # 真实后端会经 sink 推送 update_tasks 补充真实 id/状态）
        self._render_panel(self._build_callbacks(self.waiting_tasks))
        if self.page:
            self.page.show_dialog(
                ft.SnackBar(ft.Text(f"已加入队列 {submitted} 个任务"), bgcolor=Palette.SUCCESS)
            )
        return submitted

