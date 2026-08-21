"""翻译工作台页面 — UI 构建保留，功能实现已移除（重建待定）。

保留：
- build() 三行布局与全部控件（视觉结构不变）
- save_ui_state() / refresh()（layout.py 导航契约）
- build_translate() 兼容包装

移除：register_callbacks() 与其余服务/任务/提交/队列/回调功能实现
（_start_service/_stop_service/_submit_translation/_update_service_status/_on_clear/
 _on_pause_toggle/_refresh_tasks/_on_service_change/_on_task_change 均为占位，
 仅保留签名供 build() 控件回调绑定，方法体为空——对应按钮点击无响应属预期行为）。
已恢复：后端选择（_on_backend_switch/_set_backend_style，Llama/API 二选一
Switch 开关，仿 completed 页自动导出开关样式）与文件选择
（_pick_file，含文本格式验证）。
原始实现见 ui/pages/translate.py.bak。
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


# 文本文件尝试解码的编码顺序（覆盖中文与日文常见文本编码）
_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp932", "shift_jis", "gbk")


def _read_text_file(path: Path) -> str | None:
    """读取文件为文本；非文本格式（二进制 / 无法解码）返回 None。

    返回文本统一为 ``\\n`` 换行（``\\r\\n``/``\\r`` → ``\\n``）：预览进
    TextField 与后续写 temp 文件都以纯 ``\\n`` 传递，避免 Windows 下
    ``write_text`` 默认 newline=None 把 ``\\r\\n`` 写成 ``\\r\\r\\n`` 双重
    回车（executor 读回时每行多一个空行）。
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:          # NUL 字节 → 二进制特征，拒绝
        return None
    for enc in _TEXT_ENCODINGS:
        try:
            return data.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def build_translate(page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
    """兼容包装 — 创建 TranslatePage 实例并构建 UI。"""
    return TranslatePage(page, facade, file_picker).build()


class TranslatePage:
    """翻译工作台页面实例 — 长期持有状态，避免导航切换时丢失。"""

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── 服务状态 ──
        self.is_online = False
        self.is_loading = False
        self.is_paused = True  # 任务队列初始暂停：仅服务加载后才能开启
        self.current_backend = "llama"  # "llama" | "api"
        self.service_device = None  # 实际工作设备（cuda/cpu/api），状态栏着色显示

        # ── 配置选择缓存 ──
        self.selected_server = None
        self.selected_prompts = None
        self.selected_args = None
        self.selected_rule = None
        self.selected_glossary = None

        # ── 翻译默认配置（configs/defaults/translate/default.json）──
        self._default_cfg: dict = {}
        self._load_default_config()

        # ── 输入状态 ──
        self.input_path = None
        self.input_value = ""

        # ── 队列状态缓存 ──
        self.current_task = None
        self.waiting_tasks = []
        # 已入队文件的绝对路径集合（批量导入去重，Windows 大小写不敏感）
        self._queued_paths = set()  # 兼容旧版去重（已废弃，保留字段避免外部引用报错）

        # ── Refs ──
        self.status_dot = ft.Ref[ft.Container]()
        self.status_label = ft.Ref[ft.Text]()
        self.start_btn = ft.Ref[ft.TextButton]()
        self.stop_btn = ft.Ref[ft.TextButton]()
        self.input_text = ft.Ref[ft.TextField]()
        self.task_container = ft.Ref[ft.Container]()
        self.backend_switch = ft.Ref[ft.Switch]()      # 后端选择开关（关=Llama / 开=API）
        self.backend_name_label = ft.Ref[ft.Text]()    # 开关旁动态后端名（Llama/API）

        # ── 配置选择器 getter（build 时赋值）──
        self._get_server = None
        self._get_prompts = None
        self._get_args = None
        self._get_rule = None
        self._get_glossary = None

        # ── 服务配置下拉的 config_type 切换器（build 时赋值，后端切换调用）──
        self._set_server_ctype = None
        # ── 翻译参数下拉的 config_type 切换器（build 时赋值，后端切换调用）──
        self._set_args_ctype = None

        # ── UI sink 接线（后端推送 → update_service_status/update_tasks）──
        # 注册 name = current_backend（服务注册 key：'llama'/'api'），后端切换时重新注册
        self._sink = None
        if self.facade is not None and hasattr(self.facade, "register_ui_sink"):
            self._sink = PageUiSink(page, self)
            self.facade.register_ui_sink(self.current_backend, self._sink)

    # ════════════════════════════════════════════════════
    #  公开接口
    # ════════════════════════════════════════════════════

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
            self._rebuild_task_panel()

    def update_tasks(self, current, waiting):
        """后端推送：重赋值任务缓存并重建队列面板（已完成任务由 CompletedPage 管理）。"""
        self.current_task = current
        self.waiting_tasks = waiting or []
        self._rebuild_task_panel()

    def build(self) -> ft.Control:
        """构建/重建翻译页面 UI（视觉结构原样保留）。"""

        # ── 配置选择器（使用缓存的初始值）──
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

        # ── 后端选择开关（Switch 在上，后端名文字在下；无「后端」标签）──
        # 关 = Llama（本地默认后端），开 = API（远程）；切换经 _on_backend_switch。
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

        # ── 服务配置 Dropdown（config_type 默认 llama，与 current_backend 一致；
        #    由 set_config_type 在切换后端时切换目标目录）
        #    宽度与其它配置选择控件一致（PICKER_WIDTH_SM）──
        server_picker, self._get_server = config_picker(
            "服务配置", [], config_type="llama", width=Layout.PICKER_WIDTH_SM,
            value=self.selected_server,
        )
        self._set_server_ctype = self._get_server.set_config_type

        # ── 服务管理栏：两行共用 12 列网格（ResponsiveRow），同索引控件
        #    左边缘水平对齐——第一行第 2 个「服务配置」与第二行第 2 个
        #    「翻译参数」对齐；merge_backend=True：后端选择（Switch+后端名）
        #    与启动/停止按钮组合并为行尾一个复合组件；
        #    第二行 extra_items：提示词/翻译参数/规则/术语表 ──
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
            merge_backend=True,           # 后端选择 + 启停按钮组合并为行尾复合组件
            row1_cols=(3, 3, 6),          # 状态 / 服务配置 / 后端+启停
            row2_cols=(3, 3, 3, 3),       # 提示词 / 翻译参数 / 规则 / 术语表
        )

        # ── 输入区 ──
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
                # 选择文件按钮位于「源文本」标题行最右侧对齐
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

        # ── 任务队列 ──
        task_panel = ft.Container(
            ref=self.task_container,
            content=task_queue_panel(
                # 切页重建时从实例缓存渲染（与 _rebuild_task_panel 一致），避免任务丢失
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

        # ── 工作区（宽屏双栏；窄屏纵向排列，保留页面滚动）──
        # 宽屏 workspace 不再固定高度：expand 填满内容区剩余高度，
        # 保证源文本/任务队列面板底边与内容区底边对齐、完整显示。
        is_narrow = self.page.width > 0 and self.page.width < Layout.DESKTOP_MIN_WIDTH
        workspace = (
            # 窄屏同样不滚动：scroll 容器内 flex 子项高度塌缩(不显示)，
            # 面板内部(task_rows/输入框)已有滚动，外层用有界 flex 分配
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

        # 根列不滚动：flet 0.86.2 的 scroll 容器(Flutter ListView)主轴无界，
        # 内部 expand 子项高度塌缩为 0 导致组件不显示；窗口已固定，无需整页滚动。
        return ft.Column([
            service_bar,
            ft.Container(height=Layout.SECTION_GAP),
            workspace,
        ], spacing=0, expand=True,
           horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    def save_ui_state(self) -> None:
        """离开页面前从控件 refs 同步状态到实例属性。"""
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
        """导航切回/提交/启停后的兜底：从后端拉取最新快照 → update_* 渲染。"""
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
        """将输入打包为 TranslationRequest 列表（统一返回 list）。

        - str（文本）→ 写入 temp/input_{年月日时分秒}.txt 作 file_path
        - Path → 直接用该文件作 file_path
        - list[Path] → 逐个生成

        task_type 取 self.current_backend（'llama'/'api'，前端以 backend 值为
        服务单元名）；configs 为配置选择器读取的路径字典（translate_config /
        prompts / glossary / rule，glossary 未选时保留 None 键）。
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
        # str 文本 → 临时文件 temp/input_{年月日时分秒}.txt
        # 必须 newline="\n":Windows 上 write_text 默认 newline=None 会把 \n 转成
        # \r\n,若 source 本身是 \r\n(如从文件预览进 TextField),会写成 \r\r\n
        # 双重回车,executor 读回时 universal newline 解析为 2 个换行 → 每行后
        # 多一个空行(见 _read_text_file 的换行归一化)。这里统一写 \n,交由
        # 读取方(read_text universal newline)归一化。
        tmp_dir = Path("temp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"input_{time.strftime('%Y%m%d%H%M%S')}.txt"
        tmp.write_text(str(source), encoding="utf-8", newline="\n")
        return [_make(tmp, tmp.name)]

    # ════════════════════════════════════════════════════
    #  内部方法 — UI 样式（功能已移除，占位保留签名）
    # ════════════════════════════════════════════════════

    def _set_backend_style(self):
        """根据 current_backend 同步后端选择开关与动态标签。"""
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
        """打开软件时读取"翻译默认配置"（configs/system/default.ini [translate]）。

        初始化各配置选择器的默认选中项（llama 后端）；api 后端默认在切换后端时
        经 config_picker.set_value 应用（api_server / translate_args_api）。
        文件缺失或解析失败时保持 None（下拉回退首项）。
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
        """后端选择开关切换：关=Llama（本地默认），开=API（远程）。

        与原 _on_llama_click/_on_api_click 等价：联动服务配置下拉
        （llama → configs/models/llama；api → configs/models/API）、
        翻译参数下拉（translate_args / translate_args_api）、样式刷新
        与 sink 重注册。
        """
        self.current_backend = "api" if e.control.value else "llama"
        if self.current_backend == "llama":
            if self._set_server_ctype:
                self._set_server_ctype("llama")   # 服务配置下拉切换到 configs/models/llama
                if self._get_server and self._default_cfg.get("llama_server"):
                    self._get_server.set_value(self._default_cfg["llama_server"])
            if self._set_args_ctype:
                self._set_args_ctype("translate_args")   # 翻译参数下拉切换到 Llama 参数目录
                if self._get_args and self._default_cfg.get("translate_args"):
                    self._get_args.set_value(self._default_cfg["translate_args"])
        else:
            if self._set_server_ctype:
                self._set_server_ctype("api")     # 服务配置下拉切换到 configs/models/API
                if self._get_server and self._default_cfg.get("api_server"):
                    self._get_server.set_value(self._default_cfg["api_server"])
            if self._set_args_ctype:
                self._set_args_ctype("translate_args_api")   # 翻译参数下拉切换到 API 参数目录
                if self._get_args and self._default_cfg.get("translate_args_api"):
                    self._get_args.set_value(self._default_cfg["translate_args_api"])
        self._set_backend_style()
        self.service_device = self._current_device()
        self._update_service_status()   # 状态文字同步为当前后端（Llama/API × 未加载/已加载/正在加载）
        self._re_register_sink()

    def _re_register_sink(self):
        """后端切换后：sink 重新注册到当前服务的 key（'llama'/'api'）。"""
        if self._sink is not None and self.facade is not None \
                and hasattr(self.facade, "register_ui_sink"):
            self.facade.register_ui_sink(self.current_backend, self._sink)

    # ════════════════════════════════════════════════════
    #  内部方法 — 服务状态与任务（功能已移除，占位保留签名）
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
        """刷新状态点/标签/启停按钮。"""
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
        """返回 (显示文本, 颜色)；CUDA 绿 / CPU 橙 / API 蓝，未知返回 None。"""
        if not device:
            return None, None
        dv = str(device).lower()
        if dv.startswith("cuda"):
            return "CUDA", Palette.SUCCESS
        if dv.startswith("cpu"):
            return "CPU", Palette.WARNING
        return dv.upper(), Palette.PRIMARY

    def _apply_device_span(self, label: ft.Text) -> None:
        """把实际工作设备作为彩色 TextSpan 追加到状态文字。"""
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
        """从 facade 查询当前后端实际工作设备（无 facade 时返回 None）。"""
        if self.facade is not None and hasattr(self.facade, "get_service_device"):
            try:
                return self.facade.get_service_device(self.current_backend)
            except Exception:
                return None
        return None

    def _on_move_up(self, task_id):
        """上移等待任务。"""
        self._move_waiting(task_id, -1)

    def _on_move_down(self, task_id):
        """下移等待任务。"""
        self._move_waiting(task_id, 1)

    def _move_waiting(self, task_id, delta):
        """本地缓存交换位置 + 同步 facade（core 队列为最终秩序）。"""
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
        """取消任务（本地缓存移除 + 同步 facade）。

        当前任务（current_task）与等待任务（waiting_tasks）都需同步：
        facade.cancel_task 必须总是调用（core 对 running 任务发取消信号、
        对 pending 任务移除）——此前仅当 waiting_tasks 变化才调用，
        导致当前任务无法取消。
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
        """清空等待任务（保留当前运行任务与已完成任务）。"""
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
        """暂停/恢复队列：仅服务加载后可开启；暂停始终允许（服务未加载时本就暂停）。"""
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
        """从后端拉取当前/等待任务并渲染（导航切回兜底；推送接线后仍保留）。"""
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
        }

    # ════════════════════════════════════════════════════
    #  内部方法 — 服务操作（功能已移除，占位保留签名）
    # ════════════════════════════════════════════════════

    async def _start_service(self, e):
        """启动翻译服务（current_backend 即注册服务 key；config_path 用服务配置选择器）。"""
        if self.facade is None:
            log.record("warn", "[翻译] facade 未接线")
            return
        name = self.current_backend
        config_path = self._get_server() if self._get_server else None
        try:
            # core 启动为同步阻塞（llama 拉起/模型加载）→ 线程池避免阻塞事件循环
            await asyncio.to_thread(self.facade.start_service, name, None, config_path)
            # 主动刷新按钮状态（推送兜底：即使后端未回调也即时更新）
            self.service_device = self._current_device()
            self.update_service_status(True, False, self.service_device)
        except Exception as ex:
            log.record("error", f"[翻译] 启动失败: {ex}")
            self.service_device = None
            self.update_service_status(False, False)

    async def _stop_service(self, e):
        """停止翻译服务（core 内先取消当前任务；完成后主动刷新按钮状态）。"""
        if self.facade is None:
            return
        try:
            await asyncio.to_thread(self.facade.stop_service, self.current_backend)
            # 主动刷新按钮状态（推送兜底）
            self.service_device = None
            self.update_service_status(False, False)
        except Exception as ex:
            log.record("error", f"[翻译] 停止失败: {ex}")

    # ════════════════════════════════════════════════════
    #  内部方法 — 文件选择与提交（功能已移除，占位保留签名）
    # ════════════════════════════════════════════════════

    async def _pick_file(self, e):
        """选择导入方式 — AlertDialog 分流：选择文件（可多选、后缀不限）/ 选择文件夹。"""
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
        # flet 0.86.2：Page 无 dialog 属性，必须用 show_dialog()（加入对话框栈并渲染）
        self.page.show_dialog(dlg)

    def _close_import_dialog(self, e=None):
        dlg = getattr(self, "_import_dialog", None)
        if dlg is not None:
            dlg.open = False
            self.page.update()  # 与 settings 页 _close_dlg 模式一致（show_dialog 栈内关闭）

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
            # flet 0.86.2：文件夹选择 API 为 get_directory_path（直接返回路径字符串），
            # pick_directory 已移除
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
        """扫描文件夹顶层：仅 is_file、不递归子目录、按文件名排序。"""
        try:
            entries = [p for p in path.iterdir() if p.is_file()]
        except OSError as ex:
            log.record("error", f"[翻译] 扫描文件夹失败: {ex}")
            return []
        return sorted(entries, key=lambda p: p.name.lower())

    async def _process_selected(self, paths: list):
        """分发：单文件 → 验证后预览填充输入区；多文件/文件夹 → 批量入队。"""
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
                # 记录源文件（供提交文本时沿用文件名；提交成功后重置）
                try:
                    self.input_path = paths[0].resolve()
                except OSError:
                    self.input_path = paths[0].absolute()
            return
        self._enqueue_translations(paths)

    def _enqueue_translations(self, paths: list):
        """批量校验并入队：后缀不限、NUL/解码失败拒绝且不中断。

        允许同一文件多次添加（每次导入都是独立任务）；
        文本任务（_submit_translation）不经过此路径。
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
        """统一入队：facade 提交（存在时）+ 本地缓存 + 提示。

        文件批量导入与文本提交共用；文本任务不经过 _queued_paths 去重
        （每次提交为独立新任务）。
        - facade 已接线：队列展示由 core _emit → update_tasks 推送接管（不手动重建）
        - facade 未接线：本地缓存 + 手动重建兜底
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
            submitted = len(reqs)  # facade 未接线：仅本地缓存展示
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
            # flet 0.86.2：SnackBar 为 DialogControl，经 show_dialog 显示（show_snack_bar 已移除）
            self.page.show_dialog(
                ft.SnackBar(ft.Text(f"已加入队列 {submitted} 个任务"), bgcolor=Palette.SUCCESS)
            )
        return submitted

    def _rebuild_task_panel(self):
        """重建任务队列面板（本地缓存驱动；后端 update_tasks 接线后由推送接管）。"""
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
        """提交输入区文本为翻译任务（写入 temp/input_{时间戳}.txt 后统一入队）。"""
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
        # 预览读取的文本：任务文件名沿用源文件名（内容仍为输入区当前文本）
        if self.input_path is not None:
            reqs[0].file_name = self.input_path.name
        submitted = self._enqueue_requests(reqs)
        if submitted > 0 and self.input_text.current:
            self.input_text.current.value = ""
            self.input_text.current.update()
            self.input_value = ""
            self.input_path = None  # 重置预览标志，防下次误用

    # ════════════════════════════════════════════════════
    #  Facade 回调（功能已移除，占位保留签名）
    # ════════════════════════════════════════════════════

    def _on_service_change(self, status: dict):
        """服务状态变更回调（功能已移除）。"""
        pass

    def _on_task_change(self, snapshot):
        """任务状态变更回调（功能已移除）。"""
        pass
