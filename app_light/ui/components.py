"""可复用 UI 组件 — 精修版。

新增：
- _shadow() 阴影辅助
- glow_dot() 带光晕的状态指示点
- panel_header() 统一面板标题行
- service_management_bar() 服务管理组件
- toolbar_panel_header() 统一任务队列/完成任务顶栏
"""

import flet as ft
from ui.theme import Palette, Radius, Anim, Typography


# ═══════════════════════════════════════════════════════════════
# 基础辅助
# ═══════════════════════════════════════════════════════════════

def _icon(name: str, size: int = 20, color: str | None = None) -> ft.Icon:
    return ft.Icon(name, size=size, color=color)


def _text(value: str, size: int = 14, weight: str = "normal",
          color: str | None = None, italic: bool = False) -> ft.Text:
    w = ft.FontWeight.NORMAL
    if weight == "bold":
        w = ft.FontWeight.BOLD
    elif weight == "w600":
        w = ft.FontWeight.W_600
    elif weight == "w500":
        w = ft.FontWeight.W_500
    return ft.Text(value=value, size=size, weight=w, color=color, italic=italic)


def divider(color: str | None = None) -> ft.Divider:
    return ft.Divider(height=1, thickness=1, color=color or Palette.BORDER_SUBTLE)


# ═══════════════════════════════════════════════════════════════
# 带边框按钮 — 统一图标左对齐、文字右对齐
# ═══════════════════════════════════════════════════════════════

def bordered_button(
    label: str,
    icon: str,
    on_click=None,
    tooltip: str | None = None,
    width: int | None = None,
    padding=None,
) -> ft.OutlinedButton:
    """主色描边按钮（统一样式）：图标在左、文字在右。

    - 未指定 ``width``：按钮按内容自适应，图标与文字相邻（不拉伸）；
    - 指定 ``width``：按钮撑满该宽度，图标贴左缘、文字贴右缘
      （``SPACE_BETWEEN``），与参考语种/目标语种等固定宽度控件对齐。
    """
    style_kwargs = dict(
        color=Palette.PRIMARY,
        shape=ft.RoundedRectangleBorder(radius=Radius.MD),
        side=ft.BorderSide(1, Palette.PRIMARY),
    )
    if padding is not None:
        style_kwargs["padding"] = padding
    return ft.OutlinedButton(
        content=ft.Row([
            _icon(icon, size=18, color=Palette.PRIMARY),
            ft.Text(label, size=14, color=Palette.PRIMARY),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
           vertical_alignment=ft.CrossAxisAlignment.CENTER,
           tight=width is None),
        on_click=on_click,
        tooltip=tooltip,
        width=width,
        style=ft.ButtonStyle(**style_kwargs),
    )


# ═══════════════════════════════════════════════════════════════
# 阴影辅助
# ═══════════════════════════════════════════════════════════════

def _shadow(level: str = "low") -> list:
    if level == "med":
        return [
            ft.BoxShadow(blur_radius=12, spread_radius=0,
                         color="#00000030", offset=ft.Offset(0, 4)),
            ft.BoxShadow(blur_radius=4, spread_radius=0,
                         color="#00000018", offset=ft.Offset(0, 1)),
        ]
    if level == "high":
        return [
            ft.BoxShadow(blur_radius=24, spread_radius=0,
                         color="#00000040", offset=ft.Offset(0, 8)),
            ft.BoxShadow(blur_radius=8, spread_radius=0,
                         color="#00000020", offset=ft.Offset(0, 2)),
        ]
    return [
        ft.BoxShadow(blur_radius=8, spread_radius=0,
                     color="#00000025", offset=ft.Offset(0, 2)),
    ]


# ═══════════════════════════════════════════════════════════════
# 状态指示点 — 带光晕
# ═══════════════════════════════════════════════════════════════

def glow_dot(
    color: str = Palette.SUCCESS,
    size: int = 8,
    ref: ft.Ref | None = None,
) -> ft.Container:
    inner = ft.Container(
        width=size, height=size,
        border_radius=size / 2,
        bgcolor=color,
    )
    if ref is not None:
        inner.ref = ref
    return ft.Container(
        content=inner,
        width=size + 8, height=size + 8,
        border_radius=(size + 8) / 2,
        bgcolor=f"{color}25",
        alignment=ft.Alignment.CENTER,
        animate=ft.Animation(Anim.SLOW, ft.AnimationCurve.EASE),
    )


# ═══════════════════════════════════════════════════════════════
# 面板标题行 — 统一风格
# ═══════════════════════════════════════════════════════════════

def panel_header(
    title: str,
    icon_name: str | None = None,
    icon_color: str = Palette.PRIMARY,
    trailing: ft.Control | None = None,
) -> ft.Row:
    items = []
    if icon_name:
        items.append(
            ft.Container(
                content=_icon(icon_name, 18, icon_color),
                bgcolor=f"{icon_color}18",
                border_radius=Radius.SM,
                padding=ft.Padding.all(6),
            )
        )
    items.append(_text(title, Typography.HEADING_SM, "bold", Palette.TEXT))
    if trailing:
        items.append(ft.Container(expand=True))
        items.append(trailing)
    return ft.Row(items, spacing=10)


# ═══════════════════════════════════════════════════════════════
# 工具栏面板标题行 — 用于任务队列/完成任务/配置库顶栏
# ═══════════════════════════════════════════════════════════════

def toolbar_panel_header(
    title: str,
    actions: list[ft.Control] | None = None,
    icon_name: str | None = None,
) -> ft.Row:
    """统一的面板标题行：图标 + 标题 + 右侧操作按钮列表。"""
    items = []
    if icon_name:
        items.append(
            ft.Container(
                content=_icon(icon_name, 16, Palette.PRIMARY),
                bgcolor=f"{Palette.PRIMARY}18",
                border_radius=Radius.SM,
                padding=ft.Padding.all(6),
            )
        )
    items.append(_text(title, Typography.HEADING_SM, "bold", Palette.TEXT))
    items.append(ft.Container(expand=True))
    if actions:
        items.extend(actions)
    return ft.Row(items, spacing=8)


# ═══════════════════════════════════════════════════════════════
# 服务管理栏组件
# ═══════════════════════════════════════════════════════════════

def service_management_bar(
    service_type: str,           # "translate" | "transcribe"
    status_dot_ref: ft.Ref,
    status_label_ref: ft.Ref,
    start_btn_ref: ft.Ref,
    stop_btn_ref: ft.Ref,
    backend_selector: ft.Control | None = None,  # 后端选择控件（翻译页用）
    config_dropdown: ft.Control | None = None,   # 服务配置 Dropdown
    extra_items: list[ft.Control] | None = None, # 其它配置选择器（第二行）
    pre_start_actions: list[ft.Control] | None = None,  # 启停按钮组内、启动按钮左侧的操作按钮
    on_start=None,
    on_stop=None,
    merge_backend: bool = False,               # True：后端选择与启停按钮组合并为行尾复合组件（翻译页）
    row1_cols: tuple[int, ...] = (3, 3, 3, 3),   # 状态栏各控件列跨度（12 列网格）
    row2_cols: tuple[int, ...] = (3, 3, 3, 3),   # 配置栏各控件列跨度（12 列网格）
) -> ft.Container:
    """服务管理组件 — ResponsiveRow 两行 12 列网格工具条。

    第一行（状态栏）：状态指示 + 服务配置 + 行尾操作区，各控件按 ``row1_cols``
        占虚拟 12 列网格中的列数。行尾操作区（启停按钮组，组内右对齐贴行右边缘）：
        merge_backend=False 时为启停按钮组单独一个单元（转写页）；
        merge_backend=True 时后端选择与启停按钮组合并为行尾一个复合组件（翻译页）。
    第二行（配置栏）：extra_items（翻译页的提示词/参数/规则/术语表、
        转写页的转写参数/VAD/Hotwords），各控件按 ``row2_cols`` 占列。

    两行共用同一 12 列网格，相同位置（索引）的控件左边缘水平对齐——
    例如第一行第 2 个「服务配置」与第二行第 2 个「翻译参数/VAD」左对齐。

    row1_cols / row2_cols 长度必须与实际控件数一致，且每行总和应为 12
    （超出 12 会触发 ResponsiveRow 换行，破坏单行对齐与跨行列对齐）。

    backend_selector: Switch 复合控件（翻译页/转写页）或 Dropdown，可选。
    extra_items: 其它配置选择器，独立放入第二行。
    注意（flet 0.86.2）：ResponsiveRow 要求父级提供有界宽度（页面内容区
        STRETCH Column 内满足）；子控件宽度由列网格精确约束（相同 col
        的控件宽度一致），勿在 row1/row2 中放入 expand/flex 控件，列宽
        完全由 col 决定。
    """
    status_icon_map = {
        "translate": ft.Icons.TRANSLATE,
        "transcribe": ft.Icons.MIC,
        "gsv": ft.Icons.RECORD_VOICE_OVER,
    }
    icon = status_icon_map.get(service_type, ft.Icons.DNS)

    # ── 第一行（状态栏）：状态指示（icon + 状态点 + 文字）→ 服务配置 →
    #    后端（仅翻译页）→ 启停按钮组，控件顺序固定 ──
    row1_items = [
        ft.Row([
            _icon(icon, 16, Palette.PRIMARY),
            ft.Row([
                ft.Container(ref=status_dot_ref, width=8, height=8,
                             border_radius=4, bgcolor=Palette.ERROR,
                             animate=ft.Animation(500, ft.AnimationCurve.EASE)),
                ft.Text(ref=status_label_ref, value="离线",
                        size=13, weight=ft.FontWeight.W_600, color=Palette.TEXT),
            ], spacing=6),
        ], spacing=8),
    ]
    if config_dropdown:
        row1_items.append(config_dropdown)   # 第 2 位：与第二行第 2 个配置左对齐
    # 启停按钮组：行尾固定列 + 组内右对齐（贴行右边缘）
    action_row = ft.Row([
        *(pre_start_actions or []),
        ft.TextButton(
            "启动服务", ref=start_btn_ref, visible=True,
            icon=ft.Icons.PLAY_ARROW,
            on_click=on_start,
            style=ft.ButtonStyle(color=Palette.SUCCESS),
        ),
        ft.TextButton(
            "停止服务", ref=stop_btn_ref, visible=False,
            icon=ft.Icons.STOP,
            on_click=on_stop,
            style=ft.ButtonStyle(color=Palette.ERROR),
        ),
    ], spacing=8, alignment=ft.MainAxisAlignment.END)
    # 翻译页（merge_backend=True）：后端选择 + 启停按钮组合并为行尾一个复合组件
    if merge_backend and backend_selector:
        row1_items.append(
            ft.Row([backend_selector, action_row], spacing=12,
                   alignment=ft.MainAxisAlignment.END)
        )
    else:
        if backend_selector:
            row1_items.append(backend_selector)
        row1_items.append(action_row)
    if len(row1_cols) != len(row1_items):
        raise ValueError(
            f"row1_cols 长度 {len(row1_cols)} 与状态栏控件数 {len(row1_items)} 不一致"
        )
    if sum(row1_cols) != 12:
        raise ValueError(
            f"row1_cols 总和 {sum(row1_cols)} != 12（每行须占满 12 列网格）"
        )
    for ctrl, col in zip(row1_items, row1_cols):
        ctrl.col = col
    row1 = ft.ResponsiveRow(row1_items, spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # ── 第二行（配置栏）：其它配置选择器，按 row2_cols 占列 ──
    rows = [row1]
    if extra_items:
        if len(row2_cols) != len(extra_items):
            raise ValueError(
                f"row2_cols 长度 {len(row2_cols)} 与配置栏控件数 {len(extra_items)} 不一致"
            )
        if sum(row2_cols) != 12:
            raise ValueError(
                f"row2_cols 总和 {sum(row2_cols)} != 12（每行须占满 12 列网格）"
            )
        for ctrl, col in zip(extra_items, row2_cols):
            ctrl.col = col
        rows.append(
            ft.ResponsiveRow(extra_items, spacing=8,
                             vertical_alignment=ft.CrossAxisAlignment.CENTER)
        )

    return ft.Container(
        content=ft.Column(rows, spacing=8,
                          horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
        bgcolor=Palette.SURFACE,
        border_radius=Radius.LG,
        padding=ft.Padding.symmetric(horizontal=14, vertical=10),
        border=ft.Border.all(1, Palette.BORDER_SUBTLE),
        shadow=_shadow("low"),
    )


# ═══════════════════════════════════════════════════════════════
# 统计卡片
# ═══════════════════════════════════════════════════════════════

def stat_card(
    title: str, value: str, icon_name: str,
    gradient: list[str],
    subtitle: str = "",
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=_icon(icon_name, 22, "#FFFFFF"),
                    bgcolor="#FFFFFF30",
                    border_radius=Radius.LG,
                    padding=ft.Padding.all(8),
                ),
                ft.Text(value, size=Typography.DISPLAY, weight=ft.FontWeight.BOLD,
                        color="#FFFFFF"),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Text(title, size=Typography.BODY, color="#FFFFFFDD"),
            ft.Text(subtitle, size=Typography.SMALL, color="#FFFFFF88") if subtitle else ft.Container(),
        ], spacing=6, alignment=ft.MainAxisAlignment.START),
        gradient=ft.LinearGradient(
            begin=ft.Alignment.TOP_LEFT,
            end=ft.Alignment.BOTTOM_RIGHT,
            colors=gradient,
        ),
        border_radius=Radius.XL,
        padding=ft.Padding.all(20),
        shadow=_shadow("med"),
        expand=True,
        animate=ft.Animation(Anim.NORMAL, ft.AnimationCurve.EASE),
    )


# ═══════════════════════════════════════════════════════════════
# 面板容器
# ═══════════════════════════════════════════════════════════════

def section_card(title: str, content: ft.Control, icon_name: str | None = None) -> ft.Container:
    header = panel_header(title, icon_name)
    return ft.Container(
        content=ft.Column([
            header,
            divider(),
            content,
        ], spacing=12),
        bgcolor=Palette.SURFACE,
        border_radius=Radius.XL,
        padding=ft.Padding.all(20),
        border=ft.Border.all(1, Palette.BORDER_SUBTLE),
        shadow=_shadow("low"),
    )


# ═══════════════════════════════════════════════════════════════
# 状态 Chip
# ═══════════════════════════════════════════════════════════════

_STATUS_STYLES = {
    "pending":   ("#94A3B8", ft.Icons.HOURGLASS_EMPTY),
    "running":   ("#3B82F6", ft.Icons.PLAY_ARROW),
    "completed": ("#10B981", ft.Icons.CHECK),
    "failed":    ("#EF4444", ft.Icons.ERROR),
    "cancelled": ("#F59E0B", ft.Icons.CANCEL),
}


def status_chip(status: str) -> ft.Chip:
    style = _STATUS_STYLES.get(status, ("#94A3B8", ft.Icons.HELP))
    return ft.Chip(
        label=ft.Text(status.upper(), size=Typography.SMALL, weight=ft.FontWeight.BOLD,
                       color=style[0]),
        leading=ft.Icon(style[1], size=14, color=style[0]),
        bgcolor=f"{style[0]}20",
        shape=ft.RoundedRectangleBorder(radius=Radius.XS),
        padding=ft.Padding.symmetric(horizontal=8, vertical=2),
    )
