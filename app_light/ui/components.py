"""Reusable UI components — refined edition.

Added:
- _shadow() shadow helper
- glow_dot() status indicator dot with a glow
- panel_header() unified panel title row
- service_management_bar() service management component
- toolbar_panel_header() unified task queue / completed tasks top bar
"""

import flet as ft
from ui.theme import Palette, Radius, Anim, Typography


# ── Basic helpers ──

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


# ── Bordered button — icon left-aligned, text right-aligned ──

def bordered_button(
    label: str,
    icon: str,
    on_click=None,
    tooltip: str | None = None,
    width: int | None = None,
    padding=None,
) -> ft.OutlinedButton:
    """Primary-outlined button (unified style): icon on the left, text on the right.

    - Without ``width``: the button sizes to its content, icon and text adjacent (no stretch);
    - With ``width``: the button fills that width, icon flush to the left edge and text to
      the right edge (``SPACE_BETWEEN``), aligning with fixed-width controls such as
      reference/target language pickers.
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


# ── Shadow helper ──

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


# ── Status indicator dot — with glow ──

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


# ── Panel title row — unified style ──

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


# ── Toolbar panel header — for task queue / completed / config library top bar ──

def toolbar_panel_header(
    title: str,
    actions: list[ft.Control] | None = None,
    icon_name: str | None = None,
) -> ft.Row:
    """Unified panel title row: icon + title + a list of action buttons on the right."""
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


# ── Service management bar component ──

def service_management_bar(
    service_type: str,           # "translate" | "transcribe"
    status_dot_ref: ft.Ref,
    status_label_ref: ft.Ref,
    start_btn_ref: ft.Ref,
    stop_btn_ref: ft.Ref,
    backend_selector: ft.Control | None = None,  # backend selector control (translate page)
    config_dropdown: ft.Control | None = None,   # service config Dropdown
    extra_items: list[ft.Control] | None = None, # other config pickers (second row)
    pre_start_actions: list[ft.Control] | None = None,  # action buttons inside the start/stop group, left of the start button
    on_start=None,
    on_stop=None,
    merge_backend: bool = False,               # True: merge backend selector + start/stop group into one trailing composite (translate page)
    row1_cols: tuple[int, ...] = (3, 3, 3, 3),   # status bar column spans (12-col grid)
    row2_cols: tuple[int, ...] = (3, 3, 3, 3),   # config bar column spans (12-col grid)
) -> ft.Container:
    """Service management component — a two-row, 12-column ResponsiveRow toolbar.

    Row 1 (status bar): status indicator + service config + trailing action area; each
        control takes ``row1_cols`` columns of the virtual 12-col grid. The trailing area
        (start/stop button group, right-aligned within the group against the row edge):
        with merge_backend=False it is a single start/stop group cell (transcribe page);
        with merge_backend=True the backend selector and start/stop group are merged into
        one trailing composite (translate page).
    Row 2 (config bar): extra_items (translate page prompts/args/rules/glossary, transcribe
        page transcription args/VAD/hotwords), each control takes ``row2_cols`` columns.

    Both rows share the same 12-col grid, so controls at the same index are left-aligned
        horizontally — e.g. the 2nd control "Service Config" in row 1 aligns with the 2nd
        control "Translate Args / VAD" in row 2.

    row1_cols / row2_cols lengths must match the actual control counts, and each row must
        sum to 12 (exceeding 12 triggers a ResponsiveRow wrap, breaking single-row and
        cross-row column alignment).

    backend_selector: Switch composite (translate/transcribe page) or Dropdown; optional.
    extra_items: other config pickers, placed independently in row 2.
    Note (flet 0.86.2): ResponsiveRow requires the parent to provide a bounded width
        (satisfied inside the page content area's STRETCH Column); child widths are
        precisely constrained by the grid (same col = same width). Do not place
        expand/flex controls in row1/row2; column widths are fully determined by col.
    """
    status_icon_map = {
        "translate": ft.Icons.TRANSLATE,
        "transcribe": ft.Icons.MIC,
        "gsv": ft.Icons.RECORD_VOICE_OVER,
    }
    icon = status_icon_map.get(service_type, ft.Icons.DNS)

    # ── Row 1 (status bar): status indicator (icon + dot + text) → service config →
    #    backend (translate page only) → start/stop group, fixed control order ──
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
        row1_items.append(config_dropdown)   # 2nd position: left-aligned with the 2nd config in row 2
    # Start/stop group: fixed trailing column + right-aligned within the group (against the row edge)
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
    # Translate page (merge_backend=True): backend selector + start/stop group merged into one trailing composite
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

    # ── Row 2 (config bar): other config pickers, each taking row2_cols columns ──
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


# ── Stat card ──

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


# ── Panel container ──

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


# ── Status chip ──

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
