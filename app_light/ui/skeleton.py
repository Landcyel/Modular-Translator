"""Skeleton screen components — a lightweight UI rendered first during app startup / page switches.

Two-phase render strategy (avoids a blank screen on open):
- The first frame pushes only the skeleton: brand area + top-bar window controls + nav
  rail + a content-area ProgressRing (~40 controls total, serialized/rendered in
  milliseconds, so the user immediately sees the app frame rather than a blank screen).
- Once the full page tree is built on a background thread, ui/layout replaces
  content_area.content (for the home page) or replaces the page root wholesale
  (skeleton → full-UI switch).

During the skeleton phase the window control buttons (minimize/maximize/close) stay fully
functional, so the user can still drag/minimize/close the window while loading; the nav
rail does not respond to clicks (loading).
"""

import flet as ft

from ui.theme import Layout, Palette, Radius, Typography
from ui.components import _shadow


def skeleton_placeholder(
    text: str = "正在加载…",
    *,
    title: str | None = None,
    subtitle: str | None = None,
    ref: ft.Ref[ft.Container] | None = None,
    text_ref: ft.Ref[ft.Text] | None = None,
) -> ft.Container:
    """Content-area loading placeholder: shimmer bars + ProgressRing + hint text (centered both axes).

    title/subtitle are optional title and subtitle (explaining which module is loading during
    the skeleton phase); ref/text_ref let ui/layout's loading-animation coroutine drive the
    whole-area pulse and hint-text rotation. Defaults keep backward compatibility (no-arg call
    = the original "spinner + Loading...").
    """
    children: list[ft.Control] = []
    if title:
        children.append(ft.Text(title, size=Typography.BODY_LG,
                                weight=ft.FontWeight.BOLD, color=Palette.TEXT))
        children.append(ft.Container(height=8))
    # shimmer placeholder bars (rounded grey bars simulating text lines, decreasing widths)
    children.append(ft.Column(
        [
            ft.Container(width=300, height=10, border_radius=5, bgcolor=Palette.SURFACE2),
            ft.Container(width=240, height=10, border_radius=5, bgcolor=Palette.SURFACE2),
            ft.Container(width=180, height=10, border_radius=5, bgcolor=Palette.SURFACE2),
        ],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    ))
    children.append(ft.Container(height=18))
    children.append(ft.ProgressRing(width=38, height=38, stroke_width=3,
                                    color=Palette.PRIMARY))
    children.append(ft.Container(height=14))
    children.append(ft.Text(text, ref=text_ref, size=Typography.BODY,
                            color=Palette.SUBTEXT))
    if subtitle:
        children.append(ft.Container(height=6))
        children.append(ft.Text(subtitle, size=Typography.CAPTION,
                                color=Palette.TEXT_MUTED))
    return ft.Container(
        content=ft.Column(children,
                          horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                          spacing=0),
        ref=ref,
        alignment=ft.Alignment.CENTER,
        expand=True,
        bgcolor=Palette.BG,
    )


def error_placeholder(message: str) -> ft.Container:
    """Page-build-failure placeholder: error icon + message (avoids blank screen / silent crash)."""
    return ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.ERROR_OUTLINE, size=40, color=Palette.ERROR),
                ft.Container(height=14),
                ft.Text("页面加载失败", size=Typography.BODY_LG,
                        weight=ft.FontWeight.BOLD, color=Palette.TEXT),
                ft.Container(height=6),
                ft.Text(message, size=Typography.BODY, color=Palette.SUBTEXT,
                        text_align=ft.TextAlign.CENTER),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0,
        ),
        alignment=ft.Alignment.CENTER,
        expand=True,
        bgcolor=Palette.BG,
    )


def _brand_container() -> ft.Container:
    """Brand area (visually consistent with ui/layout; a standalone instance during the skeleton phase)."""
    brand_icon_box = ft.Container(
        content=ft.Text("T", size=20, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
        gradient=ft.LinearGradient(
            begin=ft.Alignment.TOP_LEFT,
            end=ft.Alignment.BOTTOM_RIGHT,
            colors=[Palette.PRIMARY, Palette.PRIMARY_DARK],
        ),
        border_radius=Radius.SM,
        width=36, height=36,
        alignment=ft.Alignment.CENTER,
        shadow=_shadow("low"),
    )
    brand_text_column = ft.Column(
        [
            ft.Text("Modular Translator", size=Typography.HEADING,
                    weight=ft.FontWeight.BOLD, color=Palette.TEXT),
            ft.Text("Translation Suite", size=Typography.CAPTION,
                    color=Palette.TEXT_MUTED,
                    style=ft.TextStyle(letter_spacing=1)),
        ],
        spacing=0,
    )
    return ft.Container(
        content=ft.Row([brand_icon_box, brand_text_column], spacing=10),
        bgcolor=Palette.SURFACE,
        border=ft.Border.only(
            right=ft.BorderSide(1, Palette.BORDER),
            bottom=ft.BorderSide(1, Palette.BORDER),
        ),
        width=Layout.SIDEBAR_WIDTH,
        height=Layout.BRAND_HEIGHT,
        padding=ft.Padding.all(16),
    )


def _nav_rail_container() -> ft.Container:
    """Navigation rail (no switching during the skeleton phase; shows the full nav structure)."""
    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        min_extended_width=180,
        group_alignment=-0.9,
        extended=True,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.MIC, selected_icon=ft.Icons.MIC,
                label="转写", padding=ft.Padding.symmetric(vertical=8)),
            ft.NavigationRailDestination(
                icon=ft.Icons.TRANSLATE, selected_icon=ft.Icons.TRANSLATE,
                label="翻译", padding=ft.Padding.symmetric(vertical=8)),
            ft.NavigationRailDestination(
                icon=ft.Icons.RECORD_VOICE_OVER, selected_icon=ft.Icons.RECORD_VOICE_OVER,
                label="语音合成", padding=ft.Padding.symmetric(vertical=8)),
            ft.NavigationRailDestination(
                icon=ft.Icons.CHECK_CIRCLE, selected_icon=ft.Icons.CHECK_CIRCLE,
                label="已完成", padding=ft.Padding.symmetric(vertical=8)),
            ft.NavigationRailDestination(
                icon=ft.Icons.SETTINGS, selected_icon=ft.Icons.SETTINGS,
                label="设置", padding=ft.Padding.symmetric(vertical=8)),
            ft.NavigationRailDestination(
                icon=ft.Icons.ARTICLE, selected_icon=ft.Icons.ARTICLE,
                label="日志", padding=ft.Padding.symmetric(vertical=8)),
        ],
        on_change=None,  # no switching while loading
        bgcolor=Palette.SURFACE,
    )
    return ft.Container(
        content=rail,
        padding=ft.Padding.only(left=16, top=12, right=16),
        bgcolor=Palette.SURFACE,
        border=ft.Border.only(right=ft.BorderSide(1, Palette.BORDER)),
        width=Layout.SIDEBAR_WIDTH,
        expand=True,
    )


def _app_bar(page: ft.Page) -> ft.Container:
    """Top bar (window controls fully functional: drag-area double-click to maximize + minimize/maximize/close)."""
    maximize_button = ft.IconButton(
        icon=ft.Icons.ASPECT_RATIO if page.window.maximized else ft.Icons.CROP_SQUARE,
        icon_size=18, icon_color=Palette.SUBTEXT,
        tooltip="还原" if page.window.maximized else "最大化",
        style=ft.ButtonStyle(padding=ft.Padding.all(6)),
    )

    def _toggle_maximize(e=None):
        w = page.window
        w.maximized = not w.maximized
        maximize_button.icon = ft.Icons.ASPECT_RATIO if w.maximized else ft.Icons.CROP_SQUARE
        maximize_button.tooltip = "还原" if w.maximized else "最大化"
        try:
            maximize_button.update()
        except RuntimeError:
            pass
        page.update()

    maximize_button.on_click = _toggle_maximize

    drag_area = ft.WindowDragArea(
        content=ft.Container(expand=True),
        expand=True,
        on_double_tap=_toggle_maximize,
    )

    minimize_button = ft.IconButton(
        icon=ft.Icons.MINIMIZE, icon_size=18, icon_color=Palette.SUBTEXT,
        tooltip="最小化",
        on_click=lambda e: setattr(page.window, "minimized", True) or page.update(),
        style=ft.ButtonStyle(padding=ft.Padding.all(6)),
    )

    async def close_window(e):
        await page.window.close()

    close_button = ft.IconButton(
        icon=ft.Icons.CLOSE, icon_size=18, icon_color=Palette.SUBTEXT,
        tooltip="关闭", on_click=close_window,
        style=ft.ButtonStyle(padding=ft.Padding.all(6)),
    )

    window_controls_row = ft.Row(
        [minimize_button, maximize_button, close_button], spacing=4)

    return ft.Container(
        content=ft.Row(
            [drag_area, window_controls_row],
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        height=Layout.APP_BAR_HEIGHT,
        padding=ft.Padding.symmetric(horizontal=16),
        bgcolor=Palette.SURFACE,
        border=ft.Border.only(bottom=ft.BorderSide(1, Palette.BORDER)),
    )


def skeleton_screen(page: ft.Page) -> ft.Control:
    """Lightweight shell: brand area + top bar + nav rail + content-area loading placeholder."""
    left_column = ft.Column(
        [_brand_container(), _nav_rail_container()],
        spacing=0, width=Layout.SIDEBAR_WIDTH,
    )
    right_column = ft.Column(
        [_app_bar(page), skeleton_placeholder()],
        spacing=0, expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )
    root_row = ft.Row(
        [left_column, right_column],
        spacing=0, expand=True,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )
    return ft.Container(
        content=root_row,
        expand=True,
        bgcolor=Palette.BG,
    )
