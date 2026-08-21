"""骨架屏组件 — 应用启动/页面切换时先行渲染的轻量界面。

两阶段渲染策略（避免打开白屏）：
- 首帧只推骨架：品牌区 + 顶栏窗口控制 + 导航栏 + 内容区 ProgressRing
  （控件总数约 40，毫秒级序列化/渲染，用户立刻看到界面框架而非白屏）。
- 完整页面树在后台线程构建完成后，由 ui/layout 替换 content_area.content
  （启动首页）或整体替换 page 根（骨架→完整界面切换）。

骨架阶段窗口控制按钮（最小化/最大化/关闭）保持真实可用，加载期间
用户仍可拖拽/最小化/关闭窗口；导航栏不响应点击（加载中）。
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
    """内容区加载占位：shimmer 占位条 + ProgressRing + 提示文本（垂直水平居中）。

    title/subtitle 为可选标题与副标题（骨架阶段说明正在加载的模块）；
    ref/text_ref 供 ui/layout 的加载动画协程驱动整区脉冲与提示文案轮换。
    默认参数保持向后兼容（无参调用 = 原「转圈 + 正在加载…」）。
    """
    children: list[ft.Control] = []
    if title:
        children.append(ft.Text(title, size=Typography.BODY_LG,
                                weight=ft.FontWeight.BOLD, color=Palette.TEXT))
        children.append(ft.Container(height=8))
    # shimmer 占位条（圆角灰条模拟文本行，宽度递减）
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
    """页面构建失败占位：错误图标 + 信息（避免白屏/静默崩溃）。"""
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
    """品牌区（与 ui/layout 视觉一致，骨架阶段独立实例）。"""
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
    """导航栏（骨架阶段不响应切换，展示完整导航结构）。"""
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
        on_change=None,  # 加载中不可切换
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
    """顶栏（窗口控制真实可用：拖拽区双击最大化 + 最小化/最大化/关闭）。"""
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
    """轻量外壳：品牌区 + 顶栏 + 导航栏 + 内容区加载占位。"""
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
