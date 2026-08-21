"""主布局骨架 — 左右双列结构（APP_test 外壳）。

左列 Column：品牌区（250px）+ 导航栏（250px，expand 占满剩余高度）
右列 Column：顶栏（34px，窗口控制）+ 内容区（expand）

六页导航：转写 / 翻译 / 语音合成 / 已完成任务 / 设置 / 日志
facade 在此层实例化并传递给各页面 builder。
页面实例模式：每个页面长期持有状态，导航时调用 build/refresh。
"""

import asyncio
import threading
import flet as ft

from app.paths import project_root
from app.log import log
from ui.theme import Anim, Layout, Palette, Radius, Typography
from ui.components import _shadow
from ui.skeleton import skeleton_placeholder, error_placeholder


async def shutdown_and_destroy(facade, page: ft.Page):
    """统一关闭流程：优雅停止服务后销毁窗口。

    供两处共用：APP.py 的 ``window.on_event`` CLOSE 分支，以及本文件顶栏
    关闭按钮（经 ``window.close()`` 走同一 CLOSE 分支）。

    - *facade* 兼容两种传参：服务实例对象，或惰性 getter（build_layout 现
      返回 get_facade 函数，未初始化时为 None）。None 时跳过 shutdown——
      服务从未启动，无残留子进程。
    - ``facade.shutdown()`` 异常只打印不抛出——否则 ``prevent_close=True``
      下唯一出口是 ``destroy()``，异常会导致窗口永远关不掉。
    - ``finally`` 保证 ``destroy()`` 必然执行。
    """
    try:
        facade_obj = facade() if callable(facade) else facade
        if facade_obj is not None:
            facade_obj.shutdown()
    except Exception as exc:
        log.record("error", f"关闭时停止服务异常: {exc}")
    finally:
        await page.window.destroy()


def build_layout(page: ft.Page):
    """构建主布局，返回 (content_area, switch_page_fn, fp_translate, fp_transcribe, fp_log, fp_gsv, get_facade)。

    facade / 页面实例 / core 重链改为惰性初始化：build_layout 同步段只构建
    轻量外壳（品牌/导航/顶栏/骨架占位），零 core import（重模块后移到
    后台线程，由 load_content 触发 ensure_initialized 导入）。

    - get_facade：惰性取 facade 的函数（未初始化返回 None），供窗口关闭时
      调用 shutdown() 优雅停止服务（防 llama-server 子进程残留）。
    - switch_page_fn(idx) 用于切换当前展示的页面内容。
    - fp_translate / fp_transcribe 为 FilePicker 实例（同步创建）。
    """

    # ── 共享 FilePicker（轻量构造，同步创建；APP.py 首帧前注册）──
    file_picker_translate = ft.FilePicker()
    file_picker_transcribe = ft.FilePicker()
    file_picker_completed = ft.FilePicker()
    file_picker_settings = ft.FilePicker()
    file_picker_log = ft.FilePicker()
    file_picker_gsv = ft.FilePicker()

    # ── 惰性初始化：facade / 页面实例 / core 重链延迟到后台线程首次触发 ──
    # 骨架首帧前零 core import（openai 等重库全部后移）。
    _init: dict = {"facade": None, "pages": None, "done": False}

    def ensure_initialized() -> None:
        """首次调用时导入 core 重链并实例化 facade + 4 个页面（后台线程执行）。

        幂等：done 置位后直接返回。页面 __init__ 不触发 update/服务注册；
        Settings 的 FilePicker 注册副作用由 load_content 在主线程处理。
        """
        if _init["done"]:
            return
        from app.torch_runtime import ensure_available

        ensure_available()  # 无 torch 运行时（冻结产物缺外挂包）时给出可读错误，由 load_content 显示
        from app.facade import AppFacade
        from core.task_que import TranslationTaskQueue, TranscriptionTaskQueue, GsvTaskQueue
        from core.service import LlamaService, APIService, GsvService
        from core.moss import MossService
        from ui.pages.translate import TranslatePage
        from ui.pages.transcribe import TranscribePage
        from ui.pages.completed import CompletedPage
        from ui.pages.settings import SettingsPage
        from ui.pages.log import LogPage
        from ui.pages.tts import TtsPage

        facade = AppFacade(
            backend_dict={
                "llama":      (LlamaService, TranslationTaskQueue),
                "api":        (APIService, TranslationTaskQueue),
                "moss":       (MossService, TranscriptionTaskQueue),
                "gsv":        (GsvService, GsvTaskQueue),
            },
            config_dict={
                "llama":      project_root / "configs/models/llama/default.json",
                "api":        project_root / "configs/models/API/default.json",
                "moss":       project_root / "configs/models/moss/default.json",
                "gsv":        project_root / "configs/models/gsv/default.json",
            },
        )
        pages = {
            0: TranscribePage(page, facade=facade, file_picker=file_picker_transcribe),
            1: TranslatePage(page, facade=facade, file_picker=file_picker_translate),
            2: TtsPage(page, facade=facade, file_picker=file_picker_gsv),
            3: CompletedPage(page, facade=facade, file_picker=file_picker_completed),
            4: SettingsPage(page, facade=facade, file_picker=file_picker_settings),
            5: LogPage(page, facade=facade, file_picker=file_picker_log),
        }
        _init["facade"] = facade
        _init["pages"] = pages
        _init["done"] = True

    def get_facade():
        """惰性获取 facade（未初始化时返回 None；供窗口关闭优雅停止服务）。"""
        return _init["facade"]

    current_index = 0

    # ── 骨架加载动画（品牌呼吸 + 占位脉冲 + 提示文案轮换）──
    # 协程以 _stop_event（threading.Event）停止：run_task 返回的 concurrent
    # Future 对已运行任务 cancel() 无效，事件标志才是可靠取消点；stop 后协程
    # 在下一轮循环退出（≤0.8s 残留无害）。仅骨架占位显示期间运行。
    brand_icon_ref = ft.Ref[ft.Container]()
    placeholder_root = ft.Ref[ft.Container]()
    placeholder_text = ft.Ref[ft.Text]()
    _PLACEHOLDER_HINTS = ("正在加载引擎…", "正在初始化…", "即将就绪…")
    _stop_event = threading.Event()
    _pulse_task = None

    def _make_placeholder(text: str = "正在加载…"):
        """构建当前骨架占位（绑定动画 refs，脉冲/文案轮换作用于最新占位）。"""
        return skeleton_placeholder(text, ref=placeholder_root, text_ref=placeholder_text)

    def _stop_loading_animation():
        nonlocal _pulse_task
        _stop_event.set()
        _pulse_task = None

    def _start_loading_animation():
        nonlocal _pulse_task
        if _pulse_task is None:
            _stop_event.clear()
            _pulse_task = page.run_task(_animate_loading)

    async def _animate_loading():
        """骨架阶段加载动画：品牌图标呼吸 + 占位整区脉冲 + 提示文案轮换。

        每次迭代对品牌/占位做一次 update（真实挂载后正常；未挂载或占位
        已被替换时抛 RuntimeError，防御捕获）。
        """
        low = False
        step = 0
        while not _stop_event.is_set():
            low = not low
            brand = brand_icon_ref.current
            if brand is not None:
                brand.scale = 1.06 if low else 1.0
                try:
                    brand.update()
                except RuntimeError:
                    pass
            ph = placeholder_root.current
            if ph is not None:
                tr = placeholder_text.current
                if tr is not None:
                    tr.value = _PLACEHOLDER_HINTS[step % len(_PLACEHOLDER_HINTS)]
                ph.opacity = 0.6 if low else 1.0
                try:
                    ph.update()
                except RuntimeError:
                    pass
            step += 1
            await asyncio.sleep(0.8)

    # ── 内容区域（初始为骨架占位；页面切换直接替换，无过渡动画）──
    content_area = ft.Container(
        content=_make_placeholder(),
        padding=ft.Padding.all(Layout.CONTENT_GAP),
        expand=True,
        bgcolor=Palette.BG,
    )

    def _set_content(tree: ft.Control):
        """切换内容区（直接替换 content，无切换动画）。"""
        content_area.content = tree
        try:
            content_area.update()
        except RuntimeError:
            pass

    # ── 页面控件树缓存（首次后台构建，切换时复用 + refresh，不重新 build）──
    views: dict = {}

    # ── 页面后台构建（两阶段渲染共用）──
    def build_content(idx: int) -> ft.Control:
        """同步构建第 idx 页完整控件树（在后台线程执行，纯对象创建）。"""
        return _init["pages"][idx].build()

    async def load_content(idx: int):
        """两阶段渲染：占位先行 → 后台初始化/构建 → 主线程替换 + 刷新。

        - 阶段 A（后台线程）：ensure_initialized 导入 core 重链并实例化
          facade + 页面（骨架首帧前零 core import）。
        - 阶段 B（主线程）：处理页面"连接/注册类副作用"（SettingsPage 的
          FilePicker 服务注册），避免跨线程 update。
        - 阶段 C（后台线程）：构建页面完整树（纯对象创建）。
        - 构建异常时内容区显示错误占位（不白屏、不崩溃）。
        """
        try:
            await asyncio.to_thread(ensure_initialized)
        except Exception as ex:
            log.record("error", f"初始化失败: {ex}")
            _stop_loading_animation()
            _set_content(error_placeholder(f"初始化失败: {ex}"))
            return
        ensure = getattr(_init["pages"][idx], "_ensure_file_picker_registered", None)
        if callable(ensure):
            ensure()
        try:
            tree = await asyncio.to_thread(build_content, idx)
        except Exception as ex:
            log.record("error", f"页面 {idx} 构建失败: {ex}")
            _stop_loading_animation()
            _set_content(error_placeholder(f"{type(ex).__name__}: {ex}"))
            return
        views[idx] = tree
        _stop_loading_animation()
        try:
            _set_content(tree)
        except Exception as ex:
            # patch 序列化/发送失败（如控件属性类型不被 msgpack 支持）→ 错误占位
            log.record("error", f"页面 {idx} 挂载失败: {ex}")
            try:
                _set_content(error_placeholder(f"{type(ex).__name__}: {ex}"))
            except Exception:
                pass
            return
        try:
            _init["pages"][idx].refresh()
        except RuntimeError:
            # flet 0.86.2：首帧 patch 刚发出，子控件 parent 链可能尚未建立，
            # refresh 的 update() 会抛 "Control must be added to the page first"。
            # 状态在后续推送/再次切换页面时自然刷新，此处静默（不刷日志）。
            pass
        except Exception as ex:
            log.record("warn", f"页面 {idx} refresh 失败: {ex}")

    # ── 切换页面函数 ──
    def switch_page(idx: int):
        nonlocal current_index
        if not _init["done"]:
            # 骨架阶段导航点击直接忽略：绝不能在此同步执行 ensure_initialized
            # （重库导入/页面构建会阻塞 UI 线程数秒）。导航在骨架结束后自然可用。
            return
        pages = _init["pages"]
        if idx not in pages:
            return
        if idx == current_index:
            try:
                pages[idx].refresh()
            except Exception:
                pass
            try:
                content_area.update()
            except RuntimeError:
                pass
        else:
            pages[current_index].save_ui_state()
            current_index = idx
            if idx in views:
                _stop_loading_animation()
                _set_content(views[idx])
                try:
                    pages[idx].refresh()
                except Exception:
                    pass
            else:
                # 未构建：先显示骨架占位，后台构建完成后替换（两阶段渲染）
                _start_loading_animation()
                _set_content(_make_placeholder())
                page.run_task(load_content, idx)

    # ══════════════════════════════════════════════════════════
    # 品牌区（top_row 左侧，250px 定宽）
    # ══════════════════════════════════════════════════════════

    # ── 品牌图标: Logo 图片（material/logo.png；骨架阶段呼吸动画经 _animate_loading）──
    brand_icon_box = ft.Container(
        content=ft.Image(
            src=str(project_root / "material/logo.png"),
            width=40, height=40,
            fit=ft.BoxFit.CONTAIN,  # flet 0.86.2：ImageFit 已移除，用 BoxFit
            filter_quality=ft.FilterQuality.HIGH,
        ),
        border_radius=Radius.SM,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,  # 圆角裁剪图片
        width=40, height=40,
        alignment=ft.Alignment.CENTER,
        shadow=_shadow("low"),
        ref=brand_icon_ref,
        animate_scale=ft.Animation(800, ft.AnimationCurve.EASE),
    )

    # ── 品牌文本列: 标题 + 副标题 ──
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

    # ── 品牌 Row: 图标 + 文本（logo 与右侧文字组件垂直居中对齐）──
    brand_row = ft.Row(
        [brand_icon_box, brand_text_column],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # ── 品牌区容器 (定宽 250px，固定高 68px，右+下边框) ──
    brand_container = ft.Container(
        content=brand_row,
        bgcolor=Palette.SURFACE,
        border=ft.Border.only(
            right=ft.BorderSide(1, Palette.BORDER),
            bottom=ft.BorderSide(1, Palette.BORDER),
        ),
        width=Layout.SIDEBAR_WIDTH,
        height=Layout.BRAND_HEIGHT,
        padding=ft.Padding.all(16),
    )

    # ══════════════════════════════════════════════════════════
    # 顶栏（右列上方，固定高 = 品牌区一半，宽占满右列剩余空间）
    # ══════════════════════════════════════════════════════════

    # ── 最大化/还原切换（最大化按钮 + 双击拖拽区共用）──
    def _toggle_maximize(e=None):
        w = page.window
        w.maximized = not w.maximized
        maximize_button.icon = ft.Icons.ASPECT_RATIO if w.maximized else ft.Icons.CROP_SQUARE
        maximize_button.tooltip = "还原" if w.maximized else "最大化"
        try:
            maximize_button.update()
        except RuntimeError:
            pass  # 未挂载 page 时跳过推送
        page.update()  # 推送 window.maximized 变化到客户端（与最小化按钮一致）

    # ── 可拖拽空白区（左），双击切换最大化/还原 ──
    drag_area = ft.WindowDragArea(
        content=ft.Container(expand=True),
        expand=True,
        on_double_tap=_toggle_maximize,
    )

    # ── 最大化按钮 ──
    maximize_button = ft.IconButton(
        icon=ft.Icons.ASPECT_RATIO if page.window.maximized else ft.Icons.CROP_SQUARE,
        icon_size=18,
        icon_color=Palette.SUBTEXT,
        tooltip="还原" if page.window.maximized else "最大化",
        on_click=_toggle_maximize,
        style=ft.ButtonStyle(padding=ft.Padding.all(6)),
    )

    # ── 最小化按钮 ──
    minimize_button = ft.IconButton(
        icon=ft.Icons.MINIMIZE,
        icon_size=18,
        icon_color=Palette.SUBTEXT,
        tooltip="最小化",
        on_click=lambda e: setattr(page.window, "minimized", True) or page.update(),
        style=ft.ButtonStyle(padding=ft.Padding.all(6)),
    )

    # ── 关闭按钮 ──
    # 走 window.close() 发起原生关闭请求：prevent_close=True 会拦截并经
    # window.on_event 以 WindowEventType.CLOSE 报告（flet 0.86.2 window.py
    # 文档保证），由 APP.py 统一执行 shutdown + destroy，避免绕过清理。
    async def close_window(e):
        await page.window.close()

    close_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        icon_size=18,
        icon_color=Palette.SUBTEXT,
        tooltip="关闭",
        on_click=close_window,
        style=ft.ButtonStyle(padding=ft.Padding.all(6)),
    )

    # ── 窗口控制按钮行 ──
    window_controls_row = ft.Row(
        [minimize_button, maximize_button, close_button],
        spacing=4,
    )

    # ── 顶栏 Row: 拖拽区 + 窗口控制 ──
    app_bar_row = ft.Row(
        [drag_area, window_controls_row],
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # ── 顶栏容器 (固定高 34px = 品牌区一半，宽度由右列 STRETCH 占满) ──
    app_bar = ft.Container(
        content=app_bar_row,
        height=Layout.APP_BAR_HEIGHT,
        padding=ft.Padding.symmetric(horizontal=16),
        bgcolor=Palette.SURFACE,
        border=ft.Border.only(bottom=ft.BorderSide(1, Palette.BORDER)),
    )

    # ══════════════════════════════════════════════════════════
    # 导航栏（左列，250px 定宽，expand 占品牌区以下剩余高度）
    # ══════════════════════════════════════════════════════════

    # ── NavigationRail ──
    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        min_extended_width=180,
        group_alignment=-0.9,
        extended=True,  # 展开填满 250px 容器，避免折叠态（仅图标）在右侧留大片空白
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.MIC,
                selected_icon=ft.Icons.MIC,
                label="转写",
                padding=ft.Padding.symmetric(vertical=8),
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.TRANSLATE,
                selected_icon=ft.Icons.TRANSLATE,
                label="翻译",
                padding=ft.Padding.symmetric(vertical=8),
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.RECORD_VOICE_OVER,
                selected_icon=ft.Icons.RECORD_VOICE_OVER,
                label="语音合成",
                padding=ft.Padding.symmetric(vertical=8),
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.CHECK_CIRCLE,
                selected_icon=ft.Icons.CHECK_CIRCLE,
                label="已完成",
                padding=ft.Padding.symmetric(vertical=8),
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.SETTINGS,
                selected_icon=ft.Icons.SETTINGS,
                label="设置",
                padding=ft.Padding.symmetric(vertical=8),
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.ARTICLE,
                selected_icon=ft.Icons.ARTICLE,
                label="日志",
                padding=ft.Padding.symmetric(vertical=8),
            ),
        ],
        on_change=lambda e: switch_page(e.control.selected_index),
        bgcolor=Palette.SURFACE,
    )

    # ── 导航容器 ──
    nav_rail_container = ft.Container(
        content=rail,
        padding=ft.Padding.only(left=16, top=12, right=16),
        bgcolor=Palette.SURFACE,
        border=ft.Border.only(right=ft.BorderSide(1, Palette.BORDER)),
        width=Layout.SIDEBAR_WIDTH,
        expand=True,  # 左列 Column 中占满品牌区以下的剩余高度
    )

    # ══════════════════════════════════════════════════════════
    # 根 Row — 左列（品牌区 + 导航栏）| 右列（顶栏 + 内容区）
    # ══════════════════════════════════════════════════════════

    left_column = ft.Column(
        [brand_container, nav_rail_container],
        spacing=0,
        # 固定侧栏宽度，不再 expand 抢占一半窗口宽（否则侧栏右侧留大片空白）
        width=Layout.SIDEBAR_WIDTH,
    )
    right_column = ft.Column(
        [app_bar, content_area],
        spacing=0, expand=True,
        # 子项横向拉伸：顶栏宽度占满剩余空间（内容区亦全宽）
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    root_row = ft.Row(
        [left_column, right_column],
        spacing=0, expand=True,
        # 纵向拉伸：固定宽度侧栏铺满窗口高度
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    # ── 根 Container — 最外层 ──
    root_container = ft.Container(
        content=root_row,
        expand=True,
        bgcolor=Palette.BG,
    )

    # ── 挂载到 page ──
    page.add(root_container)

    # ── 两阶段渲染：首帧推送外壳 + 骨架占位（APP.py 的 page.update()），
    #    随后在后台构建首页完整树并替换内容区 ──
    _start_loading_animation()
    page.run_task(load_content, 0)

    return (content_area, switch_page, file_picker_translate, file_picker_transcribe,
            file_picker_log, file_picker_gsv, get_facade)
