"""Main layout skeleton — left/right two-column structure (APP shell).

Left column: brand area (250px) + navigation rail (250px, expand to fill remaining height)
Right column: top bar (34px, window controls) + content area (expand)

Six-page navigation: Transcribe / Translate / TTS / Completed / Settings / Logs
The facade is instantiated at this layer and passed to each page builder.
Page instance pattern: each page holds state long-term; navigation calls build/refresh.
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
    """Unified shutdown flow: gracefully stop services, then destroy the window.

    Shared by two call sites: APP.py's ``window.on_event`` CLOSE branch, and the top-bar
    close button in this file (which goes through the same CLOSE branch via ``window.close()``).

    - *facade* accepts two forms: a service instance, or a lazy getter (build_layout now
      returns a get_facade function; it is None when uninitialized). None skips shutdown —
      the service never started, so no residual child processes.
    - Exceptions in ``facade.shutdown()`` are logged, not raised — otherwise, under
      ``prevent_close=True`` the only exit is ``destroy()`` and an exception would leave
      the window unclosable forever.
    - ``finally`` guarantees ``destroy()`` always runs.
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
    """Build the main layout; returns (content_area, switch_page_fn, fp_translate, fp_transcribe, fp_log, fp_gsv, get_facade).

    Facade / page instances / core chain are lazily initialized: the synchronous part of
    build_layout only builds the lightweight shell (brand/nav/top bar/skeleton placeholder)
    with zero core imports (heavy modules moved to the background thread and imported by
    ensure_initialized, triggered by load_content).

    - get_facade: a function that lazily returns the facade (None when uninitialized);
      used at window close to call shutdown() for a graceful stop (prevents llama-server
      child-process leftovers).
    - switch_page_fn(idx) switches the currently displayed page content.
    - fp_translate / fp_transcribe are FilePicker instances (created synchronously).
    """

    # ── Shared FilePickers (lightweight, created synchronously; registered by APP.py before the first frame) ──
    file_picker_translate = ft.FilePicker()
    file_picker_transcribe = ft.FilePicker()
    file_picker_completed = ft.FilePicker()
    file_picker_settings = ft.FilePicker()
    file_picker_log = ft.FilePicker()
    file_picker_gsv = ft.FilePicker()

    # ── Lazy init: facade / page instances / core chain deferred to the first background-thread trigger ──
    # Zero core imports before the skeleton's first frame (heavy libs like openai all moved later).
    _init: dict = {"facade": None, "pages": None, "done": False}

    def ensure_initialized() -> None:
        """On first call, import the core chain and instantiate the facade + pages (runs on a background thread).

        Idempotent: returns immediately once done is set. Page __init__ does not trigger
        update/service registration; the Settings FilePicker registration side effect is
        handled by load_content on the main thread.
        """
        if _init["done"]:
            return
        from app.torch_runtime import ensure_available

        ensure_available()  # gives a readable error when the torch runtime is missing (frozen build lacks the external package); displayed by load_content
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
        """Lazily get the facade (None when uninitialized; used for a graceful stop at window close)."""
        return _init["facade"]

    current_index = 0

    # ── Skeleton loading animation (brand breathing + placeholder pulse + hint text rotation) ──
    # The coroutine is stopped via _stop_event (threading.Event): the concurrent Future
    # returned by run_task cannot cancel() an already-running task; the event flag is the
    # reliable cancellation point. After stop the coroutine exits on the next loop
    # iteration (≤0.8s residue is harmless). Runs only while the skeleton placeholder is shown.
    brand_icon_ref = ft.Ref[ft.Container]()
    placeholder_root = ft.Ref[ft.Container]()
    placeholder_text = ft.Ref[ft.Text]()
    _PLACEHOLDER_HINTS = ("正在加载引擎…", "正在初始化…", "即将就绪…")
    _stop_event = threading.Event()
    _pulse_task = None

    def _make_placeholder(text: str = "正在加载…"):
        """Build the current skeleton placeholder (binds animation refs; pulse/text rotation acts on the latest placeholder)."""
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
        """Skeleton-phase loading animation: brand icon breathing + whole-placeholder pulse + hint text rotation.

        Each iteration updates the brand/placeholder once (fine once really mounted; a
        RuntimeError is raised when unmounted or the placeholder was replaced — caught defensively).
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

    # ── Content area (initially a skeleton placeholder; page switches replace it directly, no transition) ──
    content_area = ft.Container(
        content=_make_placeholder(),
        padding=ft.Padding.all(Layout.CONTENT_GAP),
        expand=True,
        bgcolor=Palette.BG,
    )

    def _set_content(tree: ft.Control):
        """Switch the content area (directly replaces content; no transition animation)."""
        content_area.content = tree
        try:
            content_area.update()
        except RuntimeError:
            pass

    # ── Page control-tree cache (first built in the background; reused + refresh on switch, no rebuild) ──
    views: dict = {}

    # ── Background page build (shared by the two-phase render) ──
    def build_content(idx: int) -> ft.Control:
        """Synchronously build the full control tree for page idx (runs on a background thread; pure object creation)."""
        return _init["pages"][idx].build()

    async def load_content(idx: int):
        """Two-phase rendering: placeholder first → background init/build → main-thread replace + refresh.

        - Phase A (background thread): ensure_initialized imports the core chain and
          instantiates the facade + pages (zero core imports before the skeleton's first frame).
        - Phase B (main thread): handles page "connect/register side effects" (SettingsPage's
          FilePicker service registration), avoiding cross-thread update.
        - Phase C (background thread): builds the page's full tree (pure object creation).
        - On a build error the content area shows an error placeholder (no blank screen, no crash).
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
            # patch serialize/send failure (e.g. control attribute type not supported by msgpack) → error placeholder
            log.record("error", f"页面 {idx} 挂载失败: {ex}")
            try:
                _set_content(error_placeholder(f"{type(ex).__name__}: {ex}"))
            except Exception:
                pass
            return
        try:
            _init["pages"][idx].refresh()
        except RuntimeError:
            # flet 0.86.2: the first-frame patch was just sent; the child's parent chain
            # may not be established yet, so refresh's update() raises
            # "Control must be added to the page first". State refreshes naturally on
            # later pushes or page switches; swallow silently here (no log).
            pass
        except Exception as ex:
            log.record("warn", f"页面 {idx} refresh 失败: {ex}")

    # ── Page switch function ──
    def switch_page(idx: int):
        nonlocal current_index
        if not _init["done"]:
            # Ignore nav clicks during the skeleton phase: never run ensure_initialized
            # synchronously here (heavy imports/page builds would block the UI thread for
            # seconds). Navigation becomes available naturally once the skeleton finishes.
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
                # Not yet built: show the skeleton placeholder first, then replace after the background build (two-phase render)
                _start_loading_animation()
                _set_content(_make_placeholder())
                page.run_task(load_content, idx)

    # ── Brand area (top_row left side, fixed 250px width) ──

    # ── Brand icon: logo image (material/logo.png; breathing animation during skeleton via _animate_loading) ──
    brand_icon_box = ft.Container(
        content=ft.Image(
            src=str(project_root / "material/logo.png"),
            width=40, height=40,
            fit=ft.BoxFit.CONTAIN,  # flet 0.86.2: ImageFit removed; use BoxFit
            filter_quality=ft.FilterQuality.HIGH,
        ),
        border_radius=Radius.SM,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,  # clip image to rounded corners
        width=40, height=40,
        alignment=ft.Alignment.CENTER,
        shadow=_shadow("low"),
        ref=brand_icon_ref,
        animate_scale=ft.Animation(800, ft.AnimationCurve.EASE),
    )

    # ── Brand text column: title + subtitle ──
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

    # ── Brand Row: icon + text (logo vertically centered with the right text column) ──
    brand_row = ft.Row(
        [brand_icon_box, brand_text_column],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # ── Brand container (fixed width 250px, fixed height 68px, right + bottom border) ──
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

    # ── Top bar (above the right column; fixed height = half the brand area, width fills the remaining right-column space) ──

    # ── Maximize/restore toggle (shared by the maximize button and the double-click drag area) ──
    def _toggle_maximize(e=None):
        w = page.window
        w.maximized = not w.maximized
        maximize_button.icon = ft.Icons.ASPECT_RATIO if w.maximized else ft.Icons.CROP_SQUARE
        maximize_button.tooltip = "还原" if w.maximized else "最大化"
        try:
            maximize_button.update()
        except RuntimeError:
            pass  # skip push when not mounted to page
        page.update()  # push window.maximized changes to the client (same as the minimize button)

    # ── Draggable blank area (left); double-click toggles maximize/restore ──
    drag_area = ft.WindowDragArea(
        content=ft.Container(expand=True),
        expand=True,
        on_double_tap=_toggle_maximize,
    )

    # ── Maximize button ──
    maximize_button = ft.IconButton(
        icon=ft.Icons.ASPECT_RATIO if page.window.maximized else ft.Icons.CROP_SQUARE,
        icon_size=18,
        icon_color=Palette.SUBTEXT,
        tooltip="还原" if page.window.maximized else "最大化",
        on_click=_toggle_maximize,
        style=ft.ButtonStyle(padding=ft.Padding.all(6)),
    )

    # ── Minimize button ──
    minimize_button = ft.IconButton(
        icon=ft.Icons.MINIMIZE,
        icon_size=18,
        icon_color=Palette.SUBTEXT,
        tooltip="最小化",
        on_click=lambda e: setattr(page.window, "minimized", True) or page.update(),
        style=ft.ButtonStyle(padding=ft.Padding.all(6)),
    )

    # ── Close button ──
    # Uses window.close() to issue a native close request: prevent_close=True intercepts it
    # and reports it via window.on_event as WindowEventType.CLOSE (guaranteed by the flet
    # 0.86.2 window.py docs); APP.py then runs shutdown + destroy uniformly, avoiding cleanup bypass.
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

    # ── Window control button row ──
    window_controls_row = ft.Row(
        [minimize_button, maximize_button, close_button],
        spacing=4,
    )

    # ── Top-bar Row: drag area + window controls ──
    app_bar_row = ft.Row(
        [drag_area, window_controls_row],
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # ── Top-bar container (fixed height 34px = half the brand area; width filled by the right column's STRETCH) ──
    app_bar = ft.Container(
        content=app_bar_row,
        height=Layout.APP_BAR_HEIGHT,
        padding=ft.Padding.symmetric(horizontal=16),
        bgcolor=Palette.SURFACE,
        border=ft.Border.only(bottom=ft.BorderSide(1, Palette.BORDER)),
    )

    # ── Navigation rail (left column, fixed 250px, expand fills remaining height below the brand area) ──

    # ── NavigationRail ──
    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        min_extended_width=180,
        group_alignment=-0.9,
        extended=True,  # extended to fill the 250px container, avoiding a big blank area on the right in collapsed (icon-only) mode
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

    # ── Navigation container ──
    nav_rail_container = ft.Container(
        content=rail,
        padding=ft.Padding.only(left=16, top=12, right=16),
        bgcolor=Palette.SURFACE,
        border=ft.Border.only(right=ft.BorderSide(1, Palette.BORDER)),
        width=Layout.SIDEBAR_WIDTH,
        expand=True,  # fills the remaining height below the brand area in the left Column
    )

    # ── Root Row — left column (brand + nav) | right column (top bar + content) ──

    left_column = ft.Column(
        [brand_container, nav_rail_container],
        spacing=0,
        # Fixed sidebar width, no longer expand (which would grab half the window and
        # leave a big blank area to the right of the sidebar)
        width=Layout.SIDEBAR_WIDTH,
    )
    right_column = ft.Column(
        [app_bar, content_area],
        spacing=0, expand=True,
        # Stretch children horizontally: the top bar fills the remaining space (content area full width too)
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    root_row = ft.Row(
        [left_column, right_column],
        spacing=0, expand=True,
        # Stretch vertically: the fixed-width sidebar fills the window height
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    # ── Root Container — outermost ──
    root_container = ft.Container(
        content=root_row,
        expand=True,
        bgcolor=Palette.BG,
    )

    # ── Mount to page ──
    page.add(root_container)

    # ── Two-phase render: first frame pushes the shell + skeleton placeholder (APP.py's page.update()),
    #    then the home page's full tree is built in the background and replaces the content area ──
    _start_loading_animation()
    page.run_task(load_content, 0)

    return (content_area, switch_page, file_picker_translate, file_picker_transcribe,
            file_picker_log, file_picker_gsv, get_facade)
