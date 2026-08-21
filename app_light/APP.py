"""
Modular Translator — Japanese-to-Chinese translation workbench.
Flet desktop application entry point.

Run: python APP.py
"""

# ── Windowed-mode fallback ──
# After PyInstaller --noconsole packaging, sys.stdout/sys.stderr are None and any
# print() would raise AttributeError and crash. Redirect to devnull here; in dev
# mode (console run of python APP.py) both are non-None, so this block is a no-op.
import os
import sys
import time

# In-process boot start point (excludes time before the exe/bootloader/Python runtime)
T0 = time.perf_counter()

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# ── Project-critical FFmpeg: set PATH first ──
# so every ffmpeg/ffprobe subprocess call (including bare-name calls from vendored
# GPT-SoVITS / UVR5) hits dependencies/FFmpeg.
from app.ffmpeg import ensure_ffmpeg_on_path  # noqa: E402

ensure_ffmpeg_on_path()

from app import torch_runtime  # noqa: F401  # pluggable torch runtime (CPU baseline / CUDA in dependencies), must precede all torch/core imports
from app.log import log  # noqa: E402
from app.paths import project_root  # noqa: E402

log.record("info", f"[boot] 进程内启动点 + ffmpeg PATH + torch_runtime setup 完成（+{time.perf_counter() - T0:.1f}s）")

import flet as ft

log.record("info", f"[boot] flet 导入完成（+{time.perf_counter() - T0:.1f}s）")

from ui.theme import Palette
from ui.layout import build_layout, shutdown_and_destroy

log.record("info", f"[boot] 进入 ft.run（+{time.perf_counter() - T0:.1f}s）")


def main(page: ft.Page):
    # ── Page base settings ──
    page.title = "Modular Translator"
    page.padding = 0
    page.spacing = 0
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.START

    # ── Taskbar/window icon (flet 0.86.2 Window.icon supports .ico only) ──
    # The build output contains an explicitly generated material/logo.ico; in dev
    # mode, when the source tree has no ico, generate one at runtime with PIL from
    # logo.png into temp/ (the ico is not persisted).
    _icon = project_root / "material" / "logo.ico"
    if not _icon.is_file():
        _png = project_root / "material" / "logo.png"
        if _png.is_file():
            try:
                from PIL import Image
                _icon = project_root / "temp" / "logo.ico"
                _icon.parent.mkdir(parents=True, exist_ok=True)
                Image.open(str(_png)).save(
                    str(_icon), format="ICO",
                    sizes=[(16, 16), (32, 32), (48, 48), (64, 64),
                           (128, 128), (256, 256)],
                )
            except Exception:
                _icon = None  # generation failed → keep default icon
    if _icon is not None and _icon.is_file():
        page.window.icon = str(_icon)

    page.window.title_bar_hidden = True
    page.window.title_bar_buttons_hidden = True

    # ── Fixed window size, manual resize disabled; maximize/restore allowed ──
    # Fix to the current window size (flet 0.86.2 Window API: width/height/resizable/min_width)
    # resizable=False prevents drag-resize; max_width/max_height unset so the window can
    # expand when maximized, with panel heights following the content area flex, and
    # returns to the fixed size on restore.
    win_w = page.window.width or 1280
    win_h = page.window.height or 800
    page.window.width = win_w
    page.window.height = win_h
    page.window.min_width = win_w
    page.window.min_height = win_h
    page.window.resizable = False

    # ── Custom theme ──
    page.theme = ft.Theme(
        color_scheme_seed=Palette.PRIMARY,
        font_family="Microsoft YaHei",
    )
    page.dark_theme = ft.Theme(
        color_scheme_seed=Palette.PRIMARY,
        font_family="Microsoft YaHei",
    )

    # ── Window close handling ──
    # flet 0.86.2: window event listener is attached to page.window.on_event
    # (Page has no on_window_event). prevent_close=True intercepts native close
    # requests (taskbar/Alt+F4/titlebar X); on WindowEventType.CLOSE we decide
    # whether to actually close.
    page.window.prevent_close = True
    closing = False
    # Fallback binding: if build_layout raises (get_facade never assigned), the
    # close event closure can still reference it (shutdown_and_destroy accepts
    # None), avoiding a secondary NameError crash.
    get_facade = None

    async def on_window_close(e):
        # Prevent a second CLOSE event during shutdown/destroy from destroying twice
        nonlocal closing
        if e.type == ft.WindowEventType.CLOSE and not closing:
            closing = True
            await shutdown_and_destroy(get_facade, page)

    page.window.on_event = on_window_close

    # ── Build layout (facade created lazily inside layout; unpacked here for graceful shutdown) ──
    # Two-phase rendering: the synchronous part of build_layout only builds a light
    # shell (brand + nav + top bar + content-area skeleton placeholders) and schedules
    # a run_task background init (core re-import + page build); the page.update() below
    # pushes the first frame (skeleton, UI appears in milliseconds), then the background
    # work replaces the content area. get_facade is a lazy getter (None until initialized),
    # compatible with shutdown_and_destroy on close.
    content_area, switch_page, fp_translate, fp_transcribe, fp_log, fp_gsv, get_facade = build_layout(page)
    log.record("info", f"[boot] 首帧骨架构建完成（+{time.perf_counter() - T0:.1f}s）")

    # ── Page update ──
    page.update()

    # ── Register FilePickers ──
    page._services.register_service(fp_translate)
    page._services.register_service(fp_transcribe)
    page._services.register_service(fp_log)
    page._services.register_service(fp_gsv)


if __name__ == "__main__":
    ft.run(main)
