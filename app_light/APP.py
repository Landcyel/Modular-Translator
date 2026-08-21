"""
Modular Translator — 日文→中文翻译工作台
Flet 桌面应用入口

运行：python APP.py
"""

# ── windowed 模式兜底 ──
# PyInstaller --noconsole 打包后 sys.stdout/sys.stderr 为 None，任何 print()
# 都会抛 AttributeError 导致崩溃。此处重定向到 devnull；开发模式（控制台
# 运行 python APP.py）下二者非 None，本块不生效，不影响正常输出。
import os
import sys
import time

# 进程内启动起点（不含 exe/bootloader/PYTHON 运行时之前的时间）
T0 = time.perf_counter()

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# ── 项目刚需 FFmpeg：最早设置 PATH，让全部 ffmpeg/ffprobe 子进程调用
#    （含 vendored GPT-SoVITS / UVR5 的裸名调用）先命中 dependencies/FFmpeg ──
from app.ffmpeg import ensure_ffmpeg_on_path  # noqa: E402

ensure_ffmpeg_on_path()

from app import torch_runtime  # noqa: F401  # 可插拔 torch 运行时（CPU 基线 / dependencies 外挂 CUDA），必须先于一切 torch/core import
from app.log import log  # noqa: E402
from app.paths import project_root  # noqa: E402

log.record("info", f"[boot] 进程内启动点 + ffmpeg PATH + torch_runtime setup 完成（+{time.perf_counter() - T0:.1f}s）")

import flet as ft

log.record("info", f"[boot] flet 导入完成（+{time.perf_counter() - T0:.1f}s）")

from ui.theme import Palette
from ui.layout import build_layout, shutdown_and_destroy

log.record("info", f"[boot] 进入 ft.run（+{time.perf_counter() - T0:.1f}s）")


def main(page: ft.Page):
    # ── 页面基础设置 ──
    page.title = "Modular Translator"
    page.padding = 0
    page.spacing = 0
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.START

    # ── 任务栏/窗口图标（flet 0.86.2 Window.icon 仅支持 .ico）──
    # 产物含构建期显式生成的 material/logo.ico；开发模式源目录无 ico 时，
    # 运行时用 PIL 从 logo.png 生成到 temp/（不持久保存 ico）
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
                _icon = None  # 生成失败 → 保持默认图标
    if _icon is not None and _icon.is_file():
        page.window.icon = str(_icon)

    page.window.title_bar_hidden = True
    page.window.title_bar_buttons_hidden = True

    # ── 窗口大小固定、禁止手动拉伸；允许最大化/恢复 ──
    # 固定为当前窗口尺寸（flet 0.86.2 Window API：width/height/resizable/min_width）
    # resizable=False 防拖拽拉伸；不设 max_width/max_height，最大化时窗口可扩展，
    # 面板高度随内容区 flex 动态变化，恢复后回到固定尺寸。
    win_w = page.window.width or 1280
    win_h = page.window.height or 800
    page.window.width = win_w
    page.window.height = win_h
    page.window.min_width = win_w
    page.window.min_height = win_h
    page.window.resizable = False

    # ── 自定义主题 ──
    page.theme = ft.Theme(
        color_scheme_seed=Palette.PRIMARY,
        font_family="Microsoft YaHei",
    )
    page.dark_theme = ft.Theme(
        color_scheme_seed=Palette.PRIMARY,
        font_family="Microsoft YaHei",
    )

    # ── 窗口关闭处理 ──
    # flet 0.86.2：窗口事件监听器挂在 page.window.on_event（Page 无 on_window_event）。
    # prevent_close=True 拦截原生关闭请求（任务栏/Alt+F4/标题栏 X），
    # 收到 WindowEventType.CLOSE 后由我们决定是否真正关闭。
    page.window.prevent_close = True
    closing = False
    # 兜底绑定：build_layout 若抛异常（get_facade 未赋值），关闭事件闭包
    # 仍可引用（shutdown_and_destroy 兼容 None），避免二次 NameError 崩溃。
    get_facade = None

    async def on_window_close(e):
        # 防止 shutdown/destroy 期间重复收到 CLOSE 而二次销毁
        nonlocal closing
        if e.type == ft.WindowEventType.CLOSE and not closing:
            closing = True
            await shutdown_and_destroy(get_facade, page)

    page.window.on_event = on_window_close

    # ── 构建布局（facade 在 layout 内部惰性创建，此处解包供关闭时优雅停止服务）──
    # 两阶段渲染：build_layout 同步段只构建轻量外壳（品牌+导航+顶栏+内容区
    # 骨架占位）并调度 run_task 后台初始化（core 重链导入+页面构建）；下方
    # page.update() 推送首帧（骨架，毫秒级出界面），后台完成后替换内容区。
    # get_facade 为惰性 getter（未初始化返回 None），关闭时经 shutdown_and_destroy 兼容。
    content_area, switch_page, fp_translate, fp_transcribe, fp_log, fp_gsv, get_facade = build_layout(page)
    log.record("info", f"[boot] 首帧骨架构建完成（+{time.perf_counter() - T0:.1f}s）")

    # ── 页面更新 ──
    page.update()

    # ── 注册 FilePicker ──
    page._services.register_service(fp_translate)
    page._services.register_service(fp_transcribe)
    page._services.register_service(fp_log)
    page._services.register_service(fp_gsv)


if __name__ == "__main__":
    ft.run(main)
