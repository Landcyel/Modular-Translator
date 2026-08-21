"""日志页面 — 显示运行日志（AppLog 环形缓冲）+ 手动导出。

布局（按需求）：
1. 顶栏：``toolbar_panel_header("日志", [导出日志按钮])``
   —— 标题"日志"左对齐、导出日志按钮右对齐。
2. 第二行：``expand=True`` 的卡片容器内 ``ft.ListView(auto_scroll=True)``
   逐行显示日志 —— 正常窗口占满内容区剩余高度（定高区域），最大化后随窗口
   增长；内容超出行时内部分行滚动、自动滚到最新。

实时监控：``build()`` 时经 ``page.run_task`` 启动 1s 轮询，用
``AppLog.snapshot_since`` 增量追加新行（避免全量重建）；``save_ui_state()``
离开页面时停止轮询，``refresh()`` 回来时重启并同步。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import List, Optional

import flet as ft

from app.log import log
from app.paths import project_root
from ui.components import bordered_button, toolbar_panel_header
from ui.theme import Layout, Palette, Radius, Typography


def build_log(page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
    """兼容包装 — 创建 LogPage 实例并构建 UI。"""
    return LogPage(page, facade, file_picker).build()


class LogPage:
    """日志页实例 — 长期持有状态，避免导航切换时丢失。"""

    POLL_INTERVAL = 1.0  # 秒

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── 状态 ──
        self._polling = False
        self._poll_task = None
        self._seen_version = log.version()  # 已显示到的日志版本
        self.list_view: Optional[ft.ListView] = None

    # ════════════════════════════════════════════════
    #  公开接口（页面实例模式：build/save_ui_state/refresh/register_callbacks）
    # ════════════════════════════════════════════════

    def build(self) -> ft.Control:
        """构建/重建日志页 UI。"""
        self._seen_version = log.version()
        header = toolbar_panel_header(
            "日志",
            actions=[self._build_export_button()],
        )
        self.list_view = ft.ListView(
            expand=True,
            auto_scroll=True,          # 自动滚到最新日志
            # 切页/新日志到达直接定位到底部，不做滚动动画（避免"滚动向下"效果）。
            # flet 0.86.2：auto_scroll_animation 为 AnimationValue（True=1s 动画 /
            # int 毫秒 / Animation），False 是非法值会使引擎端解析崩溃、整棵子树
            # 渲染失败（全部灰色）；0 = 瞬时跳转（duration 0 → jumpTo）。
            auto_scroll_animation=0,
            spacing=2,
            padding=ft.Padding.symmetric(vertical=4, horizontal=8),
        )
        for line in log.lines():
            self.list_view.controls.append(self._row(line))
        card = ft.Container(
            content=self.list_view,
            bgcolor=Palette.SURFACE,
            border=ft.Border.all(1, Palette.BORDER),
            border_radius=Radius.LG,
            expand=True,               # 正常窗口=内容区剩余高度；最大化后随窗口增长
        )
        children: List[ft.Control] = [header, ft.Container(height=Layout.CONTENT_GAP)]
        hint = self._build_dump_hint()
        if hint is not None:
            children.append(hint)
            children.append(ft.Container(height=Layout.CONTENT_GAP))
        children.append(card)
        self._start_polling()
        return ft.Column(
            children,
            spacing=0,
            expand=True,
        )

    def refresh(self) -> None:
        """导航回本页时刷新：同步新行 + 重启轮询（若已停止）。"""
        self._sync_since()
        self._start_polling()

    def save_ui_state(self) -> None:
        """离开页面：停止轮询（控件状态无需保存，日志实时读取）。"""
        self._stop_polling()

    def register_callbacks(self) -> None:
        """日志页无 facade 回调（直接读 AppLog 缓冲，不依赖事件推送）。"""
        pass

    # ════════════════════════════════════════════════
    #  内部：日志行渲染 / 增量同步 / 轮询
    # ════════════════════════════════════════════════

    _LEVEL_RE = re.compile(r"^\[[^\]]*\] \[(error|warn|info)\]")

    @staticmethod
    def _row(line: str) -> ft.Text:
        """单行日志：级别着色（error 红 / warn 橙 / info 默认），等宽字体。

        解析行首 ``[时间戳] [级别]``；兼容旧格式 ``[info] ...``（历史测试）。
        """
        m = LogPage._LEVEL_RE.match(line)
        level = m.group(1) if m else None
        if level is None:
            if line.startswith("[error]"):
                level = "error"
            elif line.startswith("[warn]"):
                level = "warn"
            elif line.startswith("[info]"):
                level = "info"
        color = Palette.TEXT
        if level == "error":
            color = Palette.ERROR
        elif level == "warn":
            color = Palette.WARNING
        return ft.Text(
            line,
            size=Typography.BODY_SM,
            color=color,
            font_family="Consolas",
            # 超长日志按区域宽度换行显示（不截断、不省略号）
            no_wrap=False,
        )

    def _sync_since(self) -> None:
        """增量同步新日志行（幂等；轮询与 refresh 共用）。"""
        if self.list_view is None:
            return
        v, new, cleared = log.snapshot_since(self._seen_version)
        if not new and not cleared:
            return
        self._seen_version = v
        if cleared:
            self.list_view.controls.clear()
        for line in new:
            self.list_view.controls.append(self._row(line))
        try:
            self.page.update()
        except RuntimeError:
            pass

    @staticmethod
    def _build_dump_hint() -> Optional[ft.Text]:
        """若存在 error/crash 落盘文件，在顶栏下方给出提示（弱化版历史入口）。"""
        log_dir = Path(project_root) / "logs"
        try:
            files = sorted(
                list(log_dir.glob("app-error-*.log")) + list(log_dir.glob("app-crash-*.log")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None
        if not files:
            return None
        latest = files[0]
        return ft.Text(
            f"检测到错误落盘文件：{latest.name}（logs/ 目录，可手动打开或导出）",
            size=Typography.CAPTION,
            color=Palette.WARNING,
        )

    def _start_polling(self) -> None:
        if self._polling:
            return
        self._polling = True
        try:
            self._poll_task = self.page.run_task(self._poll_loop)
        except Exception:
            self._polling = False

    def _stop_polling(self) -> None:
        self._polling = False
        self._poll_task = None

    async def _poll_loop(self) -> None:
        """页面存活期间每 1s 增量刷新；自动滚底由 ListView.auto_scroll 保证。"""
        while self._polling:
            await asyncio.sleep(self.POLL_INTERVAL)
            try:
                self._sync_since()
            except Exception:
                pass

    # ════════════════════════════════════════════════
    #  导出
    # ════════════════════════════════════════════════

    def _build_export_button(self) -> ft.OutlinedButton:
        return bordered_button(
            "导出日志", ft.Icons.FILE_DOWNLOAD,
            on_click=self._export_logs,
            padding=ft.Padding.symmetric(horizontal=12, vertical=6),
        )

    async def _export_logs(self, e) -> None:
        """FilePicker save_file 选路径 → AppLog.export（flet 0.86.2 直接返回路径）。"""
        if self.file_picker is None:
            log.record("warn", "导出日志失败: 未注册文件选择器")
            return
        path = await self.file_picker.save_file(
            dialog_title="导出日志",
            file_name="app.log",
            file_type=ft.FilePickerFileType.ANY,
        )
        if not path:
            return  # 用户取消
        try:
            log.export(path)
        except Exception as exc:
            log.record("error", f"导出日志失败: {exc}")
            return
        log.record("info", f"导出日志 → {path}")
