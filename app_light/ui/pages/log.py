"""Logs page — shows the running log (AppLog ring buffer) + manual export.

Layout (per requirements):
1. Top bar: ``toolbar_panel_header("Logs", [export logs button])``
   — the "Logs" title is left-aligned, the export button right-aligned.
2. Second row: a ``ft.ListView(auto_scroll=True)`` inside an ``expand=True`` card
   shows the log line by line — in a normal window it fills the content area's
   remaining height (bounded region) and grows with the window when maximized;
   when lines overflow, it scrolls internally and auto-scrolls to the latest.

Live monitoring: ``build()`` starts a 1s polling loop via ``page.run_task`` and uses
``AppLog.snapshot_since`` to append new lines incrementally (avoiding a full rebuild);
``save_ui_state()`` stops polling when leaving the page, and ``refresh()`` restarts
and syncs when returning.
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
    """Compatibility wrapper — create a LogPage instance and build the UI."""
    return LogPage(page, facade, file_picker).build()


class LogPage:
    """Logs page instance — holds state long-term so it survives navigation switches."""

    POLL_INTERVAL = 1.0  # seconds

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        self.file_picker = file_picker

        # ── State ──
        self._polling = False
        self._poll_task = None
        self._seen_version = log.version()  # log version already displayed
        self.list_view: Optional[ft.ListView] = None

    # ── Public interface (page instance pattern: build/save_ui_state/refresh/register_callbacks) ──

    def build(self) -> ft.Control:
        """Build/rebuild the logs page UI."""
        self._seen_version = log.version()
        header = toolbar_panel_header(
            "日志",
            actions=[self._build_export_button()],
        )
        self.list_view = ft.ListView(
            expand=True,
            auto_scroll=True,          # auto-scroll to the latest log line
            # Jump straight to the bottom on page switch / new log arrival, without a
            # scroll animation (avoids a "scrolling down" effect).
            # flet 0.86.2: auto_scroll_animation is an AnimationValue (True = 1s animation /
            # int ms / Animation); False is invalid and crashes engine-side parsing,
            # failing to render the whole subtree (all grey); 0 = instant jump (duration 0 → jumpTo).
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
            expand=True,               # normal window = content area remaining height; grows with the window when maximized
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
        """Refresh when navigating back to this page: sync new lines + restart polling (if stopped)."""
        self._sync_since()
        self._start_polling()

    def save_ui_state(self) -> None:
        """Leaving the page: stop polling (no control state to save; the log is read live)."""
        self._stop_polling()

    def register_callbacks(self) -> None:
        """The logs page has no facade callbacks (reads the AppLog buffer directly, no event push)."""
        pass

    # ── Internal: log row rendering / incremental sync / polling ──

    _LEVEL_RE = re.compile(r"^\[[^\]]*\] \[(error|warn|info)\]")

    @staticmethod
    def _row(line: str) -> ft.Text:
        """Single log line: level-colored (error red / warn orange / info default) with a monospace font.

        Parses the leading ``[timestamp] [level]``; also handles the old ``[info] ...`` format (legacy tests).
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
            # wrap overly long log lines to the area width (no truncation, no ellipsis)
            no_wrap=False,
        )

    def _sync_since(self) -> None:
        """Incrementally sync new log lines (idempotent; shared by polling and refresh)."""
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
        """If error/crash dump files exist, show a hint below the top bar (a de-emphasized legacy entry point)."""
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
        """While the page is alive, refresh incrementally every 1s; auto-scroll-to-bottom is handled by ListView.auto_scroll."""
        while self._polling:
            await asyncio.sleep(self.POLL_INTERVAL)
            try:
                self._sync_since()
            except Exception:
                pass

    # ── Export ──

    def _build_export_button(self) -> ft.OutlinedButton:
        return bordered_button(
            "导出日志", ft.Icons.FILE_DOWNLOAD,
            on_click=self._export_logs,
            padding=ft.Padding.symmetric(horizontal=12, vertical=6),
        )

    async def _export_logs(self, e) -> None:
        """FilePicker save_file picks a path → AppLog.export (flet 0.86.2 returns the path directly)."""
        if self.file_picker is None:
            log.record("warn", "导出日志失败: 未注册文件选择器")
            return
        path = await self.file_picker.save_file(
            dialog_title="导出日志",
            file_name="app.log",
            file_type=ft.FilePickerFileType.ANY,
        )
        if not path:
            return  # user cancelled
        try:
            log.export(path)
        except Exception as exc:
            log.record("error", f"导出日志失败: {exc}")
            return
        log.record("info", f"导出日志 → {path}")
