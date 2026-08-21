"""可复用的任务列表面板 — 拆分版。

组件：
- task_queue_panel()  等待+运行中任务队列
- task_card()         单任务卡片
- _task_status_icon() 状态图标
- _task_progress_bar() 进度条
"""

from __future__ import annotations

import flet as ft
from ui.theme import Layout, Palette, Radius, Anim, Typography
from ui.components import _icon, _text, divider, _shadow, toolbar_panel_header


_STATUS_MAP = {
    "pending":   (ft.Icons.HOURGLASS_EMPTY, "#94A3B8"),
    "running":   (ft.Icons.PLAY_ARROW,       "#3B82F6"),
    "completed": (ft.Icons.CHECK,            "#10B981"),
    "failed":    (ft.Icons.ERROR,            "#EF4444"),
    "cancelled": (ft.Icons.CANCEL,           "#F59E0B"),
}

_TYPE_ICON = {
    "translate": ft.Icons.TRANSLATE,
    "transcribe": ft.Icons.MIC,
    "gsv": ft.Icons.RECORD_VOICE_OVER,
    "moss": ft.Icons.GROUPS,
}
_TYPE_COLOR = {
    "translate": Palette.PRIMARY,
    "transcribe": Palette.INFO,
    "gsv": Palette.PRIMARY_DARK,
    "moss": Palette.INFO,
}


def _task_status_icon(status: str) -> ft.Container:
    """任务状态图标（无背景）。"""
    icon, color = _STATUS_MAP.get(status, (ft.Icons.HELP, "#94A3B8"))
    return ft.Container(
        content=_icon(icon, 14, color),
        border_radius=Radius.SM,
        padding=ft.Padding.all(5),
        tooltip=status,
    )


def _task_progress_bar(progress: float) -> ft.ProgressBar | ft.Container:
    """运行中任务显示 0-1 进度条（圆角 + 轨道色 + 末端圆点收尾）。"""
    if 0.0 < progress <= 1.0:
        return ft.ProgressBar(
            value=progress, bar_height=4,
            color=Palette.PRIMARY,
            bgcolor=Palette.BORDER,
            border_radius=4,
            stop_indicator_color=Palette.PRIMARY,
            stop_indicator_radius=3,
        )
    return ft.Container(height=0)


def _lrc_time(seconds: float) -> str:
    """LRC 风格时间戳：``[MM:SS.cs]``（方案文档格式）。"""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"


def _task_progress_text(task) -> str:
    """生成任务进度文本（running 含百分比 + 类型专属进度详情）。"""
    status = getattr(task, "status", "") if not isinstance(task, dict) else task.get("status", "")
    progress = float(getattr(task, "progress", 0) if not isinstance(task, dict) else task.get("progress", 0))
    error = getattr(task, "error", "") if not isinstance(task, dict) else task.get("error", "")
    ttype = getattr(task, "type", "") if not isinstance(task, dict) else (
        task.get("service_type", "") or task.get("type", ""))

    if status == "failed" and error:
        return error[:40]
    # MOSS 模型懒加载期 progress==0 也需显示加载状态文本（如 · loading_model），
    # 其余任务（含 Whisper）保持首段前无文案
    elif status == "running" and (progress > 0 or ttype == "moss"):
        payload = (task.get("payload", {}) if isinstance(task, dict)
                   else getattr(task, "payload", {})) or {}
        # ── GSV 合成：pos/total 是片段序号/预估片段数，不是秒 ──
        if ttype == "gsv":
            if isinstance(payload, dict) and payload.get("total"):
                return f"{int(progress * 100)}% · 片段 {int(payload.get('pos', 0))}/{int(payload['total'])}"
            return f"{int(progress * 100)}%"
        # ── 转写统一详情：pos/total（秒）→ LRC 时间戳，speed → v/s ──
        # 仅保留百分比 / [已完成时长/总时长] / 速度 v/s（unit=ratio 为时长探测失败回退）
        detail = ""
        if (isinstance(payload, dict)
                and payload.get("pos")
                and payload.get("total")
                and payload.get("unit") != "ratio"):
            pos, total = float(payload.get("pos", 0)), float(payload["total"])
            detail = f" [{_lrc_time(pos)}/{_lrc_time(total)}]"
            if payload.get("speed"):
                detail += f" {float(payload['speed']):.1f}/s"
        return f"{int(progress * 100)}%{detail}"
    elif status == "pending":
        return "等待中…"
    elif status == "completed":
        return "已完成"
    elif status == "cancelled":
        return "已取消"
    return ""


def task_card(
    task,
    role: str = "waiting",   # "current" | "waiting"
    callbacks: dict | None = None,
) -> ft.Container:
    """单个任务卡片。

    Args:
        task: TaskSnapshot 或 dict
        role: "current" = 当前运行任务（只显示取消），"waiting" = 等待中（显示上移/下移/取消）
        callbacks: {"on_cancel", "on_move_up", "on_move_down"}
    """
    cb = callbacks or {}

    if isinstance(task, dict):
        tid = task.get("id", "")
        ttype = task.get("service_type", "") or task.get("type", "")
        status = task.get("status", "")
        progress = float(task.get("progress", 0))
        summary = task.get("input_summary", "")
    else:
        tid = getattr(task, "id", "")
        ttype = getattr(task, "service_type", "") or getattr(task, "type", "")
        status = getattr(task, "status", "")
        progress = float(getattr(task, "progress", 0))
        summary = getattr(task, "input_summary", "")

    progress_widget = _task_progress_bar(progress) if status == "running" else None
    detail_text = _task_progress_text(task)
    detail_color = (
        Palette.ERROR if status == "failed" else
        Palette.SUCCESS if status in ("completed", "running") and progress > 1.0 else
        Palette.PRIMARY if status == "running" else
        Palette.SUBTEXT
    )
    if status == "cancelled":
        detail_color = "#F59E0B"
    # 运行中百分比文本加粗（与进度条呼应），其余状态常规
    detail_weight = "w600" if status == "running" else "normal"

    # 操作按钮
    actions = []
    if role == "waiting" and status == "pending":
        if cb.get("on_move_up"):
            actions.append(ft.IconButton(
                icon=ft.Icons.ARROW_UPWARD, icon_size=12, icon_color=Palette.SUBTEXT,
                tooltip="上移",
                on_click=lambda e, tid=tid: cb["on_move_up"](tid),
                style=ft.ButtonStyle(padding=ft.Padding.all(0)),
            ))
        if cb.get("on_move_down"):
            actions.append(ft.IconButton(
                icon=ft.Icons.ARROW_DOWNWARD, icon_size=12, icon_color=Palette.SUBTEXT,
                tooltip="下移",
                on_click=lambda e, tid=tid: cb["on_move_down"](tid),
                style=ft.ButtonStyle(padding=ft.Padding.all(0)),
            ))
    if status in ("pending", "running") and cb.get("on_cancel"):
        actions.append(ft.IconButton(
            icon=ft.Icons.CLOSE, icon_size=12, icon_color=Palette.ERROR,
            tooltip="取消",
            on_click=lambda e, tid=tid: cb["on_cancel"](tid),
            style=ft.ButtonStyle(padding=ft.Padding.all(0)),
        ))

    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Column([
                    ft.Text(summary or tid, size=Typography.BODY, weight=ft.FontWeight.W_600,
                            color=Palette.TEXT, no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Row([
                        _text(f"#{tid[:8]}", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                        _text("·", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                        _text(detail_text, Typography.CAPTION, detail_weight, detail_color),
                    ], spacing=4),
                    *([progress_widget] if progress_widget is not None else []),
                ], spacing=2, expand=True),
                _task_status_icon(status),
                ft.Row(actions, spacing=0) if actions else ft.Container(width=0),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ], spacing=4),
        bgcolor=Palette.SURFACE2,
        border_radius=Radius.LG,
        padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        border=ft.Border.all(1, Palette.BORDER_SUBTLE),
        animate=ft.Animation(Anim.FAST, ft.AnimationCurve.EASE),
    )


def task_queue_panel(
    current_task=None,
    waiting_tasks: list | None = None,
    callbacks: dict | None = None,
    empty_text: str = "暂无任务",
    max_items: int = 8,
    expand: bool | int = True,
) -> ft.Container:
    """任务队列面板 — 当前任务 + 等待任务列表。

    Args:
        current_task: 当前运行任务（TaskSnapshot 或 None）
        waiting_tasks: 等待中任务列表
        callbacks: {"on_cancel", "on_move_up", "on_move_down", "on_clear", "on_pause_toggle", "is_paused"}
        empty_text: 空状态文案
        max_items: 最大显示条目
        expand: 扩展比例
    """
    waiting_tasks = waiting_tasks or []
    cb = callbacks or {}

    task_rows = []

    # 当前任务卡片
    if current_task:
        task_rows.append(
            ft.Column([
                _text("当前任务", Typography.SMALL, "w600", Palette.SUBTEXT),
                task_card(current_task, role="current", callbacks=cb),
            ], spacing=4)
        )

    # 等待任务
    if waiting_tasks:
        task_rows.append(
            ft.Column([
                _text(f"等待中 ({len(waiting_tasks)})", Typography.SMALL, "w600", Palette.SUBTEXT),
                *[task_card(t, role="waiting", callbacks=cb) for t in waiting_tasks],
            ], spacing=4)
        )

    # 空状态
    if not task_rows:
        task_rows.append(
            ft.Container(
                content=ft.Column([
                    _icon(ft.Icons.INBOX, 32, Palette.TEXT_MUTED),
                    _text(empty_text, Typography.BODY, "normal", Palette.SUBTEXT),
                    _text("提交后将在此处显示", Typography.CAPTION, "normal", Palette.TEXT_MUTED),
                ], spacing=6, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                alignment=ft.Alignment.CENTER,
                bgcolor=Palette.SURFACE2,
                border_radius=Radius.LG,
                padding=ft.Padding.all(32),
                border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            )
        )

    # 暂停/恢复按钮
    is_paused = cb.get("is_paused", False)
    pause_label = "恢复" if is_paused else "暂停"
    pause_icon = ft.Icons.PLAY_ARROW if is_paused else ft.Icons.PAUSE
    pause_color = Palette.SUCCESS if is_paused else Palette.WARNING

    pause_btn = ft.TextButton(
        pause_label,
        icon=pause_icon,
        on_click=lambda e: cb.get("on_pause_toggle")() if cb.get("on_pause_toggle") else None,
        style=ft.ButtonStyle(color=pause_color, padding=ft.Padding.all(4)),
    ) if cb.get("on_pause_toggle") else None

    # 清空按钮
    clear_btn = ft.TextButton(
        "清空", icon=ft.Icons.CLEAR_ALL,
        on_click=lambda e: cb.get("on_clear")() if cb.get("on_clear") else None,
        style=ft.ButtonStyle(color=Palette.SUBTEXT, padding=ft.Padding.all(4)),
    ) if cb.get("on_clear") else None

    header_actions = []
    if pause_btn:
        header_actions.append(pause_btn)
    if clear_btn:
        header_actions.append(clear_btn)

    return ft.Container(
        content=ft.Column([
            toolbar_panel_header("任务队列", actions=header_actions),
            divider(),
            ft.Column(task_rows, spacing=8, scroll=ft.ScrollMode.AUTO, expand=True),
        ], spacing=8),
        bgcolor=Palette.SURFACE,
        border_radius=Radius.XL,
        padding=ft.Padding.all(16),
        border=ft.Border.all(1, Palette.BORDER_SUBTLE),
        shadow=_shadow("low"),
        # 高度不再固定：由父级 workspace(expand) 撑满，底边完整显示
        expand=expand,
    )
