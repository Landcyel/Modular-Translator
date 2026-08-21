"""
Core data contracts — stable dataclass types shared between Core, AppController, and UI.

These replace loose dicts and provide lightweight validation at the boundary.
Model/network errors remain the responsibility of the execution layer.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class FacadeError(Exception):
    pass


class NameNotFound(FacadeError):
    def __init__(self, name):
        super().__init__()
        self.detail = name


# ── 任务状态 ────────────────────────────────────────────

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _new_paused_event() -> threading.Event:
    """返回已 set 的 Event（默认未暂停），供 Task._pause_event 使用。"""
    event = threading.Event()
    event.set()
    return event


# ── 任务数据结构 ────────────────────────────────────────

@dataclass
class Task:
    """单个任务，携带取消信号供执行体定期检查。"""
    task_type:Optional[str]
    file_path:Optional[Path]
    configs:Optional[dict[str, Path]]
    file_name:Optional[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    index:int = -1
    status: str = TaskStatus.PENDING
    progress: float = 0.0                 # 0.0 ~ 1.0
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)   # 进度详情（转写：pos/total/speed）

    # 取消信号 — 由外部调用 cancel() 设置，执行体定期检查
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    # 暂停信号 — 初始为已设置（未暂停）；由 TaskQueue.pause()/resume() 驱动，
    # 执行体在检查点调用 wait_if_paused() 挂起/继续
    _pause_event: threading.Event = field(
        default_factory=_new_paused_event, repr=False
    )

    def cancel(self):
        """设置取消信号（非阻塞）。"""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self):
        """若已取消则抛出 CancelledError，供执行体在可中断点调用。"""
        if self.is_cancelled:
            raise CancelledError(self.id)

    def pause(self):
        """挂起任务（非阻塞）：执行体在下一个检查点阻塞等待。"""
        self._pause_event.clear()

    def resume(self):
        """恢复任务：让暂停中的执行体继续。"""
        self._pause_event.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def wait_if_paused(self) -> bool:
        """若任务被暂停则阻塞等待恢复；被取消返回 False，恢复返回 True。

        供执行体在 chunk / segment 检查点调用——暂停期间挂起执行，
        恢复后继续；暂停期间收到取消信号则立即退出（返回 False）。
        """
        if self._pause_event.is_set():
            return True
        while not self._pause_event.is_set():
            if self.is_cancelled:
                return False
            self._pause_event.wait(0.2)
        return True

    def __lt__(self, other):
        """按 ``index`` 大小比较（供 sorted() 等按队列位置排序）。"""
        if not isinstance(other, Task):
            return NotImplemented
        return self.index < other.index


class CancelledError(Exception):
    """任务被取消时抛出。"""
    def __init__(self, task_id: str):
        super().__init__(f"任务 {task_id} 已被取消")
        self.task_id = task_id


@dataclass
class TaskRequest:
    task_type:Optional[str]
    file_path:Optional[Path]
    file_name:Optional[str]
    configs:Optional[dict[str, Path]]
    payload: Optional[dict] = None

    def validate(self) -> list[str]:
        ...


@dataclass
class TranslationRequest(TaskRequest):
    def validate(self) -> list[str]:
        issues = []
        return issues


@dataclass
class TranscriptionRequest(TaskRequest):
    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.file_path or not isinstance(self.file_path, Path):
            issues.append("audio_path is required and must be a Path")
        elif not self.file_path.exists():
            issues.append(f"audio file not found: {self.file_path}")

        args:Path = self.configs.get('args', None)
        if args or not isinstance(args, Path):
            issues.append("args is required and must be a Path")
        elif not args.exists():
            issues.append(f"args file not found: {args}")
        return issues


@dataclass
class TaskSnapshot:
    """Immutable snapshot of a single task's state for UI display.

    Built by CoreFacade from an internal Task + status info, then forwarded
    to the UI via the `on_task_change` callback.
    """
    index:int

    id: str = ""
    """8-char task short-id."""

    type: str = ""
    """``"translate"`` or ``"transcribe"``."""

    status: str = ""
    """One of ``pending``, ``running``, ``completed``, ``failed``, ``cancelled``."""

    progress: float = 0.0
    """0.0–1.0 for translate; segment count as float for transcribe."""

    file_name: str = ""
    """Short preview: first 60 chars of source text, or audio filename."""

    input_summary: str = ""
    """Short preview text (first 60 chars of source / audio filename)."""

    result: Any = None
    """Translation result (str) or transcription result (dict with segments+info)."""

    error: str = ""
    """Error message if status is 'failed'."""

    created_at: float = 0.0
    """Unix timestamp when the task was submitted."""

    payload: dict = field(default_factory=dict)
    """Original task payload (text / audio_path etc.)."""

    def __eq__(self, other):
        if not isinstance(other, TaskSnapshot):
            return NotImplemented
        return self.index == other.index


    def __lt__(self, other):
        if not isinstance(other, TaskSnapshot):
            return NotImplemented
        return self.index < other.index

@dataclass
class ServiceSnapshot:
    service_type: str
    service_status: str
    queue_status: str = "stopped"
    pending_count: int = 0



if __name__ == "__main__":
    test = TaskRequest(None, Path('./'), None, None)
    dic = asdict(test)
    print(dic)

    new_test = TaskRequest(**dic)
    print(new_test)
    