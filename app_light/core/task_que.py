"""
Task queue system: provides separate sequential execution queues for translation
and transcription tasks. Supports real-time reordering of pending tasks, cancelling
a running task, and pause/resume.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Any, Optional, TYPE_CHECKING
from pathlib import Path
from .contracts import Task, TaskStatus, CancelledError, NameNotFound
from . import service
from .moss.moss_service import MossService

# Executor is used only for type annotations (and all function signatures are lazily
# evaluated), so core.executor is not imported at top level — importing it in the
# startup chain would pull in heavy libs like openai/numpy. The worker execution path
# that actually needs Executor imports it from service/executor itself.
if TYPE_CHECKING:
    from .executor import Executor


# ── Generic task queue ──

class TaskQueue:
    """Generic sequential task queue.

    Each queue is bound to one executor (an Executor instance) and runs tasks in FIFO
    order on a daemon thread. The executor instance is injected externally
    (CoreFacade); the queue does not manage its lifecycle. The default execution body
    is ``_run(task)`` — it calls ``excutor.execute(...)``; subclasses may override it
    to customize execution logic.

    Supports:
    - add():     append a task to the tail
    - remove():  remove a pending task
    - cancel():  cancel a pending task or signal a running task
    - reorder(): change a pending task's position in the queue
    - pause() / resume(): pause / resume fetching tasks
    - get_all(): get status list of all tasks
    - start() / stop(): start / stop the worker thread
    """

    def __init__(self, excutor: Optional[Executor],
                 on_status_change: Optional[Callable] = None):
        self.excutor = excutor
        self.name = type(self).__name__
        self._on_status_change = on_status_change

        self._tasks: list[Task] = []
        self._current_task: Optional[Task] = None
        self._finished_tasks: list[Task] = []
        self._lock = threading.Lock()

        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = threading.Event()
        self._paused.set()  # initial state: not paused (event set, wait() returns immediately)

    # ── Public API ──

    def set_on_status_change(self, callback: Optional[Callable]):
        """Inject/replace the status-change callback (for Facade subclasses to wire at runtime)."""
        self._on_status_change = callback

    def add(self, task: Task) -> str:
        """Append a task to the tail; returns the task_id."""
        with self._lock:
            self._tasks.append(task)
            self._order()
        self._emit()
        return task.id

    def remove(self, task_id: str) -> bool:
        """Remove a pending task (mark it cancelled and delete it from the queue)."""
        removed = False
        with self._lock:
            for i, t in enumerate(self._tasks):
                if t.id == task_id and t.status == TaskStatus.PENDING:
                    t.status = TaskStatus.CANCELLED
                    self._tasks.pop(i)
                    self._order()
                    removed = True
                    break
        if removed:
            self._emit()  # callback outside the lock, avoiding a reentrancy deadlock when the UI redraws and re-queries the queue
        return removed

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task (remove it from the queue), or signal a running task."""
        with self._lock:
            # running task → send the cancel signal
            if self._current_task and self._current_task.id == task_id:
                self._current_task.cancel()
                return True
            # pending task → remove directly      
        return self.remove(task_id)

    def reorder(self, task_id: str, new_index: int) -> bool:
        """Move a pending task to a given position (0 = next to execute).

        Note: if current_task is running, new_index=0 means the front of the pending queue.
        After a successful move, pushes a status change to drive real-time UI queue refresh.
        """
        moved = False
        with self._lock:
            for i, t in enumerate(self._tasks):
                if t.id == task_id and t.status == TaskStatus.PENDING:
                    self._tasks.pop(i)
                    new_index = max(0, min(new_index, len(self._tasks)))
                    self._tasks.insert(new_index, t)
                    self._order()
                    moved = True
                    break
        if moved:
            self._emit()  # push outside the lock, avoiding callback reentrancy
        return moved

    def get_all(self) -> list[dict]:
        """Return status snapshots of all tasks (current + pending + finished), for the frontend."""
        with self._lock:
            result = []
            if self._current_task:
                result.append(self._to_dict(self._current_task))
            for t in self._tasks:
                result.append(self._to_dict(t))
            for t in self._finished_tasks:
                result.append(self._to_dict(t))
            return result
    
    def get_current_task(self) -> Optional[Task]:
        return self._current_task if self._current_task else None

    def get_pending_tasks(self) -> list[Task]:
        return self._tasks if self._tasks else []

    def get_finished_tasks(self) -> list[Task]:
            return self._finished_tasks if self._finished_tasks else []
    
    def update_progress(self, progress: float, payload: Optional[dict] = None):
        """Update the running task's progress (0.0–1.0) and optional details (payload); thread-safe."""
        with self._lock:
            if self._current_task:
                self._current_task.progress = progress
                if payload is not None:
                    self._current_task.payload = payload
        self._emit()  # callback outside the lock, avoiding a reentrancy deadlock when the UI redraws and re-queries the queue

    def pause(self):
        """Pause the queue — stop fetching new tasks and suspend the running task (checkpoint wait).

        The running task blocks at the next chunk / segment checkpoint until
        :meth:`resume` is called.
        """
        self._paused.clear()
        with self._lock:
            current = self._current_task
        if current is not None:
            current.pause()

    def resume(self):
        """Resume the queue — resume fetching tasks and let the suspended current task continue."""
        self._paused.set()
        with self._lock:
            current = self._current_task
        if current is not None:
            current.resume()

    def clear(self):
        with self._lock:
            self._tasks.clear()
        self._emit()  # callback outside the lock, avoiding a reentrancy deadlock when the UI redraws and re-queries the queue

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def start(self):
        """Start the worker daemon thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name=f"tq-{self.name}", daemon=True
        )
        self._worker_thread.start()

    def stop(self):
        """Stop the worker daemon thread, preserving pending tasks (idempotent).

        The worker may block inside ``execute()`` (e.g. network requests), so the join
        has a timeout; the caller (Facade) guarantees the current task is awaited before
        calling this method.
        """
        if not self._running:
            return
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._tasks)

    @property
    def active_count(self) -> int:
        """Total count of running + still-waiting tasks."""
        with self._lock:
            count = len(self._tasks)
            if self._current_task:
                count += 1
            return count

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internals ──
    def _run(self, task: Task):
        """Default execution body: runs the task via the injected executor.

        Progress-callback convention: progress_callback(current, total); total<=0 is
        treated as 1. Subclasses may override to customize execution logic.
        """
        if self.excutor is None:
            raise RuntimeError(
                f"queue {self.name} has no executor — call set_excutor() before start()"
            )

        def _on_progress(current: int, total: int):
            self.update_progress(current / total if total > 0 else 1.0)

        return self.excutor.execute(
            task,
            progress_callback=_on_progress,
            cancel_event=task._cancel_event,
        )

    def _order(self):
        """Manage each Task's index by list position: pending and finished each numbered from 0.

        - ``_tasks`` (pending): index = position in the waiting queue (0 = next to execute)
        - ``_finished_tasks`` (finished): index = completion order

        Called after add / remove / reorder / worker task fetch and completion, so each
        Task's ``index`` matches its list position.
        """
        for i, t in enumerate(self._tasks):
            t.index = i
        for i, t in enumerate(self._finished_tasks):
            t.index = i

    def _worker_loop(self):
        while self._running:
            # block while paused, without interrupting the running task (checked before fetching a new task)
            self._paused.wait()

            task: Optional[Task] = None
            with self._lock:
                if self._tasks:
                    task = self._tasks.pop(0)
                    self._current_task = task
                    self._order()  # renumber remaining pending indexes (keep contiguous)

            if task is None:
                time.sleep(0.1)
                continue

            # after fetching a task, check first whether it was cancelled
            if task.is_cancelled:
                task.status = TaskStatus.CANCELLED
                self._emit()
                with self._lock:
                    self._current_task = None
                continue

            task.status = TaskStatus.RUNNING
            self._emit()

            try:
                result = self._run(task)
                if task.is_cancelled:
                    task.status = TaskStatus.CANCELLED
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
            except CancelledError:
                task.status = TaskStatus.CANCELLED
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)

            self._emit()
            with self._lock:
                # cancelled tasks do not enter the finished queue (they vanish from the
                # queue); failed ones are kept so the error can be inspected
                if task.status != TaskStatus.CANCELLED:
                    self._finished_tasks.append(task)
                self._current_task = None
                self._order()  # number finished by completion order

    def _emit(self):
        if self._on_status_change:
            info = (
                self.get_current_task(),
                self.get_pending_tasks(),
                self.get_finished_tasks()
            )
            try:
                self._on_status_change(info)
            except Exception:
                pass

    @staticmethod
    def _to_dict(task: Task) -> dict:
        return {
            "id": task.id,
            "status": task.status.value,
            "progress": task.progress,
            "file_path": task.file_path,
            "configs":task.configs,
            "file_name":task.file_name,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at,
            "index":task.index
        }


# ── Translation task queue ──

class TranslationTaskQueue(TaskQueue):
    """Translation task queue — executor injected externally.

    Usage:
        excutor = LlamaService(model_config).start().get_executor()
        queue = TranslationTaskQueue(excutor)
        queue.start()

        task = Task(task_type="translate", payload={
            "text": "こんにちは",
            "splitter": RuleSplitter(rule),
            "trans_config": {...},
            "prompts": {...},
            "glossary": {...},
        })
        queue.add(task)

        queue.pause()
        queue.resume()
        queue.stop()
    """

    def __init__(self, excutor:Optional[Executor], on_status_change: Optional[Callable] = None):
        super().__init__(excutor, on_status_change)

    def set_excutor(self, excutor:Optional[Executor]):
        self.excutor = excutor


# ── Transcription task queue ──

class TranscriptionTaskQueue(TaskQueue):
    """Transcription task queue — executor injected externally.

    Usage:
        svc = MossService(model_config)
        svc.start()
        excutor = svc.get_executor()
        queue = TranscriptionTaskQueue(excutor)
        queue.start()

        task = Task(task_type="moss", payload={
            "audio_path": "test.mp3",
            "language": "ja",
        })
        queue.add(task)
    """

    def __init__(
            self,
            excutor:Optional[Executor],
            on_status_change: Optional[Callable] = None
        ):
        super().__init__(excutor, on_status_change)

    def set_excutor(self, excutor:Optional[Executor]):
            self.excutor = excutor

    def _run(self, task: Task):
        # progress semantics: transcription reports along the timeline (pos/total/speed);
        # progress stores a 0.0–1.0 ratio, details (LRC timings/rate) go in task.payload for the UI
        def _on_progress(pos: float, total: float, speed: float, segs=None):
            ratio = min(pos / total, 1.0) if total > 0 else 1.0
            payload = {"pos": pos, "total": total, "speed": speed}
            if isinstance(segs, dict):
                # MOSS status payload (status/generated_tokens/segments/unit) → merged into payload
                payload.update(segs)
            elif segs:
                payload["segments"] = segs
            self.update_progress(ratio, payload)

        return self.excutor.execute(
            task,
            progress_callback=_on_progress,
            cancel_event=task._cancel_event,
        )


# ── GSV synthesis task queue ──

class GsvTaskQueue(TaskQueue):
    """GPT-SoVITS synthesis task queue — fragment-level 4-arg progress (pos/total/speed/payload).

    Usage (isomorphic to TranscriptionTaskQueue)::

        svc = GsvService(model_config)
        svc.start()
        queue = GsvTaskQueue(svc.get_executor())
        queue.start()
        queue.add(Task(task_type="gsv", file_path=..., configs={"args": ...}))
    """

    def __init__(
            self,
            excutor: Optional[Executor],
            on_status_change: Optional[Callable] = None
        ):
        super().__init__(excutor, on_status_change)

    def set_excutor(self, excutor: Optional[Executor]):
        self.excutor = excutor

    def _run(self, task: Task):
        # progress semantics: reported per fragment (pos/total/fragment); progress stores
        # a 0.0–1.0 ratio, details (fragment number) go in task.payload for the UI
        def _on_progress(pos: float, total: float, speed=None, payload=None):
            ratio = min(pos / total, 1.0) if total > 0 else 1.0
            self.update_progress(ratio, {"pos": pos, "total": total, **dict(payload or {})})

        return self.excutor.execute(
            task,
            progress_callback=_on_progress,
            cancel_event=task._cancel_event,
        )


# ── Backend registry ──

class Backends:
    LLAMA = (service.LlamaService, TranslationTaskQueue)
    OPENAIAPI = (service.APIService, TranslationTaskQueue)
    MOSS = (MossService, TranscriptionTaskQueue)
    GSV = (service.GsvService, GsvTaskQueue)

    TYPE:dict = {
        'llama': LLAMA,
        'openai': OPENAIAPI,
        'moss': MOSS,
        'gsv': GSV,
    }

    def resolve(self, type_name:str) -> Optional[tuple]:
        backend_type = Backends.TYPE.get(type_name, None)
        try:
            if backend_type:
                return backend_type
            else: 
                raise NameNotFound(type_name)
        except NameNotFound as e:
            print(f'Can\'t find backend:{e.detail}')
            return None
