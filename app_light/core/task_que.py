"""
任务队列系统：为翻译和转写任务提供独立的顺序执行队列。
支持实时调整未执行任务的顺序、取消正在执行的任务，以及暂停/恢复。
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Any, Optional, TYPE_CHECKING
from pathlib import Path
from .contracts import Task, TaskStatus, CancelledError, NameNotFound
from . import service
from .moss.moss_service import MossService

# Executor 仅用于类型标注（且函数签名均为惰性求值），顶层不再导入
# core.executor —— 它在启动链中会连带加载 openai/numpy 等重库。真正需要
# Executor 的 worker 执行路径由 service/executor 自身导入。
if TYPE_CHECKING:
    from .executor import Executor


# ── 通用任务队列 ────────────────────────────────────────

class TaskQueue:
    """通用顺序任务队列。

    每个队列绑定一个 executor（Executor 实例），在 daemon 线程中按 FIFO
    顺序执行任务。executor 实例由外部（CoreFacade）注入，队列不管理其
    生命周期。默认执行体为 ``_run(task)`` —— 调用 ``excutor.execute(...)``，
    子类可覆盖以定制执行逻辑。

    支持：
    - add():    添加任务到队尾
    - remove(): 移除 pending 任务
    - cancel(): 取消 pending 或标记 running 任务
    - reorder(): 改变 pending 任务在队列中的位置
    - pause() / resume(): 暂停 / 恢复取任务
    - get_all(): 获取全部任务的状态列表
    - start() / stop(): 启停 worker 线程
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
        self._paused.set()  # 初始状态：未暂停（event 已设置，wait() 立即返回）

    # ── 公开 API ─────────────────────────────────────

    def set_on_status_change(self, callback: Optional[Callable]):
        """注入/替换状态变化回调（供 Facade 子类在运行期接线）。"""
        self._on_status_change = callback

    def add(self, task: Task) -> str:
        """添加任务到队尾，返回 task_id。"""
        with self._lock:
            self._tasks.append(task)
            self._order()
        self._emit()
        return task.id

    def remove(self, task_id: str) -> bool:
        """移除一个 pending 任务（标记为 cancelled 并从队列删除）。"""
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
            self._emit()  # 锁外回调，避免 UI 重绘反查队列时重入死锁
        return removed

    def cancel(self, task_id: str) -> bool:
        """取消 pending 任务（从队列移除），或向 running 任务发送取消信号。"""
        with self._lock:
            # 正在执行的任务 → 发送取消信号
            if self._current_task and self._current_task.id == task_id:
                self._current_task.cancel()
                return True
            # pending 任务 → 直接移除      
        return self.remove(task_id)

    def reorder(self, task_id: str, new_index: int) -> bool:
        """将 pending 任务移到指定位置（0 = 下一个执行）。

        注意：如果 current_task 正在执行，new_index=0 表示排到 pending 队列最前。
        移动成功后推送状态变更，驱动 UI 队列实时刷新。
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
            self._emit()  # 锁外推送，避免回调重入
        return moved

    def get_all(self) -> list[dict]:
        """返回所有任务的状态快照（current + pending + finished），用于前端展示。"""
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
        """更新正在执行任务的进度（0.0 ~ 1.0）与可选详情（payload），线程安全。"""
        with self._lock:
            if self._current_task:
                self._current_task.progress = progress
                if payload is not None:
                    self._current_task.payload = payload
        self._emit()  # 锁外回调，避免 UI 重绘反查队列时重入死锁

    def pause(self):
        """暂停队列 — 停止取新任务，同时挂起正在执行的任务（检查点等待）。

        正在执行的任务会在下一个 chunk / segment 检查点阻塞等待，
        直到 :meth:`resume` 恢复。
        """
        self._paused.clear()
        with self._lock:
            current = self._current_task
        if current is not None:
            current.pause()

    def resume(self):
        """恢复队列 — 继续取任务，并让挂起的当前任务继续执行。"""
        self._paused.set()
        with self._lock:
            current = self._current_task
        if current is not None:
            current.resume()

    def clear(self):
        with self._lock:
            self._tasks.clear()
        self._emit()  # 锁外回调，避免 UI 重绘反查队列时重入死锁

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def start(self):
        """启动 worker daemon 线程。"""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name=f"tq-{self.name}", daemon=True
        )
        self._worker_thread.start()

    def stop(self):
        """停止 worker daemon 线程,保留 pending 任务(幂等)。

        worker 可能在 ``execute()`` 中阻塞(如网络请求),因此 join 带超时;
        由调用方(Facade)保证先等待当前任务完成再调用本方法。
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
        """正在执行 + 还在等待的任务总数。"""
        with self._lock:
            count = len(self._tasks)
            if self._current_task:
                count += 1
            return count

    @property
    def is_running(self) -> bool:
        return self._running

    # ── 内部 ─────────────────────────────────────────
    def _run(self, task: Task):
        """默认执行体：调用注入的 executor 执行任务。

        进度回调约定: progress_callback(current, total)，total<=0 时按 1 处理。
        子类可覆盖以定制执行逻辑。
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
        """按列表位置管理 Task 的 index：pending 与 finished 各自从 0 编号。

        - ``_tasks``（pending）：index = 等待队列中的位置（0 = 下一个执行）
        - ``_finished_tasks``（finished）：index = 完成顺序

        供 add / remove / reorder / worker 取任务与完成任务后调用，
        保证各列表内 Task 的 ``index`` 与其列表位置一致。
        """
        for i, t in enumerate(self._tasks):
            t.index = i
        for i, t in enumerate(self._finished_tasks):
            t.index = i

    def _worker_loop(self):
        while self._running:
            # 暂停时阻塞等待，不中断正在执行的任务（此处检查在取新任务之前）
            self._paused.wait()

            task: Optional[Task] = None
            with self._lock:
                if self._tasks:
                    task = self._tasks.pop(0)
                    self._current_task = task
                    self._order()  # 剩余 pending 的 index 重排（保持连续）

            if task is None:
                time.sleep(0.1)
                continue

            # 取到任务后，先检查是否已被取消
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
                # 取消的任务不进已完成队列（直接从队列消失）；failed 保留以便查看错误
                if task.status != TaskStatus.CANCELLED:
                    self._finished_tasks.append(task)
                self._current_task = None
                self._order()  # finished 按完成顺序编号

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


# ── 翻译任务队列 ────────────────────────────────────────

class TranslationTaskQueue(TaskQueue):
    """翻译任务队列 — 由外部注入 excutor executor。

    用法:
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


# ── 转写任务队列 ────────────────────────────────────────

class TranscriptionTaskQueue(TaskQueue):
    """转写任务队列 — 由外部注入 excutor executor。

    用法:
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
        # 进度语义：转写按时间轴上报（pos/total/speed），progress 存 0.0~1.0 比例，
        # 详情（LRC 时间/速率）存 task.payload 供 UI 显示
        def _on_progress(pos: float, total: float, speed: float, segs=None):
            ratio = min(pos / total, 1.0) if total > 0 else 1.0
            payload = {"pos": pos, "total": total, "speed": speed}
            if isinstance(segs, dict):
                # MOSS 状态载荷（status/generated_tokens/segments/unit）→ 并入 payload
                payload.update(segs)
            elif segs:
                payload["segments"] = segs
            self.update_progress(ratio, payload)

        return self.excutor.execute(
            task,
            progress_callback=_on_progress,
            cancel_event=task._cancel_event,
        )


# ── GSV 合成任务队列 ─────────────────────────────────────

class GsvTaskQueue(TaskQueue):
    """GPT-SoVITS 合成任务队列 — 片段级 4 参进度（pos/total/speed/payload）。

    用法（与 TranscriptionTaskQueue 同构）::

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
        # 进度语义：按片段上报（pos/total/fragment），progress 存 0.0~1.0 比例，
        # 详情存 task.payload（fragment 序号）供 UI 显示
        def _on_progress(pos: float, total: float, speed=None, payload=None):
            ratio = min(pos / total, 1.0) if total > 0 else 1.0
            self.update_progress(ratio, {"pos": pos, "total": total, **dict(payload or {})})

        return self.excutor.execute(
            task,
            progress_callback=_on_progress,
            cancel_event=task._cancel_event,
        )


# ── 后端注册表 ──────────────────────────────────────────

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
