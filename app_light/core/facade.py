"""
CoreFacade — single entry point that orchestrates translation & transcription
task queues, manages Service lifecycle (llama-server process, MOSS model, GSV engine),
and emits unified TaskSnapshots to the UI layer.

Owns zero Flet imports; safe to call from any thread.

Lifecycle
---------
- ``__init__(backend_dict, config_dict)`` **only registers** configuration —
  nothing is started (lazy loading).  Use
  ``start_service(name)`` / ``stop_service(name)`` /
  ``restart_service(name)`` to drive the lifecycle explicitly; every call is
  idempotent and safe to repeat.

- ``stop_service(name)`` pauses the queue, waits for the running task to
  finish (30s timeout), stops the worker thread and the backend service, but
  **keeps the queue and its pending tasks**.  Calling ``start_service()`` /
  ``restart_service()`` again re-injects the executor and resumes consuming
  the preserved pending tasks.

- ``register_service(name, backend, config_path)`` updates a service unit's
  configuration; a running service of the same type is hot-restarted, a
  different type is stopped first, then replaced.

- ``register_callback(name, callback)`` is a **stub** — the base class only
  declares the interface; subclasses override it to wrap the UI callback and
  inject it into the Service (``set_on_status_change``) and the TaskQueue
  (``on_status_change``).  Callbacks are no longer accepted at construction
  time; wire them explicitly after ``Facade`` is built.

- ``shutdown()`` stops services **and** destroys the queues.  It is
  idempotent — safe to call multiple times.
"""

from __future__ import annotations
from uuid import uuid4
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import asdict

from .contracts import (
    TaskRequest,
    TranslationRequest,
    TranscriptionRequest,
    TaskSnapshot,
    ServiceSnapshot,
    NameNotFound,
    FacadeError,
    Task,
    TaskStatus,
)
from .task_que import TranslationTaskQueue, TranscriptionTaskQueue, TaskQueue
from .service import LlamaService, APIService, Service
from .rule_splitter import RuleSplitter




def _new_translation_service(backend: str, model_config: dict) -> Service:
    """Create the translation service matching *backend* ("llama" or "api")."""
    if backend == "api":
        return APIService(model_config)
    if backend == "llama":
        return LlamaService(model_config)
    raise ValueError(f"Unknown backend: {backend}")


def _make_snapshot(raw: Task) -> TaskSnapshot:
    """Convert the queue's internal dict into an immutable TaskSnapshot."""
    payload = raw.get("payload", {})
    if raw.get("type") == "translate":
        text = payload.get("text", "")
        summary = text[:60].replace("\n", " ") if text else ""
    else:
        ap = payload.get("audio_path", "")
        summary = str(Path(ap).name) if ap else ""
    return TaskSnapshot(
        index=raw.get('index'),
        id=raw.get("id", ""),
        type=raw.get("type", ""),
        status=raw.get("status", ""),
        progress=float(raw.get("progress", 0.0)),
        input_summary=summary,
        result=raw.get("result"),
        error=raw.get("error", ""),
        created_at=float(raw.get("created_at", 0.0)),
        payload=payload,
    )


class Facade:
    def __init__(
            self,
            backend_dict: dict[str, tuple],
            config_dict: dict[str, Path],
        ):
        # 注册表:name -> (service_cls, queue_cls)
        self._backend_dict: dict[str, tuple] = {}
        self._config_dict: dict[str, Path] = {}
        self._service_dic: dict[str, Service] = {}
        self._queue_dic: dict[str, TaskQueue] = {}
        self._callbacks: dict[str, Optional[callable]] = {}
        self._uuid_pool: set[str] = set()

        # 懒加载:只注册配置,不启动任何服务
        for name, backend in (backend_dict or {}).items():
            self.register_service(name, backend, config_dict.get(name, None))

        self._init(backend_dict, config_dict)

    #=================================================
    #                   公开接口
    #=================================================

    def register_service(
            self,
            name: str,
            backend: tuple,
            config_path: Path = None,
        ):
        """注册(或更新)一个服务单元的配置;懒加载,不启动。

        - 同类型且运行中:热更新配置并 restart(保留 pending 任务)。
        - 不同类型且运行中:先 stop 旧单元,再更新注册表。
        - 已停止:仅更新注册表,下次 ``start_service`` 生效。
        """
        service_cls, queue_cls = backend
        if not (isinstance(service_cls, type) and issubclass(service_cls, Service)):
            raise FacadeError(
                f"backend[0] must be a Service subclass, got {service_cls!r}"
            )
        if not (isinstance(queue_cls, type) and issubclass(queue_cls, TaskQueue)):
            raise FacadeError(
                f"backend[1] must be a TaskQueue subclass, got {queue_cls!r}"
            )

        old_service = self._service_dic.get(name)
        if old_service is not None and old_service.is_running:
            if isinstance(old_service, service_cls):
                # 同类型热更新:更新配置并 restart,恢复消费保留的 pending
                self._backend_dict[name] = backend
                self._config_dict[name] = config_path
                old_service.restart(config_path)
                queue = self._queue_dic.get(name)
                if queue is not None:
                    queue.set_excutor(old_service.get_executor())
                    queue.start()  # 幂等
                return
            # 类型变化:先停旧单元(释放资源),再更新注册表
            self.stop_service(name)

        self._backend_dict[name] = backend
        self._config_dict[name] = config_path

    def _before_service_start(self, name: str, service) -> None:
        """子类钩子：在 ``service.start()`` 之前调用，用于注入回调（如 UI 日志/状态）。"""
        pass

    def start_service(self, name: str, backend: tuple = None, config_path: Path = None):
        """启动(或重启)一个服务单元;幂等,已运行则 no-op。

        兼容旧签名:传入 ``backend`` / ``config_path`` 时先执行注册再启动。
        """
        if backend is not None:
            self.register_service(name, backend, config_path)
        if name not in self._backend_dict:
            raise NameNotFound(name)

        service = self._service_dic.get(name)
        if service is not None and service.is_running:
            return  # 幂等

        if service is None:
            service_cls, _ = self._backend_dict[name]
            cfg = config_path or self._config_dict[name]
            if config_path is not None:
                # 选择生效：本次传入的配置覆盖注册表默认（后续 restart 沿用）
                self._config_dict[name] = config_path
            service = service_cls(cfg)
            self._service_dic[name] = service
        else:
            # 已存在(停止中):应用传入配置（若指定）或注册表里的最新配置再启动
            cfg = config_path or self._config_dict[name]
            if config_path is not None:
                self._config_dict[name] = config_path
            service.restart(cfg)

        self._before_service_start(name, service)
        service.start()
        executor = service.get_executor()

        queue = self._queue_dic.get(name)
        if queue is None:
            _, queue_cls = self._backend_dict[name]
            queue = queue_cls(executor)
            self._queue_dic[name] = queue
        else:
            queue.set_excutor(executor)
        queue.start()
        # 恢复执行：停止服务时 current 被挂起（暂停而非取消），
        # 重新加载后从暂停点继续（配合 UI「加载服务后开启队列」语义）
        queue.resume()

    def stop_service(self, name: str):
        """停止一个服务单元;幂等。

        先停取新任务,等当前任务完成(30s 超时),再真正退出 worker 线程
        (pending 任务保留),最后停止后端服务。
        """
        if name not in self._backend_dict:
            raise NameNotFound(name)
        service = self._service_dic.get(name)
        if service is None or not service.is_running:
            return  # 幂等

        queue = self._queue_dic.get(name)
        if queue is not None:
            # 停止服务 = 暂停队列并挂起当前任务（不取消、不等其完成）
            queue.pause()               # 停取新任务 + 挂起 current（执行体在检查点阻塞）
            queue.set_excutor(None)     # 解除执行体引用（暂停中不取新任务）
            # 注意：不 queue.stop()（worker daemon 保留挂起，重启 resume 后继续，
            #       避免 stop 的 join 超时后残留 worker 与重启新 worker 并发双跑）；
            #       不 queue.resume()（保持暂停位，start_service 时统一恢复）
        service.stop()

    def restart_service(self, name: str):
        """停止后立即重新启动;保留 pending 任务。"""
        self.stop_service(name)
        self.start_service(name)

    def pause_queue(self, name: str):
        """暂停队列（立即生效）：停止取新任务，并挂起正在执行的任务。

        无需等待当前任务完成——``queue.pause()`` 现在会暂停 current，
        执行体在 chunk / segment 检查点阻塞，``resume_queue`` 后继续。
        """
        queue = self.get_que(name)
        queue.pause()

    def resume_queue(self, name: str):
        queue = self.get_que(name)
        queue.resume()

    def clear_queue(self, name: str):
        queue = self.get_que(name)
        queue.clear()

    def submit_task(self, task_request: TaskRequest):
        name = task_request.task_type
        queue = self.get_que(name)

        id = self.get_new_id()
        task_request = asdict(task_request)
        task_request["id"] = id
        if task_request.get("payload") is None:
            task_request["payload"] = {}
        task = Task(**task_request)
        queue.add(task)

    def reorder_task(self, snapshot: TaskSnapshot):
        name = snapshot.type
        queue = self.get_que(name)

        id = snapshot.id
        new_index = snapshot.index
        queue.reorder(id, new_index)
        return

    def remove_task(self, snapshot: TaskSnapshot):
        name = snapshot.type
        queue = self.get_que(name)

        id = snapshot.id
        queue.remove(id)

    def get_new_id(self):
        id = str(uuid4())[:8]
        while id in self._uuid_pool:
            id = str(uuid4())[:8]
        self._uuid_pool.add(id)
        return id

    def get_que(self, name: str) -> TaskQueue:
        """返回任务队列;未注册抛 NameNotFound,已注册未启动抛 FacadeError。"""
        if name not in self._backend_dict:
            raise NameNotFound(name)
        queue = self._queue_dic.get(name)
        if queue is None:
            raise FacadeError(
                f"service '{name}' not started — call start_service('{name}') first"
            )
        return queue

    def get_service(self, name: str) -> Service:
        """返回后端服务;未注册抛 NameNotFound,已注册未启动抛 FacadeError。"""
        if name not in self._backend_dict:
            raise NameNotFound(name)
        service = self._service_dic.get(name)
        if service is None:
            raise FacadeError(
                f"service '{name}' not started — call start_service('{name}') first"
            )
        return service

    def get_status(self, name: str) -> ServiceSnapshot:
        """返回服务单元的合并状态(service + queue 两维)。"""
        if name not in self._backend_dict:
            raise NameNotFound(name)
        service = self._service_dic.get(name)
        queue = self._queue_dic.get(name)

        service_status = "running" if (service is not None and service.is_running) \
            else "stopped"

        if queue is None or not queue.is_running:
            queue_status = "stopped"
        elif queue.is_paused:
            queue_status = "paused"
        else:
            queue_status = "running"

        return ServiceSnapshot(
            service_type=self._backend_dict[name][0].__name__,
            service_status=service_status,
            queue_status=queue_status,
            pending_count=queue.pending_count if queue is not None else 0,
        )

    def register_callback(self, name: str, callback: Optional[callable]):
        """注册 *name* 服务单元的状态变化回调（**子类重写**）。

        基类只声明接口,不实现:回调的包装、注入(Service.set_on_status_change
        与 TaskQueue.on_status_change),以及"先注册后启动 / 先启动后注册"
        两种时序的接线,均由子类负责。

        约定(见 core/test_facade.py 的 TestFacade):
        - ``Service._emit`` 传 ``self._running``(bool),TaskQueue._emit 传
          ``(current, pending, finished)`` 三元组——子类按需包装后注入;
        - 已启动的单元应立即注入,未启动的存入注册表,待 ``start_service``
          创建后补注入。
        """
        ...

    def shutdown(self) -> None:
        """Idempotent: stop services, destroy queues, release all resources."""
        for name in list(self._service_dic.keys()):
            self.stop_service(name)
        self._service_dic.clear()
        self._queue_dic.clear()
        self._callbacks.clear()
        self._backend_dict.clear()
        self._config_dict.clear()

    @staticmethod
    def _wait_for_idle(queue, timeout: float):
        """Block until *queue* has no inflight task, or *timeout* expires.

        On timeout the in-flight task is marked ``failed`` so it is not lost
        silently; the caller proceeds to tear the queue down.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with queue._lock:
                if queue._current_task is None:
                    return
            time.sleep(0.1)
        with queue._lock:
            task = queue._current_task
            if task is not None:
                task.status = TaskStatus.FAILED
                task.error = f"task stopped by service stop after {timeout:.0f}s timeout"

    #=================================================
    #                 需重写的内部函数
    #=================================================
    def _init(
        self,
        backend_dict: dict[str, tuple],
        config_dict: dict[str, Path],
        ):

        ...

if __name__ == "__main__":
    ...
