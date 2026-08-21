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
        # registry: name -> (service_cls, queue_cls)
        self._backend_dict: dict[str, tuple] = {}
        self._config_dict: dict[str, Path] = {}
        self._service_dic: dict[str, Service] = {}
        self._queue_dic: dict[str, TaskQueue] = {}
        self._callbacks: dict[str, Optional[callable]] = {}
        self._uuid_pool: set[str] = set()

        # lazy loading: only register configuration, start nothing
        for name, backend in (backend_dict or {}).items():
            self.register_service(name, backend, config_dict.get(name, None))

        self._init(backend_dict, config_dict)

    # ── Public API ──

    def register_service(
            self,
            name: str,
            backend: tuple,
            config_path: Path = None,
        ):
        """Register (or update) a service unit's configuration; lazy, starts nothing.

        - Same type and running: hot-update config and restart (pending tasks preserved).
        - Different type and running: stop the old unit first, then update the registry.
        - Stopped: only update the registry; takes effect on the next ``start_service``.
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
                # same-type hot update: update config and restart, resume consuming preserved pending tasks
                self._backend_dict[name] = backend
                self._config_dict[name] = config_path
                old_service.restart(config_path)
                queue = self._queue_dic.get(name)
                if queue is not None:
                    queue.set_excutor(old_service.get_executor())
                    queue.start()  # idempotent
                return
            # type change: stop the old unit first (release resources), then update the registry
            self.stop_service(name)

        self._backend_dict[name] = backend
        self._config_dict[name] = config_path

    def _before_service_start(self, name: str, service) -> None:
        """Subclass hook: called before ``service.start()`` to inject callbacks (e.g. UI log/status)."""
        pass

    def start_service(self, name: str, backend: tuple = None, config_path: Path = None):
        """Start (or restart) a service unit; idempotent, no-op if already running.

        When ``backend`` / ``config_path`` are passed, registers first, then starts.
        """
        if backend is not None:
            self.register_service(name, backend, config_path)
        if name not in self._backend_dict:
            raise NameNotFound(name)

        service = self._service_dic.get(name)
        if service is not None and service.is_running:
            return  # idempotent

        if service is None:
            service_cls, _ = self._backend_dict[name]
            cfg = config_path or self._config_dict[name]
            if config_path is not None:
                # apply the selected config: this call's config overrides the registry default (kept for later restarts)
                self._config_dict[name] = config_path
            service = service_cls(cfg)
            self._service_dic[name] = service
        else:
            # already exists (stopped): apply the passed config (if any) or the latest
            # registry config, then start
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
        # resume execution: current is suspended (paused, not cancelled) when the
        # service stops; after reload it continues from the pause point (matching the
        # UI "start queue after loading service" semantics)
        queue.resume()

    def stop_service(self, name: str):
        """Stop a service unit; idempotent.

        First stops fetching new tasks, waits for the current task to finish (30s
        timeout), then exits the worker thread (pending tasks preserved), and finally
        stops the backend service.
        """
        if name not in self._backend_dict:
            raise NameNotFound(name)
        service = self._service_dic.get(name)
        if service is None or not service.is_running:
            return  # idempotent

        queue = self._queue_dic.get(name)
        if queue is not None:
            # stopping the service = pausing the queue and suspending the current
            # task (no cancel, no waiting for it to finish)
            queue.pause()               # stop fetching new tasks + suspend current (executor blocks at checkpoints)
            queue.set_excutor(None)     # drop the executor reference (no new tasks while paused)
            # note: no queue.stop() (the worker daemon stays suspended and resumes after
            #       restart-resume; avoids a leftover worker after stop's join timeout
            #       racing with the new worker on restart);
            #       no queue.resume() (keeps the paused state; start_service resumes uniformly)
        service.stop()

    def restart_service(self, name: str):
        """Stop then immediately restart; preserves pending tasks."""
        self.stop_service(name)
        self.start_service(name)

    def pause_queue(self, name: str):
        """Pause the queue (takes effect immediately): stop fetching new tasks and suspend the running task.

        No need to wait for the current task — ``queue.pause()`` now suspends current,
        the executor blocks at chunk / segment checkpoints, and resumes after
        ``resume_queue``.
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
        """Return the task queue; NameNotFound if unregistered, FacadeError if registered but not started."""
        if name not in self._backend_dict:
            raise NameNotFound(name)
        queue = self._queue_dic.get(name)
        if queue is None:
            raise FacadeError(
                f"service '{name}' not started — call start_service('{name}') first"
            )
        return queue

    def get_service(self, name: str) -> Service:
        """Return the backend service; NameNotFound if unregistered, FacadeError if registered but not started."""
        if name not in self._backend_dict:
            raise NameNotFound(name)
        service = self._service_dic.get(name)
        if service is None:
            raise FacadeError(
                f"service '{name}' not started — call start_service('{name}') first"
            )
        return service

    def get_status(self, name: str) -> ServiceSnapshot:
        """Return a service unit's combined status (service + queue dimensions)."""
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
        """Register the status-change callback for the *name* service unit (**overridden by subclasses**).

        The base class only declares the interface, it does not implement: callback
        wrapping, injection (Service.set_on_status_change and
        TaskQueue.on_status_change), and wiring for both "register-then-start" and
        "start-then-register" timings are all the subclass's responsibility.

        Contract (see TestFacade in core/test_facade.py):
        - ``Service._emit`` passes ``self._running`` (bool), TaskQueue._emit passes the
          ``(current, pending, finished)`` triple — subclasses wrap as needed before injecting;
        - started units should be injected immediately; not-yet-started ones are stored
          in the registry and injected after ``start_service`` creates them.
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

    # ── Internal functions to override ──
    def _init(
        self,
        backend_dict: dict[str, tuple],
        config_dict: dict[str, Path],
        ):

        ...

if __name__ == "__main__":
    ...
