"""AppFacade — CoreFacade subclass: UI sink push wiring + queue-operation convenience methods + legacy stub-API compatibility layer.

Integration design: see PLANS/appfacade-integration.md.

- core has zero flet dependency: sinks are duck-typed (on_service_status /
  on_tasks / on_finished_tasks), marshalled to the main thread by the UI-side
  PageUiSink via ``page.run_thread``.
- The old stub API (list_*/cancel_task(tid)/reorder_task(tid,i)/export_task(tid,out)/
  submit_transcription/_on_*_change callbacks) is kept as a compatibility layer for
  the transcribe / completed pages until they are migrated to sinks.
"""

from __future__ import annotations
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from core.facade import Facade
from core.contracts import Task, TaskSnapshot
from app.log import log


# Task config key → readable label (submission log shows the actual config file paths used)
_TASK_CONFIG_LABELS = {
    "translate_config": "翻译参数",
    "prompts": "提示词",
    "glossary": "术语表",
    "rule": "规则",
    "args": "转写参数",
    "vad": "VAD",
    "hotwords": "热词",
    "language": "语言",
}


def _fmt_task_configs(configs) -> str:
    """Format the configs actually used by a task (task_request.configs) for submission-log display.

    Iterates the non-None items in configs and outputs a "label: path" list; when
    all are empty (including None) returns "not specified". Distinct from the
    *service* config paths shown in start_service logs — the submission log should
    show the task-config file paths such as translate params/prompts/VAD.
    """
    if not configs:
        return "未指定"
    parts = []
    for key, value in configs.items():
        if value is None or value == "":
            continue
        label = _TASK_CONFIG_LABELS.get(key, key)
        parts.append(f"{label}: {value}")
    return ", ".join(parts) if parts else "未指定"


def _project(task) -> dict:
    """Project a core internal Task into a UI-renderable dict (consumed by the translate page task_card).

    - type = task.task_type (task type is the service name)
    - status = TaskStatus enum → .value string
    - input_summary = file_name (readable file name for display)
    - payload: Task.payload (progress details, e.g. transcribe pos/total/speed) passed through
    """
    name = task.file_name or ""
    if not name and task.file_path:
        name = Path(task.file_path).name
    return {
        "id": task.id,
        "type": task.task_type,
        "status": task.status.value if hasattr(task.status, "value") else str(task.status),
        "progress": float(task.progress),
        "file_name": name,
        "input_summary": name,
        "result": task.result,
        "error": task.error,
        "created_at": float(task.created_at),
        "index": task.index,
        "payload": getattr(task, "payload", None) or {},
    }


def _to_snapshot(task) -> TaskSnapshot:
    """Convert a core Task to a core TaskSnapshot (old pages access snap.status/snap.result compatibly)."""
    name = task.file_name or ""
    if not name and task.file_path:
        name = Path(task.file_path).name
    return TaskSnapshot(
        index=task.index,
        id=task.id,
        type=task.task_type,
        status=task.status.value if hasattr(task.status, "value") else str(task.status),
        progress=float(task.progress),
        file_name=name,
        input_summary=name,
        result=task.result,
        error=task.error,
        created_at=float(task.created_at),
        payload=getattr(task, "payload", None) or {},
    )


class PageUiSink:
    """UI-side sink wrapper: backend-thread callbacks → page.run_thread main-thread control updates.

    - TranslatePage does not manage finished tasks: on_finished_tasks duck-checks
      whether impl implements update_finished_tasks and skips it otherwise.
    """

    def __init__(self, page, impl):
        self._page = page
        self._impl = impl

    def on_service_status(self, online: bool, loading: bool = False, device: Optional[str] = None):
        if self._page is not None:
            self._page.run_thread(lambda: self._impl.update_service_status(online, loading, device))
        else:
            self._impl.update_service_status(online, loading, device)

    def on_tasks(self, current, waiting):
        if self._page is not None:
            self._page.run_thread(lambda: self._impl.update_tasks(current, waiting))
        else:
            self._impl.update_tasks(current, waiting)

    def on_finished_tasks(self, tasks):
        """Forward completion events (transcribe-page preview refresh / TTS-page result refresh)."""
        fn = getattr(self._impl, "update_finished_tasks", None)
        if fn is None:
            return
        if self._page is not None:
            self._page.run_thread(lambda: fn(tasks))
        else:
            fn(tasks)


class AppFacade(Facade):
    """Unified facade: core service/queue management + UI sink push + legacy stub-API compatibility layer.

    Constructor args match the core Facade:
    - backend_dict: {name: (service_cls, queue_cls)} (registered service keys, e.g. 'llama'/'api'/'transcribe')
    - config_dict:  {name: config_path} (service config file paths)
    """

    def __init__(self, backend_dict: Optional[dict] = None, config_dict: Optional[dict] = None):
        super().__init__(backend_dict or {}, config_dict or {})
        self._sinks: dict[str, object] = {}
        self._service_callbacks = []   # callback(status: dict)
        self._task_callbacks = []      # callback(snapshot: TaskSnapshot)
        # Incremental runtime-log tracking (used by _on_queue_event, prevents
        # high-frequency progress callbacks from spamming the log)
        self._log_current: dict[str, str] = {}        # name -> current executing task id
        self._log_seen_finished: dict[str, set] = {}  # name -> set of already-logged finished task ids

    # ── UI sink wiring (new) ──

    def register_ui_sink(self, name: str, sink) -> None:
        """Registered when UI pages are constructed; overriding registration prevents leaks; supports both timings."""
        self._sinks[name] = sink
        self._inject(name, sink)

    def _before_service_start(self, name: str, service) -> None:
        """Core startup hook: ensure log/status callbacks are wired during service.start().

        On first start the service is only created inside ``super().start_service``;
        if not injected here, the ``_log`` inside ``service.start()`` (e.g. GSV
        weight-loading logs) would degrade to print and be lost.
        """
        sink = self._sinks.get(name)
        if sink is not None:
            self._inject(name, sink)

    def _inject(self, name: str, sink):
        service = self._service_dic.get(name)
        if service is not None:
            service.set_on_status_change(
                lambda running, device=None, s=sink, n=name: self._on_service_event(n, s, bool(running), device)
            )
            # Diagnostic log wiring: core service/executor _log → AppLog (with service-name prefix)
            service.set_on_log(
                lambda level, msg, n=name: log.record(level, f"[{n}] {msg}")
            )
        queue = self._queue_dic.get(name)
        if queue is not None:
            queue.set_on_status_change(
                lambda info, s=sink, n=name: self._on_queue_event(n, s, info)
            )

    def _on_service_event(self, name: str, sink, running: bool, device: Optional[str] = None):
        if sink is not None:
            sink.on_service_status(bool(running), False, device)
        self._broadcast_service(name, running)
        if running and device:
            log.record("info", f"服务 {name} 在线（设备: {device}）")
        else:
            log.record("info", f"服务 {name} {'在线' if running else '离线'}")

    def _broadcast_service(self, name: str, running: bool):
        """Broadcast service status dict to registered callbacks (does not depend on a UI sink)."""
        for cb in list(self._service_callbacks):
            try:
                cb({"status": "running" if running else "stopped", "service_type": name})
            except Exception:
                pass

    def _on_queue_event(self, name: str, sink, info):
        current, pending, finished = info
        if sink is not None:
            sink.on_tasks(
                _project(current) if current is not None else None,
                [_project(t) for t in pending],
            )
            fn = getattr(sink, "on_finished_tasks", None)
            if fn is not None:
                fn([_project(t) for t in finished])
        # Broadcast the current task snapshot to registered callbacks on task change
        for cb in list(self._task_callbacks):
            try:
                cb(_to_snapshot(current) if current is not None else None)
            except Exception:
                pass
        # ── Runtime log: task started / completed / failed / cancelled (incremental tracking prevents duplicates) ──
        cur = _project(current) if current is not None else None
        cur_id = cur["id"] if cur else None
        if self._log_current.get(name) != cur_id:
            if cur_id:
                log.record("info", f"[{name}] 开始执行任务: {cur.get('file_name') or '?'} (id={cur_id})")
            self._log_current[name] = cur_id
        seen = self._log_seen_finished.setdefault(name, set())
        _STATUS_LABEL = {"completed": "完成", "failed": "失败", "cancelled": "取消"}
        for t in finished:
            tid = t.id
            if tid not in seen:
                seen.add(tid)
                st = t.status.value if hasattr(t.status, "value") else str(t.status)
                label = _STATUS_LABEL.get(st, st)
                if st == "failed":
                    detail = f": {t.error}" if getattr(t, "error", None) else ""
                    log.record("error", f"[{name}] 任务{label}: {t.file_name} (id={tid}){detail}")
                else:
                    log.record("info", f"[{name}] 任务{label}: {t.file_name} (id={tid})")

    def start_service(self, name: str, backend=None, config_path: Optional[Path] = None):
        cfg = config_path or self._config_dict.get(name) or "未指定"
        log.record("info", f"启动服务: {name}（配置: {cfg}）")   # E1: service-starting event
        sink = self._sinks.get(name)
        was_absent = name not in self._service_dic
        if sink is not None and not was_absent:
            # Restart/reload: inject before start to capture status pushes from service.start()
            self._inject(name, sink)
        try:
            super().start_service(name, backend, config_path)
        except Exception as ex:
            # E2: start failure does not emit (_running not set to True) → record error here + resend offline status
            log.record("error", f"服务 {name} 启动失败（配置: {cfg}）: {ex}")
            if sink is not None:
                sink.on_service_status(False, False)
            self._broadcast_service(name, False)
            raise
        # Wire queues without a UI sink too (their events still feed the broadcast callbacks)
        queue = self._queue_dic.get(name)
        if sink is None and queue is not None and queue._on_status_change is None:
            queue.set_on_status_change(
                lambda info, n=name: self._on_queue_event(n, None, info)
            )
        if sink is not None:
            self._inject(name, sink)   # re-inject (newly created service/queue)
        if was_absent:
            # First start: service.start() emits before injection and would be lost —
            # this was fixed by the _before_service_start hook. Re-emit only when
            # there is no UI sink (no early injection).
            if sink is None:
                self._on_service_event(name, sink, True, self.get_service_device(name))

    # ── Queue control (E3: pause/resume status changes are logged) ──

    def pause_queue(self, name: str):
        log.record("info", f"暂停队列: {name}")
        return super().pause_queue(name)

    def resume_queue(self, name: str):
        log.record("info", f"恢复队列: {name}")
        return super().resume_queue(name)

    def clear_queue(self, name: str):
        log.record("info", f"清空队列: {name}")
        return super().clear_queue(name)

    # ── Task submission (overrides core to return task_id) ──

    def stop_service(self, name: str, cancel_current: bool = False):
        """Stop the service. *cancel_current* is for heavy-engine backends like GSV:
        cancel the running task before stopping (pending tasks cannot resume after the
        engine is released; see app-integration-design §2.3-9)."""
        if cancel_current:
            queue = self._queue_dic.get(name)
            current = queue.get_current_task() if queue is not None else None
            if current is not None:
                queue.cancel(current.id)   # running → executor checkpoint raises CancelledError
        log.record("info", f"停止服务: {name}")
        super().stop_service(name)
        # Re-emit stop status: a duplicate is harmless if service.stop() already
        # emitted (idempotent); ensures the broadcast still reaches consumers when
        # nothing was injected (no-sink case).
        sink = self._sinks.get(name)
        if sink is not None:
            sink.on_service_status(False, False)
        self._broadcast_service(name, False)

    def switch_service_config(self, name: str, config_path: Path):
        """Restart the service with a new config (GSV role switch): hot-swap S1/S2 while
        running (base model stays resident), otherwise full restart. Cancel running tasks
        before hot-swapping (consistent with the old stop+start cancel_current=True
        semantics, avoiding a race between running tasks and weight replacement)."""
        if name == "gsv":
            service = self._service_dic.get(name)
            if service is not None and getattr(service, "is_running", False):
                queue = self._queue_dic.get(name)
                current = queue.get_current_task() if queue is not None else None
                if current is not None:
                    queue.cancel(current.id)
                service.switch_role(config_path)
                return
        self.stop_service(name, cancel_current=True)
        return self.start_service(name, None, config_path)

    def submit_task(self, task_request) -> str:
        """Submit a task to its queue, returning task_id (the core version does not).

        When the service is not started (queue not created), lazily create the queue
        (no executor, worker not started, tasks queued but not executed) — submission
        is not limited by model loading, and staged tasks are visible in the queue
        (push/query/operations all work); core start_service reuses this queue,
        injecting an executor and starting it.
        """
        name = task_request.task_type
        queue = self._queue_dic.get(name)
        if queue is None:
            _, queue_cls = self._backend_dict[name]
            queue = queue_cls(None)          # lazily created: no executor, worker not started
            self._queue_dic[name] = queue
            sink = self._sinks.get(name)
            if sink is not None:
                queue.set_on_status_change(
                    lambda info, s=sink, n=name: self._on_queue_event(n, s, info)
                )
        tid = self.get_new_id()
        data = asdict(task_request)
        data["id"] = tid
        if data.get("payload") is None:
            data["payload"] = {}
        queue.add(Task(**data))
        src = getattr(task_request, "file_name", None) or getattr(task_request, "file_path", None) or "?"
        cfg = _fmt_task_configs(getattr(task_request, "configs", None))
        log.record("info", f"提交{name}任务: {src} (id={tid})（配置: {cfg}）")
        return tid

    def submit_transcription(self, request) -> str:
        """Submit a transcription task, returning task_id."""
        return self.submit_task(request)

    # ── Queue operation convenience methods (single task_id iterating queues) ──

    def _queues(self):
        return list(self._queue_dic.values())

    def _find_task(self, task_id: str):
        """Find a task by id across all queues (get_all() dicts, including result)."""
        for q in self._queues():
            for t in q.get_all():
                if t.get("id") == task_id:
                    return t
        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task: remove if pending / send a cancel signal if running."""
        for q in self._queues():
            if q.cancel(task_id):
                log.record("info", f"取消任务: {task_id}")
                return True
        return False

    def remove_task_by_id(self, task_id: str) -> bool:
        """Remove a pending task from the queue."""
        for q in self._queues():
            if q.remove(task_id):
                log.record("info", f"移除待执行任务: {task_id}")
                return True
        return False

    def reorder_task(self, task_id: str, new_index: int) -> bool:
        """Move a pending task to the given position (overrides the core snapshot-based signature)."""
        for q in self._queues():
            if q.reorder(task_id, new_index):
                log.record("info", f"重排任务: {task_id} → 位置 {new_index}")
                return True
        return False

    def clear_completed_task(self, task_id: str) -> None:
        """Remove a single task from the finished list."""
        for q in self._queues():
            lst = q.get_finished_tasks()
            if any(t.id == task_id for t in lst):
                q.get_finished_tasks()[:] = [t for t in lst if t.id != task_id]
                log.record("info", f"清除已完成任务: {task_id}")

    def clear_all_completed(self) -> None:
        """Clear finished (completed/failed/cancelled) tasks across all queues."""
        for q in self._queues():
            q.get_finished_tasks().clear()
        log.record("info", "清空全部已完成任务")

    def export_task(self, task_id: str, output_path: Path, fmt: Optional[str] = None) -> None:
        """Export a task result to a file; the suffix determines the format (three tiers).

        - When fmt is omitted, infer from the suffix:
          .json → json snapshot; .lrc/.srt/.vtt → subtitles (segments from result);
          any other suffixed file (e.g. .ERB/.unknown) → generic txt content written
          **keeping the original suffix** (no append, no replace); no suffix → append .txt.
        - An explicit fmt takes precedence (does not depend on the suffix).
        - txt: write result directly if str, readable JSON if dict; directories auto-created.
        """
        t = self._find_task(task_id)
        if t is None or t.get("result") is None:
            log.record("warn", f"导出任务失败: {task_id} 无结果")
            return
        log.record("info", f"导出任务: {t.get('file_name')} (id={task_id}) → {output_path}")
        output_path = Path(output_path)
        if fmt is None:
            fmt = {
                ".json": "json", ".lrc": "lrc", ".srt": "srt", ".vtt": "vtt",
                ".wav": "wav",
            }.get(output_path.suffix.lower(), "txt")
        # Unmapped suffixes fall back to txt: keep the original suffix; append .txt if none
        if fmt == "txt" and not output_path.suffix:
            output_path = output_path.with_suffix(".txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = t["result"]

        if fmt == "wav":
            # GSV synthesis result {audio_path, ...} → copy the audio file (not written as text)
            audio_path = result.get("audio_path") if isinstance(result, dict) else None
            if audio_path and Path(audio_path).is_file():
                import shutil
                shutil.copy2(audio_path, output_path)
                log.record("info", f"音频已导出: {output_path}")
                return str(output_path)
            log.record("warn", f"导出音频失败: {task_id} 无 audio_path")
            return None

        if fmt == "json":
            payload = {
                "id": t.get("id"),
                "type": t.get("type"),
                "status": t.get("status"),
                "file_name": t.get("file_name"),
                "result": result,
                "segments": result.get("segments", []) if isinstance(result, dict) else None,
                "error": t.get("error"),
                "created_at": t.get("created_at"),
            }
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            output_path.write_text(text, encoding="utf-8")
            return

        if fmt in ("lrc", "srt", "vtt"):
            segments = result.get("segments", []) if isinstance(result, dict) else []
            if segments:
                from core.writer import lrc_write, srt_write, vtt_write
                {"lrc": lrc_write, "srt": srt_write, "vtt": vtt_write}[fmt](segments, str(output_path))
                return
            # No segments: fall back to writing the result text
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            output_path.write_text(text, encoding="utf-8")
            return

        # txt (covers unknown-suffix fallback)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        output_path.write_text(text, encoding="utf-8")

    # ── Compatibility queries (used by transcribe / completed pages in transition) ──

    def _all_tasks(self, name: Optional[str] = None):
        """Collect all Task objects across queues (current + pending + finished)."""
        out = []
        for qname, q in self._queue_dic.items():
            if name is not None and qname != name:
                continue
            cur = q.get_current_task()
            if cur is not None:
                out.append(cur)
            out.extend(q.get_pending_tasks())
            out.extend(q.get_finished_tasks())
        return out

    def get_service_status(self, service_type: str) -> dict:
        """Return {"status": "online|offline", "service_type": ...}."""
        if service_type not in self._backend_dict:
            return {"status": "offline", "service_type": service_type}
        try:
            snap = self.get_status(service_type)
        except Exception:
            return {"status": "offline", "service_type": service_type}
        return {
            "status": "online" if snap.service_status == "running" else "offline",
            "service_type": service_type,
        }

    def get_service_device(self, service_type: str) -> Optional[str]:
        """Return the service's actual working device (cuda/cpu/api/None) for UI status-bar coloring."""
        service = self._service_dic.get(service_type)
        if service is None:
            return None
        device = getattr(service, "actual_device", None)
        if callable(device):
            device = device()
        return device

    def list_tasks(self, task_type: Optional[str] = None) -> list[TaskSnapshot]:
        return [_to_snapshot(t) for t in self._all_tasks(task_type)]

    def list_waiting_tasks(self, task_type: Optional[str] = None) -> list[TaskSnapshot]:
        return [_to_snapshot(t) for t in self._all_tasks(task_type)
                if getattr(t.status, "value", None) == "pending"]

    def list_current_task(self, task_type: Optional[str] = None) -> Optional[TaskSnapshot]:
        for t in self._all_tasks(task_type):
            if (getattr(t.status, "value", None) or t.status) == "running":
                return _to_snapshot(t)
        return None

    def list_completed_tasks(self, task_type: Optional[str] = None) -> list[TaskSnapshot]:
        return [_to_snapshot(t) for t in self._all_tasks(task_type)
                if getattr(t.status, "value", None) in ("completed", "failed", "cancelled")]

    def _on_service_change(self, callback) -> None:
        self._service_callbacks.append(callback)

    def _on_task_change(self, callback) -> None:
        self._task_callbacks.append(callback)

    # ── Shutdown ──

    def shutdown(self) -> None:
        super().shutdown()
