"""AppFacade — CoreFacade 子类：UI sink 推送接线 + 队列操作便捷方法 + 旧存根 API 兼容层。

对接设计见 PLANS/appfacade-integration.md。

- core 零 flet 依赖：sink 为鸭子类型（on_service_status / on_tasks / on_finished_tasks），
  由 UI 侧 PageUiSink 经 ``page.run_thread`` 归队主线程。
- 旧存根 API（list_*/cancel_task(tid)/reorder_task(tid,i)/export_task(tid,out)/
  submit_transcription/_on_*_change 回调）保留为兼容层，供 transcribe / completed 页
  在 sink 化改造前过渡使用。
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


# 任务配置键 → 可读标签（提交任务日志展示实际使用的配置文件路径）
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
    """格式化任务实际使用的配置（task_request.configs），供提交日志展示。

    遍历 configs 中非 None 的项，输出「标签: 路径」列表；全空（含 None）
    时返回 "未指定"。与 start_service 日志展示的*服务*配置路径区分——
    提交日志应显示翻译参数/提示词/VAD 等任务配置文件的路径。
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
    """将 core 内部 Task 投影为 UI 可渲染 dict（translate 页 task_card 消费）。

    - type = task.task_type（任务类型即服务名）
    - status = TaskStatus enum → .value 字符串
    - input_summary = file_name（可读文件名展示）
    - payload：Task.payload（进度详情，如转写 pos/total/speed）透传
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
    """将 core Task 转为 core TaskSnapshot（旧页面属性访问兼容 snap.status/snap.result）。"""
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
    """UI 侧 sink 包装：后端线程回调 → page.run_thread 主线程控件更新。

    - TranslatePage 不管理已完成任务：on_finished_tasks 按鸭子类型检查 impl
      是否实现 update_finished_tasks，未实现则跳过。
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
        """完成事件转发（转写页预览刷新 / TTS 页结果刷新）。"""
        fn = getattr(self._impl, "update_finished_tasks", None)
        if fn is None:
            return
        if self._page is not None:
            self._page.run_thread(lambda: fn(tasks))
        else:
            fn(tasks)


class AppFacade(Facade):
    """统一 facade：core 服务/队列管理 + UI sink 推送 + 旧存根 API 兼容层。

    构造参数与 core Facade 一致：
    - backend_dict: {name: (service_cls, queue_cls)}（注册服务 key，如 'llama'/'api'/'transcribe'）
    - config_dict:  {name: config_path}（服务配置文件路径）
    """

    def __init__(self, backend_dict: Optional[dict] = None, config_dict: Optional[dict] = None):
        super().__init__(backend_dict or {}, config_dict or {})
        self._sinks: dict[str, object] = {}
        self._service_callbacks = []   # 旧 API 兼容：callback(status: dict)
        self._task_callbacks = []      # 旧 API 兼容：callback(snapshot: TaskSnapshot)
        # 运行日志增量跟踪（_on_queue_event 用，防进度高频回调刷屏）
        self._log_current: dict[str, str] = {}        # name -> 当前执行任务 id
        self._log_seen_finished: dict[str, set] = {}  # name -> 已记录完成的任务 id 集合

    # ════════════════════════════════════════════════════
    #  UI sink 接线（新）
    # ════════════════════════════════════════════════════

    def register_ui_sink(self, name: str, sink) -> None:
        """UI 页面构造时注册；覆盖注册防泄漏；支持两种时序。"""
        self._sinks[name] = sink
        self._inject(name, sink)

    def _before_service_start(self, name: str, service) -> None:
        """core 启动钩子：确保 service.start() 期间日志/状态回调已接线。

        首次启动时 service 在 ``super().start_service`` 内部才创建，若不在此
        注入，``service.start()`` 里的 ``_log``（如 GSV 权重加载日志）会退化为
        print 而丢失。
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
            # 诊断日志接线：core service/executor 的 _log → AppLog（带服务名前缀）
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
        """旧回调兼容：广播服务状态 dict（不依赖 UI sink）。"""
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
        # 旧回调兼容：任务变化广播当前任务快照
        for cb in list(self._task_callbacks):
            try:
                cb(_to_snapshot(current) if current is not None else None)
            except Exception:
                pass
        # ── 运行日志：任务开始执行 / 完成 / 失败 / 取消（增量跟踪防重复）──
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
        log.record("info", f"启动服务: {name}（配置: {cfg}）")   # E1：启动中事件
        sink = self._sinks.get(name)
        was_absent = name not in self._service_dic
        if sink is not None and not was_absent:
            # 重启/重复加载：start 前注入，捕获 service.start() 的状态推送
            self._inject(name, sink)
        try:
            super().start_service(name, backend, config_path)
        except Exception as ex:
            # E2：启动失败不 emit（_running 不置 True）→ 此处补记 error + 补发离线状态
            log.record("error", f"服务 {name} 启动失败（配置: {cfg}）: {ex}")
            if sink is not None:
                sink.on_service_status(False, False)
            self._broadcast_service(name, False)
            raise
        # 无 UI sink 的队列也接线（旧回调兼容依赖队列事件）
        queue = self._queue_dic.get(name)
        if sink is None and queue is not None and queue._on_status_change is None:
            queue.set_on_status_change(
                lambda info, n=name: self._on_queue_event(n, None, info)
            )
        if sink is not None:
            self._inject(name, sink)   # 补注入（新创建的 service/queue）
        if was_absent:
            # 首次启动：service.start() 的 emit 发生在注入之前已丢失——该问题已由
            # _before_service_start 钩子修复。仅当无 UI sink（未提前注入）时才补发。
            if sink is None:
                self._on_service_event(name, sink, True, self.get_service_device(name))

    # ════════════════════════════════════════════════════
    #  队列控制（E3：暂停/恢复状态变化记录日志）
    # ════════════════════════════════════════════════════

    def pause_queue(self, name: str):
        log.record("info", f"暂停队列: {name}")
        return super().pause_queue(name)

    def resume_queue(self, name: str):
        log.record("info", f"恢复队列: {name}")
        return super().resume_queue(name)

    def clear_queue(self, name: str):
        log.record("info", f"清空队列: {name}")
        return super().clear_queue(name)

    # ════════════════════════════════════════════════════
    #  任务提交（覆盖 core 版以返回 task_id）
    # ════════════════════════════════════════════════════

    def stop_service(self, name: str, cancel_current: bool = False):
        """停止服务。*cancel_current* 用于 GSV 等重引擎后端：停止前先取消
        运行中任务（引擎释放后挂起任务不可续跑，见 app-integration-design §2.3-9）。"""
        if cancel_current:
            queue = self._queue_dic.get(name)
            current = queue.get_current_task() if queue is not None else None
            if current is not None:
                queue.cancel(current.id)   # running → 执行体检查点抛 CancelledError
        log.record("info", f"停止服务: {name}")
        super().stop_service(name)
        # 补发停止状态：service.stop() 的 emit 若已捕获则重复无害（幂等）；
        # 未注入（无 sink 旧回调场景）时确保广播仍送达
        sink = self._sinks.get(name)
        if sink is not None:
            sink.on_service_status(False, False)
        self._broadcast_service(name, False)

    def switch_service_config(self, name: str, config_path: Path):
        """换配置重启服务（GSV 换角色）：运行中热切换 S1/S2（基础模型常驻），
        未运行/已释放则全量重启。热切换前取消运行中任务（与旧 stop+start 的
        cancel_current=True 语义一致，防运行中任务与权重替换竞态）。"""
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
        """提交任务到对应队列，返回 task_id（core 版不返回）。

        服务未启动（队列未创建）时懒创建队列（无 executor、不 start worker，
        任务排队不执行）——提交不受模型是否加载限制，且暂存任务在队列中可见
        （推送/查询/操作全通）；core start_service 会复用该队列注入 executor 并 start。
        """
        name = task_request.task_type
        queue = self._queue_dic.get(name)
        if queue is None:
            _, queue_cls = self._backend_dict[name]
            queue = queue_cls(None)          # 懒创建：无 executor，worker 不启动
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
        """提交转写任务（兼容旧存根 API），返回 task_id。"""
        return self.submit_task(request)

    # ════════════════════════════════════════════════════
    #  队列操作便捷方法（兼容旧存根签名：单 task_id 遍历队列）
    # ════════════════════════════════════════════════════

    def _queues(self):
        return list(self._queue_dic.values())

    def _find_task(self, task_id: str):
        """在全部队列中按 id 查找任务（get_all() dict，含 result）。"""
        for q in self._queues():
            for t in q.get_all():
                if t.get("id") == task_id:
                    return t
        return None

    def cancel_task(self, task_id: str) -> bool:
        """取消任务：pending 移除 / running 发取消信号。"""
        for q in self._queues():
            if q.cancel(task_id):
                log.record("info", f"取消任务: {task_id}")
                return True
        return False

    def remove_task_by_id(self, task_id: str) -> bool:
        """从队列移除 pending 任务。"""
        for q in self._queues():
            if q.remove(task_id):
                log.record("info", f"移除待执行任务: {task_id}")
                return True
        return False

    def reorder_task(self, task_id: str, new_index: int) -> bool:
        """将 pending 任务移到指定位置（覆盖 core snapshot 版签名）。"""
        for q in self._queues():
            if q.reorder(task_id, new_index):
                log.record("info", f"重排任务: {task_id} → 位置 {new_index}")
                return True
        return False

    def clear_completed_task(self, task_id: str) -> None:
        """从已完成列表移除单个任务。"""
        for q in self._queues():
            lst = q.get_finished_tasks()
            if any(t.id == task_id for t in lst):
                q.get_finished_tasks()[:] = [t for t in lst if t.id != task_id]
                log.record("info", f"清除已完成任务: {task_id}")

    def clear_all_completed(self) -> None:
        """清空全部队列的已完成（completed/failed/cancelled）任务。"""
        for q in self._queues():
            q.get_finished_tasks().clear()
        log.record("info", "清空全部已完成任务")

    def export_task(self, task_id: str, output_path: Path, fmt: Optional[str] = None) -> None:
        """导出任务结果到文件，后缀决定保存格式（三档）。

        - fmt 缺省时按后缀判断：
          .json → json 快照；.lrc/.srt/.vtt → 字幕（从 result 取 segments）；
          未在映射中的有后缀文件（如 .ERB/.unknown）→ 一般 txt 内容写入，
          **保存为原本的后缀**（不补、不换）；无后缀文件 → 补 .txt。
        - 显式传 fmt 则优先使用（不依赖后缀）。
        - txt：result 为 str 直接写，dict 写可读 JSON；目录自动创建。
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
        # 未映射后缀落为 txt：保留原本后缀；无后缀补 .txt
        if fmt == "txt" and not output_path.suffix:
            output_path = output_path.with_suffix(".txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = t["result"]

        if fmt == "wav":
            # GSV 合成结果 {audio_path, ...} → 复制音频文件（不按文本写出）
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
            # 无 segments：写 result 文本兜底
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            output_path.write_text(text, encoding="utf-8")
            return

        # txt（含未知后缀兜底）
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        output_path.write_text(text, encoding="utf-8")

    # ════════════════════════════════════════════════════
    #  兼容查询（旧存根 API；transcribe / completed 页过渡）
    # ════════════════════════════════════════════════════

    def _all_tasks(self, name: Optional[str] = None):
        """遍历队列收集全部 Task 对象（current + pending + finished）。"""
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
        """返回 {"status": "online|offline", "service_type": ...}（兼容旧存根）。"""
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
        """返回服务实际工作设备标识（cuda/cpu/api/None），供 UI 状态栏着色显示。"""
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

    # ════════════════════════════════════════════════════
    #  旧回调注册兼容（sink 化改造后移除）
    # ════════════════════════════════════════════════════

    def _on_service_change(self, callback) -> None:
        """注册旧式服务状态回调；callback(status: dict)。"""
        self._service_callbacks.append(callback)

    def _on_task_change(self, callback) -> None:
        """注册旧式任务回调；callback(snapshot: TaskSnapshot)。"""
        self._task_callbacks.append(callback)

    # ════════════════════════════════════════════════════
    #  关闭
    # ════════════════════════════════════════════════════

    def shutdown(self) -> None:
        super().shutdown()
