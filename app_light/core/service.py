"""
Service abstraction layer — unified lifecycle for translation & transcription backends.

Service owns the heavy resources (llama-server process, MOSS model, GSV engine) and
provides a ``get_executor()`` method that returns a ready-to-use worker.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union



from app.paths import project_root as _PROJECT_ROOT
from app.torch_runtime import ensure_available, describe


# ═══════════════════════════════════════════════════════════
# 抽象基类
# ═══════════════════════════════════════════════════════════

class Service(ABC):
    """Abstract backend service with start/stop lifecycle and executor access."""

    def __init__(self, config: dict, on_status_change: Optional[Callable] = None):
        self._config = config
        self._running = False
        self._on_status_change = on_status_change
        self._on_log = None  # 诊断日志回调（UI 接线；无回调退化 print）
        # 实际工作设备：'cpu' | 'cuda' | 'api'，仅在服务真正加载成功时赋值
        # （start() 内），stop()/加载失败时保持/重置为 None
        self.device: Optional[str] = None

    @abstractmethod
    def start(self):
        """Launch the backend (process / model) and prepare the executor."""
        ...

    @abstractmethod
    def stop(self):
        """Tear down the backend and release resources."""
        ...

    @abstractmethod
    def restart(self, config:dict):
        ...

    @abstractmethod
    def get_executor(self):
        """Return the ready-to-use executor instance (Translator or Transcriber)."""
        ...

    def _emit(self):
        if self._on_status_change:
            try:
                self._on_status_change(self._running, self.actual_device)
            except Exception:
                pass

    @property
    def actual_device(self) -> Optional[str]:
        """兼容别名：实际工作设备标识（cuda / cpu / api / None=未加载）。

        由 :attr:`device` 承载——子类在 start() 实际加载成功后赋值，
        stop() 重置为 None。
        """
        return self.device

    @staticmethod
    def _normalize_device(value) -> Optional[str]:
        """归一化设备值为三值域：``'cuda' | 'cpu' | 'api'``。

        - ``"cuda"`` / ``"cuda:0"`` / ``"cuda:1"`` → ``"cuda"``
        - ``"cpu"`` / ``"api"`` → 原样
        - ``"auto"`` → 按 torch 探测（cuda 可用则 cuda，否则 cpu）
        - None / 空 / 其它 → None
        """
        if value is None:
            return None
        v = str(value).strip().lower()
        if v.startswith("cuda"):
            return "cuda"
        if v in ("cpu", "api"):
            return v
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return None
        return None

    def set_on_status_change(self, callback: Optional[Callable]):
        """注入/替换状态变化回调（running, device）。"""
        self._on_status_change = callback

    def set_on_log(self, callback: Optional[Callable]):
        """注入/替换诊断日志回调（level, message）；无回调时退化 print。

        core 零 UI 依赖：回调由 AppFacade 接线到 AppLog；测试脚本/库调用
        不接线时保持原 print 行为。
        """
        self._on_log = callback

    def _log(self, level: str, message: str) -> None:
        """记录一条诊断日志（level: info/warn/error）。"""
        if self._on_log is not None:
            try:
                self._on_log(level, message)
                return
            except Exception:
                pass
        print(f"[{level}] {message}")

    @staticmethod
    def _resolve_config(config) -> dict:
        """Resolve a config value to a dict.

        Accepts a JSON file path (``Path`` or ``str``) which is loaded as UTF-8
        JSON, or an already-resolved ``dict`` which is returned as-is.
        """
        if isinstance(config, dict):
            return config
        if isinstance(config, (str, Path)):
            with open(Path(config), "r", encoding="utf-8") as fh:
                return json.load(fh)
        raise TypeError(
            f"config must be a dict, Path, or str, got {type(config).__name__}"
        )

    @property
    def is_running(self) -> bool:
        return self._running


# ═══════════════════════════════════════════════════════════
# LlamaService — manages llama-server process + Translator
# ═══════════════════════════════════════════════════════════

class LlamaService(Service):
    """Manage a llama-server subprocess and provide a Translator executor.

    Usage::

        svc = LlamaService(model_config)
        svc.start()
        translator = svc.get_executor()
        ...
        svc.stop()
    """

    def __init__(self, model_config: dict, on_status_change: Optional[Callable] = None):
        super().__init__(model_config, on_status_change)
        self._process: Optional[subprocess.Popen] = None
        self._output_thread: Optional[threading.Thread] = None
        self._executor = None

        # Resolve paths / args for llama-server
        self._llama_path: Optional[Path] = None
        self._server_args: dict = {}
        self._url: str = ""
        self._config_path = model_config if isinstance(model_config, (str, Path)) else None
        self._devices: Optional[list] = None   # 初次加载探测的 CUDA 设备列表（重载沿用）

        self._resolve_config(model_config)

    # ── Config resolution ─────────────────────────────

    def _resolve_config(self, config) -> dict:
        """Load a llama-server config and derive server launch settings.

        The dict resolved by :meth:`Service._resolve_config` (JSON file path or
        raw dict) is inspected for ``llama_path`` / ``server_arg``; the derived
        ``_llama_path``, ``_server_args`` and ``_url`` are stored on the
        instance.
        """
        cfg = super()._resolve_config(config)
        self._config = cfg
        # 空 llama_path 视为未配置（None）→ _start_llama_server 报"llama_path 未配置"；
        # 否则解析为项目根下的绝对路径。
        lp = cfg.get("llama_path", "")
        self._llama_path = None if not lp else (_PROJECT_ROOT / lp).resolve()
        self._server_args = cfg.get("server_arg", {})
        host = self._server_args.get("--host", "127.0.0.1")
        port = self._server_args.get("--port", "8080")
        self._url = f"http://{host}:{port}"
        return cfg

    # ── Service interface ─────────────────────────────

    def start(self):
        if self._running:
            return

        # 设备可用性（CUDA 列表）探测仅在初次加载执行——机器级事实，重载沿用
        # 初次结果，避免每次重载都拉起 llama-server --list-devices 探测（慢）
        if self._devices is None and self._config_path is not None:
            try:
                self._devices = LlamaService.get_device_list(self._config_path)
                if self._devices:
                    names = ", ".join(f"#{i} {n}" for i, n in self._devices)
                    self._log("info", f"llama CUDA 设备: {names}")
                else:
                    self._log("info", "llama 未探测到 CUDA 设备（-ngl auto 将不使用 GPU 层）")
            except Exception as ex:
                self._log("warn", f"llama 设备探测失败（视为无 CUDA，将使用 CPU）: {ex}")
                self._devices = []

        self._start_llama_server()
        from .executor import LlamaTranslator
        self._executor = LlamaTranslator(self._config)
        self._executor.set_on_log(self._log)  # 诊断日志透传（服务 → executor）
        try:
            self._executor._wait_for_preparing(timeout=120, is_print=True)
        except Exception as ex:
            self._log("error", f"llama 模型准备超时/失败: {ex}")
            raise

        self.device = self._resolve_llama_device()  # 实际加载成功后才赋值
        self._log("info", self._describe_actual_device())
        self._running = True
        self._emit()  # 状态推送：start 完成 → running=True

    def stop(self):
        if not self._running:
            return

        if self._process is not None:
            self._stop_llama_server()

        self._executor = None
        self._running = False
        self.device = None  # 服务已停止，设备不再有意义
        self._emit()  # 状态推送：stop 完成 → running=False

    def restart(self, config: dict):
        """Stop the running server (if any), apply *config*, and start again."""
        if self._running:
            self.stop()
        self._resolve_config(config)
        self.start()

    def get_executor(self):
        if self._executor is None:
            raise RuntimeError("LlamaService not started. Call start() first.")
        return self._executor

    # ── llama-server process management (mirrors LlamaServer) ──

    def _start_llama_server(self):
        """Launch the llama-server subprocess."""
        if self._llama_path is None:
            self._log("error", "llama-server 启动失败: llama_path 未配置")
            raise RuntimeError("llama_path not configured")

        server_exe = self._llama_path / "llama-server.exe"
        if not server_exe.exists():
            self._log("error", f"llama-server 启动失败: 未找到 {server_exe}")
            raise FileNotFoundError(f"llama-server.exe not found at {server_exe}")

        cmd = [str(server_exe)]
        for key, value in self._server_args.items():
            cmd.append(key)
            cmd.append(str(value))

        # Windows 下隐藏 llama-server 的控制台窗口（主程序 --noconsole 无控制台，
        # 控制台子进程否则会各自新建 cmd 窗口）；CREATE_NO_WINDOW 不影响
        # stdout=PIPE 日志接管与 terminate()/kill() 生命周期。
        popen_kwargs: dict = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                **popen_kwargs,
            )
        except Exception as ex:
            self._log("error", f"llama-server 进程启动失败: {ex}")
            raise
        # Daemon thread to drain stdout so the pipe doesn't block the child
        self._output_thread = threading.Thread(
            target=self._drain_output, daemon=True
        )
        self._output_thread.start()

    def _stop_llama_server(self):
        """Terminate (or kill) the llama-server subprocess."""
        if self._process is None:
            return
        if self._process.poll() is not None:
            # Already exited
            self._process = None
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

        self._process = None

    def _describe_actual_device(self) -> str:
        """服务加载完成后的实际工作设备描述（供 start() 记录日志）。"""
        server_args = self._server_args or {}
        if "--device" in server_args:
            return f"llama 实际工作设备: {server_args['--device']}（server_arg 显式指定）"
        if self._devices:
            names = ", ".join(f"#{i} {n}" for i, n in self._devices)
            return f"llama 实际工作设备: CUDA（{names}，-ngl auto）"
        if self._devices is None:
            return "llama 实际工作设备: 未知（未执行 CUDA 探测）"
        return "llama 实际工作设备: CPU（未探测到 CUDA，-ngl auto 不加载 GPU 层）"

    def _resolve_llama_device(self) -> Optional[str]:
        """加载完成后解析实际工作设备：显式 --device 优先，否则按 CUDA 探测结果。"""
        server_args = self._server_args or {}
        if "--device" in server_args:
            dv = str(server_args["--device"]).lower()
            return "cuda" if "cuda" in dv else "cpu" if "cpu" in dv else (dv or None)
        if self._devices:
            return "cuda"
        if self._devices is None:
            return None
        return "cpu"

    def _drain_output(self):
        """Continuously read subprocess stdout to prevent pipe buffer deadlock.

        同时把子进程输出转发到诊断日志：error/fail 行记 error、cuda 相关行记
        warn（如 CUDA 初始化失败），进程退出时记录退出码——原实现把服务器
        自身错误全部丢弃。
        """
        try:
            if self._process and self._process.stdout:
                for _line in self._process.stdout:
                    line = _line.rstrip("\n")
                    if not line:
                        continue
                    lower = line.lower()
                    if "cuda" in lower or "vulkan" in lower:
                        self._log("warn", f"llama-server: {line}")
                    elif "error" in lower or "fail" in lower:
                        self._log("error", f"llama-server: {line}")
                rc = self._process.poll()
                self._log("info", f"llama-server 进程退出，退出码: {rc}")
        except Exception:
            pass

    # ── Convenience ───────────────────────────────────

    @property
    def url(self) -> str:
        """Return the llama-server base URL."""
        return self._url

    @staticmethod
    def get_device_list(llama_config_path):
        """Query llama-server for available CUDA devices via ``--list-devices``."""
        from .utils import load_json_file
        import re
        import subprocess

        llama_path = _PROJECT_ROOT / load_json_file(llama_config_path).get("llama_path", "")
        llama_path = llama_path / "llama-server.exe"

        cmd = [str(llama_path), "--list-devices"]
        # 打包后主程序无控制台：隐藏探测子进程的 cmd 窗口（同 _start_llama_server）
        run_kwargs: dict = {}
        if os.name == "nt":
            run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **run_kwargs,
        )
        output = p.stdout.strip() or "No devices found"
        pattern = r"CUDA(\d+):\s+([^(]+?)\s*\("
        devices = []
        for mm in re.finditer(pattern, output):
            device_id = int(mm.group(1))
            device_name = mm.group(2).strip()
            devices.append((device_id, device_name))
        return devices


# ═══════════════════════════════════════════════════════════
# APIService — manages an OpenAI-compatible API Translator
# ═══════════════════════════════════════════════════════════

class APIService(Service):
    """Provide a Translator executor backed by an OpenAI-compatible cloud API.

    Unlike :class:`LlamaService`, no local process is managed — the executor
    talks to a remote endpoint configured in the model config.

    Usage::

        svc = APIService(model_config)
        svc.start()
        translator = svc.get_executor()
        ...
        svc.stop()
    """

    def __init__(self, model_config: dict, on_status_change: Optional[Callable] = None):
        super().__init__(model_config, on_status_change)
        self._executor = None

    # ── Config resolution ─────────────────────────────

    def _resolve_config(self, config) -> dict:
        """Load an API model config; kwargs pass through unchanged."""
        cfg = Service._resolve_config(config)
        self._config = cfg
        return cfg

    def start(self):
        if self._running:
            return

        from .executor import APITranslator
        executor = APITranslator(self._config)
        executor.set_on_log(self._log)  # 诊断日志透传（服务 → executor）
        # 启动即检测连通性（短超时）：失败抛 RuntimeError（_running 不置 True、不 emit，
        # 由 UI 层捕获并刷新为离线状态）；成功后置懒检测标志避免首次翻译重复检测
        ok, msg = executor.check_connection(timeout=10)
        if not ok:
            self._log("error", f"API 连通性检测失败: {msg}")
            raise RuntimeError(f"API 连通性检测失败：{msg}")
        executor._connection_checked = True
        self._executor = executor
        self.device = "api"  # 云端 API：无本地计算设备，连通即视为加载成功
        self._log("info", "API 实际工作设备: 云端 API（本地无计算设备）")
        self._running = True
        self._emit()  # 状态推送：start 完成 → running=True

    def stop(self):
        if not self._running:
            return

        self._executor = None
        self._running = False
        self.device = None
        self._emit()  # 状态推送：stop 完成 → running=False

    def restart(self, config: dict):
        """Stop (if running), apply *config*, and start again."""
        if self._running:
            self.stop()
        self._resolve_config(config)
        self.start()

    def get_executor(self):
        if self._executor is None:
            raise RuntimeError("APIService not started. Call start() first.")
        return self._executor


# ── Module-level forwards (kept for core.* import compatibility) ──
get_device_list = LlamaService.get_device_list


# ═══════════════════════════════════════════════════════════
# GsvService — GPT-SoVITS 文本合成（进程内模型型（重引擎服务生命周期模式））
# ═══════════════════════════════════════════════════════════

class GsvService(Service):
    """GPT-SoVITS 文本合成服务（包装 ``core/gsv.GsvEngine``，进程内常驻）。

    一个 Service 单元 = 一个角色配置（权重组合 + 默认参数）；``get_executor()``
    产出可复用 worker。三方案情绪复刻（single/aux/dual）由任务级 ``args`` 的
    ``ref_mode`` 表达，分发逻辑见 ``GsvTTSExecutor``（core/executor.py）。
    引擎内部已封装 vendor CWD / RLock 串行 / 3~10s 参考校验 / numpy 2.x 垫片，
    服务层只负责生命周期。
    """

    def __init__(self, model_config: dict, on_status_change=None):
        super().__init__(model_config, on_status_change)
        self._engine = None
        self._executor = None
        self._resolve_config(model_config)

    def _resolve_config(self, config) -> dict:
        cfg = Service._resolve_config(config)
        # 相对路径基于项目根解析（仿 LlamaService._resolve_config, service.py:137-155）
        for key in ("t2s_weights_path", "vits_weights_path",
                    "bert_base_path", "cnhuhbert_base_path", "sv_path"):
            if cfg.get(key):
                cfg[key] = str((_PROJECT_ROOT / cfg[key]).resolve())
        self._config = cfg
        return cfg

    def _log_weights_status(self) -> None:
        """GSV 服务启动后记录角色权重（S1/S2）是否实际加载。"""
        status = self._engine.weights_status()

        for prefix, specified, role_loaded, configured, used in (
            ("S1", status["t2s_specified"], status["t2s_role_loaded"],
             status["t2s_configured"], status["t2s_used"]),
            ("S2", status["vits_specified"], status["vits_role_loaded"],
             status["vits_configured"], status["vits_used"]),
        ):
            if not specified:
                self._log("info", f"GSV 未指定角色 {prefix} 权重，使用默认权重: {used}")
            elif role_loaded:
                self._log("info", f"GSV 角色 {prefix} 权重已加载: {used}")
            else:
                self._log(
                    "warn",
                    f"GSV 角色 {prefix} 权重未找到，已回退默认权重: {used}"
                    f"（配置: {configured}）",
                )

    def start(self):
        if self._running:
            return
        ensure_available()  # 运行时缺失时给出安装指引，而不是难懂的 ModuleNotFoundError
        from .gsv import GsvEngine  # 惰性导入（引擎 import 链重）

        self._log("info", "GSV 引擎加载中…（库导入+权重+模型构建，约 15-60s）")
        try:
            self._engine = GsvEngine(self._config)  # 构造即全量加载（10~20s）
        except Exception as ex:
            self._log("error", f"GSV 模型加载失败: {ex}")
            raise
        self.device = self._normalize_device(self._engine.device)  # 引擎加载成功后才赋值
        self._log_weights_status()
        from .executor import GsvTTSExecutor

        self._executor = GsvTTSExecutor(
            self._engine, defaults=self._config.get("defaults", {})
        )
        self._executor.set_on_log(self._log)
        self._running = True
        self._log(
            "info",
            f"GSV 实际工作设备: {self.device}（runtime={describe()}）",
        )
        self._emit()

    def stop(self):
        if not self._running:
            return
        self._log("info", "GSV 服务停止中…")
        self._executor = None
        if self._engine is not None:
            try:
                self._engine.release()  # del 模型引用 + empty_cache（幂等）
            except Exception:
                pass
            self._engine = None
        self._running = False
        self.device = None
        self._log("info", "GSV 服务已停止")
        self._emit()

    def restart(self, config):
        if self._running:
            self.stop()
        self._resolve_config(config)
        self.start()

    def switch_role(self, config):
        """在线切换角色：引擎运行中仅热切换 S1/S2（基础模型常驻，秒级）；
        服务未运行/已释放时回退全量重启（现状 10~20s）。

        调用方（app/facade.switch_service_config）负责先取消运行中任务。
        """
        cfg = self._resolve_config(config)   # 相对路径按项目根绝对化
        if self._engine is not None and not self._engine._released:
            self._engine.apply_role(cfg)
            self._log_weights_status()
            self._log(
                "info",
                "GSV 角色已热切换（S1/S2 权重重载，基础模型常驻）",
            )
            self._emit()
            return
        self.restart(cfg)

    def get_executor(self):
        if self._executor is None:
            raise RuntimeError("GsvService not started. Call start() first.")
        return self._executor

    @property
    def engine(self):
        """Direct access to the underlying GsvEngine (for advanced use)."""
        return self._engine
