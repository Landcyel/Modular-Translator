"""In-process MOSS-Transcribe-Diarize service (primary form, modeled on FasterWhisperService).

The MOSS libraries (``moss_transcribe_diarize`` + ``transformers>=5.6``) are installed with the main environment;
models live under ``dependencies/models/moss`` (local HF snapshot), and loading is deferred to ``start()``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

from app.paths import project_root
from app.torch_runtime import ensure_available, describe
from ..service import Service
from .moss_transcriber import MossTranscriber


def resolve_model_path(model_path) -> str:
    """Resolve a relative model path to an absolute path (a local directory under dependencies/models/).

    Absolute paths are returned as-is; relative paths try the project root and then
    ``dependencies/models/`` subdirectories, resolving on the first hit; if none
    match, the path is returned as-is (preserving the HF repo id usage).
    """
    if not model_path or os.path.isabs(str(model_path)):
        return str(model_path)
    for base in (Path(project_root), Path(project_root) / "dependencies" / "models"):
        rooted = base / str(model_path)
        if rooted.exists():
            return str(rooted.resolve())
    return str(model_path)


class MossService(Service):
    """Manage a MOSS-Transcribe-Diarize ModelRunner and provide a MossTranscriber.

    Usage::
        svc = MossService(model_config)   # dict or JSON file path
        svc.start()
        tx = svc.get_executor()
        ...
        svc.stop()

    Config highlights::
        model_path        model directory (relative to project root or absolute)
        device / dtype    default auto / bf16
        lazy_load         default true (lazy: model loads on the first transcription task);
                          false → start() loads immediately (lazy off, device/memory fixed early)
        prompt            transcription prompt (single-speaker recipe in configs/models/moss/default.json)
        single_speaker    result-side speaker normalization (default true)
        max_new_tokens / max_len / decoding   service-level transcription defaults
    """

    def __init__(self, config: dict, on_status_change: Optional[Callable] = None):
        super().__init__(config, on_status_change)
        self._runner = None
        self._executor = None
        self._resolve_config(config)

    def _resolve_config(self, config) -> dict:
        cfg = Service._resolve_config(config)
        model_path = cfg.get("model_path")
        if model_path:
            cfg = {**cfg, "model_path": resolve_model_path(model_path)}
        self._config = cfg
        return cfg

    def start(self):
        if self._running:
            return
        import time as _time
        _st_t0 = _time.perf_counter()  # start() origin (includes first import of torch/transformers)
        ensure_available()  # give install guidance on missing runtime instead of a cryptic ModuleNotFoundError
        from moss_transcribe_diarize.app.model_runner import ModelRunner  # lazy import

        cfg = self._config
        try:
            # Prefer the custom StreamingModelRunner (partial text → live segment preview,
            # matching Whisper preview logic); fall back to the vendor original when its
            # import/construction fails (no feature loss, preview just refreshes on completion).
            from .streaming_runner import StreamingModelRunner

            runner_cls = StreamingModelRunner
        except Exception as ex:
            self._log(
                "warning",
                f"MOSS 流式预览不可用，回退标准 ModelRunner: {ex}",
            )
            runner_cls = ModelRunner
        try:
            self._runner = runner_cls(
                cfg["model_path"],
                device=cfg.get("device", "auto"),
                dtype=cfg.get("dtype", "bf16"),
            )
        except Exception as ex:
            self._log("error", f"MOSS 模型加载失败: {ex}")
            raise
        # Lazy-load switch: lazy_load=false → start() loads the model immediately (lazy off).
        # vendor ModelRunner._ensure_loaded() is idempotent (skips when is_loaded),
        # sharing the same _lock for mutual exclusion with transcribe; load failure
        # propagates up (service not marked running).
        if not cfg.get("lazy_load", True):
            t0 = time.time()
            self._log("info", "MOSS 模型加载中…")
            try:
                with self._runner._lock:
                    self._runner._ensure_loaded()
            except Exception as ex:
                self._runner = None
                self._log("error", f"MOSS 模型加载失败: {ex}")
                raise
            self._log(
                "info",
                f"MOSS 模型已加载（lazy_load=false，模型用时 {time.time() - t0:.1f}s，"
                f"start 总含 import={time.time() - _st_t0:.1f}s，"
                f"实际设备={self._runner._device}，dtype={self._runner._dtype}）",
            )
        self._executor = MossTranscriber(
            self._runner, defaults=cfg, on_first_load=self._on_moss_first_load,
        )
        self._executor.set_on_log(self._log)   # pass diagnostic logs through (service → executor)
        self._running = True
        actual_device = getattr(self._runner, "_device", None)
        if actual_device is not None:
            self._log(
                "info",
                f"MOSS 模型已加载（runtime={describe()}，实际设备={actual_device}）",
            )
            # Model actually loaded: assign the real device
            self.device = self._normalize_device(actual_device)
        else:
            # ModelRunner is lazy: _device is only resolved in _ensure_loaded() of the first
            # transcription task (auto → cuda:0/cpu). At startup only config and runtime are
            # recorded; the real device is backfilled by MossTranscriber after the first
            # transcription, to avoid misleadingly showing None.
            self._log(
                "info",
                f"MOSS 服务已就绪（runtime={describe()}，"
                f"配置设备={cfg.get('device', 'auto')}，"
                f"dtype={cfg.get('dtype', 'bf16')}；模型将在首个转写任务时加载）",
            )
            # Lazy: resolve the expected device from config (auto → torch probe),
            # corrected to the real device by _on_moss_first_load after the first transcription
            self.device = self._normalize_device(cfg.get("device", "auto"))
        self._emit()

    def _on_moss_first_load(self, device, dtype):
        """After the first transcription (model actually loaded), correct the real device and push status."""
        resolved = self._normalize_device(device)
        if resolved is not None:
            self.device = resolved
        self._emit()

    def stop(self):
        if not self._running:
            return
        self._log("info", "MOSS 服务停止中…")
        self._executor = None
        self._runner = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._running = False
        self.device = None
        self._log("info", "MOSS 服务已停止")
        self._emit()

    def restart(self, config: dict):
        """Stop the loaded runner (if any), apply *config*, and start again."""
        if self._running:
            self.stop()
        self._resolve_config(config)
        self.start()

    def get_executor(self):
        if self._executor is None:
            raise RuntimeError("MossService not started. Call start() first.")
        return self._executor

    @property
    def model(self):
        """Direct access to the underlying ModelRunner (for advanced use)."""
        return self._runner
