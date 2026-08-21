"""MOSS-Transcribe-Diarize 进程内服务（主形态，仿 FasterWhisperService）。

MOSS 库（``moss_transcribe_diarize`` + ``transformers>=5.6``）随主环境安装；
模型位于 ``dependencies/models/moss``（HF 本地快照），加载延迟到 ``start()``。
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
    """相对模型路径解析为绝对路径（dependencies/models/ 下本地目录）。

    绝对路径原样返回；相对路径依次尝试项目根、``dependencies/models/``
    下目录，命中则解析；均未命中则原样返回（保留 HF repo id 的用法）。
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
        svc = MossService(model_config)   # dict 或 JSON 文件路径
        svc.start()
        tx = svc.get_executor()
        ...
        svc.stop()

    config 要点::
        model_path        模型目录（相对项目根或绝对路径）
        device / dtype    默认 auto / bf16
        lazy_load         默认 true（懒加载：首个转写任务时才加载模型）；
                          false → start() 即加载（关闭懒加载，设备/内存提前确定）
        prompt            转写提示词（单说话人配方见 configs/models/moss/default.json）
        single_speaker    结果侧说话人归一化（默认 true）
        max_new_tokens / max_len / decoding   服务级转写默认参数
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
        _st_t0 = _time.perf_counter()  # start 起点（含首次 import torch/transformers）
        ensure_available()  # 运行时缺失时给出安装指引，而不是难懂的 ModuleNotFoundError
        from moss_transcribe_diarize.app.model_runner import ModelRunner  # 延迟导入

        cfg = self._config
        try:
            # 优先使用自定义 StreamingModelRunner（部分文本 → 实时段预览，
            # 与 Whisper 预览逻辑一致）；其导入/构造失败时回退 vendor 原版
            # （功能不退化，仅运行中预览退化为完成后刷新）。
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
        # 懒加载开关：lazy_load=false → start() 即加载模型（关闭懒加载）。
        # vendor ModelRunner._ensure_loaded() 幂等（is_loaded 则跳过），
        # 持同一把 _lock 保证与 transcribe 互斥；加载失败向上抛（服务不置 running）。
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
        self._executor.set_on_log(self._log)   # 诊断日志透传（服务 → executor）
        self._running = True
        actual_device = getattr(self._runner, "_device", None)
        if actual_device is not None:
            self._log(
                "info",
                f"MOSS 模型已加载（runtime={describe()}，实际设备={actual_device}）",
            )
            # 模型已实际加载：按真实设备赋值
            self.device = self._normalize_device(actual_device)
        else:
            # ModelRunner 是懒加载：_device 在首个转写任务的 _ensure_loaded()
            # 中才解析（auto → cuda:0/cpu）。启动阶段只记录配置与运行时，
            # 真实设备由 MossTranscriber 首次转写完成后补记，避免误导为 None。
            self._log(
                "info",
                f"MOSS 服务已就绪（runtime={describe()}，"
                f"配置设备={cfg.get('device', 'auto')}，"
                f"dtype={cfg.get('dtype', 'bf16')}；模型将在首个转写任务时加载）",
            )
            # 懒加载：按配置解析预期设备（auto → torch 探测），
            # 首次转写后由 _on_moss_first_load 校正为真实设备
            self.device = self._normalize_device(cfg.get("device", "auto"))
        self._emit()

    def _on_moss_first_load(self, device, dtype):
        """首次转写（模型实际加载）后校正真实设备并推送状态。"""
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
