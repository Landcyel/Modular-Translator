"""Resident MOSS subprocess form (backup): web_cli subprocess + HTTP client.

Once the MOSS libraries are installed in the main environment, spawn a resident
``moss_transcribe_diarize.app.web_cli`` subprocess with the main python and submit
transcription tasks via the REST API (process-level isolation, hot restartable).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from app.paths import project_root
from ..contracts import CancelledError
from ..executor import Executor
from ..service import Service
from .audio_utils import probe_duration
from .moss_service import resolve_model_path
from .speaker_utils import force_single_speaker

# Timeout for subprocess readiness probing (model loading can take a while)
PREPARE_TIMEOUT_SEC = 600.0
POLL_INTERVAL = 0.5


class MossSubprocessService(Service):
    """Manage a MOSS web server subprocess and provide a MossHttpTranscriber.

    Usage::
        svc = MossSubprocessService(model_config)
        svc.start()          # Popen web_cli → poll /api/runtime for readiness
        tx = svc.get_executor()
        ...
        svc.stop()           # terminate → wait(5) → kill

    Config highlights::
        python              interpreter path (default sys.executable, i.e. the main environment)
        model_path          model directory (relative to project root or absolute)
        runs_dir            task output directory (default runs/moss, relative to project root)
        host / port         default 127.0.0.1 / 7861
        device / dtype      default auto / bf16
        prompt              single-speaker prompt (defaults in configs/models/moss_subprocess.json)
        single_speaker      result-side speaker normalization (default true)
    """

    def __init__(self, config: dict, on_status_change: Optional[Callable] = None):
        super().__init__(config, on_status_change)
        self._proc: Optional[subprocess.Popen] = None
        self._executor = None
        self._resolve_config(config)

    def _resolve_config(self, config) -> dict:
        cfg = Service._resolve_config(config)
        model_path = cfg.get("model_path")
        if model_path:
            cfg = {**cfg, "model_path": resolve_model_path(model_path)}
        runs_dir = cfg.get("runs_dir")
        if runs_dir and not os.path.isabs(str(runs_dir)):
            cfg = {**cfg, "runs_dir": str(Path(project_root) / str(runs_dir))}
        self._config = cfg
        return cfg

    @property
    def base_url(self) -> str:
        cfg = self._config
        return f"http://{cfg.get('host', '127.0.0.1')}:{cfg.get('port', 7861)}"

    def start(self):
        if self._running:
            return
        cfg = self._config
        cmd = [
            cfg.get("python") or sys.executable,
            "-m", "moss_transcribe_diarize.app.web_cli",
            "--model", cfg["model_path"],
            "--runs-dir", cfg.get("runs_dir", str(Path(project_root) / "runs" / "moss")),
            "--host", cfg.get("host", "127.0.0.1"),
            "--port", str(cfg.get("port", 7861)),
            "--device", cfg.get("device", "auto"),
            "--dtype", cfg.get("dtype", "bf16"),
            "--max-new-tokens", str(cfg.get("max_new_tokens", 65536)),
            "--max-len", str(cfg.get("max_len", 131072)),
            "--decoding", cfg.get("decoding", "greedy"),
        ]
        prompt = cfg.get("prompt")
        if prompt:
            cmd += ["--prompt", prompt]
        self._log("info", f"启动 MOSS 子进程: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        threading.Thread(target=self._drain_output, daemon=True).start()
        self._wait_for_preparing()
        self._executor = MossHttpTranscriber(self.base_url, defaults=cfg)
        self._executor.set_on_log(self._log)
        self._running = True
        self._emit()

    def _drain_output(self):
        """Forward the subprocess stdout to the service log (background thread)."""
        for line in self._proc.stdout:
            self._log("info", line.rstrip())

    def _wait_for_preparing(self, timeout: float = PREPARE_TIMEOUT_SEC):
        """Poll GET /api/runtime until the subprocess service is ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"MOSS 子进程提前退出, code={self._proc.returncode}")
            try:
                resp = requests.get(f"{self.base_url}/api/runtime", timeout=2)
                if resp.status_code == 200:
                    self._log("info", f"MOSS 服务就绪: {self.base_url}")
                    return
            except Exception:
                pass
            time.sleep(POLL_INTERVAL)
        raise RuntimeError(f"MOSS 服务 {timeout:.0f}s 内未就绪")

    def stop(self):
        if not self._running:
            return
        self._executor = None
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._running = False
        self._emit()

    def restart(self, config: dict):
        """Stop the subprocess (if any), apply *config*, and start again."""
        if self._running:
            self.stop()
        self._resolve_config(config)
        self.start()

    def get_executor(self):
        if self._executor is None:
            raise RuntimeError("MossSubprocessService not started. Call start() first.")
        return self._executor


class MossHttpTranscriber(Executor):
    """HTTP client: multipart upload → poll task status → fetch segments.

    Terminal status handling: both ``waiting_review`` (transcription complete, pending review)
    and ``done`` are treated as complete.
    """

    def __init__(self, base_url: str, defaults: Optional[dict] = None):
        self.base_url = base_url
        self.defaults = defaults or {}

    def execute(
        self,
        task,
        progress_callback: Optional[Callable[[float, float, float, Optional[dict]], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> dict:
        """Run transcription from *task* (contract same as MossTranscriber)."""
        cfg = self._resolve_task(task)
        args = cfg.get("transcribe_config") or {}
        merged = {**self.defaults, **args}
        audio_path = cfg["audio_path"]
        data = {}
        for key in ("prompt", "max_new_tokens", "max_len", "decoding", "temperature", "top_p", "top_k"):
            val = merged.get(key)
            if val is not None:
                data[key] = val
        with open(audio_path, "rb") as fh:
            resp = requests.post(
                f"{self.base_url}/api/jobs",
                files={"file": (Path(audio_path).name, fh, "application/octet-stream")},
                data=data, timeout=60,
            )
        resp.raise_for_status()
        job = resp.json()
        job_id = job["id"]
        duration = probe_duration(audio_path)
        try:
            detail = self._poll(job_id, progress_callback, cancel_event, task, duration)
            segs = requests.get(
                f"{self.base_url}/api/jobs/{job_id}/segments", timeout=30).json()
            segments = segs.get("segments", []) if isinstance(segs, dict) else segs
            if merged.get("single_speaker", False):
                segments = force_single_speaker(segments)
            info = {k: detail.get(k) for k in (
                "model", "prompt_len", "generated_tokens", "elapsed_sec",
                "inference", "usage", "files") if detail.get(k) is not None}
            # Final progress: consistent with the in-process form, progress topped at 100%
            # (the queue flips status only after execute() returns, so the bar is full at completion).
            if progress_callback is not None:
                payload = {"status": detail.get("status") or "done",
                           "generated_tokens": detail.get("generated_tokens")}
                if segments:
                    payload["segments"] = segments
                if duration:
                    progress_callback(duration, duration, None, payload)
                else:
                    payload["unit"] = "ratio"
                    progress_callback(1.0, 1.0, None, payload)
            return {"segments": segments, "info": info}
        except Exception:
            # Clean up the server-side task on cancel or failure
            try:
                requests.delete(f"{self.base_url}/api/jobs/{job_id}", timeout=10)
            except Exception:
                pass
            raise

    def _poll(self, job_id, progress_callback, cancel_event, task,
              duration: Optional[float] = None) -> dict:
        """Poll task status; on cancel → DELETE + CancelledError; return job detail at a terminal state.

        Progress callbacks align with the in-process MossTranscriber: with an audio
        duration, report on the timeline (pos/total in seconds); otherwise fall back
        to ratio semantics (unit=ratio).
        """
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError(task.id)
            resp = requests.get(f"{self.base_url}/api/jobs/{job_id}", timeout=10)
            resp.raise_for_status()
            detail = resp.json()
            status = detail.get("status")
            if status in ("failed", "cancelled"):
                raise RuntimeError(detail.get("error") or f"job {status}")
            if status in ("waiting_review", "done"):
                return detail
            if progress_callback:
                stage = float(detail.get("progress", 0.0))
                payload = dict(detail)
                if duration:
                    payload["unit"] = "seconds"
                    progress_callback(duration * stage, duration, None, payload)
                else:
                    payload["unit"] = "ratio"
                    progress_callback(stage, 1.0, None, payload)
            time.sleep(POLL_INTERVAL)

    def _resolve_task(self, task):
        """Transcription semantics: file_path as the audio path, configs["args"] providing params."""
        _source, configs = super()._resolve_task(task)
        args = configs.get("args")
        return {
            "audio_path": str(task.file_path),
            "transcribe_config": args if isinstance(args, dict) else None,
        }
