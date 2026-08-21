"""In-process MOSS transcription executor (calls ModelRunner directly, token-level progress/cancel + live segment preview).

The MOSS libraries (``moss_transcribe_diarize`` + ``transformers>=5.6``) are installed with the main environment;
this class follows the ``core.executor.Transcriber`` contract, and the four-arg progress
callback semantics align with ``TranscriptionTaskQueue`` (pos/total/speed/segs).

When ``StreamingModelRunner`` (SUPPORTS_PARTIAL_TEXT) is available, pos/total are
audio-timeline seconds and segs are confirmed subtitle segments in progress — fully
consistent with Whisper's progress bar and preview logic; older runners fall back
to token-ratio semantics (unit=ratio).

Long-audio protection (CUDA OOM): MOSS's Qwen3 decoder stuffs all audio tokens of a
whole clip into the sequence at once with full attention (12.5 token/s), so 20 minutes
of audio ≈ 16k tokens and the attention-score tensor alone ≈ 15 GiB — a 6 GB GPU will
definitely OOM. Chunking scheme (three layers of protection; see
PLANS/gsv-moss/plan-moss-long-audio-chunking.md for details):

1. **VRAM-budget window** (``vram_auto_fit``): solve the quadratic peak model
   ``peak ≈ W + C1·t + C2·t²`` against free VRAM to derive a safe single-window
   duration, clamped to ``[min_window_sec, max_audio_sec]`` — large-VRAM cards
   automatically widen the window, small ones shrink it, no manual tuning;
   ``max_audio_sec`` is the hard cap.
2. **Silence-aware boundary** (``silence_boundary``): within each candidate cut
   band ``[target−boundary_lookback_sec, target]``, pick the longest silence run
   (energy envelope, silence judged from 0.35s by default), so cuts land on natural
   pauses and never truncate normal speech; if the band has no silence, fall back
   to an arithmetic hard cut flagged as hard. Windows only shrink, never grow, and
   single-window duration is strictly ≤ the VRAM-budget value.
3. **Boundary text repair + runtime OOM retreat**: tail segments truncated at hard
   cuts are paired by text similarity with continuing head segments of the next
   window and stitched into complete segments; if a single-window transcription
   still OOMs, shrink the window (×0.7, floor 45s), replan the remaining audio,
   and retry.

After mapping per-window segments back to the global timeline, they are deduplicated
and merged; per-window ``max_new_tokens`` also converges with window duration to
prevent runaway generation.
"""
from __future__ import annotations

import difflib
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from app.paths import project_root
from ..contracts import CancelledError
from ..executor import Executor, _wait_paused
from ..writer import format_lrc_time
from .audio_utils import probe_duration
from .silence_probe import find_silence_cut, load_band_rms
from .speaker_utils import force_single_speaker

# Long-audio window defaults: a 180s window (~2340 decoder tokens) on a 6 GB GPU
# has an attention peak ≈ 350 MB, leaving ample room for model weights/KV/activations; 10s overlap for boundary continuation.
DEFAULT_WINDOW_SEC = 180.0
DEFAULT_OVERLAP_SEC = 10.0
# Per-window generation budget: 16 token/s (generous upper bound for dense JA/ZH speech + timestamp/speaker overhead)
_WINDOW_TOKEN_RATE = 16

# ── VRAM budget constants (bf16, Qwen3 28-layer full attention, ~13 token/s) ──
# Peak model: peak(t) ≈ W + C1·t + C2·t² (t = window seconds)
#   C2 = 64.5 B/token² × (13 token/s)² ≈ 10900 B/s²
#        (quadratic transient allocations: attention scores/softmax/masks, accumulated across layers by the allocator)
#   C1 = 114688 B/token × 13 token/s ≈ 1.49 MB/s (KV-cache linear term)
# Calibration anchors: 1217s → peak ≈ 15.4 GiB (OOM on a 6 GB card); 180s → ≈ 620 MB.
_ATTN_BYTES_PER_SEC2 = 10900.0
_KV_BYTES_PER_SEC = 1_490_000.0
_WEIGHTS_ESTIMATE_BYTES = 1_365_000_000  # weight estimate when lazy load is not ready (Qwen3-0.6B bf16 + encoder)
_FIXED_SLACK_BYTES = 512 * 1024 * 1024   # fixed overhead: mel features/output logits/allocator fragmentation, etc.
_DEFAULT_MIN_WINDOW_SEC = 60.0
# OOM retreat: window shrink factor / single-window floor / max retreats
_OOM_SHRINK = 0.7
_OOM_FLOOR_SEC = 45.0
_OOM_MAX_RETREATS = 3
# Hard-cut boundary repair tolerance (seconds): a tail within 1.2s of the cut is
# considered possibly truncated; a head within 0.5s after the cut is a continuation at the cut.
_HARD_CUT_TAIL_TOL = 1.2
_HARD_CUT_HEAD_TOL = 0.5


class MossTranscriber(Executor):
    """Execute transcription using an externally-managed MOSS ModelRunner.

    Usage via TaskQueue::
        runner = ModelRunner(model_path, device="auto", dtype="bf16")
        tx = MossTranscriber(runner, defaults=config)
        # TaskQueue calls tx.execute(task, progress_callback, cancel_event)

    Parameters fall into four groups (``defaults`` are service-level; task-level ``configs["args"]`` override them):
    - Service params (configs/models/moss/default.json): model_path / device / dtype (effective at load time)
    - Transcription params: max_new_tokens / max_len / decoding / temperature / top_p /
      top_k / single_speaker (temperature/top_p/top_k only take effect when decoding="sample")
    - Long-audio chunking: max_audio_sec (hard per-window cap) / overlap_sec / vram_auto_fit /
      vram_safety_ratio / min_window_sec / silence_boundary /
      silence_min_sec / boundary_lookback_sec (see the module docstring)
    - prompt: transcription prompt (service default, overridable by task args)
    - hotwords: task-level ``configs["hotwords"]`` (configs/transcribe/hotwords/*.json),
      appended to the prompt per the official recipe ("热词提示：词1, 词2…")
    """

    def __init__(self, runner, defaults: Optional[dict] = None,
                 on_first_load: Optional[Callable[[Optional[str], Optional[str]], None]] = None):
        self.runner = runner
        self.defaults = defaults or {}
        self.on_first_load = on_first_load  # callback(device, dtype) after the first transcription (model actually loaded)
        self._device_logged = False   # backfill the real device after the first transcription (ModelRunner lazy load)

    def execute(
        self,
        task,
        progress_callback: Optional[Callable[[float, float, float, Optional[list]], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> dict:
        """Run transcription from *task*.

        Task contract::
            task.file_path             audio file path
            task.configs["args"]       transcription params (max_new_tokens/max_len/decoding…,
                                       path → JSON dict, overriding service defaults)

        Returns {"segments": [...], "info": TranscriptionResult.to_dict()}.

        Progress callbacks unify with the Whisper ``Transcriber`` as
        ``(pos, total, speed, payload)``: pos/total are audio-timeline seconds,
        pos is the end of the latest confirmed segment (same as Whisper's ``seg.end``,
        0 before the first segment), and one final pos=total is emitted at completion to
        fill the bar; payload carries status/generated_tokens and the confirmed in-progress
        segments (only provided by ``SUPPORTS_PARTIAL_TEXT`` runners).
        """
        cfg = self._resolve_task(task)
        args = cfg.get("transcribe_config") or {}
        merged = {**self.defaults, **args}
        hotwords = cfg.get("hotwords")
        pause_event = getattr(task, "_pause_event", None)
        audio_path = cfg["audio_path"]
        total_sec = probe_duration(audio_path)
        started = time.time()
        state = {
            "stage": 0.0,          # ratio-progress fallback value (used when duration probing fails)
            "tokens": 0,           # tokens generated by the current window
            "tokens_base": 0,      # token accumulation across completed windows (chunked mode)
            "pos_base": 0.0,       # start of the current window on the global timeline
            "pos_floor": 0.0,      # monotonic lower bound of emitted progress (prevents rollback across windows)
            "gen_started": None,   # decode start time (first generated token)
        }

        def _check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError(task.id)
            if not _wait_paused(pause_event, cancel_event):
                raise CancelledError(task.id)

        def _emit(status: str, stage_progress: float, generated_tokens: int,
                  partial_text: Optional[str] = None):
            """Unified progress emission: cancel/pause checkpoints + Whisper-isomorphic four-arg callback.

            Token-level progress updates are removed (progress is driven only by confirmed
            segments): with no confirmed segments and status transcribing, no callback is
            sent, avoiding flicker from pushing advanced progress back to 0%; load
            milestones such as loading_model keep pos=current window start (load-phase
            progress is inherently 0).
            """
            _check_cancel()
            state["stage"] = float(stage_progress or 0.0)
            if generated_tokens is not None:
                state["tokens"] = int(generated_tokens)
            tokens_total = int(state.get("tokens_base") or 0) + int(state["tokens"] or 0)
            if state["tokens"] > 0 and state["gen_started"] is None:
                # Aligned with Whisper's start_wall: speed is measured from decode start
                # (first generated token), excluding lazy model-load time.
                state["gen_started"] = time.time()
            if progress_callback is None:
                return

            now = time.time()
            segments = None
            if partial_text:
                segments = self._segments_from_text(
                    partial_text, merged,
                    offset=float(state.get("pos_base") or 0.0),
                    apply_single=False,
                )
                if segments:
                    state["pos_floor"] = max(
                        state.get("pos_floor") or 0.0,
                        max(float(s["end"]) for s in segments),
                    )
            payload = {"status": status, "generated_tokens": tokens_total}
            if segments:
                payload["segments"] = segments
            if total_sec:
                if segments:
                    # Whisper-isomorphic: pos = end of the latest confirmed segment (real timeline),
                    # speed measured from decode start, consistent with Whisper's pos/elapsed semantics.
                    pos = max(float(s["end"]) for s in segments)
                    speed = None
                    if state["gen_started"] is not None:
                        speed = pos / max(now - state["gen_started"], 1e-6)
                    progress_callback(pos, total_sec, speed, payload)
                elif status != "transcribing":
                    # Load-phase milestones (loading_model, etc.): progress stays at the current
                    # window start, only the status text refreshes (no real progress during load, no flicker).
                    pos = max(
                        float(state.get("pos_base") or 0.0),
                        float(state.get("pos_floor") or 0.0),
                    )
                    progress_callback(pos, total_sec, None, payload)
                # No confirmed segments and transcribing: emit nothing (token-level progress is
                # gone; load milestones and occasional segment-parse failures no longer push progress back to 0%)
            else:
                # Duration probe failed: fall back to token-ratio semantics (unit=ratio lets the UI distinguish)
                payload["unit"] = "ratio"
                progress_callback(state["stage"], 1.0, None, payload)

        def _emit_window_end(window_end: float):
            """In chunked mode, emit one more progress event after each window completes (monotonically advancing to the window end)."""
            _check_cancel()
            if progress_callback is None:
                return
            pos = max(
                min(float(window_end), float(total_sec or window_end)),
                float(state.get("pos_floor") or 0.0),
            )
            pos = min(pos, float(total_sec or pos))
            state["pos_floor"] = pos
            payload = {
                "status": "transcribing",
                "generated_tokens": int(state.get("tokens_base") or 0) + int(state["tokens"] or 0),
            }
            if total_sec:
                speed = None
                if state["gen_started"] is not None:
                    speed = pos / max(time.time() - state["gen_started"], 1e-6)
                progress_callback(pos, total_sec, speed, payload)
            else:
                payload["unit"] = "ratio"
                progress_callback(1.0, 1.0, None, payload)

        def _emit_final(seg_dicts: list, generated_total: int):
            """Final progress: aligned with Whisper's last-segment end≈duration, progress topped at 100%.

            The queue flips status only after execute() returns, so the bar is visible at full at the moment of completion.
            """
            _check_cancel()
            if progress_callback is None:
                return
            payload = {
                "status": "transcribing",
                "generated_tokens": int(generated_total or 0),
            }
            if seg_dicts:
                payload["segments"] = seg_dicts
            if total_sec:
                speed = None
                if state["gen_started"] is not None:
                    speed = total_sec / max(time.time() - state["gen_started"], 1e-6)
                progress_callback(total_sec, total_sec, speed, payload)
            else:
                payload["unit"] = "ratio"
                progress_callback(1.0, 1.0, None, payload)

        def _status(status: str, progress: float, generated_tokens: int):
            _emit(status, progress, generated_tokens)

        def _partial_text(partial_text: str, generated_tokens: int):
            _emit("transcribing", state["stage"], generated_tokens, partial_text)

        transcribe_kwargs = self._build_transcribe_kwargs(merged, hotwords)
        transcribe_kwargs["status_callback"] = _status
        if getattr(self.runner, "SUPPORTS_PARTIAL_TEXT", False):
            transcribe_kwargs["partial_text_callback"] = _partial_text
        window_sec = self._resolve_window_sec(merged)
        envelope_getter = None
        if (window_sec and total_sec and total_sec > window_sec + 1e-3
                and merged.get("silence_boundary", True)):
            envelope_getter = partial(load_band_rms, audio_path)
        windows, hard_flags = self._plan_windows_with_flags(
            total_sec, merged, envelope_getter, window_sec,
        )

        if windows:
            overlap = self._as_float(merged.get("overlap_sec"), DEFAULT_OVERLAP_SEC)
            hard_count = sum(1 for f in hard_flags if f)
            self._log(
                "info",
                f"[transcribe] 长音频分窗转写：总长={total_sec:.1f}s，"
                f"窗口={len(windows)}，单窗≤{max(e - s for s, e in windows):.1f}s，"
                f"重叠={overlap:.1f}s，静音切分={len(windows) - hard_count}，"
                f"硬切={hard_count}（硬切边界自动文本修复）",
            )
            seg_dicts, info = self._transcribe_windowed(
                task=task,
                audio_path=audio_path,
                windows=windows,
                hard_flags=hard_flags,
                total_sec=total_sec,
                merged=merged,
                base_kwargs=transcribe_kwargs,
                state=state,
                started=started,
                check_cancel=_check_cancel,
                emit_window_end=_emit_window_end,
                envelope_getter=envelope_getter,
                window_sec=window_sec,
            )
            generated_total = int((info or {}).get("generated_tokens") or 0)
        else:
            try:
                result = self.runner.transcribe(audio_path, **transcribe_kwargs)
            except CancelledError:
                self._empty_cuda_cache()  # release the allocator cache after cancellation
                raise
            self._note_first_load(started)
            # Final result and live preview share the same segmentation pipeline, so the preview's last frame = the final result.
            seg_dicts = self._segments_from_text(result.text, merged)
            info = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
            if not isinstance(info, dict):
                info = {}
            generated_total = int(
                info.get("generated_tokens", state["tokens"]) or 0
            )

        _emit_final(seg_dicts, generated_total)
        return {"segments": seg_dicts, "info": info}

    # ── Long-Audio Chunking ──

    def _resolve_window_sec(self, merged: dict) -> Optional[float]:
        """VRAM budget → safe single-window duration (seconds); None/≤0 disables chunking.

        With ``vram_auto_fit=false``, a non-CUDA device, or a probe failure, fall back to
        ``max_audio_sec`` (existing fixed-window behavior). On CUDA, solve the quadratic
        peak model ``C2·t² + C1·t = budget`` for the duration (budget = free VRAM ×
        ``vram_safety_ratio`` − fixed overhead − weight estimate; when the model is already
        loaded the weights are already deducted from free VRAM and not subtracted again),
        clamping the result to ``[min_window_sec, max_audio_sec]``.
        """
        max_sec = self._as_float(merged.get("max_audio_sec"), DEFAULT_WINDOW_SEC)
        if max_sec <= 0:
            return None
        if not merged.get("vram_auto_fit", True):
            return max_sec
        free = self._probe_free_vram(merged)
        if free is None:
            return max_sec
        ratio = self._as_float(merged.get("vram_safety_ratio"), 0.7)
        ratio = max(0.1, min(ratio, 0.95))
        budget = free * ratio - _FIXED_SLACK_BYTES
        # Minimum budget guarantee (GB, 0 = no floor): with small VRAM still pick the
        # window by the given budget, preferring a larger window (better to trigger the
        # OOM shrink fallback than make the window too small)
        min_gb = self._as_float(merged.get("min_vram_budget_gb"), 0.0)
        if min_gb > 0:
            budget = max(budget, int(min_gb * 1024 ** 3))
        if getattr(self.runner, "_model", None) is None:
            budget -= _WEIGHTS_ESTIMATE_BYTES  # lazy load: weights not yet deducted from free VRAM
        min_sec = min(
            self._as_float(merged.get("min_window_sec"), _DEFAULT_MIN_WINDOW_SEC),
            max_sec,
        )
        if budget <= 0:
            return max(min_sec, 1.0)  # extremely low budget: use the minimum window and let the OOM retreat handle it
        import math

        disc = _KV_BYTES_PER_SEC ** 2 + 4.0 * _ATTN_BYTES_PER_SEC2 * budget
        sec = (-_KV_BYTES_PER_SEC + math.sqrt(disc)) / (2.0 * _ATTN_BYTES_PER_SEC2)
        return max(min(sec, max_sec), min_sec)

    def _probe_free_vram(self, merged: dict) -> Optional[int]:
        """Probe free CUDA VRAM (bytes); None for non-CUDA configs / probe failures."""
        device_cfg = str(merged.get("device") or "auto").lower()
        if device_cfg == "cpu":
            return None
        actual = getattr(self.runner, "_device", None)
        if actual is not None and actual.type != "cuda":
            return None
        try:
            import torch
        except Exception:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            device_idx = None
            if actual is not None and actual.type == "cuda":
                device_idx = getattr(actual, "index", None)
            elif device_cfg.startswith("cuda:"):
                device_idx = int(device_cfg.split(":", 1)[1])
            free, _total = (
                torch.cuda.mem_get_info(device_idx)
                if device_idx is not None
                else torch.cuda.mem_get_info()
            )
            return int(free)
        except Exception:
            return None

    def _plan_windows(self, total_sec: Optional[float], merged: dict) -> list:
        """Compatibility entry: audio exceeding max_audio_sec → arithmetic sliding windows (no silence probing).

        Adjacent windows overlap by overlap_sec; returns an empty list when the total
        length is within the threshold / duration is unknown / threshold is 0 (caller
        takes the single full-clip transcription path).
        """
        return self._plan_windows_with_flags(total_sec, merged)[0]

    def _plan_windows_with_flags(self, total_sec: Optional[float], merged: dict,
                                 envelope_getter=None, window_sec: Optional[float] = None,
                                 start_from: float = 0.0) -> tuple[list, list]:
        """Silence-aware sliding window → ([(start, end), ...], [hard_flag, ...]).

        Single-window length = ``window_sec`` (the VRAM-budget value) or ``max_audio_sec``.
        Each internal boundary looks for the best silence segment as the cut within the
        ``[target − lookback, target]`` band (target = window start + window length): cuts
        can only move earlier, never later — **single-window duration is strictly ≤ the
        budget value** (a constructive VRAM-safety guarantee). No silence in the band →
        arithmetic hard cut (flag=True, left to the boundary text repair fallback).

        ``start_from`` replans the remaining audio starting from a failed window during OOM retreat;
        hard_flag aligns with windows: flag[i] = whether window i's right boundary is a hard cut
        (the last window is always False).
        """
        max_sec = (
            window_sec
            if window_sec is not None
            else self._as_float(merged.get("max_audio_sec"), DEFAULT_WINDOW_SEC)
        )
        overlap = self._as_float(merged.get("overlap_sec"), DEFAULT_OVERLAP_SEC)
        if total_sec is None or max_sec <= 0:
            return [], []
        start_from = max(0.0, float(start_from))
        if start_from <= 0 and total_sec <= max_sec + 1e-3:
            return [], []
        overlap = max(0.0, min(overlap, max_sec * 0.5))
        if max_sec - overlap < 5.0:
            return [], []
        lookback = max(0.0, self._as_float(merged.get("boundary_lookback_sec"), 30.0))
        lookback = min(lookback, max_sec * 0.5)
        silence_min = max(0.15, self._as_float(merged.get("silence_min_sec"), 0.35))
        use_silence = bool(merged.get("silence_boundary", True)) and envelope_getter is not None

        windows: list[tuple[float, float]] = []
        hard_flags: list[bool] = []
        start = start_from
        while start < total_sec - 1e-3:
            if total_sec - start <= max_sec + 1e-3:
                windows.append((start, total_sec))
                hard_flags.append(False)
                break
            target = start + max_sec
            band_lo = max(start + 5.0, target - lookback)
            band_hi = min(target, total_sec - 0.5)
            cut, hard = target, True
            if use_silence and band_hi > band_lo + 1e-3:
                rms = self._load_band(envelope_getter, band_lo, band_hi)
                if rms is not None:
                    found = find_silence_cut(
                        rms, band_lo, target=target, silence_min_sec=silence_min,
                    )
                    if (found is not None and band_lo <= found <= band_hi
                            and found >= start + 5.0):
                        cut, hard = float(found), False
            windows.append((start, cut))
            hard_flags.append(hard)
            start = cut - overlap
        return windows, hard_flags

    @staticmethod
    def _load_band(getter, lo: float, hi: float):
        """Fault-tolerant envelope getter call (any exception → None, falls back to a hard cut)."""
        try:
            return getter(float(lo), float(hi))
        except Exception:
            return None

    @staticmethod
    def _as_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _make_window_clip(self, audio_path, task, idx: int,
                          start: float, end: float) -> Path:
        """Use ffmpeg to cut a window-level 16kHz mono WAV (for in-window model decoding).

        Output goes to the project temp dir (named by task id) and is cleaned up by the caller after the whole task.
        """
        from app.ffmpeg import run_ffmpeg

        tid = str(getattr(task, "id", None) or "moss_task")
        safe_tid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tid)
        out = (project_root / "temp" / "moss_windows" / safe_tid
               / f"win_{idx:03d}_{int(start):06d}.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        duration = max(float(end) - float(start), 0.1)
        proc = run_ffmpeg(
            [
                "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{float(start):.3f}",
                "-i", str(audio_path),
                "-t", f"{duration:.3f}",
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-3:])
            raise RuntimeError(
                f"长音频分窗切片失败（{float(start):.1f}-{float(end):.1f}s）: {tail}"
            )
        return out

    def _window_max_new_tokens(self, base, window_sec: float) -> int:
        """Per-window generation budget: the smaller of this and the user cap, to avoid runaway generation with the whole-clip budget.

        The default whole-clip budget of 65536 is meant for very long audio; after chunking,
        16 token/s per window already covers dense speech + timestamp/speaker overhead.
        """
        budget = max(1024, int(float(window_sec) * _WINDOW_TOKEN_RATE) + 256)
        try:
            base = int(base or 0)
        except (TypeError, ValueError):
            base = 0
        if base <= 0:
            return budget
        return max(1, min(base, budget))

    def _transcribe_windowed(self, *, task, audio_path, windows, hard_flags,
                             total_sec, merged, base_kwargs, state, started,
                             check_cancel, emit_window_end, envelope_getter=None,
                             window_sec=None) -> tuple[list, dict]:
        """Transcribe window by window → shift to timeline → hard-cut repair + dedup merge → combined info.

        Pipeline (GPU optimization plan A): a background thread runs window cutting + decode/mel/
        tokenize in parallel (``prepare_clip``, no model lock), while the main thread runs
        ``generate_with`` (holds the model lock) — window i's inference overlaps with window
        i+1's input preparation, eliminating GPU idle periods caused by CPU overhead such as
        ffmpeg cutting / feature extraction.

        Runtime OOM retreat: when a single window still exceeds VRAM, shrink the window ×0.7
        (floor 45s), replan the **remaining** audio (completed windows are kept), rebuild the
        prefetch queue and thread, then retry, up to 3 times; when retreats are exhausted, raise
        an error with guidance.
        """
        import queue as _queue
        import threading

        window_segments: list[list[dict]] = []
        infos: list[dict] = []
        clip_dir: Optional[Path] = None
        effective_sec = float(window_sec or max(
            (float(e) - float(s) for s, e in windows), default=DEFAULT_WINDOW_SEC,
        ))
        retreats = 0
        stop_event = threading.Event()
        ready = _queue.Queue(maxsize=2)  # prefetch at most 2 windows

        prepare_prompt = base_kwargs.get("prompt")
        prepare_max_length = int(base_kwargs.get("max_length", 131072))

        def _prepare_worker(window_list: list, start_idx: int) -> None:
            """Background thread: window cutting + prepare_clip (no model lock), parallel with the main thread's generate.

            When the queue is full, wait in a timeout loop (0.2s polling stop_event) so
            cancellation can exit; never block on put and hold GPU input tensors forever.
            """
            for widx in range(start_idx, len(window_list)):
                if stop_event.is_set():
                    return
                wstart, wend = window_list[widx]
                try:
                    clip = self._make_window_clip(audio_path, task, widx, wstart, wend)
                    inputs, prompt_len = self.runner.prepare_clip(
                        clip, prompt=prepare_prompt, max_length=prepare_max_length,
                    )
                except Exception as exc:
                    while not stop_event.is_set():
                        try:
                            ready.put((widx, None, exc, None), timeout=0.2)
                            return
                        except _queue.Full:
                            continue
                while not stop_event.is_set():
                    try:
                        ready.put((widx, inputs, prompt_len, clip), timeout=0.2)
                        break
                    except _queue.Full:
                        continue

        def _start_worker(window_list: list, start_idx: int) -> None:
            stop_event.clear()
            threading.Thread(
                target=_prepare_worker, args=(window_list, start_idx), daemon=True
            ).start()

        _start_worker(windows, 0)
        try:
            idx = 0
            while idx < len(windows):
                check_cancel()
                widx, inputs, prompt_len, clip = ready.get()
                if inputs is None:
                    raise prompt_len  # upstream prepare failed; the payload is the exception object
                if clip_dir is None:
                    clip_dir = clip.parent
                start, end = windows[idx]
                state["pos_base"] = float(start)
                kwargs = dict(base_kwargs)
                kwargs["max_new_tokens"] = self._window_max_new_tokens(
                    kwargs.get("max_new_tokens"), end - start,
                )
                generate_kwargs = {
                    "max_new_tokens": kwargs.get("max_new_tokens"),
                    "do_sample": str(kwargs.get("decoding", "greedy")) == "sample",
                    "temperature": kwargs.get("temperature"),
                    "top_p": kwargs.get("top_p"),
                    "top_k": kwargs.get("top_k"),
                    "status_callback": kwargs.get("status_callback"),
                    "partial_text_callback": kwargs.get("partial_text_callback"),
                }
                try:
                    result = self.runner.generate_with(
                        inputs, prompt_len, **generate_kwargs
                    )
                except Exception as exc:
                    if isinstance(exc, CancelledError):
                        raise  # task cancellation is not a transcription failure; do not log an error
                    if "out of memory" not in str(exc).lower():
                        self._log(
                            "error",
                            f"[transcribe] 长音频窗口 {idx + 1}/{len(windows)} 转写失败"
                            f"（{float(start):.1f}-{float(end):.1f}s）: {exc}",
                        )
                        raise
                    # ── Runtime OOM retreat: stop prefetch, shrink and replan, rebuild queue and thread ──
                    retreats += 1
                    if retreats > _OOM_MAX_RETREATS or effective_sec <= _OOM_FLOOR_SEC + 1e-3:
                        self._log(
                            "error",
                            f"[transcribe] 长音频窗口 {float(start):.1f}-{float(end):.1f}s "
                            f"连续 OOM（退避 {retreats - 1} 次后单窗已到 {_OOM_FLOOR_SEC:.0f}s 下限），"
                            "请把 MOSS 服务 device 改为 cpu，或释放显存后重试",
                        )
                        raise
                    stop_event.set()
                    effective_sec = max(_OOM_FLOOR_SEC, effective_sec * _OOM_SHRINK)
                    tail, tail_flags = self._plan_windows_with_flags(
                        total_sec, merged, envelope_getter,
                        window_sec=effective_sec, start_from=float(start),
                    )
                    if not tail:
                        raise
                    self._empty_cuda_cache()
                    self._log(
                        "warning",
                        f"[transcribe] 窗口 {float(start):.1f}-{float(end):.1f}s OOM："
                        f"单窗缩至 {effective_sec:.0f}s，剩余音频重规划为 "
                        f"{len(tail)} 个窗口后重试",
                    )
                    windows = windows[:idx] + tail
                    hard_flags = hard_flags[:idx] + tail_flags
                    ready = _queue.Queue(maxsize=2)  # drop old prefetch, rebuild the queue
                    _start_worker(windows, idx)      # restart prefetch from the current window
                    continue
                self._note_first_load(started)
                text = getattr(result, "text", "")
                info = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
                if not isinstance(info, dict):
                    info = {}
                # No single-speaker merge inside a window (cross-window boundaries are handled by the final merge)
                segs = self._segments_from_text(
                    text, merged, offset=float(start), apply_single=False,
                )
                window_segments.append(segs)
                infos.append(info)
                state["tokens_base"] = int(state.get("tokens_base") or 0) + int(
                    info.get("generated_tokens") or 0
                )
                state["tokens"] = 0
                self._log(
                    "info",
                    f"[transcribe] 长音频窗口 {idx + 1}/{len(windows)} 完成"
                    f"（{float(start):.1f}-{float(end):.1f}s，用时="
                    f"{time.time() - started:.1f}s 累计）",
                )
                emit_window_end(end)
                idx += 1
            seg_dicts = self._merge_window_segments(
                window_segments, windows, merged, hard_flags=hard_flags,
            )
        finally:
            stop_event.set()  # stop the prefetch thread (daemon, does not block exit)
            # Drain the prefetch queue and release GPU input tensor references (prevents VRAM leaks on the cancel path)
            while True:
                try:
                    _widx, _inputs, _plen, _clip = ready.get_nowait()
                    if _inputs is not None:
                        del _inputs
                except _queue.Empty:
                    break
            if clip_dir is not None:
                shutil.rmtree(clip_dir, ignore_errors=True)
            self._empty_cuda_cache()  # clear the allocator cache uniformly on cancel/OOM/normal exit
        info = self._combine_window_info(
            audio_path, windows, infos, seg_dicts,
            hard_flags=hard_flags, window_sec=effective_sec,
        )
        return seg_dicts, info

    @staticmethod
    def _empty_cuda_cache():
        """Clear the CUDA allocator cache during OOM retreat (failures silent)."""
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _merge_window_segments(self, window_segments: list, windows: list,
                               merged: dict, hard_flags: Optional[list] = None) -> list:
        """Window segment merge: hard-cut repair → overlap drop/trim → residual overlap resolution → normalization.

        Rules (paired with the chunking strategy):
        - Hard-cut boundaries are text-repaired first (``_repair_boundary_cuts``): tails
          truncated at the cut are stitched with continuing heads of the next window into
          complete segments (marked ``_repaired``, skipping later start-trimming to keep the real start);
        - Window 0 is fully kept;
        - For window k>0: segments with end ≤ previous window's end (i.e. fully inside the
          overlap) are dropped; segments with start < previous end but end crossing the
          boundary have start trimmed to the boundary;
        - Finally resolve residual overlaps on the timeline (timestamp jitter across windows):
          first-come wins; a later segment fully covered or highly text-similar is dropped,
          otherwise its start is trimmed to the previous segment's end.
        - When single_speaker=true, normalize the merged result uniformly (after merge/sort).
        """
        if hard_flags:
            window_segments = self._repair_boundary_cuts(
                window_segments, windows, hard_flags,
            )
        out: list[dict] = []
        for idx, (_wstart, _wend) in enumerate(windows):
            segs = window_segments[idx] if idx < len(window_segments) else []
            for seg in segs:
                seg = dict(seg)
                repaired = bool(seg.pop("_repaired", False))
                if idx > 0 and not repaired:
                    prev_end = float(windows[idx - 1][1])
                    if float(seg.get("end", 0.0)) <= prev_end + 0.01:
                        continue
                    if float(seg.get("start", 0.0)) < prev_end:
                        seg["start"] = prev_end
                if float(seg.get("end", 0.0)) > float(seg.get("start", 0.0)):
                    out.append(seg)
        out = self._resolve_segment_overlaps(out)
        if merged.get("single_speaker", False):
            out = force_single_speaker(out)
        out.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
        for i, seg in enumerate(out, 1):
            seg["id"] = f"seg_{i:04d}"
        return out

    def _repair_boundary_cuts(self, window_segments: list, windows: list,
                              hard_flags: list) -> list:
        """Hard-cut boundary text repair: truncated tail + continuing head → stitched into a complete segment.

        Only handles boundaries with hard_flag=True (the cut did not land in silence, so
        speech may be truncated). Pairing conditions: the previous window has a tail
        segment whose end is within 1.2s of the cut, and the next window has a head segment
        starting ≈ at the cut and ending past it (a continuation of the same sentence),
        with sufficient text similarity (the truncated tail's suffix appears at the head's
        front, or overall similarity ≥0.5). Stitched text is joined after prefix/suffix
        dedup; the time span is [tail start, head end]; the head is removed from the next
        window to avoid double counting.
        """
        segs = [[dict(s) for s in w] for w in window_segments]
        for k in range(min(len(windows), len(segs)) - 1):
            if k >= len(hard_flags) or not hard_flags[k]:
                continue
            b = float(windows[k][1])
            ov = max(0.0, b - float(windows[k + 1][0]))
            tails = sorted(
                [
                    i for i, s in enumerate(segs[k])
                    if (float(s.get("start", 0.0)) < b
                        and float(s.get("end", 0.0)) >= b - _HARD_CUT_TAIL_TOL)
                ],
                key=lambda i: float(segs[k][i].get("end", 0.0)),
                reverse=True,
            )
            heads = sorted(
                [
                    j for j, s in enumerate(segs[k + 1])
                    if (float(s.get("start", 0.0)) <= b + _HARD_CUT_HEAD_TOL
                        and float(s.get("end", 0.0)) > b + _HARD_CUT_HEAD_TOL)
                ],
                key=lambda j: float(segs[k + 1][j].get("start", 0.0)),
            )
            used_heads = set()
            for ti in tails:
                tail = segs[k][ti]
                hj = None
                for j in heads:
                    if j in used_heads:
                        continue
                    if self._boundary_pair_score(tail, segs[k + 1][j]) > 0.0:
                        hj = j
                        break
                if hj is None:
                    continue
                head = segs[k + 1][hj]
                used_heads.add(hj)
                merged = dict(tail)
                merged["start"] = min(
                    float(tail.get("start", 0.0)), float(head.get("start", 0.0)),
                )
                merged["end"] = float(head.get("end", tail.get("end", b)))
                merged["text"] = self._join_texts(
                    tail.get("text", ""), head.get("text", ""),
                )
                merged["_repaired"] = True
                segs[k][ti] = merged
            if used_heads:
                segs[k + 1] = [
                    s for j, s in enumerate(segs[k + 1]) if j not in used_heads
                ]
        return segs

    def _boundary_pair_score(self, tail: dict, head: dict) -> float:
        """Pairing confidence of a truncated tail / continuing head (0 = no pair).

        The tail's trailing characters should appear at the front of the head text
        (only the prefix up to twice the head-segment length is covered by the window);
        otherwise degrade to pairing only when overall similarity ≥0.5.
        """
        a = self._norm_text(tail.get("text", ""))
        b = self._norm_text(head.get("text", ""))
        if not a or not b:
            return 0.0
        prefix = b[: min(len(b), max(6, len(a) * 2))]
        # The last 1-3 chars of a truncated tail should appear at the head's front (a cut often lands mid-word)
        if len(a) >= 2:
            for length in (min(3, len(a)), 2):
                if a[-length:] in prefix:
                    return 1.0
        elif a in prefix:
            return 1.0
        if a in b or b in a:
            return 0.9
        try:
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
        except Exception:
            return 0.0
        return ratio if ratio >= 0.5 else 0.0

    @staticmethod
    def _norm_text(text) -> str:
        return "".join(ch for ch in str(text or "") if ch.isalnum())

    @staticmethod
    def _join_texts(a: str, b: str) -> str:
        """Join two texts by deduping the longest prefix/suffix overlap (plain concatenation when there is no overlap)."""
        a, b = (a or "").strip(), (b or "").strip()
        if not a:
            return b
        if not b:
            return a
        for length in range(min(len(a), len(b), 20), 0, -1):
            if a[-length:] == b[:length]:
                return a + b[length:]
        return a + b

    def _resolve_segment_overlaps(self, segments: list) -> list:
        """Resolve timestamp overlaps (for chunk boundaries): first-come wins, preventing duplicate/covered double counting."""
        segs = [dict(s) for s in segments if s]
        segs.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
        resolved: list[dict] = []
        for seg in segs:
            if resolved:
                prev = resolved[-1]
                if float(seg.get("start", 0.0)) < float(prev.get("end", 0.0)) - 0.02:
                    if (float(seg.get("end", 0.0)) <= float(prev.get("end", 0.0)) + 0.02
                            or self._same_utterance(seg.get("text", ""), prev.get("text", ""))):
                        continue
                    seg["start"] = float(prev["end"])
            if float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)) > 0.02:
                resolved.append(seg)
        return resolved

    @staticmethod
    def _same_utterance(text_a, text_b) -> bool:
        """Text-similarity check for boundary dedup (containment or high similarity)."""
        def _norm(text) -> str:
            return "".join(ch for ch in str(text or "") if ch.isalnum())

        a, b = _norm(text_a), _norm(text_b)
        if not a or not b:
            return False
        if a in b or b in a:
            return True
        try:
            return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85
        except Exception:
            return False

    def _combine_window_info(self, audio_path, windows: list,
                             infos: list, seg_dicts: list,
                             hard_flags: Optional[list] = None,
                             window_sec: Optional[float] = None) -> dict:
        """Combine multiple windows into a TranscriptionResult.to_dict() (token/elapsed sums)."""
        base = dict(infos[0]) if infos else {}
        base.update({
            "text": self._segments_to_transcript_text(seg_dicts),
            "generated_tokens": sum(int(i.get("generated_tokens") or 0) for i in infos),
            "elapsed_sec": round(
                sum(float(i.get("elapsed_sec") or 0.0) for i in infos), 3
            ),
            "audio": str(audio_path),
            "windows": len(windows),
            "chunking": {
                "strategy": "sliding_window_silence_aware",
                "window_sec": round(float(window_sec), 1) if window_sec else None,
                "silence_boundaries": sum(1 for f in (hard_flags or []) if not f),
                "hard_boundaries": sum(1 for f in (hard_flags or []) if f),
            },
        })
        return base

    @staticmethod
    def _segments_to_transcript_text(seg_dicts: list) -> str:
        """Segment dicts → standard LRC transcript text ([mm:ss.cs]<speaker>body, compatible with info.text display)."""
        parts = []
        for s in seg_dicts or []:
            speaker = str(s.get("speaker") or "S01")
            parts.append(
                f"[{format_lrc_time(float(s.get('start', 0.0)))}]<{speaker}>"
                f"{s.get('text', '')}"
            )
        return "\n".join(parts)

    def _build_transcribe_kwargs(self, merged: dict, hotwords) -> dict:
        """Assemble runner.transcribe(**kwargs) by parameter group."""
        decoding = merged.get("decoding", "greedy")
        transcribe_kwargs = {
            "max_length": int(merged.get("max_len", 131072)),
            "max_new_tokens": int(merged.get("max_new_tokens", 65536)),
            "decoding": decoding,
        }
        # sampling params only take effect with sample decoding (greedy would ignore them
        # anyway; simply don't pass them, matching the vendor CLI behavior)
        if decoding == "sample":
            for key, cast in (("temperature", float), ("top_p", float), ("top_k", int)):
                value = merged.get(key)
                if value not in (None, ""):
                    transcribe_kwargs[key] = cast(value)
        # Hotwords: append to the prompt per the official recipe (fall back to the vendor default prompt when no explicit prompt)
        prompt = merged.get("prompt")
        hotword_text = self._normalize_hotwords(hotwords)
        if hotword_text:
            if not prompt:
                from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
                prompt = DEFAULT_PROMPT
            prompt = f"{prompt}\n热词提示：{hotword_text}"
        if prompt:
            transcribe_kwargs["prompt"] = prompt
        return transcribe_kwargs

    def _note_first_load(self, started: float):
        """Backfill the real device after the first transcription completes (ModelRunner lazy load)."""
        if self._device_logged:
            return
        self._device_logged = True
        device = getattr(self.runner, "_device", None)
        dtype = getattr(self.runner, "_dtype", None)
        self._log(
            "info",
            f"MOSS 首次转写完成（模型实际加载，首窗/首段用时={time.time() - started:.1f}s；"
            f"分窗模式后续窗口逐窗另记）：实际设备={device}，dtype={dtype}，",
        )
        if self.on_first_load is not None:
            try:
                self.on_first_load(device, dtype)
            except Exception:
                pass  # callback failure does not affect the transcription result

    def _segments_from_text(self, text: str, merged: dict,
                            offset: float = 0.0, apply_single: bool = True) -> list:
        """Native compact transcript → subtitle segment list (shared by final result and live preview).

        postprocess=False keeps the model's original segmentation; speaker normalization
        is left to the force_single_speaker fallback; returns an empty list when the text
        is empty or parsing fails. ``offset`` maps chunked segments back to the global timeline.
        """
        if not text:
            return []
        try:
            from moss_transcribe_diarize.subtitle import subtitle_segments_from_transcript

            segments = subtitle_segments_from_transcript(text, postprocess=False)
            seg_dicts = [s.to_dict() for s in segments]
        except Exception as exc:
            self._log("warning", f"MOSS 段切分失败（跳过本次刷新）: {exc}")
            return []
        if offset:
            for seg in seg_dicts:
                seg["start"] = float(seg.get("start", 0.0)) + float(offset)
                seg["end"] = float(seg.get("end", 0.0)) + float(offset)
        if apply_single and merged.get("single_speaker", False):
            seg_dicts = force_single_speaker(seg_dicts)
        return seg_dicts

    @staticmethod
    def _normalize_hotwords(hotwords) -> str:
        """Normalize hotwords to a comma-separated string (accepts dict {"hotwords": [...]} / list / str)."""
        if not hotwords:
            return ""
        if isinstance(hotwords, dict):
            hotwords = hotwords.get("hotwords", [])
        if isinstance(hotwords, list):
            return ",".join(str(h) for h in hotwords if str(h).strip())
        return str(hotwords).strip()

    def _resolve_task(self, task):
        """Transcription semantics: file_path as the audio path, configs["args"] providing params,
        configs["hotwords"] providing hotwords (None when "none" is selected)."""
        _source, configs = super()._resolve_task(task)
        args = configs.get("args")
        return {
            "audio_path": str(task.file_path),
            "transcribe_config": args if isinstance(args, dict) else None,
            "hotwords": configs.get("hotwords"),
        }
