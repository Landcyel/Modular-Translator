"""Silence probing for MOSS long-audio chunking (energy-envelope method, zero extra dependencies).

Purpose: during chunk planning, find a silence cut within each candidate boundary
band so cuts land on natural pauses and never truncate normal speech. Only the
boundary bands are decoded (tens of seconds each), never the whole clip, so memory
usage is constant-order (frame-level RMS array, ~0.6 MB for 1 hour of audio).

Implementation: ffmpeg precise band cut → PyAV resample to 16kHz mono → per-25ms-frame
RMS(dBFS) → adaptive threshold (noise floor + dynamic-range share) → silence runs
(bridging sporadic loud pulses).

All failure paths (missing file / decode failure / insufficient dynamic range) return None;
the caller falls back to arithmetic hard cuts + boundary text repair, so transcription
availability is unaffected.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from app.paths import project_root

FRAME_SEC = 0.025          # frame length (seconds)
SAMPLE_RATE = 16000        # resample rate
BRIDGE_TOL_SEC = 0.15      # max loud pulse bridgeable between silence runs (tolerates noise clicks)
NOISE_PERCENTILE = 15      # noise floor percentile
SPEECH_PERCENTILE = 85     # speech energy percentile
RANGE_FRACTION = 0.4       # threshold = noise floor + dynamic range × this share
MIN_DYNAMIC_DB = 8.0       # below this dynamic range → no reliable silence (pure noise/speech)
ABS_SILENCE_DB = -50.0     # absolute silence floor (always silent below this)
MIN_CUT_RUN_SEC = 0.30     # min silence length for a valid cut (seconds; shorter is meaningless)


def load_band_rms(audio_path, start_sec: float, end_sec: float,
                  frame_sec: float = FRAME_SEC,
                  sr: int = SAMPLE_RATE) -> Optional[list]:
    """Decode the ``[start_sec, end_sec)`` audio band → per-frame RMS list (dBFS).

    Always returns None on missing file / ffmpeg failure / decode error.
    """
    if not audio_path or not os.path.isfile(str(audio_path)):
        return None
    duration = float(end_sec) - float(start_sec)
    if duration <= 0.1:
        return None
    tmp_dir = project_root / "temp" / "moss_bands"
    wav: Optional[Path] = None
    try:
        import numpy as np

        tmp_dir.mkdir(parents=True, exist_ok=True)
        wav = tmp_dir / f"band_{uuid.uuid4().hex}.wav"
        from app.ffmpeg import run_ffmpeg

        proc = run_ffmpeg(
            [
                "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{float(start_sec):.3f}",
                "-i", str(audio_path),
                "-t", f"{duration:.3f}",
                "-vn", "-ac", "1", "-ar", str(int(sr)),
                "-c:a", "pcm_s16le",
                str(wav),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            return None
        samples = _read_mono_f32(wav, sr=sr)
        if samples is None or samples.size == 0:
            return None
        frame_len = max(1, int(round(frame_sec * sr)))
        padded = int(np.ceil(samples.size / frame_len)) * frame_len
        if padded < 2 * frame_len:
            return None
        if samples.size < padded:
            samples = np.concatenate(
                [samples, np.zeros(padded - samples.size, dtype=np.float32)]
            )
        rms = np.sqrt(
            np.mean(samples.reshape(-1, frame_len).astype(np.float32) ** 2, axis=1)
        )
        db = 20.0 * np.log10(rms + 1e-9)
        return db.astype(np.float32).tolist()
    except Exception:
        return None
    finally:
        if wav is not None:
            try:
                wav.unlink(missing_ok=True)
            except Exception:
                pass


def _read_mono_f32(wav: Path, sr: int):
    """Read 16kHz mono float samples ([-1, 1]) via PyAV; None on failure."""
    try:
        import av
        import numpy as np
    except Exception:
        return None
    chunks = []
    try:
        with av.open(str(wav)) as container:
            stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if stream is None:
                return None
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=sr,
            )
            for frame in container.decode(stream):
                frames = resampler.resample(frame)
                if frames is None:
                    continue
                if not isinstance(frames, list):
                    frames = [frames]
                for resampled in frames:
                    chunks.append(resampled.to_ndarray().reshape(-1))
    except Exception:
        return None
    if not chunks:
        return None
    return (np.concatenate(chunks).astype(np.float32) / 32768.0).astype(
        np.float32, copy=False
    )


def find_silence_cut(rms_db, band_start: float, target: float,
                     silence_min_sec: float = 0.35,
                     frame_sec: float = FRAME_SEC) -> Optional[float]:
    """Find the best silence cut within the band (absolute seconds); None if none is found.

    Score = silence duration (capped at 4s) ×2 − distance to the target cut,
    with a slight preference for before the target (windows only shrink, never grow,
    keeping the VRAM budget).
    """
    if not rms_db or len(rms_db) < 2:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    arr = np.asarray(rms_db, dtype=np.float32)
    noise = float(np.percentile(arr, NOISE_PERCENTILE))
    speech = float(np.percentile(arr, SPEECH_PERCENTILE))
    if speech - noise < MIN_DYNAMIC_DB:
        return None  # insufficient dynamic range: possibly continuous pure speech; a cut is meaningless
    thr = noise + RANGE_FRACTION * (speech - noise)
    silent = (arr < thr) | (arr < ABS_SILENCE_DB)
    bridge_frames = max(1, int(round(BRIDGE_TOL_SEC / frame_sec)))
    runs = _silence_runs(silent, bridge_frames)
    min_frames = max(2, int(round(max(silence_min_sec, MIN_CUT_RUN_SEC) / frame_sec)))

    best_cut: Optional[float] = None
    best_score = float("-inf")
    for lo, hi in runs:
        if hi - lo + 1 < min_frames:
            continue
        mid = band_start + (lo + hi + 1) * 0.5 * frame_sec
        dur = (hi - lo + 1) * frame_sec
        score = min(dur, 4.0) * 2.0 - abs(mid - target) + (0.4 if mid <= target else -0.8)
        if score > best_score:
            best_score, best_cut = score, mid
    return best_cut


def _silence_runs(silent, bridge_frames: int) -> list:
    """Silence boolean sequence → [(start frame, end frame), ...] (loud pulses ≤ bridge frames are merged in)."""
    runs = []
    start = None
    loud_streak = 0
    for i, is_silent in enumerate(silent):
        if is_silent:
            if start is None:
                start = i
            loud_streak = 0
        elif start is not None:
            loud_streak += 1
            if loud_streak > bridge_frames:
                runs.append((start, i - loud_streak))
                start = None
                loud_streak = 0
    if start is not None:
        runs.append((start, len(silent) - 1))
    return runs
