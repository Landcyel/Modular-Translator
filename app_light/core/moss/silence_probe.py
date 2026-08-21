"""MOSS 长音频切块的静音探测（能量包络法，零额外依赖）。

用途：切块规划时在候选边界带内寻找静音切点，让切点落在自然停顿上，
避免截断正常说话。只解码边界带（每带几十秒），绝不整段解码，
内存占用为常数级（帧级 RMS 数组，1 小时音频 ≈ 0.6 MB）。

实现：ffmpeg 精确切带 → PyAV 重采样 16kHz 单声道 → 逐 25ms 帧 RMS(dBFS)
→ 自适应阈值（噪声底 + 动态范围份额）→ 静音段（含零星脉冲桥接）。

所有失败路径（文件缺失/解码失败/动态范围不足）返回 None，
调用方回退算术硬切 + 边界文本修复，保证转写可用性不受影响。
"""
from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from app.paths import project_root

FRAME_SEC = 0.025          # 帧长（秒）
SAMPLE_RATE = 16000        # 重采样率
BRIDGE_TOL_SEC = 0.15      # 静音段间可桥接的响亮脉冲上限（容忍噪声咔嗒）
NOISE_PERCENTILE = 15      # 噪声底分位
SPEECH_PERCENTILE = 85     # 语音能量分位
RANGE_FRACTION = 0.4       # 阈值 = 噪声底 + 动态范围 × 该份额
MIN_DYNAMIC_DB = 8.0       # 动态范围低于此值 → 无可靠静音（纯噪声/纯语音）
ABS_SILENCE_DB = -50.0     # 绝对静音下限（低于此值恒为静音）
MIN_CUT_RUN_SEC = 0.30     # 有效切点所需最短静音时长（秒，低于此值切点无意义）


def load_band_rms(audio_path, start_sec: float, end_sec: float,
                  frame_sec: float = FRAME_SEC,
                  sr: int = SAMPLE_RATE) -> Optional[list]:
    """解码 ``[start_sec, end_sec)`` 音频带 → 逐帧 RMS 列表（dBFS）。

    文件缺失 / ffmpeg 失败 / 解码异常一律返回 None。
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
    """PyAV 读取 16kHz 单声道浮点样本（[-1, 1]）；失败返回 None。"""
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
    """在带内寻找最合适的静音切点（绝对秒）；找不到返回 None。

    评分 = 静音时长（上限 4s 封顶）×2 − 距目标切点的距离，
    并轻度偏向目标切点之前（窗口只缩不长，守住显存预算）。
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
        return None  # 动态范围不足：可能是纯语音连读，切点无意义
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
    """静音布尔序列 → [(起帧, 止帧), ...]（≤bridge 帧的响亮脉冲并入）。"""
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
