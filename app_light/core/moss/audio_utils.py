"""MOSS audio duration probe (reads only container metadata, does not decode audio).

All MOSS whitelist formats (mp3/wav/m4a/flac/ogg/opus) can be parsed with PyAV;
PyAV is an existing dependency of the MOSS vendor, so no new install is required.
"""
from __future__ import annotations

from typing import Optional


def probe_duration(audio_path) -> Optional[float]:
    """Return the audio duration in seconds; None when it cannot be probed (caller falls back to ratio progress).

    Tries in order: audio stream duration → frames/sample rate → container duration.
    """
    if not audio_path:
        return None
    try:
        import av  # lazy import: only loaded on the first probe of a transcription task

        with av.open(str(audio_path)) as container:
            stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if stream is None:
                return None
            # Stream-level duration (unit: stream.time_base, seconds)
            if stream.duration is not None:
                duration = float(stream.duration * stream.time_base)
                if duration > 0:
                    return duration
            # Frames / sample rate
            if stream.frames and stream.rate:
                duration = float(stream.frames) / float(stream.rate)
                if duration > 0:
                    return duration
            # Container-level duration (unit: av.time_base, microseconds)
            if container.duration is not None:
                duration = float(container.duration / av.time_base)
                if duration > 0:
                    return duration
    except Exception as exc:
        print(f"[moss] 音频时长探测失败 {audio_path}: {exc}")
    return None
