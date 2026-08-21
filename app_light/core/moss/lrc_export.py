"""LRC lyrics export (standard [mm:ss.cs] timestamps + optional speaker prefix).

Server-exported srt/ass have no LRC; this tool generates one client-side from in-memory segments.
"""
from __future__ import annotations

from pathlib import Path


def format_lrc_time(seconds: float) -> str:
    """Seconds → LRC timestamp ``mm:ss.cs`` (minutes:seconds.centiseconds)."""
    seconds = max(0.0, float(seconds))
    mm = int(seconds) // 60
    ss = int(seconds) % 60
    cs = min(int(round((seconds - int(seconds)) * 100)), 99)
    return f"{mm:02d}:{ss:02d}.{cs:02d}"


def export_lrc(segments, out_path, show_speaker: bool = True) -> str:
    """Write segments to an LRC lyrics file and return the written file path.

    Line format: ``[mm:ss.cs]<S01>text`` (with speaker prefix when ``show_speaker=True``).
    """
    lines = []
    for seg in segments:
        ts = format_lrc_time(float(seg["start"]))
        prefix = f"<{seg['speaker']}>" if show_speaker else ""
        lines.append(f"[{ts}]{prefix}{seg['text']}")
    out = Path(out_path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)
