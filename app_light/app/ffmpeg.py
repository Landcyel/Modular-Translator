"""Bundled FFmpeg locator — the unified entry point for in-project FFmpeg/ffprobe/ffplay calls.

FFmpeg is required by the project: a copy is bundled into ``dependencies/FFmpeg/``,
so no system installation is needed.
- New code should prefer this module's explicit path constants ``FFMPEG_BIN`` /
  ``FFPROBE_BIN`` / ``FFPLAY_BIN``, or call ``run_ffmpeg`` / ``run_ffprobe`` directly.
- Vendored code (GPT-SoVITS / UVR5) calls bare ``ffmpeg`` / ``ffprobe`` internally;
  ``ensure_ffmpeg_on_path()`` prepends this directory to ``PATH`` so those calls
  also resolve to the bundled copy. Called once at APP.py startup, it applies globally.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.paths import project_root

__all__ = [
    "FFMPEG_DIR",
    "FFMPEG_BIN",
    "FFPROBE_BIN",
    "FFPLAY_BIN",
    "ensure_ffmpeg_on_path",
    "configure_pydub",
    "run_ffmpeg",
    "run_ffprobe",
]

FFMPEG_DIR = project_root / "dependencies" / "FFmpeg"

_EXE_SUFFIX = ".exe" if os.name == "nt" else ""
FFMPEG_BIN = FFMPEG_DIR / f"ffmpeg{_EXE_SUFFIX}"
FFPROBE_BIN = FFMPEG_DIR / f"ffprobe{_EXE_SUFFIX}"
FFPLAY_BIN = FFMPEG_DIR / f"ffplay{_EXE_SUFFIX}"


def ensure_ffmpeg_on_path() -> Path:
    """Prepend ``dependencies/FFmpeg`` to ``PATH`` (idempotent), returning that directory.

    Vendored code spawns subprocesses via ``os.system('ffmpeg ...')`` /
    ``ffmpeg-python`` ``cmd=["ffmpeg", ...]``; both Windows and POSIX resolve
    executables through ``PATH``, so after prepending they hit the bundled copy.
    """
    ffmpeg_dir = str(FFMPEG_DIR.resolve())
    current = os.environ.get("PATH", "")
    entries = [e for e in current.split(os.pathsep) if e]
    if not any(os.path.normcase(e) == os.path.normcase(ffmpeg_dir) for e in entries):
        os.environ["PATH"] = ffmpeg_dir + (os.pathsep + current if current else "")
    return FFMPEG_DIR


def _creationflags() -> int:
    """Do not pop a console window on Windows (CREATE_NO_WINDOW); return 0 on other platforms."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def configure_pydub() -> bool:
    """Point pydub's ffmpeg converter at the bundled FFmpeg and ensure it is usable.

    pydub's ``AudioSegment.converter`` uses the bare name ``ffmpeg``;
    ``mediainfo_json`` and the playback modules look up ``ffprobe`` / ``ffplay``
    via ``PATH``. This function also prepends ``dependencies/FFmpeg`` to ``PATH``,
    so all of pydub's low-level calls hit the bundled copy. Returns ``False`` if
    pydub is missing or fails to import.
    """
    try:
        from pydub import AudioSegment  # noqa: PLC0415

        AudioSegment.converter = str(FFMPEG_BIN)
        ensure_ffmpeg_on_path()
        return True
    except Exception:
        return False


def run_ffmpeg(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run the bundled ffmpeg via explicit path (no console window by default)."""
    kwargs.setdefault("creationflags", _creationflags())
    return subprocess.run([str(FFMPEG_BIN), *args], **kwargs)


def run_ffprobe(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run the bundled ffprobe via explicit path (no console window by default)."""
    kwargs.setdefault("creationflags", _creationflags())
    return subprocess.run([str(FFPROBE_BIN), *args], **kwargs)


# Takes effect on import (idempotent): any module referencing this module's
# constants means bare ``ffmpeg`` / ``ffprobe`` subprocess calls from vendored
# code also resolve through PATH to the bundled copy.
ensure_ffmpeg_on_path()
