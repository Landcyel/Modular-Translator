"""Bundled FFmpeg locator — 项目内 FFmpeg/ffprobe/ffplay 调用的统一入口。

项目刚需 FFmpeg：随项目携带一份到 ``dependencies/FFmpeg/``，不依赖系统安装。
- 新代码请优先使用本模块的显式路径常量 ``FFMPEG_BIN`` / ``FFPROBE_BIN`` /
  ``FFPLAY_BIN``，或直接调用 ``run_ffmpeg`` / ``run_ffprobe``。
- vendored 代码（GPT-SoVITS / UVR5）内部以裸名 ``ffmpeg`` / ``ffprobe`` 调用，
  通过 ``ensure_ffmpeg_on_path()`` 把本目录前插到 ``PATH``，使这些调用同样
  解析到项目自带副本。APP.py 启动时调用一次即可全局生效。
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
    """把 ``dependencies/FFmpeg`` 前插到 ``PATH``（幂等），返回该目录。

    vendored 代码通过 ``os.system('ffmpeg ...')`` / ``ffmpeg-python`` 的
    ``cmd=["ffmpeg", ...]`` 启动子进程，Windows 与 POSIX 均按 ``PATH``
    解析可执行文件；前插后即命中项目自带副本。
    """
    ffmpeg_dir = str(FFMPEG_DIR.resolve())
    current = os.environ.get("PATH", "")
    entries = [e for e in current.split(os.pathsep) if e]
    if not any(os.path.normcase(e) == os.path.normcase(ffmpeg_dir) for e in entries):
        os.environ["PATH"] = ffmpeg_dir + (os.pathsep + current if current else "")
    return FFMPEG_DIR


def _creationflags() -> int:
    """Windows 下不弹控制台窗口（CREATE_NO_WINDOW）；其它平台返回 0。"""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def configure_pydub() -> bool:
    """把 pydub 的 ffmpeg 转换器指向项目自带 FFmpeg，并确保其可用。

    pydub 的 ``AudioSegment.converter`` 使用裸名 ``ffmpeg``；``mediainfo_json``
    与播放模块则通过 ``PATH`` 查找 ``ffprobe`` / ``ffplay``。本函数同时把
    ``dependencies/FFmpeg`` 前插到 ``PATH``，因此 pydub 的全部底层调用都会
    命中项目自带副本。pydub 缺失/导入失败时返回 ``False``。
    """
    try:
        from pydub import AudioSegment  # noqa: PLC0415

        AudioSegment.converter = str(FFMPEG_BIN)
        ensure_ffmpeg_on_path()
        return True
    except Exception:
        return False


def run_ffmpeg(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """以显式路径运行项目自带 ffmpeg（默认不弹控制台窗口）。"""
    kwargs.setdefault("creationflags", _creationflags())
    return subprocess.run([str(FFMPEG_BIN), *args], **kwargs)


def run_ffprobe(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """以显式路径运行项目自带 ffprobe（默认不弹控制台窗口）。"""
    kwargs.setdefault("creationflags", _creationflags())
    return subprocess.run([str(FFPROBE_BIN), *args], **kwargs)


# 导入即生效（幂等）：任何模块只要引用本模块常量，vendored 代码的裸名
# ``ffmpeg`` / ``ffprobe`` 子进程调用也会经 PATH 命中项目自带副本。
ensure_ffmpeg_on_path()
