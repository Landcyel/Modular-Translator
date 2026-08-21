"""LRC 歌词导出（标准 [mm:ss.cs] 时间戳 + 可选说话人前缀）。

服务端导出的 srt/ass 不带 LRC，此工具从内存 segments 客户端生成。
"""
from __future__ import annotations

from pathlib import Path


def format_lrc_time(seconds: float) -> str:
    """秒 → LRC 时间戳 ``mm:ss.cs``（分钟:秒.厘秒）。"""
    seconds = max(0.0, float(seconds))
    mm = int(seconds) // 60
    ss = int(seconds) % 60
    cs = min(int(round((seconds - int(seconds)) * 100)), 99)
    return f"{mm:02d}:{ss:02d}.{cs:02d}"


def export_lrc(segments, out_path, show_speaker: bool = True) -> str:
    """把 segments 写成 LRC 歌词文件，返回写入的文件路径。

    每行格式：``[mm:ss.cs]<S01>文本``（``show_speaker=True`` 时带说话人前缀）。
    """
    lines = []
    for seg in segments:
        ts = format_lrc_time(float(seg["start"]))
        prefix = f"<{seg['speaker']}>" if show_speaker else ""
        lines.append(f"[{ts}]{prefix}{seg['text']}")
    out = Path(out_path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)
