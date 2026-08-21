from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""   # 说话人（如 MOSS 段的 "S01"；无 speaker 的段自动省略前缀）

def merge_repeated_segments(
    segments,
    max_gap=0.15,
    short_dur=0.4,
    sim_threshold=0.9,
):
    if not segments:
        return []

    merged = []
    group = [segments[0]]
    def text_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.strip(), b.strip()).ratio()

    def should_merge(prev, cur):
        gap = cur.start - prev.end
        prev_dur = prev.end - prev.start
        cur_dur = cur.end - cur.start
        sim = text_similarity(prev.text, cur.text)

        return (
            gap <= max_gap
            and sim >= sim_threshold
            and (prev_dur <= short_dur or cur_dur <= short_dur)
        )

    def flush_group(group):
        if len(group) == 1:
            return group[0]

        longest = max(group, key=lambda s: s.end - s.start)

        return Segment(
            start=group[0].start,
            end=group[-1].end,
            text=longest.text,
            speaker=group[0].speaker,   # 同组说话人一致，保留首段
        )

    for seg in segments[1:]:
        prev = group[-1]
        if should_merge(prev, seg):
            group.append(seg)
        else:
            merged.append(flush_group(group))
            group = [seg]

    merged.append(flush_group(group))
    return merged


def format_lrc_time(seconds: float) -> str:
    """LRC 时间轴：MM:SS.xx（分:秒.百分秒）。"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def lrc_write(segments, output_path: str):
    """写标准 LRC 歌词：``[mm:ss.xx]<说话人>文本``（有 speaker 字段时带说话人前缀）。

    MOSS 段带 speaker（如 ``[00:07.44]<S01>文本``）；无 speaker 的段
    自动省略前缀。只有开始时间，无结束时间。
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    segments = merge_repeated_segments(_coerce_segments(segments))
    with open(output_file, 'w', encoding='utf-8') as f:
        for seg in segments:
            start_time = format_lrc_time(seg.start)
            prefix = f"<{seg.speaker}>" if seg.speaker else ""
            f.write(f"[{start_time}]{prefix}{seg.text}\n")


def _coerce_segments(segments) -> list:
    """将 dict 段（{"start","end","text","speaker"?}）或 Segment 对象统一归一化为 Segment 列表。"""
    out = []
    for s in segments:
        if isinstance(s, dict):
            out.append(Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")),
                speaker=str(s.get("speaker", "") or ""),
            ))
        else:
            out.append(s)
    return out


def _fmt_srt_time(seconds: float) -> str:
    """SRT 时间轴：HH:MM:SS,mmm。"""
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    """VTT 时间轴：HH:MM:SS.mmm。"""
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def srt_write(segments, output_path: str):
    """写 SRT 字幕：序号 + HH:MM:SS,mmm --> HH:MM:SS,mmm + 文本。"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    segments = merge_repeated_segments(_coerce_segments(segments))
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}\n")
            f.write(f"{seg.text}\n\n")


def vtt_write(segments, output_path: str):
    """写 VTT 字幕：WEBVTT 头 + HH:MM:SS.mmm --> HH:MM:SS.mmm。"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    segments = merge_repeated_segments(_coerce_segments(segments))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{_fmt_vtt_time(seg.start)} --> {_fmt_vtt_time(seg.end)}\n")
            f.write(f"{seg.text}\n\n")


