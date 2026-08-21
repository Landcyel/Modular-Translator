from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""   # speaker (e.g. "S01" for MOSS segments; segments without a speaker omit the prefix)

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
            speaker=group[0].speaker,   # all segments in the group share the speaker; keep the first
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
    """LRC timeline: MM:SS.xx (minutes:seconds.centiseconds)."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def lrc_write(segments, output_path: str):
    """Write standard LRC lyrics: ``[mm:ss.xx]<speaker>text`` (with the speaker prefix when a speaker field exists).

    MOSS segments carry a speaker (e.g. ``[00:07.44]<S01>text``); segments without a
    speaker automatically omit the prefix. Only the start time is used; no end time.
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
    """Normalize dict segments ({"start","end","text","speaker"?}) or Segment objects into a Segment list."""
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
    """SRT timeline: HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    """VTT timeline: HH:MM:SS.mmm."""
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def srt_write(segments, output_path: str):
    """Write SRT subtitles: index + HH:MM:SS,mmm --> HH:MM:SS,mmm + text."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    segments = merge_repeated_segments(_coerce_segments(segments))
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}\n")
            f.write(f"{seg.text}\n\n")


def vtt_write(segments, output_path: str):
    """Write VTT subtitles: WEBVTT header + HH:MM:SS.mmm --> HH:MM:SS.mmm."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    segments = merge_repeated_segments(_coerce_segments(segments))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{_fmt_vtt_time(seg.start)} --> {_fmt_vtt_time(seg.end)}\n")
            f.write(f"{seg.text}\n\n")


