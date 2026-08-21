"""core.writer 字幕导出测试：标准 LRC 格式（参考 MOSS subtitle.lrc）。"""
from __future__ import annotations

from core.writer import (
    Segment,
    format_lrc_time,
    lrc_write,
    merge_repeated_segments,
)


def test_format_lrc_time():
    assert format_lrc_time(7.44) == "00:07.44"
    assert format_lrc_time(61.46) == "01:01.46"
    assert format_lrc_time(185.62) == "03:05.62"


def test_lrc_write_with_speaker(tmp_path):
    """MOSS 段（带 speaker）→ 标准 LRC：只有开始时间 + <说话人>前缀。"""
    segments = [
        {"id": "seg_0001", "start": 7.44, "end": 8.84,
         "speaker": "S01", "text": "はい、すぐりです。"},
        {"id": "seg_0002", "start": 11.32, "end": 13.32,
         "speaker": "S01", "text": "何かありました？"},
        {"id": "seg_0008", "start": 61.46, "end": 65.31,
         "speaker": "S01", "text": "私もちょっと会いたいなと思っていたので嬉しいです。"},
    ]
    out = tmp_path / "subtitle.lrc"
    lrc_write(segments, str(out))
    assert out.read_text(encoding="utf-8") == (
        "[00:07.44]<S01>はい、すぐりです。\n"
        "[00:11.32]<S01>何かありました？\n"
        "[01:01.46]<S01>私もちょっと会いたいなと思っていたので嬉しいです。\n"
    )


def test_lrc_write_without_speaker(tmp_path):
    """Whisper 段（无 speaker）→ 省略说话人前缀，仍为标准 LRC 单时间戳。"""
    segments = [
        {"start": 0.0, "end": 1.2, "text": "Hello."},
        {"start": 1.2, "end": 2.5, "text": "World."},
    ]
    out = tmp_path / "plain.lrc"
    lrc_write(segments, str(out))
    assert out.read_text(encoding="utf-8") == (
        "[00:00.00]Hello.\n"
        "[00:01.20]World.\n"
    )


def test_merge_repeated_segments_keeps_speaker():
    merged = merge_repeated_segments([
        Segment(0.0, 0.3, "重复", "S01"),
        Segment(0.3, 1.0, "重复", "S01"),
        Segment(1.2, 2.0, "其他", "S01"),
    ])
    assert len(merged) == 2
    assert merged[0].speaker == "S01"
    assert merged[0].text == "重复"
    assert merged[1].speaker == "S01"
