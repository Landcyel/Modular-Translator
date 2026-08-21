"""MOSS 长音频分窗转写测试（不加载真实模型/不调用 ffmpeg）。

背景：1217s 音频整段转写时，Qwen3 全注意力一次性分配 ≈15.4 GiB
（6GB 卡 CUDA OOM）。分窗策略：>max_audio_sec 按滑动窗口转写，
段时间戳平移回全局轴、边界去重合并、每窗 max_new_tokens 收敛。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.moss.moss_transcriber import (
    DEFAULT_OVERLAP_SEC,
    DEFAULT_WINDOW_SEC,
    MossTranscriber,
)


class _FakeTask:
    id = "t1"

    def __init__(self, path="audio.mp3", args=None):
        self.file_path = path
        self.configs = {"args": args or {}}
        self._pause_event = None


class _FakeResult:
    def __init__(self, text, generated_tokens=8, elapsed_sec=1.0):
        self.text = text
        self.generated_tokens = generated_tokens
        self.elapsed_sec = elapsed_sec

    def to_dict(self):
        return {
            "text": self.text,
            "generated_tokens": self.generated_tokens,
            "elapsed_sec": self.elapsed_sec,
        }


class _WindowFakeRunner:
    """按调用顺序返回预设窗口转录文本（本地时间轴）。"""

    SUPPORTS_PARTIAL_TEXT = False
    _device = None
    _dtype = None

    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append((str(audio_path), dict(kwargs)))
        return _FakeResult(
            self.transcripts[len(self.calls) - 1],
            generated_tokens=20 + len(self.calls),
        )


def _make_fake_clip(tmp_path):
    def _clip(self, audio_path, task, idx, start, end):
        base = Path(tmp_path) / "moss_clips"
        base.mkdir(parents=True, exist_ok=True)
        p = base / f"win_{idx:03d}_{int(start):06d}.wav"
        p.write_bytes(b"RIFF")
        return p

    return _clip


# ── 窗口规划（纯函数）────────────────────────────────────────

def test_plan_windows_sliding_with_overlap():
    tx = MossTranscriber(object())
    windows = tx._plan_windows(610.0, {"max_audio_sec": 180, "overlap_sec": 10})
    assert windows == [(0.0, 180.0), (170.0, 350.0), (340.0, 520.0), (510.0, 610.0)]


def test_plan_windows_short_audio_single_path():
    tx = MossTranscriber(object())
    assert tx._plan_windows(180.0, {"max_audio_sec": 180, "overlap_sec": 10}) == []
    assert tx._plan_windows(30.0, {"max_audio_sec": 180, "overlap_sec": 10}) == []


def test_plan_windows_defaults_and_disable():
    tx = MossTranscriber(object())
    assert tx._plan_windows(610.0, {})[0][1] == DEFAULT_WINDOW_SEC
    assert tx._plan_windows(610.0, {"max_audio_sec": 0}) == []
    assert tx._plan_windows(None, {}) == []


# ── 执行级分窗：时间轴平移 + token 收敛 + 进度单调 ────────────

def test_execute_windows_long_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 610.0
    )
    monkeypatch.setattr(MossTranscriber, "_make_window_clip", _make_fake_clip(tmp_path))

    runner = _WindowFakeRunner([
        "[0.00][S01]第一段[10.00]\n[10.00][S01]第二段[20.00]",   # win0: 0-180
        "[0.00][S01]重叠段[5.00]\n[10.00][S01]第三段[20.00]",    # win1: 170-350
        "[15.00][S01]第四段[25.00]",                             # win2: 340-520
        "[15.00][S01]第五段[25.00]",                             # win3: 510-610
    ])
    tx = MossTranscriber(
        runner,
        # vram_auto_fit=False：固定窗口，保证测试不受探测环境影响
        defaults={"max_audio_sec": 180, "overlap_sec": 10, "vram_auto_fit": False},
    )
    progress = []
    result = tx.execute(
        _FakeTask("audio.mp3", args={"max_new_tokens": 65536}),
        progress_callback=lambda *args: progress.append(args),
    )

    # 4 个窗口各转写一次；窗口音频为切出的 wav
    assert len(runner.calls) == 4
    for path, _kw in runner.calls:
        assert path.endswith(".wav")

    segs = result["segments"]
    texts = [s["text"] for s in segs]
    # 窗口 1 中全局 170-175 的“重叠段”应被丢弃（完全落在重叠区）
    assert "重叠段" not in texts
    assert texts == ["第一段", "第二段", "第三段", "第四段", "第五段"]
    # 段时间戳已平移回全局时间轴并排序
    assert [s["start"] for s in segs] == pytest.approx([0.0, 10.0, 180.0, 355.0, 525.0])
    # 合并后 id 重排，无跨窗口重复
    assert [s["id"] for s in segs] == [f"seg_{i:04d}" for i in range(1, 6)]

    # 每窗 max_new_tokens 按窗口时长收敛（180s → 3136，100s → 1856）
    assert [c[1]["max_new_tokens"] for c in runner.calls] == [3136, 3136, 3136, 1856]

    info = result["info"]
    assert info["windows"] == 4
    assert info["generated_tokens"] == sum(20 + i for i in range(1, 5))
    assert info["elapsed_sec"] == pytest.approx(4.0)

    # 进度：每窗完成推进到窗尾 + 收尾补满，全程单调不减
    poss = [p[0] for p in progress if p[3].get("status") == "transcribing"]
    assert poss == sorted(poss)
    assert progress[-1][0] == pytest.approx(610.0)
    assert progress[-1][1] == pytest.approx(610.0)


def test_execute_short_audio_remains_single_call(monkeypatch, tmp_path):
    """未超阈值：不改动既有单次整段转写行为。"""
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 30.0
    )
    monkeypatch.setattr(MossTranscriber, "_make_window_clip", _make_fake_clip(tmp_path))
    runner = _WindowFakeRunner(["[0.00][S01]短音频[2.00]"])
    tx = MossTranscriber(
        runner, defaults={"max_audio_sec": 180, "vram_auto_fit": False},
    )
    result = tx.execute(_FakeTask("audio.wav", args={"max_new_tokens": 65536}))
    assert len(runner.calls) == 1
    assert runner.calls[0][1]["max_new_tokens"] == 65536
    assert [s["text"] for s in result["segments"]] == ["短音频"]


# ── 边界合并（纯函数）─────────────────────────────────────────

def test_merge_drops_segments_inside_overlap():
    tx = MossTranscriber(object())
    windows = [(0.0, 10.0), (8.0, 18.0)]
    # 窗口段已平移回全局时间轴（_transcribe_windowed 的输入契约）
    window_segs = [
        [{"start": 0.0, "end": 10.0, "speaker": "S01", "text": "A"}],
        [
            {"start": 8.0, "end": 10.0, "speaker": "S01", "text": "A"},   # 重叠区内 → 丢弃
            {"start": 10.0, "end": 18.0, "speaker": "S01", "text": "B"},  # 核心区 → 保留
        ],
    ]
    out = tx._merge_window_segments(window_segs, windows, {})
    assert [(s["start"], s["end"], s["text"]) for s in out] == [
        (0.0, 10.0, "A"), (10.0, 18.0, "B"),
    ]


def test_merge_cross_boundary_segment_gets_trimmed():
    tx = MossTranscriber(object())
    windows = [(0.0, 10.0), (8.0, 18.0)]
    window_segs = [
        [{"start": 0.0, "end": 10.0, "speaker": "S01", "text": "A"}],
        [
            # 9-13 跨边界保留但起点裁到 10；其后 10-18 段再裁到 13
            {"start": 9.0, "end": 13.0, "speaker": "S01", "text": "AB"},
            {"start": 10.0, "end": 18.0, "speaker": "S01", "text": "C"},
        ],
    ]
    out = tx._merge_window_segments(window_segs, windows, {})
    assert [(s["start"], s["end"], s["text"]) for s in out] == [
        (0.0, 10.0, "A"), (10.0, 13.0, "AB"), (13.0, 18.0, "C"),
    ]


def test_merge_drops_highly_similar_duplicate():
    tx = MossTranscriber(object())
    out = tx._merge_window_segments(
        [
            [{"start": 0.0, "end": 10.0, "speaker": "S01", "text": "今天天气很好"}],
            [
                {"start": 8.0, "end": 10.0, "speaker": "S01", "text": "今天天气很好"},
                {"start": 10.0, "end": 18.0, "speaker": "S01", "text": "然后出发"},
            ],
        ],
        [(0.0, 10.0), (8.0, 18.0)],
        {},
    )
    texts = [s["text"] for s in out]
    assert texts == ["今天天气很好", "然后出发"]


def test_window_token_budget():
    tx = MossTranscriber(object())
    assert tx._window_max_new_tokens(65536, 180.0) == 3136
    assert tx._window_max_new_tokens(0, 180.0) == 3136
    assert tx._window_max_new_tokens(100, 180.0) == 100
    assert tx._window_max_new_tokens(None, 10.0) == 1024
