"""MOSS 切块方案新行为测试（显存预算 / 静音边界 / 硬切修复 / OOM 退避）。

不加载真实模型、不调用真实 ffmpeg：包络取数器与切片器均为假实现。
校准常量引用自 ``core.moss.moss_transcriber`` 模块私有常量，保证
预算公式与常量同步演进。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.moss import moss_transcriber as mt
from core.moss.moss_transcriber import MossTranscriber
from core.moss.silence_probe import find_silence_cut


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


def _make_fake_clip(tmp_path):
    def _clip(self, audio_path, task, idx, start, end):
        base = Path(tmp_path) / "moss_clips"
        base.mkdir(parents=True, exist_ok=True)
        p = base / f"win_{idx:03d}_{int(start):06d}.wav"
        p.write_bytes(b"RIFF")
        return p

    return _clip


# ── 显存预算窗口计算 ──────────────────────────────────────────

def _budget_free_for_sec(sec: float, ratio: float = 0.7) -> float:
    """给定目标窗口秒数，反推所需的空闲显存字节数。"""
    need = mt._ATTN_BYTES_PER_SEC2 * sec * sec + mt._KV_BYTES_PER_SEC * sec
    return (mt._WEIGHTS_ESTIMATE_BYTES + mt._FIXED_SLACK_BYTES + need) / ratio


def test_resolve_window_sec_solves_quadratic(monkeypatch):
    tx = MossTranscriber(object())
    monkeypatch.setattr(
        MossTranscriber, "_probe_free_vram",
        lambda self, m: int(_budget_free_for_sec(180.0)),
    )
    sec = tx._resolve_window_sec({"max_audio_sec": 600, "min_window_sec": 60})
    assert sec == pytest.approx(180.0, abs=2.0)


def test_resolve_window_sec_clamps_to_max_and_min(monkeypatch):
    tx = MossTranscriber(object())
    # 显存充裕 → 钳制到用户上限
    monkeypatch.setattr(
        MossTranscriber, "_probe_free_vram",
        lambda self, m: int(_budget_free_for_sec(1000.0)),
    )
    assert tx._resolve_window_sec({"max_audio_sec": 180}) == pytest.approx(180.0)
    # 显存极少 → 保底最小窗
    monkeypatch.setattr(MossTranscriber, "_probe_free_vram", lambda self, m: 0)
    assert tx._resolve_window_sec({"max_audio_sec": 180, "min_window_sec": 60}) == pytest.approx(60.0)
    # min > max 时不突破用户上限
    assert tx._resolve_window_sec({"max_audio_sec": 30, "min_window_sec": 60}) == pytest.approx(30.0)


def test_resolve_window_sec_fallbacks(monkeypatch):
    tx = MossTranscriber(object())
    # CPU 配置 → 不探测（走真实 _probe_free_vram 的设备分支）
    assert tx._resolve_window_sec({"max_audio_sec": 180, "device": "cpu"}) == pytest.approx(180.0)
    # 探测失败 → 回落 max_audio_sec
    monkeypatch.setattr(MossTranscriber, "_probe_free_vram", lambda self, m: None)
    assert tx._resolve_window_sec({"max_audio_sec": 180}) == pytest.approx(180.0)
    # 关闭自适应 → 直接使用上限
    assert tx._resolve_window_sec({"max_audio_sec": 180, "vram_auto_fit": False}) == pytest.approx(180.0)
    # 阈值为 0 → 禁用分窗
    assert tx._resolve_window_sec({"max_audio_sec": 0}) is None


# ── 静音感知边界规划 ──────────────────────────────────────────

def test_plan_windows_with_flags_silence_cut():
    """首个边界带内 158-166s 存在静音 → 切点移入静音（flag=False）。"""
    import numpy as np

    tx = MossTranscriber(object())

    def getter(lo, hi):
        if not (149.9 < lo < 150.1):
            return None
        rms = np.full(1200, -15.0, dtype=np.float32)   # 30s 带 @25ms
        rms[320:641] = -45.0                           # 158-166s 静音
        return rms.tolist()

    windows, flags = tx._plan_windows_with_flags(
        610.0, {"max_audio_sec": 180, "overlap_sec": 10},
        getter, window_sec=180.0,
    )
    assert flags[0] is False
    assert 161.5 <= windows[0][1] <= 162.5   # 静音段中点 ≈ 162s
    assert windows[1][0] == pytest.approx(windows[0][1] - 10.0)
    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(610.0)
    # 单窗时长硬性 ≤ 预算窗长（显存安全的构造性保证）
    for s, e in windows:
        assert e - s <= 180.0 + 1e-6
    # 无静音数据的边界 → 硬切标记
    assert flags[-1] is False
    assert sum(flags) == len(flags) - 2   # 仅首边界为静音切分


def test_plan_windows_hard_cut_matches_legacy_arithmetic():
    """无包络数据时与旧算术滑动窗口完全一致（含 flags）。"""
    tx = MossTranscriber(object())
    windows, flags = tx._plan_windows_with_flags(
        610.0, {"max_audio_sec": 180, "overlap_sec": 10}, None, window_sec=180.0,
    )
    assert windows == [(0.0, 180.0), (170.0, 350.0), (340.0, 520.0), (510.0, 610.0)]
    assert flags == [True, True, True, False]


def test_plan_windows_silence_never_exceeds_budget():
    """静音点在带回看范围的最右端时，切点仍不越过预算窗长。"""
    import numpy as np

    tx = MossTranscriber(object())

    def getter(lo, hi):
        if not (149.9 < lo < 150.1):
            return None
        rms = np.full(1200, -15.0, dtype=np.float32)
        rms[1000:1200] = -45.0         # 175-180s 静音（带右缘，占带 16.7%）
        return rms.tolist()

    windows, flags = tx._plan_windows_with_flags(
        610.0, {"max_audio_sec": 180, "overlap_sec": 10},
        getter, window_sec=180.0,
    )
    assert windows[0][1] <= 180.0 + 1e-6
    assert flags[0] is False
    assert 176.5 <= windows[0][1] <= 178.5   # 静音段中点 ≈ 177.5s


def test_plan_windows_short_audio_returns_empty():
    tx = MossTranscriber(object())
    assert tx._plan_windows_with_flags(180.0, {"max_audio_sec": 180}) == ([], [])
    assert tx._plan_windows_with_flags(None, {"max_audio_sec": 180}) == ([], [])
    assert tx._plan_windows_with_flags(610.0, {"max_audio_sec": 0}) == ([], [])


# ── 硬切边界文本修复 ──────────────────────────────────────────

def test_repair_merges_truncated_tail_with_head():
    tx = MossTranscriber(object())
    windows = [(0.0, 180.0), (170.0, 400.0)]
    window_segs = [
        [{"start": 0.0, "end": 10.0, "speaker": "S01", "text": "A"},
         {"start": 168.0, "end": 180.0, "speaker": "S01", "text": "我们去看"}],
        [{"start": 168.0, "end": 190.0, "speaker": "S01", "text": "去看电影了"},
         {"start": 190.0, "end": 210.0, "speaker": "S01", "text": "B"}],
    ]
    out = tx._merge_window_segments(
        window_segs, windows, {}, hard_flags=[True, False],
    )
    assert [s["text"] for s in out] == ["A", "我们去看电影了", "B"]
    merged = out[1]
    assert merged["start"] == pytest.approx(168.0)   # 保留真实起点（不被裁剪到切点）
    assert merged["end"] == pytest.approx(190.0)


def test_repair_skips_silence_boundary():
    """静音边界（flag=False）不做修复：段原样保留。"""
    tx = MossTranscriber(object())
    windows = [(0.0, 180.0), (170.0, 400.0)]
    window_segs = [
        [{"start": 168.0, "end": 180.0, "speaker": "S01", "text": "我们去看"}],
        [{"start": 179.8, "end": 185.0, "speaker": "S01", "text": "看电影了"}],
    ]
    out = tx._merge_window_segments(
        window_segs, windows, {}, hard_flags=[False, False],
    )
    assert [s["text"] for s in out] == ["我们去看", "看电影了"]
    # 非修复段仍按重叠裁剪规则处理（179.8 → 180）
    assert out[1]["start"] == pytest.approx(180.0)


def test_repair_does_not_pair_unrelated_segments():
    """句尾/句头文本无关 → 不配对，各自保留。"""
    tx = MossTranscriber(object())
    windows = [(0.0, 180.0), (170.0, 400.0)]
    window_segs = [
        [{"start": 168.0, "end": 180.0, "speaker": "S01", "text": "今天天气很好"}],
        [{"start": 180.0, "end": 190.0, "speaker": "S01", "text": "完全不同的话题"}],
    ]
    out = tx._merge_window_segments(
        window_segs, windows, {}, hard_flags=[True, False],
    )
    assert [s["text"] for s in out] == ["今天天气很好", "完全不同的话题"]


# ── 运行时 OOM 退避 ───────────────────────────────────────────

class _OomThenOkRunner:
    SUPPORTS_PARTIAL_TEXT = False
    _device = None
    _dtype = None

    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append((str(audio_path), dict(kwargs)))
        n = len(self.calls)
        if n == 1:
            raise RuntimeError("CUDA out of memory")
        return _FakeResult(f"[100.00][S01]第{n}段[103.00]", generated_tokens=10)


def test_execute_oom_retreat_replans_tail(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 610.0
    )
    monkeypatch.setattr(MossTranscriber, "_make_window_clip", _make_fake_clip(tmp_path))
    cache_clears = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_clears.append(1))

    runner = _OomThenOkRunner()
    tx = MossTranscriber(
        runner,
        defaults={"max_audio_sec": 180, "overlap_sec": 10, "vram_auto_fit": False},
    )
    progress = []
    result = tx.execute(
        _FakeTask("audio.mp3", args={"max_new_tokens": 65536}),
        progress_callback=lambda *args: progress.append(args),
    )

    # 第 1 次调用 OOM → 缩窗至 126s 重规划：1 次失败 + 6 个成功窗口
    assert len(runner.calls) == 7
    assert result["info"]["windows"] == 6
    assert result["info"]["chunking"]["hard_boundaries"] == 5
    assert cache_clears, "OOM 退避应清空 CUDA 缓存"
    texts = [s["text"] for s in result["segments"]]
    assert texts == [f"第{n}段" for n in range(2, 8)]
    # 进度单调不减且收尾满格
    poss = [p[0] for p in progress if p[3].get("status") == "transcribing"]
    assert poss == sorted(poss)
    assert progress[-1][0] == pytest.approx(610.0)
    # 缩窗后每窗 max_new_tokens 收敛（126s → 2271/2272 浮点邻域，末窗 30s → 1024 保底）
    budgets = [c[1]["max_new_tokens"] for c in runner.calls[1:]]
    assert all(b in (2271, 2272) for b in budgets[:5])
    assert budgets[5] == 1024


class _AlwaysOomRunner(_OomThenOkRunner):
    def transcribe(self, audio_path, **kwargs):
        self.calls.append((str(audio_path), dict(kwargs)))
        raise RuntimeError("CUDA out of memory")


def test_execute_oom_retreat_exhausted_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 610.0
    )
    monkeypatch.setattr(MossTranscriber, "_make_window_clip", _make_fake_clip(tmp_path))
    runner = _AlwaysOomRunner()
    tx = MossTranscriber(
        runner,
        defaults={"max_audio_sec": 180, "overlap_sec": 10, "vram_auto_fit": False},
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        tx.execute(_FakeTask("audio.mp3"))
    # 初始 1 次 + 3 次退避重试（180 → 126 → 88.2 → 61.7s），随后抛出
    assert len(runner.calls) == 4


# ── 静音探测纯函数 ────────────────────────────────────────────

def test_find_silence_cut_scores_runs():
    import numpy as np

    rms = np.full(800, -15.0, dtype=np.float32)
    rms[200:400] = -45.0    # 5-10s 静音（长但远离目标）
    rms[600:700] = -45.0    # 15-17.5s 静音（短但更接近目标 118s）
    cut = find_silence_cut(
        rms.tolist(), band_start=100.0, target=118.0, silence_min_sec=0.35,
    )
    assert 115.5 < cut < 117.0   # 选中第二段静音中点 ≈ 116.25s


def test_find_silence_cut_returns_none_when_no_dynamics():
    import numpy as np

    rms = np.full(400, -15.0, dtype=np.float32)   # 动态范围为零 → 无可靠静音
    assert find_silence_cut(rms.tolist(), 0.0, 8.0) is None
    assert find_silence_cut(None, 0.0, 8.0) is None
