"""GsvTTSExecutor 自适应重试单元测试（假引擎注入，无 torch 依赖）。

覆盖场景:
- 输出过短（< 参考时长×0.6，疑似提前 EOS）→ 阶梯下调 repetition_penalty 重试
- 正常输出不重试 / rp 已到底不重试 / 重试耗尽返回最后结果
- min_ref_ratio 可配置覆盖（跳过自适应路径）
- 重试期间进度回调单调不减
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import soundfile as sf

from core.contracts import Task
from core.executor import GsvTTSExecutor

SR = 32000


class _FakeGsvEngine:
    """按预设 specs 产出假音频的引擎：每次 synth 调用 yield 指定时长的片段。"""

    version = "v2ProPlus"     # 免 v3/v4 的 prompt_text 强校验
    last_seed = 42

    def __init__(self, calls_specs):
        # calls_specs: list[list[float]] — 每次调用的片段时长列表（秒）
        self._specs = list(calls_specs)
        self.calls: list[tuple[str, dict]] = []   # (method, kwargs)
        self.stop_called = False

    def _synth(self, method: str, **kwargs):
        self.calls.append((method, kwargs))
        spec = self._specs.pop(0) if self._specs else [5.0]
        for d in spec:
            yield SR, np.zeros(int(SR * d), dtype=np.int16)

    def synth_stream(self, text, text_lang, ref_audio_path=None,
                     prompt_text="", prompt_lang="", **params):
        yield from self._synth(
            "stream", text=text, text_lang=text_lang,
            ref_audio_path=ref_audio_path, prompt_text=prompt_text,
            prompt_lang=prompt_lang, **params,
        )

    def synth_cross_speaker(self, text, text_lang, emotion_ref_audio,
                            emotion_text, emotion_lang, role_ref_audio, **params):
        yield from self._synth(
            "cross", text=text, text_lang=text_lang,
            emotion_ref_audio=emotion_ref_audio, emotion_text=emotion_text,
            emotion_lang=emotion_lang, role_ref_audio=role_ref_audio,
            **params,
        )

    def stop(self):
        self.stop_called = True


def _make_ref_wav(tmp_path, seconds: float = 4.0) -> str:
    """生成 3~10s 范围内的参考音频（4.0s → 过短阈值 2.4s）。"""
    path = tmp_path / "ref.wav"
    t = np.linspace(0.0, seconds, int(SR * seconds), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(path), y, SR)
    return str(path)


def _make_task(tmp_path, engine, args=None, text="由我来做路线规划吗？没问题，我看一眼就能知道怎么走最快。"):
    ref = _make_ref_wav(tmp_path)
    merged = {
        "ref_mode": "dual",
        "ref_audio_path": ref,
        "role_ref_audio": ref,      # 同一 4s wav 同时充当情绪/角色参考
        "prompt_text": text,
        "prompt_lang": "zh",
        "text_lang": "zh",
        "repetition_penalty": 1.35,
    }
    merged.update(args or {})
    task = Task(
        task_type="gsv",
        file_path=text,
        file_name="t.txt",
        configs={"args": merged},
        id="t1",
    )
    return GsvTTSExecutor(engine), task


def _run(executor, task):
    logs = []
    executor.set_on_log(lambda level, msg: logs.append((level, msg)))
    result = executor.execute(task)
    return result, logs


def test_short_output_retries_with_lower_rp(tmp_path):
    """第一次过短（1.0s < 2.4s）→ 自动降 rp（0.05 步长）重试；第二次完整 → 成功。"""
    engine = _FakeGsvEngine([[1.0], [4.5]])
    executor, task = _make_task(tmp_path, engine)

    result, logs = _run(executor, task)

    assert [m for m, _ in engine.calls] == ["cross", "cross"]
    assert engine.calls[0][1]["repetition_penalty"] == pytest.approx(1.35)
    assert engine.calls[1][1]["repetition_penalty"] == pytest.approx(1.30)
    assert result["info"]["retries"] == 1
    assert result["duration"] == pytest.approx(4.5, abs=0.01)
    # 重试警告已记录（用户可在日志页看到）
    assert any(level == "warn" and "repetition_penalty 1.35" in msg for level, msg in logs)


def test_normal_output_no_retry(tmp_path):
    engine = _FakeGsvEngine([[4.5]])
    executor, task = _make_task(tmp_path, engine)

    result, _ = _run(executor, task)

    assert len(engine.calls) == 1
    assert result["info"]["retries"] == 0


def test_rp_at_floor_no_retry(tmp_path):
    """rp 已到下限 0.75 仍过短 → 不重试，直接返回（短文本等正常场景）。"""
    engine = _FakeGsvEngine([[1.0]])
    executor, task = _make_task(tmp_path, engine, args={"repetition_penalty": 0.75})

    result, _ = _run(executor, task)

    assert len(engine.calls) == 1
    assert result["info"]["retries"] == 0
    assert result["duration"] == pytest.approx(1.0, abs=0.01)


def test_rp_10_still_retries_down_to_095(tmp_path):
    """rp=1.0 仍过短 → 继续降到 0.95 重试（1.0 不再是下限）。"""
    engine = _FakeGsvEngine([[1.0], [4.5]])
    executor, task = _make_task(tmp_path, engine, args={"repetition_penalty": 1.0})

    result, _ = _run(executor, task)

    assert len(engine.calls) == 2
    rps = [kw["repetition_penalty"] for _, kw in engine.calls]
    assert rps == [pytest.approx(1.0), pytest.approx(0.95)]
    assert result["info"]["retries"] == 1
    assert result["duration"] == pytest.approx(4.5, abs=0.01)


def test_retries_exhausted_returns_last_result(tmp_path):
    """重试次数耗尽（1.35→1.30→1.25，max_retries=2 上限）仍过短 → 返回最后一次结果。"""
    engine = _FakeGsvEngine([[1.0], [1.0], [1.0]])
    executor, task = _make_task(tmp_path, engine, args={"max_retries": 2})

    result, _ = _run(executor, task)

    assert len(engine.calls) == 3
    rps = [kw["repetition_penalty"] for _, kw in engine.calls]
    assert rps == [pytest.approx(1.35), pytest.approx(1.30), pytest.approx(1.25)]
    assert result["info"]["retries"] == 2
    assert result["duration"] == pytest.approx(1.0, abs=0.01)


def test_long_output_retries_with_higher_rp(tmp_path):
    """第一次过长（60s > 4s×2.0，生成失控）→ 自动升 rp（0.03 步长）重试；第二次正常。"""
    engine = _FakeGsvEngine([[60.0], [4.5]])
    executor, task = _make_task(tmp_path, engine)

    result, logs = _run(executor, task)

    assert len(engine.calls) == 2
    rps = [kw["repetition_penalty"] for _, kw in engine.calls]
    assert rps == [pytest.approx(1.35), pytest.approx(1.38)]
    assert result["info"]["retries"] == 1
    assert result["duration"] == pytest.approx(4.5, abs=0.01)
    assert any("疑似生成失控" in msg for level, msg in logs)


def test_long_output_at_ceiling_no_retry(tmp_path):
    """rp 已到上限 2.25 仍过长 → 不重试，直接返回。"""
    engine = _FakeGsvEngine([[60.0]])
    executor, task = _make_task(tmp_path, engine, args={"repetition_penalty": 2.25})

    result, _ = _run(executor, task)

    assert len(engine.calls) == 1
    assert result["info"]["retries"] == 0
    assert result["duration"] == pytest.approx(60.0, abs=0.01)


def test_long_then_short_bounces_back(tmp_path):
    """过长升 rp 后过短 → 再降 rp（互素步长双向可回，visited 不误伤新值）。"""
    engine = _FakeGsvEngine([[60.0], [1.0], [4.5]])
    executor, task = _make_task(tmp_path, engine)

    result, _ = _run(executor, task)

    rps = [kw["repetition_penalty"] for _, kw in engine.calls]
    assert rps == [pytest.approx(1.35), pytest.approx(1.38), pytest.approx(1.33)]
    assert result["info"]["retries"] == 2
    assert result["duration"] == pytest.approx(4.5, abs=0.01)


def test_min_ref_ratio_zero_skips_retry(tmp_path):
    """min_ref_ratio=0 → 过短判定恒不触发（跳过路径等价验证）。"""
    engine = _FakeGsvEngine([[1.0]])
    executor, task = _make_task(tmp_path, engine, args={"min_ref_ratio": 0.0})

    result, _ = _run(executor, task)

    assert len(engine.calls) == 1
    assert result["info"]["retries"] == 0


def test_progress_monotonic_across_retries(tmp_path):
    """重试后片段计数全局累计 → 进度回调单调不减（UI 进度不回退）。"""
    engine = _FakeGsvEngine([[1.0], [2.0, 2.0]])
    executor, task = _make_task(tmp_path, engine)

    ratios = []

    def _on_progress(pos, total, speed=None, payload=None):
        ratios.append(min(pos / total, 1.0) if total > 0 else 1.0)

    result = executor.execute(task, progress_callback=_on_progress)

    assert len(engine.calls) == 2
    assert result["info"]["retries"] == 1
    assert ratios == sorted(ratios)          # 单调不减
    assert ratios[0] < ratios[-1]            # 确实推进了


def test_cancel_stops_engine_and_raises(tmp_path):
    """取消检查点：消费中途取消 → engine.stop() + CancelledError。"""
    from core.contracts import CancelledError

    engine = _FakeGsvEngine([[2.0, 2.0, 2.0]])
    executor, task = _make_task(tmp_path, engine)
    cancel_event = threading.Event()
    cancel_event.set()      # 首个检查点即取消

    with pytest.raises(CancelledError):
        executor.execute(task, cancel_event=cancel_event)
    assert engine.stop_called
