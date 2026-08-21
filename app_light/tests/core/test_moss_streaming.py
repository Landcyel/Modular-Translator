"""MOSS 进度条/预览统一契约测试（不加载真实模型）。

验证：
- StreamingModelRunner（SUPPORTS_PARTIAL_TEXT）路径下，MossTranscriber 的
  进度回调与 Whisper 同构：pos/total 为音频时间轴秒数，payload 携带
  status/generated_tokens/segments；
- 老版 runner 回退 token 比例语义且不传 partial_text_callback；
- TranscriptionTaskQueue 对 dict payload 的合并；
- 任务卡统一进度文案（Whisper / MOSS 时间轴 / MOSS 比例回退）。
"""
from __future__ import annotations

import pytest

from core.contracts import Task
from core.moss.audio_utils import probe_duration
from core.moss.moss_transcriber import MossTranscriber
from core.task_que import TranscriptionTaskQueue
from ui.widgets.task_list import _task_progress_text


_TRANSCRIPT = "[0.00][S01]你好[1.20]\n[1.20][S01]世界[2.50]"


class _FakeResult:
    text = _TRANSCRIPT

    def to_dict(self):
        return {"text": self.text, "generated_tokens": 8}


class _FakeTask:
    id = "t1"

    def __init__(self, path, args=None, hotwords=None):
        self.file_path = path
        self.configs = {"args": args or {}}
        if hotwords is not None:
            self.configs["hotwords"] = hotwords
        self._pause_event = None


class _StreamingFakeRunner:
    """模拟 StreamingModelRunner 的回调时序。"""

    SUPPORTS_PARTIAL_TEXT = True
    _device = None
    _dtype = None

    def transcribe(self, audio_path, **kwargs):
        status = kwargs["status_callback"]
        partial_text = kwargs.get("partial_text_callback")
        status("loading_model", 0.05, None)
        status("transcribing", 0.10, None)
        status("transcribing", 0.25, None)
        assert partial_text is not None
        partial_text(_TRANSCRIPT, 8)
        return _FakeResult()


class _LegacyFakeRunner:
    """模拟不支持部分文本的老版 ModelRunner。"""

    _device = None
    _dtype = None

    def transcribe(self, audio_path, **kwargs):
        assert "partial_text_callback" not in kwargs
        kwargs["status_callback"]("transcribing", 0.25, 3)
        return _FakeResult()


def test_progress_callback_unified_timeline_contract(monkeypatch):
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 30.0
    )
    calls = []
    tx = MossTranscriber(_StreamingFakeRunner())
    result = tx.execute(
        _FakeTask("audio.wav"),
        progress_callback=lambda *args: calls.append(args),
    )

    # 加载期（无已确认段）：与 Whisper 首段前一致，进度保持 0
    assert calls[0][0] == 0.0 and calls[0][1] == 30.0
    assert calls[0][3]["status"] == "loading_model"
    assert calls[0][3]["generated_tokens"] == 0

    # 实时预览：payload 携带已确认 segments，pos = 最新段尾（真实时间轴）
    with_segments = [c for c in calls if (c[3] or {}).get("segments")]
    assert with_segments
    pos, total, speed, payload = with_segments[0]
    assert pos == pytest.approx(2.50)   # _TRANSCRIPT 最后段 end
    assert total == 30.0
    assert speed is not None
    assert payload["status"] == "transcribing"
    assert payload["generated_tokens"] == 8
    assert [s["text"] for s in payload["segments"]] == ["你好", "世界"]

    # 收尾回调：进度补满 100%（与 Whisper 末段 end≈duration 对齐）
    pos, total, speed, payload = calls[-1]
    assert pos == pytest.approx(30.0) and total == pytest.approx(30.0)
    assert payload["status"] == "transcribing"
    assert [s["text"] for s in payload["segments"]] == ["你好", "世界"]

    # 防闪烁回归：所有 transcribing 回调的 pos 单调不减、不含 0
    # （token 级更新已删除，无段回调不再把进度打回 0%）
    transcribing = [c for c in calls if c[3].get("status") == "transcribing"]
    assert transcribing
    poss = [c[0] for c in transcribing]
    assert poss == sorted(poss)
    assert 0.0 not in poss

    # 最终结果与实时预览共用同一段切分管线
    assert [s["text"] for s in result["segments"]] == ["你好", "世界"]


def test_legacy_runner_falls_back_to_ratio_semantics(monkeypatch):
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: None
    )
    calls = []
    tx = MossTranscriber(_LegacyFakeRunner())
    result = tx.execute(
        _FakeTask("audio.wav"),
        progress_callback=lambda *args: calls.append(args),
    )

    assert calls
    # 生成中回退回调：token 比例语义（unit=ratio，无 segments）
    mid = [c for c in calls if c[3].get("unit") == "ratio"
           and "segments" not in c[3]][0]
    pos, total, speed, payload = mid
    assert pos == pytest.approx(0.25)
    assert total == 1.0
    assert speed is None

    # 收尾回调：进度补满 1.0，payload 携带最终 segments
    pos, total, speed, payload = calls[-1]
    assert pos == 1.0 and total == 1.0
    assert payload["unit"] == "ratio"
    assert [s["text"] for s in payload["segments"]] == ["你好", "世界"]
    assert [s["text"] for s in result["segments"]] == ["你好", "世界"]


def test_transcription_queue_merges_dict_payload():
    class _Executor:
        def execute(self, task, progress_callback=None, cancel_event=None):
            progress_callback(
                7.5, 30.0, 2.0,
                {
                    "status": "transcribing",
                    "generated_tokens": 8,
                    "segments": [{"id": "seg_0001", "start": 0.0,
                                 "end": 1.2, "speaker": "S01", "text": "你好"}],
                },
            )
            return {"segments": []}

    task = Task(task_type="moss", file_path=None, configs=None, file_name=None)
    queue = TranscriptionTaskQueue(_Executor())
    queue._emit = lambda: None
    queue._current_task = task

    queue._run(task)

    assert task.progress == pytest.approx(0.25)
    assert task.payload["pos"] == 7.5
    assert task.payload["total"] == 30.0
    assert task.payload["speed"] == 2.0
    assert task.payload["status"] == "transcribing"
    assert task.payload["segments"][0]["text"] == "你好"


def test_task_progress_text_unified():
    whisper = {
        "type": "transcribe", "status": "running", "progress": 0.5,
        "payload": {"pos": 10.0, "total": 20.0, "speed": 2.0},
    }
    assert _task_progress_text(whisper) == "50% [00:10.00/00:20.00] 2.0x"

    moss_timeline = {
        "type": "moss", "status": "running", "progress": 0.5,
        "payload": {"pos": 10.0, "total": 20.0, "speed": 2.0,
                    "status": "transcribing", "generated_tokens": 100},
    }
    assert _task_progress_text(moss_timeline) == (
        "50% [00:10.00/00:20.00] 2.0x · transcribing（100 tokens）"
    )

    moss_ratio = {
        "type": "moss", "status": "running", "progress": 0.5,
        "payload": {"pos": 0.5, "total": 1.0, "speed": None,
                    "status": "transcribing", "generated_tokens": 100,
                    "unit": "ratio"},
    }
    assert _task_progress_text(moss_ratio) == "50% · transcribing（100 tokens）"

    # MOSS 模型懒加载期：progress==0 仍显示加载状态文本（无 LRC 时间戳）
    moss_loading = {
        "type": "moss", "status": "running", "progress": 0.0,
        "payload": {"pos": 0.0, "total": 30.0, "speed": None,
                    "status": "loading_model"},
    }
    assert _task_progress_text(moss_loading) == "0% · loading_model"


def test_probe_duration_from_wav_metadata():
    pytest.importorskip("av")
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "Audio02.wav"
    duration = probe_duration(path)
    assert duration is not None and duration > 0


# ── 四类参数：hotwords / sampling 参数接线 ──


class _CapturingFakeRunner:
    """记录每次 transcribe 收到的 kwargs（hotwords/sampling 参数验证）。"""

    SUPPORTS_PARTIAL_TEXT = False
    _device = None
    _dtype = None

    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs)
        kwargs["status_callback"]("transcribing", 0.25, 3)
        return _FakeResult()


def test_hotwords_appended_to_prompt(monkeypatch):
    """热词（dict）→ prompt 末尾附加"热词提示：词1,词2"。"""
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 30.0
    )
    runner = _CapturingFakeRunner()
    tx = MossTranscriber(runner, defaults={"prompt": "基础提示词"})
    tx.execute(_FakeTask("audio.wav", hotwords={"hotwords": ["熱詞A", "热词B"]}))
    prompt = runner.calls[0]["prompt"]
    assert prompt.startswith("基础提示词")
    assert prompt.endswith("热词提示：熱詞A,热词B")


def test_hotwords_resolved_from_json_file(monkeypatch, tmp_path):
    """热词 JSON 文件（configs/transcribe/hotwords/*.json 形态）→ 同样附加。"""
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 30.0
    )
    hw_file = tmp_path / "hotwords.json"
    hw_file.write_text('{"hotwords": ["東京", "大阪"]}', encoding="utf-8")
    runner = _CapturingFakeRunner()
    tx = MossTranscriber(runner, defaults={})
    tx.execute(_FakeTask("audio.wav", hotwords=hw_file))
    assert runner.calls[0]["prompt"].endswith("热词提示：東京,大阪")


def test_hotwords_without_prompt_uses_default_prompt(monkeypatch):
    """无显式 prompt + 热词 → 以 vendor DEFAULT_PROMPT 为基底附加。"""
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 30.0
    )
    runner = _CapturingFakeRunner()
    tx = MossTranscriber(runner, defaults={})
    tx.execute(_FakeTask("audio.wav", hotwords={"hotwords": ["专有名词"]}))
    prompt = runner.calls[0]["prompt"]
    assert prompt.endswith("热词提示：专有名词")
    assert len(prompt) > len("热词提示：专有名词")


def test_sample_decoding_wires_sampling_params(monkeypatch):
    """decoding=sample → temperature/top_p/top_k 传入 runner。"""
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 30.0
    )
    runner = _CapturingFakeRunner()
    tx = MossTranscriber(runner, defaults={
        "decoding": "sample", "temperature": 0.8,
        "top_p": 0.9, "top_k": 50,
    })
    tx.execute(_FakeTask("audio.wav"))
    kwargs = runner.calls[0]
    assert kwargs["decoding"] == "sample"
    assert kwargs["temperature"] == 0.8
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 50


def test_greedy_decoding_omits_sampling_params(monkeypatch):
    """decoding=greedy → 不传 sampling 参数（即使配置了也忽略）。"""
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 30.0
    )
    runner = _CapturingFakeRunner()
    tx = MossTranscriber(runner, defaults={
        "decoding": "greedy", "temperature": 0.8,
        "top_p": 0.9, "top_k": 50,
    })
    tx.execute(_FakeTask("audio.wav"))
    kwargs = runner.calls[0]
    assert kwargs["decoding"] == "greedy"
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_no_prompt_no_hotwords_no_prompt_kwarg(monkeypatch):
    """无 prompt 且无热词 → 不传 prompt kwarg（沿用 runner 默认提示词）。"""
    monkeypatch.setattr(
        "core.moss.moss_transcriber.probe_duration", lambda _p: 30.0
    )
    runner = _CapturingFakeRunner()
    tx = MossTranscriber(runner, defaults={})
    tx.execute(_FakeTask("audio.wav"))
    assert "prompt" not in runner.calls[0]
