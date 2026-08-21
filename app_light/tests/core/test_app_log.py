"""回归测试：AppLog 级别归一化、版本增量、snapshot_since 重置语义、error 落盘节流。"""
import pytest

import app.log as app_log_module
from app.log import log


@pytest.fixture(autouse=True)
def _clean_log(monkeypatch):
    log.clear()
    log._last_error_dump = 0.0
    log._suppressed_errors = 0
    # 测试期间不写真实落盘文件（节流测试会自行覆盖此桩）
    monkeypatch.setattr(log, "_dump_to_file", lambda reason, suppressed: None)
    yield
    log.clear()
    log._last_error_dump = 0.0
    log._suppressed_errors = 0


def test_record_normalizes_levels():
    log.record("warning", "w")
    log.record("critical", "c")
    log.record("debug", "d")
    lines = log.lines()
    assert any("[warn]" in line and line.endswith(" w") for line in lines)
    assert any("[error]" in line and line.endswith(" c") for line in lines)
    assert any("[info]" in line and line.endswith(" d") for line in lines)


def test_record_multiline_exc_info_increments_version_per_line():
    log.clear()
    v0 = log.version()
    try:
        raise ValueError("boom")
    except ValueError as exc:
        log.record("error", "捕获异常", exc_info=exc)
    lines = log.lines()
    assert len(lines) >= 3  # 主行 + 堆栈行
    assert log.version() - v0 == len(lines)
    assert lines[0].startswith("[") and "[error]" in lines[0]


def test_snapshot_since_incremental():
    log.clear()
    log.record("info", "a")
    v = log.version()
    log.record("info", "b")
    current, new, reset = log.snapshot_since(v)
    assert current == log.version()
    assert len(new) == 1 and "b" in new[0]
    assert reset is False


def test_snapshot_since_reset_after_clear():
    log.clear()
    log.record("info", "a")
    v = log.version()
    log.clear()
    current, new, reset = log.snapshot_since(v)
    assert reset is True
    assert new == []


def test_error_dump_throttled(monkeypatch):
    """error 落盘 5 秒节流：首次立即落，节流窗口内只计数不落。"""
    calls = []
    monkeypatch.setattr(log, "_dump_to_file", lambda reason, suppressed: calls.append((reason, suppressed)))
    clock = {"t": 100.0}
    monkeypatch.setattr(app_log_module.time, "monotonic", lambda: clock["t"])

    log.record("error", "e1")
    log.record("error", "e2")
    assert len(calls) == 1 and calls[0][0] == "error"

    clock["t"] = 106.0
    log.record("error", "e3")
    assert len(calls) == 2
    assert calls[1][1] == 1  # 节流期间另有 1 条 error 未单独落盘
