"""AppLog — loguru-based runtime logging: in-memory ring buffer (for the log page) + on-demand disk dump for major errors/crashes + manual export.

- Module-level singleton: any module can ``from app.log import log`` and call ``log.record(...)``.
- Backend: ``loguru.logger`` (the default stderr sink is removed and replaced with a custom in-memory sink).
- Line format: ``[YYYY-MM-DD HH:MM:SS] [level] message`` (levels: info / warn / error).
- Ring buffer: oldest entries are evicted past the cap (2000 lines by default).
- Version counter ``version``: incremented per line written, for the log page's incremental refresh (``snapshot_since``).
- On-demand dump: **only on an error-level log or process crash (uncaught exception)** the current
  buffer is written to ``logs/app-error-*.log`` / ``logs/app-crash-*.log``; info / warn
  only enter the memory buffer, so normal operation has zero disk overhead.
- ``export(path)``: writes the current buffer to a UTF-8 text file (for the "export log" button).

Thread-safety note: loguru handles multi-thread serialization; buffer append and dump each hold their own lock.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from app.paths import project_root

MAX_LINES = 2000
LEVELS = ("info", "warn", "error")

# error dump throttling: consecutive errors within the interval dump once; later
# errors are counted and annotated on the next dump.
_ERROR_DUMP_INTERVAL = 5.0

# internal level → loguru level name
_LOGURU_LEVELS = {"info": "INFO", "warn": "WARNING", "error": "ERROR"}

# loguru level name → UI/export line level name (WARNING → warn, others lowercase)
_LEVEL_LABELS = {"WARNING": "warn"}


def _level_label(level_name: str) -> str:
    return _LEVEL_LABELS.get(level_name, level_name.lower())


class _AppLog:
    """loguru-driven in-memory log buffer with on-demand disk dump on error/crash."""

    def __init__(self, max_lines: int = MAX_LINES):
        self._max = max_lines
        self._lines: List[str] = []
        self._version = 0
        self._cleared_version = 0
        self._lock = threading.Lock()
        self._last_error_dump = 0.0
        self._suppressed_errors = 0

    # ── loguru sink ──

    def _memory_sink(self, message) -> None:
        """Custom loguru sink: append each log line into the ring buffer in UI format.

        ``message`` is a ``loguru.Message``; take time/level/message directly from
        its ``record`` fields, append exception tracebacks as continuation lines
        (4-space indent), incrementing the version per line.
        """
        rec = message.record
        ts = rec["time"].strftime("%Y-%m-%d %H:%M:%S")
        lv = _level_label(rec["level"].name)
        text = rec["message"]
        lines = [f"[{ts}] [{lv}] {text}"]
        exc = rec.get("exception")
        if exc:
            tb = "".join(traceback.format_exception(*exc)).rstrip("\n")
            lines.extend("    " + part for part in tb.split("\n"))
        self._append_lines(lines)

    def _append_lines(self, lines: List[str]) -> None:
        with self._lock:
            for line in lines:
                self._lines.append(line)
            if len(self._lines) > self._max:
                del self._lines[: len(self._lines) - self._max]
            self._version += len(lines)

    # ── Writing ──

    @staticmethod
    def _normalize_level(level: str) -> str:
        """Normalize a level into the three LEVELS tiers; unknown levels fall back to info (logging never raises)."""
        lv = str(level).strip().lower()
        if lv in ("warn", "warning"):
            return "warn"
        if lv in ("error", "critical", "fatal", "exception"):
            return "error"
        return "info"

    def record(self, level: str, message: str, *, exc_info: bool = False) -> None:
        """Append a log line; ``level`` is normalized to info/warn/error.

        When ``exc_info=True``, attach the current exception traceback (or the
        passed exception object); traceback lines are appended as indented
        continuation lines (multi-line entries each get their own line, version
        incremented per line). error-level lines trigger the throttled on-demand dump.
        """
        lv = self._normalize_level(level)
        loguru_level = _LOGURU_LEVELS[lv]
        text = str(message)
        if exc_info is not None and exc_info is not False:
            exc = sys.exc_info()[1] if exc_info is True else exc_info
            if exc is not None:
                logger.opt(exception=exc).log(loguru_level, text)
            else:
                logger.log(loguru_level, text)
        else:
            logger.log(loguru_level, text)

        if lv == "error":
            self._dump_on_error()

    def clear(self) -> None:
        """Clear the buffer (version increments; the UI uses this to fully refresh and drop old lines)."""
        with self._lock:
            self._lines.clear()
            self._version += 1
            self._cleared_version = self._version

    # ── Reading ──

    def lines(self) -> List[str]:
        """All current log lines (a copy)."""
        with self._lock:
            return list(self._lines)

    def snapshot_since(self, version: int) -> Tuple[int, List[str], bool]:
        """Return ``(current version, new lines, whether the UI list must be reset)``.

        Used by the log page polling: when version equals the current version,
        return an empty list (no change); when buffer eviction dropped old lines,
        return the tail based on the line-count delta so the UI catches up; after
        a ``clear()`` or ring eviction (new line count exceeding current buffer
        length), the third item is True and the UI must remove all old lines
        before appending.
        """
        with self._lock:
            if version >= self._version:
                return (self._version, [], False)
            start = max(0, len(self._lines) - (self._version - version))
            reset = (
                self._cleared_version > version
                or start == 0 and (self._version - version) > len(self._lines)
            )
            return (self._version, list(self._lines[start:]), reset)

    def version(self) -> int:
        with self._lock:
            return self._version

    # ── Disk dump (only on error / crash) ──

    def _dump_on_error(self) -> None:
        """On-demand dump triggered by an error-level log (5-second throttle)."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_error_dump < _ERROR_DUMP_INTERVAL:
                self._suppressed_errors += 1
                return
            self._last_error_dump = now
            suppressed = self._suppressed_errors
            self._suppressed_errors = 0
        self._dump_to_file("error", suppressed)

    def crash_dump(self, message: str = "未捕获异常", exc_info=True) -> Optional[Path]:
        """Crash fallback: record an error and force a dump to ``app-crash-*.log`` (unthrottled).

        Called by ``sys.excepthook`` / ``threading.excepthook``; can also be
        called proactively on critical paths (e.g. failed shutdown sequence).
        """
        if exc_info is not None and exc_info is not False:
            exc = sys.exc_info()[1] if exc_info is True else exc_info
        else:
            exc = None
        if exc is not None:
            logger.opt(exception=exc).error(str(message))
        else:
            logger.error(str(message))
        return self._dump_to_file("crash", 0)

    def _dump_to_file(self, reason: str, suppressed: int) -> Optional[Path]:
        """Write the current buffer to ``logs/app-<reason>-<timestamp>.log`` (atomic write)."""
        lines = self.lines()
        if suppressed:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{ts}] [warn] （期间另有 {suppressed} 条 error 未单独落盘，已合并到本次）")
        p = self._dump_path(reason)
        self._write_atomic(p, lines)
        return p

    def _dump_path(self, reason: str) -> Path:
        log_dir = Path(project_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        p = log_dir / f"app-{reason}-{stamp}.log"
        n = 1
        while p.exists():
            n += 1
            p = log_dir / f"app-{reason}-{stamp}-{n}.log"
        return p

    @staticmethod
    def _write_atomic(path: Path, lines: List[str]) -> None:
        """Atomic write via temp file + os.replace, avoiding truncated files."""
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, path)

    # ── Export ──

    def export(self, path: Path | str) -> Path:
        """Write the current buffer to a file (UTF-8, atomic); return the written path."""
        lines = self.lines()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._write_atomic(p, lines)
        return p


# Module-level singleton: `from app.log import log`
log = _AppLog()

# ── loguru init: remove default stderr sink, attach UI in-memory sink ──
try:
    logger.remove(0)
except ValueError:
    pass
logger.add(log._memory_sink, level="INFO", format="{message}")

# ── Full info disk dump (diagnostics) ──
# When TRANSLATOR_INFO_LOG=1, add a file sink writing the complete startup/loading
# timeline to logs/app-info-*.log (the 2000-line in-memory ring would evict it).
if os.environ.get("TRANSLATOR_INFO_LOG") == "1":
    _info_log_path = Path(project_root) / "logs" / f"app-info-{datetime.now():%Y%m%d-%H%M%S}.log"
    _info_log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(_info_log_path),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        encoding="utf-8",
    )

# ── Global exception hooks: force dump on crash (preserving the original hook chain) ──
_prev_excepthook = getattr(sys, "excepthook", None)
_prev_thread_excepthook = getattr(threading, "excepthook", None)


def _on_uncaught_exception(exc_type, exc, tb):
    try:
        log.crash_dump(f"未捕获异常: {exc_type.__name__}: {exc}", exc_info=exc)
    except Exception:
        pass
    if _prev_excepthook is not None:
        _prev_excepthook(exc_type, exc, tb)


def _on_uncaught_thread_exception(args):
    try:
        exc = getattr(args, "exc_value", None)
        name = getattr(getattr(args, "thread", None), "name", "?")
        log.crash_dump(f"未捕获线程异常（线程: {name}）: {exc}", exc_info=exc)
    except Exception:
        pass
    if _prev_thread_excepthook is not None:
        _prev_thread_excepthook(args)


sys.excepthook = _on_uncaught_exception
if hasattr(threading, "excepthook"):
    threading.excepthook = _on_uncaught_thread_exception
