"""AppLog — 基于 loguru 的运行日志：内存环形缓冲（供日志页）+ 重大错误/崩溃按需落盘 + 手动导出。

- 模块级单例：任意模块 ``from app.log import log`` 后直接 ``log.record(...)``。
- 底层：``loguru.logger``（默认 stderr sink 已移除，改挂自定义内存 sink）。
- 每行格式：``[YYYY-MM-DD HH:MM:SS] [级别] 消息``（级别：info / warn / error）。
- 环形缓冲：超出上限自动淘汰最旧（默认 2000 行）。
- 版本号 ``version``：每写入一行自增，供日志页增量刷新（``snapshot_since``）。
- 按需落盘：**仅当出现 error 级别日志或进程崩溃（未捕获异常）时**，把当前
  缓冲写入 ``logs/app-error-*.log`` / ``logs/app-crash-*.log``；info / warn
  只进内存缓冲，正常运行零磁盘开销。
- ``export(path)``：将当前缓冲写入 UTF-8 文本文件（供"导出日志"按钮）。

线程安全说明：loguru 负责多线程序列化；内存缓冲追加与落盘各自持锁。
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

# error 落盘节流：间隔内连续 error 只落一次，后续 error 计数并在下次落盘时标注。
_ERROR_DUMP_INTERVAL = 5.0

# 内部级别 → loguru 级别名
_LOGURU_LEVELS = {"info": "INFO", "warn": "WARNING", "error": "ERROR"}

# loguru 级别名 → UI/导出行级别名（WARNING → warn，其余小写）
_LEVEL_LABELS = {"WARNING": "warn"}


def _level_label(level_name: str) -> str:
    return _LEVEL_LABELS.get(level_name, level_name.lower())


class _AppLog:
    """loguru 驱动的内存日志缓冲，并在 error/崩溃时按需落盘。"""

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
        """loguru 自定义 sink：把每条日志按 UI 格式追加进环形缓冲。

        ``message`` 为 ``loguru.Message``；直接从 ``record`` 字段取时间/级别/
        消息，异常堆栈按续行（4 空格缩进）追加，每行版本号自增。
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

    # ── 写入 ──

    @staticmethod
    def _normalize_level(level: str) -> str:
        """归一化级别到 LEVELS 三档；未知级别按 info 处理（日志不抛错）。"""
        lv = str(level).strip().lower()
        if lv in ("warn", "warning"):
            return "warn"
        if lv in ("error", "critical", "fatal", "exception"):
            return "error"
        return "info"

    def record(self, level: str, message: str, *, exc_info: bool = False) -> None:
        """追加一条日志；``level`` 归一化为 info/warn/error。

        ``exc_info=True`` 时附加当前异常堆栈（或传入异常对象），堆栈行以缩进
        续行形式追加到缓冲（多行条目各占一行，版本号按行自增）。
        error 级别会触发按需落盘（带节流）。
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
        """清空缓冲（版本号自增，UI 据此全量刷新并移除旧行）。"""
        with self._lock:
            self._lines.clear()
            self._version += 1
            self._cleared_version = self._version

    # ── 读取 ──

    def lines(self) -> List[str]:
        """当前全部日志行（副本）。"""
        with self._lock:
            return list(self._lines)

    def snapshot_since(self, version: int) -> Tuple[int, List[str], bool]:
        """返回 ``(当前版本, 新增行, 是否需要重置 UI 列表)``。

        日志页轮询用：version 等于当前版本时返回空列表（无变化）；
        缓冲淘汰导致旧行消失时按"行数差"返回尾部，保证 UI 追平；
        发生过 ``clear()`` 或缓冲环形淘汰（新增行数超过当前缓冲长度）时，
        第三项为 True，UI 需先移除全部旧行再追加。
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

    # ── 落盘（仅 error / 崩溃时触发）──

    def _dump_on_error(self) -> None:
        """error 级日志触发的按需落盘（5 秒节流）。"""
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
        """崩溃兜底：记录一条 error 并强制落盘为 ``app-crash-*.log``（不节流）。

        由 ``sys.excepthook`` / ``threading.excepthook`` 调用；也可在关键路径
        （如关闭流程失败）主动调用。
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
        """把当前缓冲写入 ``logs/app-<reason>-<时间戳>.log``（原子写）。"""
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
        """临时文件 + os.replace 原子写，避免半截文件。"""
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, path)

    # ── 导出 ──

    def export(self, path: Path | str) -> Path:
        """把当前缓冲写入文件（UTF-8，原子写）；返回写入路径。"""
        lines = self.lines()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._write_atomic(p, lines)
        return p


# 模块级单例：`from app.log import log`
log = _AppLog()

# ── loguru 初始化：移除默认 stderr sink，挂接 UI 内存 sink ──
try:
    logger.remove(0)
except ValueError:
    pass
logger.add(log._memory_sink, level="INFO", format="{message}")

# ── info 全量落盘开关（诊断用）：TRANSLATOR_INFO_LOG=1 时追加文件 sink，
#    完整启动/加载时间线写入 logs/app-info-*.log（内存环形缓冲 2000 行会淘汰）──
if os.environ.get("TRANSLATOR_INFO_LOG") == "1":
    _info_log_path = Path(project_root) / "logs" / f"app-info-{datetime.now():%Y%m%d-%H%M%S}.log"
    _info_log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(_info_log_path),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        encoding="utf-8",
    )

# ── 全局异常钩子：崩溃时强制落盘（保留原钩子链）──
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
