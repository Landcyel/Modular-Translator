"""
Executor — abstract base and concrete implementations for translation and transcription.

Every executor exposes a single ``execute(task, progress_callback, cancel_event)``
entry point so that ``TaskQueue`` can schedule work uniformly.

Contents
--------
- ``Executor``         abstract base
- ``Translator``       abstract translation base (chunking, prompt rendering, merging)
- ``LlamaTranslator``  local llama-server translation
- ``APITranslator``    OpenAI-compatible cloud API translation
"""
from __future__ import annotations

import inspect
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .contracts import CancelledError
from .utils import load_json_file
from .rule_splitter import RuleSplitter, SentenceInfo

# Note: openai / requests / numpy are all heavy libraries (several seconds combined);
# they are now lazily imported inside the functions that use them. core.executor is
# imported by core modules (task_que / moss_transcriber) in the startup chain, so
# top-level imports would slow the app shell's first frame.


def _wait_paused(pause_event, cancel_event) -> bool:
    """Pause checkpoint: block while paused until resumed; return False if cancelled.

    Returns True immediately when ``pause_event`` is None (scenarios outside direct
    execute calls) or not paused.
    """
    # cancel takes priority: must respond to cancel even when not paused (otherwise
    # cancel would only affect paused tasks)
    if cancel_event is not None and cancel_event.is_set():
        return False
    if pause_event is None or pause_event.is_set():
        return True
    while not pause_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            return False
        pause_event.wait(0.2)
    return True


# ── Executor — abstract base ──

class Executor(ABC):
    """Abstract execution unit consumed by a TaskQueue worker."""

    # diagnostic log callback (injected by the UI via service; falls back to print
    # without a callback, so library callers are unaffected)
    _on_log = None

    def set_on_log(self, callback: Optional[Callable]):
        """Inject/replace the diagnostic log callback (level, message); falls back to print without one."""
        self._on_log = callback

    def _log(self, level: str, message: str) -> None:
        """Record a diagnostic log entry (level: info/warn/error)."""
        if self._on_log is not None:
            try:
                self._on_log(level, message)
                return
            except Exception:
                pass
        print(f"[{level}] {message}")

    @abstractmethod
    def execute(
        self,
        task,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Any:
        """Run the task and return its result."""
        ...

    # ── Task resolution (multi-format reading, like Service._resolve_config) ──

    @staticmethod
    def _resolve_value(value):
        """Read a single value in multiple formats.

        - ``dict`` → returned as-is (already a parsed result)
        - an existing file path (``Path``/``str``) → read as UTF-8; parsed as JSON
          dict when possible, otherwise returned as text; binary files (e.g. audio)
          are not force-decoded and returned as-is
        - anything else (e.g. non-file strings like ``"ja"``) → returned as-is
        """
        if isinstance(value, dict):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    return value  # binary file (audio etc.), return as-is
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return value
        return value

    def _resolve_task(self, task):
        """Base-class generic resolution: only multi-format file reading, no semantics.

        Returns ``(source, configs_dict)``:

        - ``source`` — result of :meth:`_resolve_value` on ``task.file_path``
        - ``configs_dict`` — result of :meth:`_resolve_value` on each value in ``task.configs``

        Subclasses should override this: call ``super()._resolve_task(task)`` for the
        generic reading, then implement semantic parsing (audio paths, RuleSplitter
        construction, etc.).
        """
        source = self._resolve_value(task.file_path)
        configs = {}
        for key, value in (task.configs or {}).items():
            configs[key] = self._resolve_value(value)
        return source, configs


# ── Translator — abstract translation base ──

class Translator(Executor):
    """Abstract translator with chunking, prompt rendering, and merge logic.

    Subclasses must implement ``_translate()`` for a single request."""

    # retry count for a single failed translation: chunks interrupted by a service
    # stop are re-translated after recovery via retries
    # (_translate returns None on connection drop/timeout; before retrying it re-waits for backend readiness)
    _translate_retries = 3

    def __init__(self, config=None):
        super().__init__()
        self.config = None
        if config is not None:
            self.config = load_json_file(config)

    # ── Executor interface ──

    def execute(
        self,
        task,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str | None:
        """Run translation from *task*.

        Task contract::

            task.file_path                    source text file (Path/str/read text)
            task.configs["translate_config"]  translation args (path → JSON dict)
            task.configs["prompts"]           prompt templates (path → JSON dict)
            task.configs["glossary"]          glossary (path → JSON dict, optional)
            task.configs["rule"]              split rules (path → JSON dict, optional)
        """
        cfg = self._resolve_task(task)
        pause_event = getattr(task, "_pause_event", None)
        return self.translate(
            text=cfg["text"],
            trans_config=cfg.get("translate_config"),
            prompts=cfg.get("prompts"),
            glossary=cfg.get("glossary"),
            splitter=cfg.get("splitter"),
            cancel_event=cancel_event,
            pause_event=pause_event,
            progress_callback=progress_callback,
        )

    def _resolve_task(self, task):
        """Translation-specific resolution: read source text from file_path, build a RuleSplitter from configs["rule"]."""
        source, configs = super()._resolve_task(task)

        if isinstance(source, str):
            text = source
        else:
            # fallback: re-read binary/unresolved paths as text (not reached in normal flow)
            text = Path(task.file_path).read_text(encoding="utf-8")

        splitter = None
        rule = configs.get("rule")
        if rule:
            splitter = RuleSplitter(rule)

        return {
            "text": text,
            "translate_config": configs.get("translate_config"),
            "prompts": configs.get("prompts"),
            "glossary": configs.get("glossary"),
            "splitter": splitter,
        }

    # ── Translatable-segment check (shared by send/count/backfill) ──

    @staticmethod
    def _translatable_segments(line_info: list) -> list:
        """Return the list of translatable structure segments in this line.

        The check matches _merge_chunks' has_content: non-skip and
        (has prefix/suffix structure or a non-blank body). Pure-whitespace filler
        segments (no structure, blank body, e.g. indentation) do not count — they are
        structural placeholders kept with their original body, not sent for
        translation, and consume no translated lines. Shared by
        _text_split / translate / _merge_chunks so that "translated segment count =
        counted segment count = backfilled segment count" stays one-to-one.
        """
        return [
            si for si in line_info
            if not si.is_skip
            and (si.prefix or si.suffix or (si.body and si.body.strip()))
        ]

    # ── Public API ──

    def translate(self, text, trans_config, prompts, glossary=None,
                  splitter=None,
                  timeout=None, cancel_event=None, pause_event=None,
                  progress_callback=None):
        """Unified translation entry point.

        With *splitter* → chunked translate → merge.
        Without *splitter* (rules=off) → use a "passthrough splitter" to go through
        the same chunking path: no structure is recognized/skipped, chunks are split
        by line + token budget — avoiding a single request larger than the model
        context (e.g. llama -c 1024) that would fail translation with HTTP 400.
        """
        # missing config (not selected / invalid item) → explicit error instead of a
        # low-level TypeError: the task is marked FAILED in the queue with a readable error.
        if trans_config is None:
            raise ValueError("翻译参数配置未选择：任务 configs['translate_config'] 缺失或选了无效项")
        if prompts is None:
            raise ValueError("提示词配置未选择：任务 configs['prompts'] 缺失或选了无效项")
        trans_config = load_json_file(trans_config)
        prompts = load_json_file(prompts)
        if glossary:
            glossary = load_json_file(glossary)

        # rules=off: passthrough splitter — empty prefix/suffix/skip/placeholder,
        # every non-empty line is translatable; _text_split chunks only by max_token,
        # _merge_chunks backfills line by line, and empty lines are preserved
        # (identical to the with-rules path).
        if splitter is None:
            splitter = RuleSplitter({
                "prefix": [], "suffix": [], "skip": [], "placeholder": [],
            })

        max_token = self._resolve_max_token(None, trans_config)
        max_lines_per_chunk = self._resolve_max_lines(None, trans_config)

        chunk_infos_list, chunks = self._text_split(
            text, splitter, trans_config, prompts, glossary,
            max_token, max_lines_per_chunk=max_lines_per_chunk,
        )
        if not chunks:
            # splitter produced no chunks (e.g. empty text) → return as-is
            return text or ""

        total = len(chunks)
        if progress_callback:
            progress_callback(0, total)

        _chunks = []
        for i, chunk in enumerate(chunks):
            if cancel_event and cancel_event.is_set():
                self._log("info", "[translate] 任务已取消，停止翻译")
                break
            # pause checkpoint: block while paused, continue after resume; exit on cancel
            if not _wait_paused(pause_event, cancel_event):
                self._log("info", "[translate] 任务已取消，停止翻译")
                break
            _glossary = self._match_glossary(chunk, glossary)
            # translatable segments of this chunk (consistent across send/count/backfill)
            segs = [
                si for li in chunk_infos_list[i]
                for si in self._translatable_segments(li)
            ]
            expected = len(segs)
            result = None
            for attempt in range(self._translate_retries):
                if cancel_event and cancel_event.is_set():
                    break
                # pause checkpoint: block while paused, continue after resume; exit on cancel
                if not _wait_paused(pause_event, cancel_event):
                    break
                result = self._translate(
                    chunk, trans_config, prompts, _glossary,
                    timeout=timeout, cancel_event=cancel_event,
                    pause_event=pause_event,
                )
                if result is not None:
                    break
                # a None caused by cancellation (already logged as "task cancelled"
                # inside _translate) must not trigger retries/false alarms
                if cancel_event and cancel_event.is_set():
                    break
                if attempt < self._translate_retries - 1:
                    self._log("warn", f"[translate] chunk {i + 1}/{total} 翻译失败，重试（{attempt + 1}/{self._translate_retries - 1}）...")
                    time.sleep(1)
            if result is None:
                if cancel_event and cancel_event.is_set():
                    self._log("info", f"[translate] chunk {i + 1}/{total} 已取消，回退原文")
                else:
                    self._log("error", f"[translate] chunk {i + 1}/{total} 翻译失败，重试耗尽 → 该块回退原文")
                _chunks.append(None)
            else:
                actual = len(result.split('\n'))
                if actual != expected:
                    # translated line count mismatch (LLM merged/split/dropped lines)
                    # → fall back to per-sentence translation for this chunk: each
                    # segment body is translated and received separately, guaranteeing
                    # one-to-one segment correspondence.
                    self._log("warn", f"[translate] chunk {i + 1}/{total} 译文行数不匹配"
                              f"（期望 {expected}，实际 {actual}）→ 回退逐句翻译")
                    per_line = []
                    for si in segs:
                        if cancel_event and cancel_event.is_set():
                            per_line.append(si.body)
                            continue
                        if not _wait_paused(pause_event, cancel_event):
                            per_line.append(si.body)
                            continue
                        if not si.body:
                            per_line.append("")  # structural segment with empty body needs no translation
                            continue
                        seg_result = self._translate(
                            si.body, trans_config, prompts, _glossary,
                            timeout=timeout, cancel_event=cancel_event,
                            pause_event=pause_event,
                        )
                        per_line.append(
                            seg_result if seg_result is not None else si.body
                        )
                    _chunks.append('\n'.join(per_line))
                else:
                    _chunks.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return self._merge_chunks(splitter, chunk_infos_list[:len(_chunks)], _chunks)

    # ── Subclass overrides ──

    def _translate(self, text, trans_config, prompts, glossary=None,
                   timeout=None, cancel_event=None, pause_event=None):
        raise NotImplementedError("子类必须实现 _translate 方法")

    def get_token_count(self, text, trans_config, prompts, glossary=None):
        return None

    def _resolve_max_token(self, max_token, trans_config):
        return max_token

    def _resolve_max_lines(self, max_lines, trans_config):
        """Resolve the per-chunk translatable-line cap; the base class does not parse it (only the llama subclass reads the max_lines entry)."""
        return max_lines

    # ── Prompt rendering ──

    def _render_glossary(self, glossary: dict | None) -> str:
        fmt = glossary["format"]
        lines = []
        for item in glossary.get("entries", []):
            if item.get("info"):
                line = fmt["with_info"].format(**item)
            else:
                line = fmt["without_info"].format(**item)
            lines.append(line)
        return fmt.get("separator", "\n").join(lines)

    def render_prompt(self, text, trans_args, prompts, glossary=None):
        """Build a /v1/chat/completions request body."""
        trans_args = load_json_file(trans_args)
        prompts = load_json_file(prompts)
        if glossary:
            glossary = load_json_file(glossary)

        request_body = dict(trans_args['request'])

        if glossary:
            user_content = prompts['user_with_glossary'].format(
                GLOSSARY_TEXT=self._render_glossary(glossary),
                ORIGINAL_TEXT=text,
            )
        else:
            user_content = prompts['user_without_glossary'].format(
                ORIGINAL_TEXT=text,
            )

        request_body["messages"] = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": user_content},
        ]
        return request_body

    # ── Glossary matching ──

    def _match_glossary(self, text: str, glossary: dict | None) -> dict | None:
        """Filter glossary to entries actually present in *text*."""
        text = "".join(text)
        if glossary is None:
            return None
        matched_entries = [
            entry for entry in glossary.get("entries", [])
            if entry.get("src", "") in text
        ]
        if not matched_entries:
            return None
        return {
            "name": glossary.get("name"),
            "format": glossary.get("format"),
            "entries": matched_entries,
        }

    # ── Text chunking ──

    def _text_split(self, text, splitter: RuleSplitter, trans_config, prompts,
                    glossary=None, max_token=None, max_lines_per_chunk=None):
        """Greedy line-level chunking respecting *max_token* and *max_lines_per_chunk*.

        Only non-skip lines are included in the translatable text.  Skip lines
        (code, headers, tags) are kept in the line_infos structure but excluded
        from the chunk text sent to the LLM.  This prevents the LLM from
        modifying ERB / script markup and keeps the line count stable.

        Returns ``(chunk_line_infos_list, chunk_pure_texts_list)``.
        """
        line_infos = splitter.split(text)
        if not line_infos:
            return [], []

        # ── Separate translatable vs skip lines ──
        # translatable_lines[i] = body text for the i-th translatable row
        # translatable_to_global[i] = index in line_infos
        translatable_lines: list[str] = []
        translatable_to_global: list[int] = []

        for gi, li in enumerate(line_infos):
            if not li:
                continue  # empty lines are never translated
            # per-segment line joining: multiple translatable segments within one
            # source line each occupy their own body line ('\n'-separated), so the
            # LLM keeps translations line-aligned and merge backfills per line,
            # preventing conditional-branch boundaries (e.g. 「\@ … ? … # … \@」)
            # from being merged or misaligned.
            # structural segments with empty body (e.g. full-width space lines)
            # produce empty lines, consistent with the expected count.
            bodies = [si.body for si in self._translatable_segments(li)]
            if bodies:
                translatable_lines.append('\n'.join(bodies))
                translatable_to_global.append(gi)

        if not translatable_lines:
            # rules mark everything as skip (e.g. an ERB template rule wrongly applied
            # to plain text): fall back to whole-text translation — a single non-skip
            # segment carries the full text and merge backfills the whole translation;
            # empty text returns empty (translate() falls back to the original)
            if not text or not text.strip():
                return [], []
            # whole-text fallback: build non-skip lines matching the original line
            # count (translation backfilled line by line).
            # returns a single chunk (chunk = list of lines, line = [SentenceInfo])
            lines = text.split("\n")
            return [[[SentenceInfo(body=l)] for l in lines]], [text]

        matched_glossary = None
        if glossary:
            glossary_data = load_json_file(glossary)
            matched_glossary = self._match_glossary(text, glossary_data)

        # ── Chunk the translatable lines ──
        chunk_line_infos_list = []
        chunk_texts_list = []
        chunk_start = 0          # index into translatable_lines
        prev_end = 0             # global right boundary of the previous chunk (first chunk starts at file offset 0)
        i = 0

        def _global_end(idx: int) -> int:
            """Global right boundary of a chunk's line range: up to the next translatable
            line (including empty lines between), and to the end of the file at the tail
            (including trailing empty lines), so empty lines are not dropped outside chunk
            boundaries."""
            if idx < len(translatable_lines):
                return translatable_to_global[idx]
            return len(line_infos)

        while i < len(translatable_lines):
            trial_text = '\n'.join(translatable_lines[chunk_start:i + 1])

            token_count = self.get_token_count(
                trial_text, trans_config, prompts, matched_glossary,
            )
            token_ok = (max_token is None or token_count is None
                        or token_count <= max_token)
            lines_ok = (max_lines_per_chunk is None
                        or (i - chunk_start + 1) <= max_lines_per_chunk)

            if token_ok and lines_ok:
                i += 1
            else:
                if chunk_start < i:
                    # Build chunk covering global line range.
                    # The left boundary starts at prev_end (0 for the first chunk), not
                    # at the first translatable-line index — otherwise skip/comment lines
                    # at the file head (before the first translatable line) fall outside
                    # every chunk and are lost when merging.
                    g_start = prev_end
                    g_end = _global_end(i)
                    chunk_line_infos_list.append(line_infos[g_start:g_end])
                    chunk_texts_list.append(
                        '\n'.join(translatable_lines[chunk_start:i])
                    )
                    prev_end = g_end
                    chunk_start = i
                else:
                    # Single translatable line too large, force one line
                    g_start = prev_end
                    g_end = _global_end(chunk_start + 1)
                    chunk_line_infos_list.append(
                        line_infos[g_start:g_end]
                    )
                    chunk_texts_list.append(translatable_lines[chunk_start])
                    prev_end = g_end
                    chunk_start = i + 1
                    i += 1

        # ── Remainder ──
        if chunk_start < len(translatable_lines):
            g_start = prev_end
            g_end = _global_end(len(translatable_lines))
            chunk_line_infos_list.append(line_infos[g_start:g_end])
            chunk_texts_list.append(
                '\n'.join(translatable_lines[chunk_start:])
            )

        return chunk_line_infos_list, chunk_texts_list

    def _merge_chunks(self, splitter: RuleSplitter, chunk_line_infos_list: list,
                      translated_chunks: list) -> str:
        """Reassemble translated chunks back into a full file.

        Translated text is mapped exclusively to non-skip lines.  Skip lines
        (code / headers) keep their original body unchanged.  If a translated
        chunk has fewer lines than expected the remaining non-skip lines fall
        back to their original body.
        """
        # ── Flatten ──
        all_line_infos = []
        for chunk_infos in chunk_line_infos_list:
            all_line_infos.extend(chunk_infos)

        # ── Collect per-chunk translated lines ──
        all_translated_lines: list[str] = []
        for chunk_idx, chunk_text in enumerate(translated_chunks):
            if chunk_text is None:
                chunk_text = ""
            chunk_lines = chunk_text.split('\n')

            # consumption-unit placeholders: translatable-segment body + "empty-line
            # segments" built by the all-skip fallback branch (no structure and empty
            # body; produced only by that branch, corresponding to blank lines in the
            # chunk text).
            # matches the translated-segment count in _text_split; padding fills in
            # the original text via placeholders.
            chunk_infos = chunk_line_infos_list[chunk_idx]
            placeholders: list[str] = []
            for li in chunk_infos:
                segs = self._translatable_segments(li)
                if segs:
                    placeholders.extend(si.body for si in segs)
                elif (li and len(li) == 1
                        and not li[0].prefix and not li[0].suffix
                        and li[0].body == ''):
                    placeholders.append('')  # empty-line segment placeholder
            expected = len(placeholders)
            if len(chunk_lines) != expected:
                self._log(
                    "warn",
                    f"[translate] _merge_chunks: chunk #{chunk_idx} 行数不匹配——"
                    f"期望 {expected} 段, 实际 {len(chunk_lines)} 行",
                )
                # defensive: pad with original text / truncate (translate() already
                # guarantees a match on the normal path).
                # when there are extra lines, absorb leading empty lines (extra blank
                # lines emitted by the model) before truncating — so the tail
                # translation is not cut off and later consumption units stay aligned.
                if len(chunk_lines) < expected:
                    missing_start = len(chunk_lines)
                    for j in range(missing_start, expected):
                        chunk_lines.append(
                            placeholders[j] if j < len(placeholders) else ''
                        )
                else:
                    while chunk_lines and chunk_lines[0] == '' and len(chunk_lines) > expected:
                        chunk_lines.pop(0)
                    chunk_lines = chunk_lines[:expected]

            all_translated_lines.extend(chunk_lines)

        # ── Rebuild: iterate line_infos, pull translated lines per segment ──
        # per-segment line backfill: each translatable segment consumes one translated
        # line into its body; skip / pure-whitespace filler / empty lines consume no
        # translated lines and keep the whole line as-is. When a translated line is
        # empty, the segment's original body is kept (guards against model omissions);
        # structure (prefix/suffix) is always preserved.
        adjusted_line_infos = []
        ti = 0  # index into all_translated_lines

        for line_info in all_line_infos:
            if not line_info:
                adjusted_line_infos.append([SentenceInfo()])
                continue

            segs = self._translatable_segments(line_info)
            if not segs:
                # lines with no translatable segments (skip lines / pure-whitespace
                # structural lines): the whole line keeps its original text.
                # empty-line segments (built by the all-skip fallback branch) consume a
                # translated line only when it is an empty string, preserving empty-line
                # semantics without shifting later lines; when the translated line is
                # non-empty (the model merged blank lines away) it is not consumed and
                # is left for later translatable segments.
                if (len(line_info) == 1
                        and not line_info[0].prefix and not line_info[0].suffix
                        and line_info[0].body == ''
                        and ti < len(all_translated_lines)
                        and all_translated_lines[ti] == ''):
                    ti += 1
                adjusted_line_infos.append(line_info)
                continue

            for si in segs:
                translated_line = (
                    all_translated_lines[ti]
                    if ti < len(all_translated_lines)
                    else ''
                )
                ti += 1
                if translated_line:
                    si.body = translated_line
                # empty translation → keep original body (guards against omission); prefix/suffix structure unchanged
            adjusted_line_infos.append(line_info)

        return splitter.merge(adjusted_line_infos)


# ── LlamaTranslator — local llama-server ──

class LlamaTranslator(Translator):
    """Translator backed by a local llama-server process."""

    def __init__(self, config):
        super().__init__(config)
        server_arg = self.config.get("server_arg", {})
        host = server_arg.get("--host", "127.0.0.1")
        port = server_arg.get("--port", "8080")
        self.base_url = f"http://{host}:{port}"
        self.translate_timeout = None
        self._context_size = int(server_arg.get("-c", 4096))
        self._server_ready = False

    def _resolve_max_token(self, max_token, trans_config):
        if max_token is not None:
            return max_token
        ratio = trans_config.get("max_token_ratio", 0.4)
        return int(ratio * self._context_size)

    def _resolve_max_lines(self, max_lines, trans_config):
        """Only the llama backend parses the max_lines entry in trans_config (positive int applies; otherwise None)."""
        if max_lines is not None:
            return max_lines
        value = trans_config.get("max_lines")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None

    def _get_token_count(self, request_body):
        import requests

        try:
            self._wait_for_preparing()
            url = self.base_url + "/tokenize"
            # count the concatenated content length of all messages (system + user,
            # including rendered glossary); counting only the user message would
            # underestimate actual input tokens and loosen the chunk budget
            body = {"content": "\n".join(
                m.get("content", "") for m in request_body["messages"]
            )}
            response = requests.post(url, json=body, timeout=30)
            response.raise_for_status()
            return len(response.json()['tokens'])
        except requests.RequestException:
            self._log("warn", "[LlamaTranslator] tokenize 请求失败（服务可能未就绪）")
            return None

    def get_token_count(self, text, trans_config, prompts, glossary=None):
        request_body = self.render_prompt(text, trans_config, prompts, glossary)
        return self._get_token_count(request_body=request_body)

    def _wait_for_preparing(self, timeout=60, is_print=False):
        import requests

        if self._server_ready:
            return
        health_url = self.base_url + '/health'
        wait_time = 0
        while True:
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    self._server_ready = True
                    if is_print:
                        self._log("info", f"llama-server 就绪（Status Code:{response.status_code}）")
                    break
                if wait_time >= timeout:
                    raise TimeoutError
            except requests.RequestException:
                if is_print and wait_time % 10 == 0:
                    self._log("info", f"llama-server 尚未就绪，已等待 {wait_time}s")
            time.sleep(1)
            wait_time += 1

    def _translate(self, text, trans_config, prompts, glossary=None,
                   timeout=None, cancel_event=None, pause_event=None):
        """Streaming translation: the request runs on a background thread while the
        main thread polls pause/cancel for prompt interruption.

        requests' ``iter_lines`` is a blocking read: if the request ran on the main
        thread, a stuck model would only reach cancel/pause checkpoints at the next
        token, so the task could hang forever. Therefore the POST + SSE parsing runs on
        a daemon child thread; the main thread polls events and, on cancel/pause,
        breaks the child thread's blocking read via ``response.close()`` then joins it.
        """
        import requests

        _timeout = timeout if timeout is not None else self.translate_timeout
        try:
            self._wait_for_preparing()
            url = self.base_url + "/v1/chat/completions"
            request_body = self.render_prompt(text, trans_config, prompts, glossary)
            # streaming is forced on: pause/cancel checkpoints between tokens rely on
            # SSE chunks arriving (so the stream field in args config is dead config,
            # already removed from default.json)
            request_body["stream"] = True
        except Exception:
            return None

        parts: list[str] = []
        errors: list[BaseException] = []
        done = threading.Event()
        response_holder: dict = {}

        def _stream():
            try:
                with requests.post(url, json=request_body, timeout=_timeout, stream=True) as response:
                    response_holder["response"] = response
                    response.raise_for_status()
                    # explicit UTF-8 decoding: SSE responses often lack a charset
                    # (requests falls back to latin-1/bytes); line bytes must be decoded
                    # as UTF-8 or CJK tokens become garbled
                    for raw in response.iter_lines(decode_unicode=False):
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="replace")
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                                continue
                            if delta:
                                parts.append(delta)
            except Exception as e:  # includes Timeout/ConnectionError/HTTPError/RequestException
                errors.append(e)
            finally:
                done.set()

        t = threading.Thread(target=_stream, daemon=True)
        t.start()

        # main thread polls pause/cancel; on trigger, close the connection to interrupt
        # the child thread's read, join, then return None
        while not done.is_set():
            if not _wait_paused(pause_event, cancel_event):
                self._log("info", "[LlamaTranslator] 任务已取消，停止请求")
                resp = response_holder.get("response")
                if resp is not None:
                    resp.close()
                t.join(timeout=5)
                return None
            done.wait(0.1)

        t.join()
        if errors:
            e = errors[0]
            if isinstance(e, requests.Timeout):
                self._log("error", f"[LlamaTranslator] 请求超时（{_timeout}s）")
            elif isinstance(e, requests.ConnectionError):
                self._log("error", f"[LlamaTranslator] 连接失败: {e}")
            elif isinstance(e, requests.HTTPError):
                self._log("error", f"[LlamaTranslator] HTTP 错误 {e.response.status_code}: {e.response.text[:200]}")
            elif isinstance(e, requests.RequestException):
                self._log("error", f"[LlamaTranslator] 请求异常: {e}")
            else:
                self._log("error", f"[LlamaTranslator] 请求异常: {e}")
            return None
        return "".join(parts).strip()


# ── APITranslator — OpenAI-compatible cloud API ──

class APITranslator(Translator):
    """Translator backed by an OpenAI-compatible cloud API."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = self.config.get("base_url", "")
        self.model = self.config.get("model", "gpt-4o")
        self.timeout = self.config.get("timeout", 120)
        self._api_key = self.config.get("api_key", "")
        self._client = None                     # lazily-created openai SDK client
        self._connection_checked = False        # lazily check connectivity once before the first translation

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Normalize base_url for the openai SDK: strip trailing slash, append /v1 (SDK 2.x does not do this automatically).

        - ``https://host``    → ``https://host/v1``
        - ``https://host/v1`` → ``https://host/v1`` (no duplication)
        """
        base = (base_url or "").rstrip("/")
        if base.endswith("/v1"):
            return base
        return base + "/v1"

    def _get_client(self, timeout: Optional[float] = None):
        """Lazily create the openai SDK client (created once, reused)."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key or "EMPTY",
                base_url=self._normalize_base_url(self.base_url),
                timeout=timeout if timeout is not None else self.timeout,
                max_retries=2,
            )
        return self._client

    def check_connection(self, timeout: float = 10.0):
        """Check API connectivity + validate the model name (openai SDK /models list).

        Returns: (ok: bool, message: str)
        """
        import openai

        if not self._api_key or self._api_key == "YOUR_API_KEY_HERE":
            return False, "API Key 未配置"
        base = (self.base_url or "").rstrip("/")
        if not base:
            return False, "base_url 未配置"
        try:
            models = self._get_client(timeout=timeout).models.list()
        except openai.AuthenticationError as e:
            return False, f"API Key 无效：{e}"
        except openai.NotFoundError as e:
            return False, f"端点不存在（检查 base_url={base}）：{e}"
        except openai.APITimeoutError:
            return False, f"连接超时（{timeout}s）：{base}"
        except openai.APIConnectionError as e:
            return False, f"无法连接 API：{e}"
        except openai.APIStatusError as e:
            return False, f"HTTP {e.status_code}：{e}"
        except openai.OpenAIError as e:
            return False, f"请求异常：{e}"
        # model-name validation (lenient list format: parse failure is not reported as an error)
        try:
            ids = [getattr(m, "id", None) or getattr(m, "model", None) for m in models.data]
            ids = [str(i) for i in ids if i]
        except Exception:
            return True, "API 连接正常（模型列表无法解析，跳过校验）"
        if ids and self.model not in ids:
            return False, f"模型名 '{self.model}' 不在服务端可用列表（示例：{ids[:5]}）"
        return True, "API 连接正常"

    def _translate(self, text, trans_config, prompts, glossary=None,
                   timeout=None, cancel_event=None, pause_event=None):
        import openai

        if not self._api_key or self._api_key == "YOUR_API_KEY_HERE":
            self._log("error", "[APITranslator] API Key 未配置")
            return None
        # lazily check connectivity once before the first translation (skipped if the startup check already ran)
        if not self._connection_checked:
            ok, msg = self.check_connection(timeout=10)
            self._connection_checked = True
            if not ok:
                self._log("error", f"[APITranslator] {msg}")
                return None

        request_body = self.render_prompt(text, trans_config, prompts, glossary)
        # parameter whitelist: pass only OpenAI-compatible params (SDK validates strictly;
        # strips llama-specific ones like repeat_penalty)
        kwargs = {
            "model": self.model,
            "messages": request_body["messages"],
            "stream": True,
            "temperature": request_body.get("temperature"),
            "top_p": request_body.get("top_p"),
            "presence_penalty": request_body.get("presence_penalty"),
            "frequency_penalty": request_body.get("frequency_penalty"),
            "max_tokens": request_body.get("max_tokens"),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        _timeout = timeout if timeout is not None else self.timeout

        if cancel_event and cancel_event.is_set():
            return None
        try:
            stream = self._get_client(timeout=_timeout).chat.completions.create(**kwargs)
            parts = []
            for chunk in stream:
                # pause/cancel checkpoint (active during token generation)
                if not _wait_paused(pause_event, cancel_event):
                    self._log("info", "[APITranslator] 任务已取消，停止请求")
                    return None
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    parts.append(delta.content)
            return "".join(parts).strip()
        except openai.AuthenticationError as e:
            self._log("error", f"[APITranslator] API Key 无效: {e}")
            return None
        except openai.RateLimitError as e:
            self._log("error", f"[APITranslator] 速率限制: {e}")
            return None
        except openai.APITimeoutError as e:
            self._log("error", f"[APITranslator] 请求超时（{_timeout}s）: {e}")
            return None
        except openai.APIConnectionError as e:
            self._log("error", f"[APITranslator] 连接失败: {e}")
            return None
        except openai.APIStatusError as e:
            self._log("error", f"[APITranslator] HTTP {e.status_code}: {str(e)[:200]}")
            return None
        except openai.OpenAIError as e:
            self._log("error", f"[APITranslator] 请求异常: {e}")
            return None


# ── GsvTTSExecutor — GPT-SoVITS TTS (three-mode emotion replication) ──

class GsvTTSExecutor(Executor):
    """GPT-SoVITS text-to-speech executor (wraps ``core/gsv.GsvEngine``).

    Task contract (matches docs/plan-gsv-service-executor.md §4)::

        task.file_path          target text (plain string / .txt path / JSON containing a "text" key)
        task.configs["args"]    synth args (JSON file path → dict, or a direct dict):
            ref_mode            "default"|"aux"|"dual" (default: default)
            ref_audio_path      reference audio (emotion reference; required for default/aux/dual; single ref for default)
            prompt_text         reference text (optional for default, empty = ref_free; required for aux/dual; required for v4)
            prompt_lang         reference language (default: ja)
            role_ref_audio      role reference audio (timbre anchor; required for aux/dual)
            text_lang           target language (default: zh)
            + whitelisted synth args (top_k/top_p/temperature/... see ``_SYNTH_KEYS``)

    Three-mode dispatch (engine unchanged): default/aux → ``engine.synth_stream``
    (aux additionally passes ``aux_ref_audio_paths=[role_ref]``); dual →
    ``engine.synth_cross_speaker`` (emotion audio → S1 semantics, role audio → S2
    spectrogram/SV prompt_cache orchestration).

    Adaptive retry (bidirectional, coprime step sizes): when the target text equals or
    closely resembles the reference text, repetition_penalty suppresses tokens
    semantically duplicating the reference, causing S1 to EOS early and output only
    ~40% of the duration (the tail disappears); conversely, when RP is too low the
    model keeps repeating/continuing, and generation runs into the vendor limit
    (``early_stop_num = hz×max_sec``, 60s on this machine). After synthesis:
    - output duration < emotion ref duration×``min_ref_ratio`` (default 0.6)
      → judged too short (early EOS) → **lower** repetition_penalty by 0.05 and retry;
    - output duration > emotion ref duration×``dur_cap_ratio`` (default 2.0)
      → judged runaway generation (target ≈ continuation of reference)
      → **raise** repetition_penalty by 0.03 and retry.
    The step sizes 0.05/0.03 are coprime (gcd=0.01); together with ``visited`` dedup
    they traverse every 0.01 grid point in ``[rp_floor, rp_ceil]=[0.75, 2.25]``, so
    the search is never stuck on a dead end in a single direction; retry cap is
    ``max_retries`` (default 12). Thresholds/step sizes/bounds can all be overridden
    via args (not part of the engine whitelist); if reference-duration probing fails,
    adaptation is skipped.

    Cancel/pause checkpoints are inserted after each fragment yield (fragment-level
    granularity, consistent with the transcription queue); cancel =
    ``engine.stop()`` + ``CancelledError``, with no leftover inference.
    """

    REF_MODES = ("default", "aux", "dual")

    # whitelisted synth args (passed through to GsvEngine; unknown keys silently ignored to avoid engine TypeError)
    _SYNTH_KEYS = (
        "top_k", "top_p", "temperature", "repetition_penalty", "speed_factor",
        "sample_steps", "text_split_method", "seed", "batch_size",
        "parallel_infer", "super_sampling", "split_bucket", "fragment_interval",
    )

    # ── Adaptive retry params (bidirectional, coprime steps) ──
    _MIN_REF_RATIO = 0.6   # output < ref×this ratio → too short (early EOS) → lower RP
    _MAX_REF_RATIO = 2.0   # output > ref×this ratio → too long (runaway continuation) → raise RP
    _RP_STEP_DOWN = 0.05   # RP step down (too short)
    _RP_STEP_UP = 0.03     # RP step up (too long; coprime with _RP_STEP_DOWN → 0.01 grid covers the range)
    _RP_FLOOR = 0.75       # rp lower bound
    _RP_CEIL = 2.25        # rp upper bound
    _MAX_RETRIES = 12      # max attempts for the bidirectional search

    def __init__(self, engine, defaults: Optional[dict] = None):
        self.engine = engine
        self.defaults = defaults or {}

    @staticmethod
    def _check_ref_duration(path: str, label: str) -> None:
        """Pre-validate the 3–10s reference duration (run outside the lock to avoid triggering run()'s internal auto-reload).

        Reads metadata only, without loading waveforms; skips on read failure (e.g. mp3),
        relying on the engine's 3–10s hard validation as a fallback (same as vendored
        TTS.py:814-816).
        """
        try:
            import soundfile

            info = soundfile.info(path)
            dur = info.frames / info.samplerate
        except Exception:
            return
        if not (3.0 <= dur <= 10.0):
            raise ValueError(f"{label}时长 {dur:.2f}s 超出 3~10s 范围: {path}")

    @staticmethod
    def _probe_ref_duration(path: str) -> Optional[float]:
        """Probe the emotion reference audio duration (baseline for adaptive-retry decisions).

        Returns None when the path is empty or probing fails (some formats are not
        supported by soundfile) — the adaptive too-short check is skipped, preserving
        original behavior.
        """
        if not path:
            return None
        try:
            import soundfile

            info = soundfile.info(path)
            return info.frames / info.samplerate
        except Exception:
            return None

    def execute(
        self,
        task,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        import numpy as np

        cfg = self._resolve_task(task)  # multi-format resolution into (source, configs)
        source, configs = cfg
        args = {**self.defaults, **dict(configs.get("args") or {})}
        pause_event = getattr(task, "_pause_event", None)

        # ── Text: args["text"] takes precedence, else file_path (string text / .txt content / JSON "text" key) ──
        text = args.pop("text", None)
        if text is None:
            if isinstance(source, dict):
                text = source.get("text", "")
            elif isinstance(source, str):
                text = source
        if not text or not text.strip():
            raise ValueError("目标文本为空（请通过 file_path 提供文本或设置 args['text']）")

        # ── Pre-validation (outside the lock, avoiding run()'s expensive auto-reload) ──
        ref_mode = args.get("ref_mode", "default")
        if ref_mode not in self.REF_MODES:
            raise ValueError(f"未知 ref_mode: {ref_mode!r}（可选: {', '.join(self.REF_MODES)}）")
        ref_audio = args.get("ref_audio_path") or ""
        role_ref = args.get("role_ref_audio") or ""
        if ref_mode in ("aux", "dual") and not ref_audio:
            raise ValueError(f"ref_mode={ref_mode} 需要 ref_audio_path（情绪主参考）")
        if ref_mode in ("aux", "dual") and not role_ref:
            raise ValueError(f"ref_mode={ref_mode} 需要 role_ref_audio（角色参考音频）")
        for path, label in ((ref_audio, "参考音频"), (role_ref, "角色参考音频")):
            if path:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"{label}不存在: {path}")
                self._check_ref_duration(path, label)
        prompt_text = args.get("prompt_text", "") or ""
        if self.engine.version in ("v3", "v4") and not prompt_text.strip():
            raise ValueError("v3/v4 版本要求 prompt_text（参考文本）")

        # ── Three-mode dispatch (all streaming, for fragment-level progress/cancel) ──
        text_lang = args.get("text_lang", "zh")
        prompt_lang = args.get("prompt_lang", "ja")
        synth_params = {k: v for k, v in args.items() if k in self._SYNTH_KEYS}

        # ── Adaptive retry (bidirectional): too short → lower RP; too long (runaway) → raise RP ──
        # default mode: synthesis (inference) duration is determined by the target text
        # and need not be proportional to reference audio length, so no ratio anchoring
        # is done (ref_dur=None skips the bidirectional check, no retry)
        ref_dur = None if ref_mode == "default" else self._probe_ref_duration(ref_audio)
        min_ratio = float(args.get("min_ref_ratio", self._MIN_REF_RATIO))
        max_ratio = float(args.get("dur_cap_ratio", self._MAX_REF_RATIO))
        max_retries = int(args.get("max_retries", self._MAX_RETRIES))
        rp = float(synth_params.get("repetition_penalty", 1.35))
        retries = 0
        visited = {rp}   # tried rp values: coprime steps guarantee full grid coverage, so repeats are pointless
        fragments: list[np.ndarray] = []
        sr = None
        frag_total = 0        # global fragment counter (not reset on retry → progress callback is monotonic)
        total = max(1, (len(text) + 9) // 10)  # heuristic fragment estimate (sentence-level split, not exact)
        t0 = time.time()
        while True:
            if ref_mode == "dual":
                gen = self.engine.synth_cross_speaker(
                    text, text_lang,
                    emotion_ref_audio=ref_audio, emotion_text=prompt_text,
                    emotion_lang=prompt_lang, role_ref_audio=role_ref,
                    **synth_params,
                )
            else:
                if ref_mode == "aux":
                    synth_params["aux_ref_audio_paths"] = [role_ref]
                gen = self.engine.synth_stream(
                    text, text_lang,
                    ref_audio_path=ref_audio or None,
                    prompt_text=prompt_text, prompt_lang=prompt_lang,
                    **synth_params,
                )

            # ── Stream consumption (fragment-level progress/cancel checkpoints) ──
            try:
                for _i, (_sr, frag) in enumerate(gen):
                    if not _wait_paused(pause_event, cancel_event):
                        self.engine.stop()
                        raise CancelledError(task.id)
                    if cancel_event is not None and cancel_event.is_set():
                        self.engine.stop()
                        raise CancelledError(task.id)
                    sr = _sr
                    fragments.append(frag)
                    frag_total += 1
                    if progress_callback:
                        progress_callback(
                            frag_total, total, None,
                            {"fragment": frag_total, "attempt": retries + 1},
                        )
            finally:
                gen.close()  # triggers vendor-side empty_cache (TTS.py:1527-1528)

            # ── Bidirectional decision: too short → lower RP; too long (runaway) → raise RP ──
            dur = (sum(len(f) for f in fragments) / sr) if (sr and fragments) else 0.0
            too_short = ref_dur is not None and dur < ref_dur * min_ratio
            too_long = ref_dur is not None and dur > ref_dur * max_ratio
            if (not (too_short or too_long) or retries >= max_retries):
                break
            if too_short and rp > self._RP_FLOOR + 1e-9:
                new_rp = max(rp - self._RP_STEP_DOWN, self._RP_FLOOR)
                why = "疑似提前 EOS（目标文本与参考文本相似时常见）"
                ratio = min_ratio
                cmp = "<"
            elif too_long and rp < self._RP_CEIL - 1e-9:
                new_rp = min(rp + self._RP_STEP_UP, self._RP_CEIL)
                why = "疑似生成失控（输出远超参考，目标≈参考续写）"
                ratio = max_ratio
                cmp = ">"
            else:
                break  # still anomalous at the bound → accept current result (alarm on the evaluation side)
            if new_rp in visited:
                break  # this rp value was already tried and still anomalous; stop (prevents infinite loop)
            visited.add(new_rp)
            self._log(
                "warn",
                f"输出时长 {dur:.2f}s {cmp} 参考音频 {ref_dur:.2f}s×{ratio:.2f}，"
                f"{why}，repetition_penalty {rp:.2f}→{new_rp:.2f} 重试（第 {retries + 1} 次）",
            )
            rp = new_rp
            synth_params["repetition_penalty"] = rp
            fragments = []
            retries += 1

        if sr is None or not fragments:
            raise RuntimeError("合成未产出任何音频片段")

        # ── Output: output/gsv/{task_id}.wav ──
        from app.paths import project_root

        audio = np.concatenate(fragments)
        out_dir = project_root / "output" / "gsv"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task.id}.wav"
        import soundfile

        soundfile.write(str(out_path), audio, sr, subtype="PCM_16")
        return {
            "audio_path": str(out_path),
            "sample_rate": sr,
            "duration": len(audio) / sr,
            "info": {
                "version": self.engine.version,
                "ref_mode": ref_mode,
                "fragments": len(fragments),
                "seed": self.engine.last_seed,
                "elapsed_sec": round(time.time() - t0, 2),
                "retries": retries,
            },
        }


# ── Module-level lazy attributes (PEP 562) ──
# core.executor.requests / core.executor.openai remain reachable (tests monkeypatch
# this path). First access imports the real module and caches it.

def __getattr__(name):
    if name in ("requests", "openai"):
        import importlib

        module = importlib.import_module(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
