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

# 注意：openai / requests / numpy 均为重库（合计数秒），已改为各使用点
# 函数内惰性导入。core.executor 被 task_que / moss_transcriber 等核心模块
# 在启动链中导入，顶层 import 会拖慢应用骨架首帧。


def _wait_paused(pause_event, cancel_event) -> bool:
    """暂停检查点：任务被暂停则阻塞等待恢复；被取消返回 False。

    ``pause_event`` 为 None（直接调用 execute 之外场景）或未暂停时立即返回 True。
    """
    # 取消优先：未暂停时也必须响应 cancel（否则取消仅对暂停中任务生效）
    if cancel_event is not None and cancel_event.is_set():
        return False
    if pause_event is None or pause_event.is_set():
        return True
    while not pause_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            return False
        pause_event.wait(0.2)
    return True


# ═══════════════════════════════════════════════════════════
# Executor — abstract base
# ═══════════════════════════════════════════════════════════

class Executor(ABC):
    """Abstract execution unit consumed by a TaskQueue worker."""

    # 诊断日志回调（UI 经 service 透传注入；无回调退化 print，库调用不受影响）
    _on_log = None

    def set_on_log(self, callback: Optional[Callable]):
        """注入/替换诊断日志回调（level, message）；无回调时退化 print。"""
        self._on_log = callback

    def _log(self, level: str, message: str) -> None:
        """记录一条诊断日志（level: info/warn/error）。"""
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

    # ── 任务解析（多格式读取，类似 Service._resolve_config）──────────

    @staticmethod
    def _resolve_value(value):
        """多格式读取单个值。

        - ``dict`` → 原样返回（已是解析结果）
        - 存在的文件路径（``Path``/``str``）→ UTF-8 读入；可解析为 JSON 则返回
          dict，否则返回文本内容；二进制文件（如音频）不强行解码，原样返回
        - 其余（如 ``"ja"`` 这类非文件字符串）→ 原样返回
        """
        if isinstance(value, dict):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    return value  # 二进制文件（音频等），原样返回
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return value
        return value

    def _resolve_task(self, task):
        """父类通用解析：仅实现多格式文件读取，不做语义解析。

        返回 ``(source, configs_dict)``：

        - ``source`` — 对 ``task.file_path`` 做 :meth:`_resolve_value` 的结果
        - ``configs_dict`` — 对 ``task.configs`` 中每个值做 :meth:`_resolve_value`

        子类应覆盖本方法：先调用 ``super()._resolve_task(task)`` 拿到通用读取
        结果，再自行实现语义解析（如音频路径、RuleSplitter 构建等）。
        """
        source = self._resolve_value(task.file_path)
        configs = {}
        for key, value in (task.configs or {}).items():
            configs[key] = self._resolve_value(value)
        return source, configs


# ═══════════════════════════════════════════════════════════
# Translator — abstract translation base
# ═══════════════════════════════════════════════════════════

class Translator(Executor):
    """Abstract translator with chunking, prompt rendering, and merge logic.

    Subclasses must implement ``_translate()`` for a single request."""

    # 单次翻译失败重试次数：服务停止中断的 chunk 在恢复后经重试补译
    # （_translate 对连接中断/超时返回 None，重试前重新等待后端就绪）
    _translate_retries = 3

    def __init__(self, config=None):
        super().__init__()
        self.config = None
        if config is not None:
            self.config = load_json_file(config)

    # ── Executor interface ───────────────────────────

    def execute(
        self,
        task,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str | None:
        """Run translation from *task*.

        Task 契约::

            task.file_path                    待翻译文本文件（Path/str/已读文本）
            task.configs["translate_config"]  翻译参数（路径 → JSON dict）
            task.configs["prompts"]           prompt 模板（路径 → JSON dict）
            task.configs["glossary"]          术语表（路径 → JSON dict，可选）
            task.configs["rule"]              分割规则（路径 → JSON dict，可选）
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
        """翻译语义解析：file_path 读原文文本，configs["rule"] 构建 RuleSplitter。"""
        source, configs = super()._resolve_task(task)

        if isinstance(source, str):
            text = source
        else:
            # 兜底：二进制/未解析路径按文本重读（正常场景不会走到）
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

    # ── 可翻译段判定（送译/计数/回填三方共用） ──────────────

    @staticmethod
    def _translatable_segments(line_info: list) -> list:
        """返回该行中【可翻译】的结构段列表。

        判定与 _merge_chunks 的 has_content 一致：non-skip 且
        (有 prefix/suffix 结构 或 body 非空白)。纯空白填充段
        （无结构、body 空白，如缩进）不算——它们只是结构占位，
        保留原 body、不送译、不消费译文行。
        供 _text_split / translate / _merge_chunks 三方共用，
        保证"送译段数 = 计数段数 = 回填段数"一一对应。
        """
        return [
            si for si in line_info
            if not si.is_skip
            and (si.prefix or si.suffix or (si.body and si.body.strip()))
        ]

    # ── Public API ───────────────────────────────────

    def translate(self, text, trans_config, prompts, glossary=None,
                  splitter=None,
                  timeout=None, cancel_event=None, pause_event=None,
                  progress_callback=None):
        """Unified translation entry point.

        With *splitter* → chunked translate → merge.
        Without *splitter* (规则=无) → 用「直通 splitter」走同一分块路径：
        不识别/跳过任何结构，仅按行 + token 预算分块——避免整段文本一次性
        请求超出模型 context（如 llama -c 1024）导致 HTTP 400 翻译失败。
        """
        # 配置缺失（未选择/选了无效项）→ 显式报错而非底层 TypeError：
        # 任务在队列中标记 FAILED，error 信息对用户可读。
        if trans_config is None:
            raise ValueError("翻译参数配置未选择：任务 configs['translate_config'] 缺失或选了无效项")
        if prompts is None:
            raise ValueError("提示词配置未选择：任务 configs['prompts'] 缺失或选了无效项")
        trans_config = load_json_file(trans_config)
        prompts = load_json_file(prompts)
        if glossary:
            glossary = load_json_file(glossary)

        # 规则=无：直通 splitter —— 空 prefix/suffix/skip/placeholder，
        # 所有非空行都可翻译，_text_split 只按 max_token 分块、_merge_chunks
        # 逐行回填，空行原样保留（与有规则路径完全一致）。
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
            # splitter 无法产出任何 chunk（如空文本）→ 原样返回
            return text or ""

        total = len(chunks)
        if progress_callback:
            progress_callback(0, total)

        _chunks = []
        for i, chunk in enumerate(chunks):
            if cancel_event and cancel_event.is_set():
                self._log("info", "[translate] 任务已取消，停止翻译")
                break
            # 暂停检查点：暂停期间阻塞等待，恢复后继续；取消则退出
            if not _wait_paused(pause_event, cancel_event):
                self._log("info", "[translate] 任务已取消，停止翻译")
                break
            _glossary = self._match_glossary(chunk, glossary)
            # 该 chunk 的可翻译段（送译/计数/回填三方一致）
            segs = [
                si for li in chunk_infos_list[i]
                for si in self._translatable_segments(li)
            ]
            expected = len(segs)
            result = None
            for attempt in range(self._translate_retries):
                if cancel_event and cancel_event.is_set():
                    break
                # 暂停检查点：暂停期间阻塞等待，恢复后继续；取消则退出
                if not _wait_paused(pause_event, cancel_event):
                    break
                result = self._translate(
                    chunk, trans_config, prompts, _glossary,
                    timeout=timeout, cancel_event=cancel_event,
                    pause_event=pause_event,
                )
                if result is not None:
                    break
                # 取消导致的 None（_translate 内部已记录“任务已取消”）不应进入重试/误报
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
                    # 译文行数不匹配（LLM 合并/拆行/漏行）→ 该块回退逐句：
                    # 每段 body 单独送翻译、单独收译文，保证段级一一对应。
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
                            per_line.append("")  # 空 body 结构段无需翻译
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

    # ── Subclass overrides ───────────────────────────

    def _translate(self, text, trans_config, prompts, glossary=None,
                   timeout=None, cancel_event=None, pause_event=None):
        raise NotImplementedError("子类必须实现 _translate 方法")

    def get_token_count(self, text, trans_config, prompts, glossary=None):
        return None

    def _resolve_max_token(self, max_token, trans_config):
        return max_token

    def _resolve_max_lines(self, max_lines, trans_config):
        """解析每 chunk 的可翻译行数上限;基类不解析(仅 llama 子类读取 max_lines 词条)。"""
        return max_lines

    # ── Prompt rendering ─────────────────────────────

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

    # ── Glossary matching ────────────────────────────

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

    # ── Text chunking ────────────────────────────────

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
            # 段级分行拼接：同一源文件行内的多个可翻译段，body 各自占一行
            # （'\n' 分隔）。LLM 输出译文时按行保持，merge 时逐行回填——
            # 避免旧版 ''（无分隔）拼接导致 LLM 无法分辨
            # 「\@ … ? … # … \@」等条件分支边界而把分支内容合并/错位。
            # 空 body 的结构段（如全角空格行）产生空行，与 expected 计数一致。
            bodies = [si.body for si in self._translatable_segments(li)]
            if bodies:
                translatable_lines.append('\n'.join(bodies))
                translatable_to_global.append(gi)

        if not translatable_lines:
            # 规则把全部内容判为 skip（如 ERB 模板规则误用于普通文本）：
            # 回退为整段翻译——non-skip 单 segment 承载全文，merge 时整段译文回填；
            # 空文本则返回空（调用方 translate() 兜底返回原文）
            if not text or not text.strip():
                return [], []
            # 整段回退：按原文行数构造 non-skip 行（译文逐行回填）。
            # 返回单个 chunk（chunk = 行列表，行 = [SentenceInfo]）
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
        prev_end = 0             # 上一个 chunk 的全局行右边界（首个 chunk 从文件头 0 开始）
        i = 0

        def _global_end(idx: int) -> int:
            """chunk 的全局行范围右边界：到下一个翻译行（含其间空行），
            末尾则到文件末尾（含尾部空行），避免空行落在 chunk 边界外丢失。"""
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
                    # 左边界从 prev_end（首个 chunk 为 0）开始，而不是第一个
                    # 可翻译行索引——否则文件头部（第一个可翻译行之前）的
                    # skip/注释行落在所有 chunk 之外，合并重建时整段丢失。
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

            # 消费单元占位：可翻译段 body + 全 skip 回退分支构造的"空行段"
            # （无结构且 body 为空串，仅该分支产生，对应 chunk 文本空行）。
            # 与 _text_split 送译段数一致，padding 按占位补原文。
            chunk_infos = chunk_line_infos_list[chunk_idx]
            placeholders: list[str] = []
            for li in chunk_infos:
                segs = self._translatable_segments(li)
                if segs:
                    placeholders.extend(si.body for si in segs)
                elif (li and len(li) == 1
                        and not li[0].prefix and not li[0].suffix
                        and li[0].body == ''):
                    placeholders.append('')  # 空行段占位
            expected = len(placeholders)
            if len(chunk_lines) != expected:
                self._log(
                    "warn",
                    f"[translate] _merge_chunks: chunk #{chunk_idx} 行数不匹配——"
                    f"期望 {expected} 段, 实际 {len(chunk_lines)} 行",
                )
                # 防御：补原文占位 / 截断（正常路径 translate() 已保证匹配）。
                # 多出时先吸收前导空行（模型额外输出的空行），再截断——
                # 避免把末尾译文截掉导致后续消费单元错位。
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
        # 段级逐行回填：每个可翻译段消费一行译文赋给其 body；skip 段/纯空白
        # 填充段/空行不消费译文行，整行原样保留。译文行空串时保留该段原文
        # body（防模型漏译），结构（prefix/suffix）始终保留。
        adjusted_line_infos = []
        ti = 0  # index into all_translated_lines

        for line_info in all_line_infos:
            if not line_info:
                adjusted_line_infos.append([SentenceInfo()])
                continue

            segs = self._translatable_segments(line_info)
            if not segs:
                # 无可翻译段行（skip 行/纯空白结构行）：整行保留原文。
                # 空行段（全 skip 回退分支构造）仅在译文行为空串时消费，
                # 保持空行语义且后续行不错位；译文行非空（模型合并掉空行）
                # 时不消费，留给后续可翻译段。
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
                # 译空 → 保留原文 body（防漏译），结构 prefix/suffix 不变
            adjusted_line_infos.append(line_info)

        return splitter.merge(adjusted_line_infos)


# ═══════════════════════════════════════════════════════════
# LlamaTranslator — local llama-server
# ═══════════════════════════════════════════════════════════

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
        """仅 llama 后端解析 trans_config 的 max_lines 词条(正整数生效;其余 None)。"""
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
            # 统计全部消息（system + user，含术语表渲染）的 content 拼接长度；
            # 只统计 user 消息会低估实际输入 token，chunk 预算偏松
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
        """流式翻译：请求在后台线程执行，主线程轮询暂停/取消，可即时中断。

        requests 的 ``iter_lines`` 是阻塞读：若把请求放在主线程，模型卡住时
        cancel/pause 检查点要等下一个 token 才生效，任务可能永久挂起。因此
        这里把 POST + SSE 解析放进 daemon 子线程，主线程轮询事件，取消/暂停
        时通过 ``response.close()`` 打断子线程的阻塞读取并 join 回收。
        """
        import requests

        _timeout = timeout if timeout is not None else self.translate_timeout
        try:
            self._wait_for_preparing()
            url = self.base_url + "/v1/chat/completions"
            request_body = self.render_prompt(text, trans_config, prompts, glossary)
            # 流式强制开启：token 间暂停/取消检查点依赖 SSE 逐块到达
            # （args 配置里的 stream 字段因此是死配置，已从 default.json 移除）
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
                    # 显式 UTF-8 解码：SSE 响应常无 charset（requests 会退化为
                    # latin-1/bytes），必须按 UTF-8 解码行字节，否则中文 token 变乱码
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
            except Exception as e:  # 含 Timeout/ConnectionError/HTTPError/RequestException
                errors.append(e)
            finally:
                done.set()

        t = threading.Thread(target=_stream, daemon=True)
        t.start()

        # 主线程轮询暂停/取消；触发时关闭连接打断子线程读取，join 后返回 None
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


# ═══════════════════════════════════════════════════════════
# APITranslator — OpenAI-compatible cloud API
# ═══════════════════════════════════════════════════════════

class APITranslator(Translator):
    """Translator backed by an OpenAI-compatible cloud API."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = self.config.get("base_url", "")
        self.model = self.config.get("model", "gpt-4o")
        self.timeout = self.config.get("timeout", 120)
        self._api_key = self.config.get("api_key", "")
        self._client = None                     # 懒创建 openai SDK client
        self._connection_checked = False        # 首次翻译前懒检测一次连通性

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """规范化 base_url 供 openai SDK 使用：去尾斜杠，补 /v1 后缀（SDK 2.x 不自动补）。

        - ``https://host``    → ``https://host/v1``
        - ``https://host/v1`` → ``https://host/v1``（不重复）
        """
        base = (base_url or "").rstrip("/")
        if base.endswith("/v1"):
            return base
        return base + "/v1"

    def _get_client(self, timeout: Optional[float] = None):
        """懒创建 openai SDK client（一次创建复用）。"""
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
        """检测 API 连通性 + 模型名校验（openai SDK /models 列表）。

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
        # 模型名校验（列表格式宽容：解析失败不误报）
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
        # 首次翻译前懒检测一次连通性（启动检测已置位则跳过）
        if not self._connection_checked:
            ok, msg = self.check_connection(timeout=10)
            self._connection_checked = True
            if not ok:
                self._log("error", f"[APITranslator] {msg}")
                return None

        request_body = self.render_prompt(text, trans_config, prompts, glossary)
        # 参数白名单：仅传 OpenAI 兼容参数（SDK 严格校验，剥离 llama 特有 repeat_penalty 等）
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
                # 暂停/取消检查点（token 生成期间生效）
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


# ═══════════════════════════════════════════════════════════
# GsvTTSExecutor — GPT-SoVITS 文本合成（三方案情绪复刻）
# ═══════════════════════════════════════════════════════════

class GsvTTSExecutor(Executor):
    """GPT-SoVITS 文本合成执行器（包装 ``core/gsv.GsvEngine``）。

    任务契约（与 docs/plan-gsv-service-executor.md §4 一致）::

        task.file_path          目标文本（直接字符串 / .txt 路径 / 含 "text" 键的 JSON）
        task.configs["args"]    合成参数（JSON 文件路径 → dict，或直接 dict）:
            ref_mode            "default"|"aux"|"dual"（默认 default）
            ref_audio_path      参考音频（主参考; default/aux/dual 必填，default 单参考）
            prompt_text         参考文本（default 可选，空 = ref_free; aux/dual 必填; v4 必填）
            prompt_lang         参考语种（默认 ja）
            role_ref_audio      角色参考音频（音色锚定; aux/dual 必填）
            text_lang           目标语种（默认 zh）
            + 白名单合成参数（top_k/top_p/temperature/... 见 ``_SYNTH_KEYS``）

    三方案分发（引擎零改动）: default/aux → ``engine.synth_stream``（aux 附加
    ``aux_ref_audio_paths=[role_ref]``）; dual → ``engine.synth_cross_speaker``
    （情绪音频 → S1 语义、角色音频 → S2 谱/SV 的 prompt_cache 编排）。

    自适应重试（双向，互素步长）: 目标文本与参考文本相同/高度相似时，
    repetition_penalty 会压制与参考语义重复的 token，导致 S1 提前 EOS、
    输出只剩约 40% 时长（后半段消失）；反之 RP 过低时模型复读/续写不停，
    生成冲到 vendor 上限（``early_stop_num = hz×max_sec``，本机 60s）。
    合成完成后：
    - 输出时长 < 情绪参考时长×``min_ref_ratio``（默认 0.6）→ 判定过短
      （提前 EOS）→ **下调** repetition_penalty 0.05 重试；
    - 输出时长 > 情绪参考时长×``dur_cap_ratio``（默认 2.0）→ 判定生成失控
      （目标≈参考续写）→ **上调** repetition_penalty 0.03 重试。
    升降步长 0.05/0.03 互素（gcd=0.01），配合 ``visited`` 去重可遍历
    ``[rp_floor, rp_ceil]=[0.75, 2.25]`` 区间内的全部 0.01 网格点，不会
    被困在单一方向的边界死路；重试上限 ``max_retries``（默认 12）。
    阈值/步长/边界均可用 args 覆盖（不进引擎白名单）；参考时长探测失败
    时跳过自适应。

    取消/暂停检查点插在每片段 yield 后（片段级粒度，与转写队列一致）; 取消 =
    ``engine.stop()`` + ``CancelledError``，无残留推理。
    """

    REF_MODES = ("default", "aux", "dual")

    # 白名单合成参数（透传 GsvEngine; 未知键静默忽略，避免引擎 TypeError）
    _SYNTH_KEYS = (
        "top_k", "top_p", "temperature", "repetition_penalty", "speed_factor",
        "sample_steps", "text_split_method", "seed", "batch_size",
        "parallel_infer", "super_sampling", "split_bucket", "fragment_interval",
    )

    # ── 自适应重试参数（双向，互素步长）──
    _MIN_REF_RATIO = 0.6   # 输出 < 参考×此比例 → 过短（提前 EOS）→ 降 RP
    _MAX_REF_RATIO = 2.0   # 输出 > 参考×此比例 → 过长（生成失控续写）→ 升 RP
    _RP_STEP_DOWN = 0.05   # 降 RP 步长（过短）
    _RP_STEP_UP = 0.03     # 升 RP 步长（过长；与 _RP_STEP_DOWN 互素 → 0.01 粒度覆盖区间）
    _RP_FLOOR = 0.75       # rp 下限
    _RP_CEIL = 2.25        # rp 上限
    _MAX_RETRIES = 12      # 双向搜索最多尝试次数

    def __init__(self, engine, defaults: Optional[dict] = None):
        self.engine = engine
        self.defaults = defaults or {}

    @staticmethod
    def _check_ref_duration(path: str, label: str) -> None:
        """3~10s 参考时长预校验（锁外执行，防触发 run() 内部自动重载）。

        只读元数据不加载波形; 读取失败（如 mp3）时跳过，由引擎侧 3~10s
        硬校验兜底（vendored TTS.py:814-816 同款）。
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
        """探测情绪参考音频时长（自适应重试判定基准）。

        路径为空或探测失败（个别格式 soundfile 不支持）返回 None ——
        自适应过短检测跳过，保持原行为。
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

        cfg = self._resolve_task(task)  # (source, configs) 多格式解析
        source, configs = cfg
        args = {**self.defaults, **dict(configs.get("args") or {})}
        pause_event = getattr(task, "_pause_event", None)

        # ── 文本: args["text"] 优先，否则 file_path（字符串文本 / .txt 内容 / JSON 的 "text" 键）
        text = args.pop("text", None)
        if text is None:
            if isinstance(source, dict):
                text = source.get("text", "")
            elif isinstance(source, str):
                text = source
        if not text or not text.strip():
            raise ValueError("目标文本为空（请通过 file_path 提供文本或设置 args['text']）")

        # ── 预校验（锁外，防触发 run() 内部昂贵的自动重载）─────────
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

        # ── 三方案分发（均流式，便于片段级进度/取消）───────────────
        text_lang = args.get("text_lang", "zh")
        prompt_lang = args.get("prompt_lang", "ja")
        synth_params = {k: v for k, v in args.items() if k in self._SYNTH_KEYS}

        # ── 自适应重试（双向）：过短→降 RP；过长（失控续写）→升 RP ──
        # default 模式：合成（推理）时长由目标文本决定，不要求与参考音频
        # 长度成比例，故不做比例锚定（ref_dur=None 跳过双向判定，不重试）
        ref_dur = None if ref_mode == "default" else self._probe_ref_duration(ref_audio)
        min_ratio = float(args.get("min_ref_ratio", self._MIN_REF_RATIO))
        max_ratio = float(args.get("dur_cap_ratio", self._MAX_REF_RATIO))
        max_retries = int(args.get("max_retries", self._MAX_RETRIES))
        rp = float(synth_params.get("repetition_penalty", 1.35))
        retries = 0
        visited = {rp}   # 已试过的 rp 值：互素步长保证网格全覆盖，重复尝试无意义
        fragments: list[np.ndarray] = []
        sr = None
        frag_total = 0        # 全局累计片段数（重试不重置 → 进度回调单调不减）
        total = max(1, (len(text) + 9) // 10)  # 启发式预估片段数（句级切分，非精确）
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

            # ── 流式消费（片段级进度/取消检查点）──────────────────
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
                gen.close()  # 触发 vendor 端 empty_cache（TTS.py:1527-1528）

            # ── 双向判定：过短降 RP；过长（失控）升 RP ──
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
                break  # 已到边界仍异常 → 接受当前结果（评估侧报警）
            if new_rp in visited:
                break  # 该 rp 值已试过仍异常，停止（防死循环）
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

        # ── 输出: output/gsv/{task_id}.wav ───────────────────────
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


# ── 模块级惰性属性（PEP 562）──────────────────────────────────
# 保持 core.executor.requests / core.executor.openai 旧有访问方式可用
# （tests 与旧代码 monkeypatch 依赖该路径）。首次访问时导入真实模块并缓存。

def __getattr__(name):
    if name in ("requests", "openai"):
        import importlib

        module = importlib.import_module(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
