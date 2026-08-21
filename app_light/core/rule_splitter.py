"""
基于规则的文本分割器 / Rule-based Text Splitter

- split(text) → line_infos
- merge(line_infos) → text

rule 结构::
    {
        'prefix':      [entry, ...],   # 前缀条目列表（字面量或正则，见下）
        'suffix':      [entry, ...],   # 后缀条目列表（字面量或正则，见下）
        'placeholder': [(entry, dst), ...],  # entry=匹配条目, dst=替换文本
        'skip':        [entry, ...],   # 可选：跳过条目列表
        'recognize':   [entry, ...],   # 可选：识别条目列表
    }

entry 条目形态::
    str                  → 字面量：按原文精确匹配（内部 re.escape，特殊字符无需转义）
    {"literal": "..."}  → 字面量：同上（显式写法，与 str 等价）
    {"regex": "..."}    → 正则表达式

行为（v2 独立识别算法）:
    - prefix 与 suffix 在行内【分别独立识别】，互不要求配对
    - 逐 token 扫描：每个 prefix/suffix 正则独立匹配（最长匹配优先），
      其余文本作为 body
    - 聚合：连续 prefix 归为一个结构段的前缀部分、连续 suffix 归为其后缀部分，
      输出形态仍为 [前缀段][body][后缀段]
    - 例如「\tPRINTFORML「こんにちは」」被识别为
      prefix="\tPRINTFORML「"、body="こんにちは"、suffix="」"
    - 例如「PRINTFORML あああ」（无后缀）被识别为 prefix="PRINTFORML"、body=" あああ"
      （旧版会把整行当作普通文本翻译，v2 不再如此）

   废弃: rule['multi'] 字段（旧版多级算法的开关）不再参与算法分派，
    仅保留解析以兼容旧配置。
"""

import re
from dataclasses import dataclass, field


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class SentenceInfo:
    """每句的格式元数据。

    Fields:
        prefix:       匹配到的前缀文本，无则为 ""
        suffix:       匹配到的后缀文本，无则为 ""
        placeholders: [(原始占位符, 替换符), ...] 该句中匹配到的占位符列表
        body:         当前文本（split 后为原文；翻译后由 translator 回填译文）
        is_skip:      True = 跳过不翻译，merge 时直接取 body；False = 需要翻译
    """
    prefix: str = ""
    suffix: str = ""
    placeholders: list = field(default_factory=list)
    body: str = ""
    is_skip: bool = False


# ── RuleSplitter ──────────────────────────────────────────

class RuleSplitter:
    """基于正则 rule 的文本分割/合并器（v2：prefix/suffix 独立识别；v2.1：条目支持字面量与正则）。

    rule 结构::
        {
            'prefix':      [entry, ...],      # 前缀条目列表
            'suffix':      [entry, ...],      # 后缀条目列表
            'placeholder': [(entry, dst), ...],  # entry=匹配条目, dst=替换文本
            'skip':        [entry, ...],      # 可选：跳过条目列表
            'recognize':   [entry, ...],      # 可选：识别条目列表
            'multi':       bool               # 已废弃，仅兼容解析
        }

    entry 条目形态（v2.1 起，prefix/suffix/skip/recognize/placeholder-src 通用）::
        str                  → 字面量：按原文精确匹配（内部 re.escape，特殊字符无需转义）
        {"literal": "..."}  → 字面量：同上（显式写法，与 str 等价）
        {"regex": "..."}    → 正则表达式

    Usage::
        splitter = RuleSplitter(rule)
        line_infos = splitter.split(text)
        # ... 翻译器对 is_skip=False 的句子翻译后回填 body ...
        result = splitter.merge(line_infos)
    """

    @staticmethod
    def _compile_entry(entry):
        """统一编译一条 rule 条目为编译后的正则。

        - str                → 字面量（re.escape 后编译）
        - {"literal": "..."} → 字面量（显式写法，与 str 等价）
        - {"regex": "..."}   → 正则表达式
        - 其它 → TypeError
        """
        if isinstance(entry, str):
            return re.compile(re.escape(entry))
        if isinstance(entry, dict):
            if isinstance(entry.get('literal'), str):
                return re.compile(re.escape(entry['literal']))
            if isinstance(entry.get('regex'), str):
                return re.compile(entry['regex'])
        raise TypeError(
            f"rule 条目必须为 str(字面量)、{{\"literal\": ...}} 或 "
            f"{{\"regex\": ...}},收到 {type(entry).__name__}: {entry!r}"
        )

    def __init__(self, rule: dict):
        # prefix / suffix：条目列表统一编译（v2.1：字符串=字面量，{"regex": ...}=正则）
        prefix_raw: list = rule['prefix']
        suffix_raw: list = rule['suffix']
        # multi 字段：旧版多级算法开关，v2 起废弃，仅保留解析以兼容
        self._multi: bool = rule.get('multi', False)
        self._prefix_list: list[re.Pattern] = [
            self._compile_entry(p) for p in prefix_raw
        ]
        self._suffix_list: list[re.Pattern] = [
            self._compile_entry(s) for s in suffix_raw
        ]

        # placeholder：(编译后匹配条目, 替换文本)
        self._placeholder_pairs: list[tuple[re.Pattern, str]] = [
            (self._compile_entry(src), dst)
            for src, dst in rule.get('placeholder', [])
        ]

        # skip / recognize（可选）
        skip_raw: list = rule.get('skip', [])
        self._skip_res: list[re.Pattern] | None = (
            [self._compile_entry(s) for s in skip_raw] if skip_raw else None
        )
        rec_raw: list = rule.get('recognize', [])
        self._recognize_res: list[re.Pattern] | None = (
            [self._compile_entry(r) for r in rec_raw] if rec_raw else None
        )

    # ── placeholder 处理 ──────────────────────────────────

    def _replace_placeholders(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """按顺序替换占位符（entry 已编译为 pattern）。

        同一 dst 条目的每次匹配各生成**唯一**替换标记（dst + 序号），matched
        记录 (原始文本, 唯一标记) 对——正则条目匹配到不同文本（如任意
        ``<<<x...>>>``）时，恢复阶段也能逐一还原；旧实现只记第一个匹配的
        actual_text，多占位符会把所有标记恢复成第一个。

        Returns:
            (replaced_text, matched_pairs) — matched_pairs 每项为
            (实际匹配到的原始文本, 其唯一替换标记)
        """
        if not self._placeholder_pairs:
            return text, []

        matched: list[tuple[str, str]] = []
        result = text
        for pat, dst in self._placeholder_pairs:
            matches = list(pat.finditer(result))
            if not matches:
                continue
            # 从后往前替换：前面的匹配位置不受已替换标记长度变化影响
            for idx in range(len(matches) - 1, -1, -1):
                m = matches[idx]
                unique_dst = dst if len(matches) == 1 else f"{dst}#{idx}"
                result = result[:m.start()] + unique_dst + result[m.end():]
                matched.append((m.group(0), unique_dst))
            matched.reverse()
        return result, matched

    def _restore_placeholders(self, text: str, placeholders: list[tuple[str, str]]) -> str:
        """将已替换的占位符恢复为原始形式（反序恢复，纯字符串替换）。"""
        result = text
        for src, dst in reversed(placeholders):
            result = result.replace(dst, src)
        return result

    # ── 分句（栈式扫描） ──────────────────────────────────

    _PAIR_MAP = {
        '「': '」', '『': '』', '（': '）', '(': ')',
        '【': '】', '《': '》', '〈': '〉', '〔': '〕',
    }
    _SAME_CHARS = {'"', "'", '`'}

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        """按句末标点（。！？!?）分句，跳过括号/引号配对内的标点。"""
        if not text:
            return []

        stack: list[str] = []
        results: list[str] = []
        start = 0

        for i, ch in enumerate(text):
            if stack:
                top = stack[-1]
                if top in cls._PAIR_MAP and ch == cls._PAIR_MAP[top]:
                    stack.pop()
                    continue
                if top in cls._SAME_CHARS and ch == top:
                    stack.pop()
                    continue

            if ch in cls._PAIR_MAP or ch in cls._SAME_CHARS:
                stack.append(ch)
            elif not stack and ch in '。！？!?':
                results.append(text[start:i + 1])
                start = i + 1

        if start < len(text):
            results.append(text[start:])

        return results

    # ── skip / recognize 判断 ─────────────────────────────

    def _should_skip(self, line: str) -> bool:
        """检查该行是否匹配任一跳过符号正则。"""
        if not self._skip_res:
            return False
        return any(pat.search(line) for pat in self._skip_res)

    def _should_recognize(self, line: str) -> bool:
        """检查该行是否匹配任一识别符号正则。
        若未配置 recognize 列表则默认全部识别。
        """
        if not self._recognize_res:
            return True
        return any(pat.search(line) for pat in self._recognize_res)

    # ── token 级独立扫描（v2 核心） ───────────────────────

    def _longest_match(self, patterns: list[re.Pattern], text: str, pos: int):
        """在 pos 处尝试各正则锚定匹配，返回匹配文本最长的那个。

        - 最长匹配优先：解决 PRINTFORML / PRINTFORMDL 类前缀冲突
          （PRINTFORMDL 不会被 PRINTFORML 截断）
        - 平局取列表顺序
        - 允许零宽匹配（如行尾 $）；调用方需自行处理
        """
        best = None
        best_len = -1
        for pat in patterns:
            m = pat.match(text, pos)
            if m:
                ln = m.end() - m.start()
                if ln > best_len:
                    best = m
                    best_len = ln
        return best

    def _scan_tokens(self, text: str) -> list[tuple[str, str]]:
        """token 级独立扫描：prefix / suffix 分别识别，互不要求配对。

        Returns:
            [(kind, matched_text), ...]，kind ∈ {'prefix', 'suffix', 'body'}
        """
        if not text:
            return []

        n = len(text)
        tokens: list[tuple[str, str]] = []
        pos = 0

        while pos < n:
            pm = self._longest_match(self._prefix_list, text, pos)
            sm = self._longest_match(self._suffix_list, text, pos)

            # 同一位置 prefix/suffix 同时命中：更长者胜，等长 prefix 优先
            chosen = None
            if pm is not None and sm is not None:
                if sm.end() - sm.start() > pm.end() - pm.start():
                    chosen = ('suffix', sm)
                else:
                    chosen = ('prefix', pm)
            elif pm is not None:
                chosen = ('prefix', pm)
            elif sm is not None:
                chosen = ('suffix', sm)

            if chosen is not None:
                kind, m = chosen
                if m.end() == m.start():
                    # 零宽匹配（如行尾 $）：不产 token，忽略并继续普通文本收集
                    chosen = None
                else:
                    tokens.append((kind, text[m.start():m.end()]))
                    pos = m.end()
                    continue

            # 普通文本：收集到下一个【正长度】prefix/suffix 命中点
            next_pos = n
            for pat in self._prefix_list:
                m = pat.search(text, pos)
                if m and m.end() > m.start() and m.start() < next_pos:
                    next_pos = m.start()
            for pat in self._suffix_list:
                m = pat.search(text, pos)
                if m and m.end() > m.start() and m.start() < next_pos:
                    next_pos = m.start()
            if next_pos <= pos:
                next_pos = pos + 1  # 防御：避免死循环
            tokens.append(('body', text[pos:next_pos]))
            pos = next_pos

        return tokens

    # ── 结构聚合（v2 核心） ────────────────────────────────

    def _aggregate_tokens(self, tokens: list[tuple[str, str]]) -> list[SentenceInfo]:
        """把 token 流聚合成结构段列表。

        形态：连续 prefix → [前缀段]，body → [body]，连续 suffix → [后缀段]。
        - prefix 与 prefix 相邻 → 并入同一段的前缀位
        - body 跟在 prefix 后 / body 后 → 并入同一段
        - suffix 跟在 body 后 / suffix 后 → 并入同一段（后缀位）
        - body 跟在 suffix 后、prefix 跟在 body/suffix 后 → 关闭当前段，开新段
        """
        segments: list[SentenceInfo] = []
        cur: SentenceInfo | None = None
        stage = 0  # 0=无段, 1=前缀位, 2=body位, 3=后缀位, 4=已关闭

        for kind, text in tokens:
            if kind == 'prefix':
                if stage in (0, 4):
                    cur = SentenceInfo(prefix=text)
                    segments.append(cur)
                    stage = 1
                elif stage == 1:
                    cur.prefix += text
                else:  # stage in (2, 3)：body/后缀位后出现 prefix → 开新段
                    cur = SentenceInfo(prefix=text)
                    segments.append(cur)
                    stage = 1
            elif kind == 'body':
                if stage in (0, 4):
                    cur = SentenceInfo(body=text)
                    segments.append(cur)
                    stage = 2
                elif stage in (1, 2):
                    cur.body += text
                    stage = 2
                else:  # stage == 3：后缀位后出现 body → 开新段
                    cur = SentenceInfo(body=text)
                    segments.append(cur)
                    stage = 2
            else:  # kind == 'suffix'
                if stage in (0, 4):
                    cur = SentenceInfo(suffix=text)
                    segments.append(cur)
                    stage = 3
                elif stage in (1, 2, 3):
                    cur.suffix += text
                    stage = 3

        return segments

    # ── 单行分割（v2 入口） ────────────────────────────────

    def _split_text(self, text: str) -> list[SentenceInfo]:
        """独立识别 + 结构聚合，纯 body 段按句号分句。"""
        if not text:
            return []

        tokens = self._scan_tokens(text)
        segments = self._aggregate_tokens(tokens)

        result: list[SentenceInfo] = []
        for seg in segments:
            if seg.prefix == '' and seg.suffix == '':
                # 纯 body 段：按句末标点分句（沿用栈式配对分句）
                for sentence in self._split_sentences(seg.body):
                    body_clean, matched = self._replace_placeholders(sentence)
                    result.append(SentenceInfo(
                        body=body_clean, placeholders=matched, is_skip=False,
                    ))
            else:
                # 结构段：body 不分句，整段为一个 SentenceInfo
                body_clean, matched = self._replace_placeholders(seg.body)
                # 分隔空白归位：prefix/suffix 与正文之间的空格（如
                # "PRINTFORML こんにちは"）是结构的一部分——若留在 body 里
                # 会随正文送进 LLM，模型输出的译文不含该空格，merge 后丢失。
                # 前导空白并入 prefix、尾部空白并入 suffix（仅当对应结构存在；
                # 纯 body 段无此问题，行首/行尾空白语义另论）。
                if seg.prefix:
                    leading = len(body_clean) - len(body_clean.lstrip(" \t"))
                    if leading:
                        seg.prefix += body_clean[:leading]
                        body_clean = body_clean[leading:]
                if seg.suffix:
                    trailing = len(body_clean) - len(body_clean.rstrip(" \t"))
                    if trailing:
                        seg.suffix = body_clean[-trailing:] + seg.suffix
                        body_clean = body_clean[:-trailing]
                seg.body = body_clean
                seg.placeholders = matched
                seg.is_skip = False
                result.append(seg)

        return result

    # ── 核心方法 ──────────────────────────────────────────

    def split(self, text: str) -> list[list[SentenceInfo]]:
        """先按换行符分行（保留空行），再逐行应用规则。

        Returns:
            line_infos: 二维结构，外层每行。空行为空列表。
        """
        if not text:
            return []

        lines = text.split('\n')
        line_infos: list[list[SentenceInfo]] = []

        for line in lines:
            # 空行保留
            if line == "":
                line_infos.append([])
                continue

            # skip 优先
            if self._should_skip(line):
                line_infos.append([SentenceInfo(body=line, is_skip=True)])
                continue

            # recognize 过滤
            if not self._should_recognize(line):
                line_infos.append([SentenceInfo(body=line, is_skip=True)])
                continue

            # 正常分割
            line_infos.append(self._split_text(line))

        return line_infos

    def merge(self, line_infos: list[list[SentenceInfo]]) -> str:
        """按行重建并 \\n 拼接。直接使用每个 SentenceInfo.body。

        Args:
            line_infos: split 返回的二维结构，翻译器已回填非 skip 句子的 body。

        Returns:
            还原后的完整文本。
        """
        if not line_infos:
            return ""

        line_results: list[str] = []
        for line_info in line_infos:
            parts: list[str] = []
            for si in line_info:
                body_text = si.body if si.body is not None else ""
                body_with_ph = self._restore_placeholders(body_text, si.placeholders)
                parts.append(si.prefix + body_with_ph + si.suffix)
            line_results.append(''.join(parts))

        return '\n'.join(line_results)
