"""Rule-based text splitter (v2) — LineInfo 恒为物理行 + region 区域 + ps_pair 成对 + 占位符跨行按行片段发标记。

- split(text) → line_infos（外层 = 物理行）
- merge(line_infos) → text（逐行还原，与源逐字节一致）

rule structure::
    {
        'prefix':      [entry, ...],                    # 前缀条目（字面量/正则）
        'suffix':      [entry, ...],                    # 后缀条目
        'placeholder': [(entry, dst), ...],             # 占位符（DOTALL 编译，跨 \\n 按行片段发标记）
        'skip':        [entry, ...],                    # 可选：整行透传不译
        'recognize':   [entry, ...],                    # 可选：仅匹配行参与翻译
        'ps_pair':     [{open, close}, ...],            # 可选：成对 prefix-suffix（成对优先）
        'region':      [{label:{open,close}, concat:[...]}, ...],  # 可选：跨行区域（虚拟行应用规则再分行）
    }

entry forms::
    str                  → literal（内部 re.escape）
    {"literal": "..."}  → literal（显式形式）
    {"regex": "..."}    → 正则

region 语义::
    - label：跨行符号对（如 { } 块定界）；开行/闭行不入内容（普通行走 recognize/skip）
    - concat：行尾续行符（补充性质：不独立开启区域，仅延续已开启区域）
    - 区域包裹部分 = 虚拟行（内部 \\n 保留）→ 应用规则 → 再分行回物理行

跨行占位符::
    - 匹配文本含 \\n 时按行片段各发一标记（序号放入 <<<>>> 内部：<<<AT#0>>>），\\n 保留
    - 记录每片段 (原文, 标记)；再分行后每行只携带本行标记，merge 逐行还原

向后兼容：无 ps_pair/region 时行为与 v1（rule_splitter_v1.py）一致（物理行 + 独立前缀/后缀扫描）。
"""

import re
from dataclasses import dataclass, field

_MARKER_RE = re.compile(r"<<<[^>\n]+>>>")


def _ordinal_marker(dst: str, ordinal: int) -> str:
    """唯一标记：序号放入 <<<>>> 内部（如 <<<AT#0>>>），整体为单个不可变 token。"""
    if dst.startswith("<<<") and dst.endswith(">>>"):
        return dst[:-3] + f"#{ordinal}" + dst[-3:]
    return f"{dst}#{ordinal}"


def _compile_dotall(entry):
    """str/{"literal"}/{"regex"} 编译（DOTALL：占位符允许跨 \\n 匹配）。"""
    if isinstance(entry, str):
        return re.compile(re.escape(entry), re.DOTALL)
    if isinstance(entry, dict):
        if isinstance(entry.get('literal'), str):
            return re.compile(re.escape(entry['literal']), re.DOTALL)
        if isinstance(entry.get('regex'), str):
            return re.compile(entry['regex'], re.DOTALL)
    raise TypeError(
        f"rule 条目必须为 str(字面量)、{{\"literal\": ...}} 或 "
        f"{{\"regex\": ...}},收到 {type(entry).__name__}: {entry!r}"
    )


# ── Data structures ──


@dataclass
class SentenceInfo:
    """Per-sentence format metadata.

    Fields:
        prefix:       匹配的前缀结构文本
        suffix:       匹配的后缀结构文本
        placeholders: [(原文, 唯一标记), ...] 本段占位符
        body:         正文（翻译后由执行器回填译文）
        is_skip:      True = 跳过翻译，merge 直接取 body
        logical_span: (起始物理行, 结束物理行)（区域再分行后每行 = 自身）
    """
    prefix: str = ""
    suffix: str = ""
    placeholders: list = field(default_factory=list)
    body: str = ""
    is_skip: bool = False
    logical_span: tuple | None = None


# ── RuleSplitter ──


class RuleSplitter:
    """Regex-rule based text splitter/merger (v2).

    rule structure::
        {
            'prefix':      [entry, ...],      # 前缀条目
            'suffix':      [entry, ...],      # 后缀条目
            'placeholder': [(entry, dst), ...],  # 占位符
            'skip':        [entry, ...],      # 可选
            'recognize':   [entry, ...],      # 可选
            'ps_pair':     [{open, close}, ...],  # 可选：成对 prefix-suffix
            'region':      [{label, concat}, ...],  # 可选：跨行区域
        }

    Usage::
        splitter = RuleSplitter(rule)
        line_infos = splitter.split(text)
        # ... the translator translates is_skip=False sentences and backfills body ...
        result = splitter.merge(line_infos)
    """

    @staticmethod
    def _compile_entry(entry):
        """Compile a single rule entry into a compiled regex.

        - str                → literal (compiled after re.escape)
        - {"literal": "..."} → literal (explicit form, equivalent to str)
        - {"regex": "..."}   → regular expression
        - anything else      → TypeError
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
        prefix_raw: list = rule.get('prefix', [])
        suffix_raw: list = rule.get('suffix', [])
        self._multi: bool = rule.get('multi', False)
        self._prefix_list: list[re.Pattern] = [
            self._compile_entry(p) for p in prefix_raw
        ]
        self._suffix_list: list[re.Pattern] = [
            self._compile_entry(s) for s in suffix_raw
        ]

        # placeholder: (compiled matching entry with DOTALL, replacement text)
        self._placeholder_pairs: list[tuple[re.Pattern, str]] = [
            (_compile_dotall(src), dst)
            for src, dst in rule.get('placeholder', [])
        ]

        # skip / recognize (optional)
        skip_raw: list = rule.get('skip', [])
        self._skip_res: list[re.Pattern] | None = (
            [self._compile_entry(s) for s in skip_raw] if skip_raw else None
        )
        rec_raw: list = rule.get('recognize', [])
        self._recognize_res: list[re.Pattern] | None = (
            [self._compile_entry(r) for r in rec_raw] if rec_raw else None
        )

        # ps_pair: 成对 prefix-suffix（open/close 支持 str/literal/regex）
        self._paired: list[tuple[re.Pattern, re.Pattern]] = [
            (self._compile_entry(p["open"]), self._compile_entry(p["close"]))
            for p in rule.get('ps_pair', [])
        ]

        # region: 跨行区域（label 跨行符号对 + concat 行尾续行补充）
        self._regions: list[dict] = []
        for r in rule.get('region', []):
            label = r.get('label') or {}
            self._regions.append({
                "open": self._compile_entry(label["open"]),
                "close": self._compile_entry(label["close"]),
                "concat": [self._compile_entry(c) for c in r.get('concat', [])],
            })

    # ── Placeholder handling（DOTALL + 跨行按行片段发标记）──

    def _replace_placeholders(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Replace placeholders; cross-\\n matches emit one marker per line piece (ordinal inside <<<>>>).

        Returns:
            (replaced_text, matched_pairs) — each pair is (piece original text, unique marker)
        """
        if not self._placeholder_pairs:
            return text, []

        matched: list[tuple[str, str]] = []
        result = text
        for pat, dst in self._placeholder_pairs:
            matches = list(pat.finditer(result))
            if not matches:
                continue
            per_match = [
                m.group(0).split("\n") if "\n" in m.group(0) else [m.group(0)]
                for m in matches
            ]
            total = sum(len(p) for p in per_match)
            # replace from the back: earlier match positions unaffected by length changes
            for mi in range(len(matches) - 1, -1, -1):
                m = matches[mi]
                pieces = per_match[mi]
                start_idx = sum(len(per_match[k]) for k in range(mi))
                markers = []
                for k, p in enumerate(pieces):
                    mk = dst if total == 1 else _ordinal_marker(dst, start_idx + k)
                    markers.append(mk)
                    matched.append((p, mk))
                result = result[:m.start()] + "\n".join(markers) + result[m.end():]
            matched.reverse()
        return result, matched

    def _restore_placeholders(self, text: str, placeholders: list[tuple[str, str]]) -> str:
        """Restore replaced placeholders to their original form (pure string replacement)."""
        result = text
        for src, dst in reversed(placeholders):
            result = result.replace(dst, src)
        return result

    # ── Sentence splitting (stack scan) ──

    _PAIR_MAP = {
        '「': '」', '『': '』', '（': '）', '(': ')',
        '【': '】', '《': '》', '〈': '〉', '〔': '〕',
    }
    _SAME_CHARS = {'"', "'", '`'}

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        """Split by sentence-ending punctuation (。！？!?), skipping punctuation inside paired brackets/quotes."""
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

    # ── Skip / recognize checks ──

    def _should_skip(self, line: str) -> bool:
        """Check whether the line matches any skip regex."""
        if not self._skip_res:
            return False
        return any(pat.search(line) for pat in self._skip_res)

    def _should_recognize(self, line: str) -> bool:
        """Check whether the line matches any recognize regex.
        Everything is recognized by default when no recognize list is configured.
        """
        if not self._recognize_res:
            return True
        return any(pat.search(line) for pat in self._recognize_res)

    # ── Token scan（成对优先：close > open > 独立 prefix > 独立 suffix > body）──

    def _longest_match(self, patterns: list[re.Pattern], text: str, pos: int):
        """Try each regex anchored at pos, returning the one with the longest matched text."""
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

    def _find_paired(self, text: str):
        """ps_pair 平衡区域：[(open_pos, open_text, close_pos, close_text), ...]，支持嵌套/重叠。"""
        regions = []
        n = len(text)
        for ropen, rclose in self._paired:
            i = 0
            while i < n:
                mo = ropen.search(text, i)
                if not mo:
                    break
                open_pos, otxt = mo.start(), mo.group(0)
                depth, k, close = 1, mo.end(), None
                while k < n:
                    a = ropen.match(text, k)
                    if a and a.end() > a.start():
                        depth += 1
                        k = a.end()
                        continue
                    b = rclose.match(text, k)
                    if b and b.end() > b.start():
                        depth -= 1
                        if depth == 0:
                            close = (k, b.group(0))
                            break
                        k = b.end()
                        continue
                    k += 1
                if close is not None:
                    cpos, ctxt = close
                    regions.append((open_pos, otxt, cpos, ctxt))
                    i = cpos + len(ctxt)
                else:
                    break
        regions.sort(key=lambda r: r[0])
        return regions

    def _scan_tokens(self, text: str) -> list[tuple[str, str]]:
        """Token scan with paired-first precedence (close > open > independent prefix > suffix > body)."""
        if not text:
            return []

        paired = self._find_paired(text)
        open_at = {p[0]: (p[1], p[2]) for p in paired}
        close_at = {p[2]: p[3] for p in paired}
        special = sorted(set(open_at) | set(close_at))

        tokens: list[tuple[str, str]] = []
        i, n = 0, len(text)

        def _next_special(pos: int) -> int:
            for sp in special:
                if sp > pos:
                    return sp
            return n

        while i < n:
            # 成对优先：close 最高（区域必须闭合），其次 open；之后独立 prefix > suffix
            if i in close_at:
                ctxt = close_at[i]
                sm = self._longest_match(self._suffix_list, text, i)
                if sm and (sm.end() - sm.start()) > len(ctxt):
                    tokens.append(("suffix", sm.group(0)))
                    i = sm.end()
                else:
                    tokens.append(("suffix", ctxt))
                    i += len(ctxt)
                continue
            if i in open_at:
                ot, _cp = open_at[i]
                tokens.append(("prefix", ot))
                i += len(ot)
                continue

            pm = self._longest_match(self._prefix_list, text, i)
            sm = self._longest_match(self._suffix_list, text, i)
            chosen, is_pre = None, None
            if pm is not None and sm is not None:
                lp, ls = pm.end() - pm.start(), sm.end() - sm.start()
                chosen, is_pre = (pm, True) if lp >= ls else (sm, False)
            elif pm is not None:
                chosen, is_pre = pm, True
            elif sm is not None:
                chosen, is_pre = sm, False
            if chosen is not None and chosen.end() > chosen.start():
                tokens.append(("prefix" if is_pre else "suffix", chosen.group(0)))
                i = chosen.end()
                continue

            # body: until the next positive-width regex match or paired special position
            nxt = n
            for pat in self._prefix_list + self._suffix_list:
                m = pat.search(text, i)
                if m and m.end() > m.start() and m.start() < nxt:
                    nxt = m.start()
            sp = _next_special(i)
            if sp < nxt:
                nxt = sp
            if nxt <= i:
                nxt = i + 1  # guard: avoid infinite loop
            tokens.append(("body", text[i:nxt]))
            i = nxt

        return tokens

    # ── Structure aggregation ──

    def _aggregate_tokens(self, tokens: list[tuple[str, str]]) -> list[SentenceInfo]:
        """Aggregate the token stream into a list of structural segments."""
        segments: list[SentenceInfo] = []
        cur: SentenceInfo | None = None
        stage = 0  # 0=no segment, 1=prefix slot, 2=body slot, 3=suffix slot

        for kind, text in tokens:
            if kind == 'prefix':
                if stage in (0, 4):
                    cur = SentenceInfo(prefix=text)
                    segments.append(cur)
                    stage = 1
                elif stage == 1:
                    cur.prefix += text
                else:  # stage in (2, 3)
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
                else:  # stage == 3
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

    # ── Single-line split ──

    def _split_text(self, text: str) -> list[SentenceInfo]:
        """Structure recognition + aggregation; pure body segments are split by sentence-end punctuation."""
        if not text:
            return []

        tokens = self._scan_tokens(text)
        segments = self._aggregate_tokens(tokens)

        result: list[SentenceInfo] = []
        for seg in segments:
            if seg.prefix == '' and seg.suffix == '':
                for sentence in self._split_sentences(seg.body):
                    body_clean, matched = self._replace_placeholders(sentence)
                    result.append(SentenceInfo(
                        body=body_clean, placeholders=matched, is_skip=False,
                    ))
            else:
                body_clean, matched = self._replace_placeholders(seg.body)
                # separator whitespace reassignment: leading whitespace into prefix,
                # trailing whitespace into suffix (only when the structure exists)
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

    # ── Region detection（concat 为补充：不独立开启区域，仅延续已开启区域）──

    def _find_regions(self, physical: list[str]):
        """返回 [(内容起始行, 内容结束行, 虚拟行文本), ...]；开行/闭行不入内容。"""
        regions = []
        n = len(physical)
        i = 0
        while i < n:
            opener = None
            for r in self._regions:
                if self._unbalanced(r, physical[i]):
                    opener = r
                    break
            if opener is None:
                i += 1
                continue
            acc, total = [], physical[i]
            j = i + 1
            while j < n:
                total += "\n" + physical[j]
                if not self._unbalanced(opener, total) and not self._concat_tail(opener, total):
                    break  # 闭行（配平且无 concat 补充）
                acc.append(physical[j])
                j += 1
            regions.append((i + 1, j - 1, "\n".join(acc)))
            i = j + 1
        return regions

    def _unbalanced(self, r: dict, text: str) -> bool:
        no = len(r["open"].findall(text))
        nc = len(r["close"].findall(text))
        if r["open"].pattern == r["close"].pattern:
            return no % 2 == 1  # 自定界奇偶
        return no > nc  # 成对开多闭

    def _concat_tail(self, r: dict, buf: str) -> bool:
        """行尾续行（补充）：concat 在 buf 尾部命中（m.end() >= 最后非空白字符后）。"""
        tail_len = len(buf.rstrip())
        for pat in r["concat"]:
            m = pat.search(buf)
            if m and m.end() >= tail_len:
                return True
        return False

    # ── split：分行 → 区域 → 虚拟行应用规则 → 再分行（LineInfo=物理行）──

    def split(self, text: str) -> list[list[SentenceInfo]]:
        """Split by newlines (preserving empty lines); cross-line regions processed as virtual lines then re-split.

        Returns:
            line_infos: outer level = physical lines; empty lines are empty lists.
        """
        if not text:
            return []
        physical = text.split('\n')
        if not self._regions:
            return self._split_physical(physical)  # 纯物理行（ps_pair 仍生效于 _scan_tokens）

        regions = self._find_regions(physical)
        covered = set()
        for s, e, _v in regions:
            covered.update(range(s, e + 1))

        out: list[list[SentenceInfo]] = [[] for _ in physical]
        for i, line in enumerate(physical):
            if i in covered:
                continue
            if line == "":
                out[i] = []
            elif self._should_skip(line):
                out[i] = [SentenceInfo(body=line, is_skip=True)]
            elif not self._should_recognize(line):
                out[i] = [SentenceInfo(body=line, is_skip=True)]
            else:
                out[i] = self._split_text(line)
            for si in out[i]:
                si.logical_span = (i, i)

        for s, e, vtext in regions:
            if s > e:
                continue  # 无内容行（如空块）
            segs = self._process_virtual(vtext)
            out[s:e + 1] = self._resplit(segs, e - s + 1)
            for k, line_segs in enumerate(out[s:e + 1]):
                for si in line_segs:
                    si.logical_span = (s + k, s + k)
        return out

    def _split_physical(self, physical: list[str]) -> list[list[SentenceInfo]]:
        out: list[list[SentenceInfo]] = []
        for line in physical:
            if line == "":
                out.append([])
            elif self._should_skip(line):
                out.append([SentenceInfo(body=line, is_skip=True)])
            elif not self._should_recognize(line):
                out.append([SentenceInfo(body=line, is_skip=True)])
            else:
                out.append(self._split_text(line))
        return out

    def _process_virtual(self, vtext: str) -> list[SentenceInfo]:
        if vtext == "":
            return []
        if self._should_skip(vtext):
            return [SentenceInfo(body=vtext, is_skip=True)]
        if not self._should_recognize(vtext):
            return [SentenceInfo(body=vtext, is_skip=True)]
        return self._split_text(vtext)

    def _resplit(self, segments: list[SentenceInfo], n_lines: int) -> list[list[SentenceInfo]]:
        """再分行：段按 prefix/body/suffix 各自切 \\n，逐片按行序分配（占位符随片走）。"""
        lines: list[list[SentenceInfo]] = [[] for _ in range(n_lines)]
        li = 0
        for si in segments:
            if si.is_skip:
                pieces = (si.body or "").split("\n")
                for k, p in enumerate(pieces):
                    if li + k < n_lines:
                        lines[li + k].append(SentenceInfo(body=p, is_skip=True))
                li += len(pieces)
                continue
            stream = []
            for role, text in (("prefix", si.prefix), ("body", si.body or ""), ("suffix", si.suffix)):
                parts = text.split("\n")
                for idx, p in enumerate(parts):
                    stream.append((role, p, idx == len(parts) - 1))
            for role, p, is_last in stream:
                if li >= n_lines:
                    break
                ns = SentenceInfo(is_skip=False, placeholders=self._piece_placeholders(si.placeholders, p))
                if role == "prefix":
                    ns.prefix = p
                elif role == "body":
                    ns.body = p
                else:
                    ns.suffix = p
                lines[li].append(ns)
                if not is_last:
                    li += 1
        return lines

    def _piece_placeholders(self, placeholders: list[tuple[str, str]], piece: str) -> list[tuple[str, str]]:
        if not placeholders or not piece:
            return []
        ph_map = {mk: orig for orig, mk in placeholders}
        found = []
        for m in _MARKER_RE.finditer(piece):
            mk = m.group(0)
            if mk in ph_map:
                found.append((ph_map[mk], mk))
        return found

    # ── merge：prefix/body/suffix 统一还原（占位符逐行）──

    def merge(self, line_infos: list[list[SentenceInfo]]) -> str:
        """Rebuild by line and join with \\n. Restores placeholders in prefix/body/suffix."""
        if not line_infos:
            return ""

        line_results: list[str] = []
        for line_info in line_infos:
            parts: list[str] = []
            for si in line_info:
                p = self._restore_placeholders(si.prefix, si.placeholders)
                b = self._restore_placeholders(si.body if si.body is not None else "", si.placeholders)
                s = self._restore_placeholders(si.suffix, si.placeholders)
                parts.append(p + b + s)
            line_results.append(''.join(parts))

        return '\n'.join(line_results)
