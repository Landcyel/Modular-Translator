"""
Rule-based text splitter.

- split(text) → line_infos
- merge(line_infos) → text

rule structure::
    {
        'prefix':      [entry, ...],   # prefix entry list (literal or regex, see below)
        'suffix':      [entry, ...],   # suffix entry list (literal or regex, see below)
        'placeholder': [(entry, dst), ...],  # entry=matching entry, dst=replacement text
        'skip':        [entry, ...],   # optional: skip entry list
        'recognize':   [entry, ...],   # optional: recognize entry list
    }

entry forms::
    str                  → literal: exact match against the source (internally re.escaped; special chars need no escaping)
    {"literal": "..."}  → literal: same as above (explicit form, equivalent to str)
    {"regex": "..."}    → regular expression

Behavior (v2 independent-recognition algorithm):
    - prefix and suffix are recognized independently within a line; no pairing required
    - token-level scan: each prefix/suffix regex matches independently (longest match first),
      remaining text becomes body
    - aggregation: consecutive prefixes form one segment's prefix part, consecutive
      suffixes form its suffix part; output shape stays [prefix][body][suffix]
    - e.g. 「\tPRINTFORML「こんにちは」」is recognized as
      prefix="\tPRINTFORML「", body="こんにちは", suffix="」"
    - e.g. 「PRINTFORML あああ」(no suffix) is recognized as prefix="PRINTFORML", body=" あああ"
"""

import re
from dataclasses import dataclass, field


# ── Data structures ──

@dataclass
class SentenceInfo:
    """Per-sentence format metadata.

    Fields:
        prefix:       matched prefix text, "" if none
        suffix:       matched suffix text, "" if none
        placeholders: [(original placeholder, replacement), ...] placeholders matched in this sentence
        body:         current text (original after split; the translator backfills the translation after translating)
        is_skip:      True = skip translation, merge takes body directly; False = needs translation
    """
    prefix: str = ""
    suffix: str = ""
    placeholders: list = field(default_factory=list)
    body: str = ""
    is_skip: bool = False


# ── RuleSplitter ──

class RuleSplitter:
    """Regex-rule based text splitter/merger (v2: prefix/suffix independently recognized; v2.1: entries support literals and regexes).

    rule structure::
        {
            'prefix':      [entry, ...],      # prefix entry list
            'suffix':      [entry, ...],      # suffix entry list
            'placeholder': [(entry, dst), ...],  # entry=matching entry, dst=replacement text
            'skip':        [entry, ...],      # optional: skip entry list
            'recognize':   [entry, ...],      # optional: recognize entry list
        }

    entry forms (since v2.1, shared by prefix/suffix/skip/recognize/placeholder-src)::
        str                  → literal: exact match against the source (internally re.escaped; special chars need no escaping)
        {"literal": "..."}  → literal: same as above (explicit form, equivalent to str)
        {"regex": "..."}    → regular expression

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
        # prefix / suffix: entry lists compiled uniformly (v2.1: string = literal, {"regex": ...} = regex)
        prefix_raw: list = rule['prefix']
        suffix_raw: list = rule['suffix']
        self._multi: bool = rule.get('multi', False)
        self._prefix_list: list[re.Pattern] = [
            self._compile_entry(p) for p in prefix_raw
        ]
        self._suffix_list: list[re.Pattern] = [
            self._compile_entry(s) for s in suffix_raw
        ]

        # placeholder: (compiled matching entry, replacement text)
        self._placeholder_pairs: list[tuple[re.Pattern, str]] = [
            (self._compile_entry(src), dst)
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

    # ── Placeholder handling ──

    def _replace_placeholders(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Replace placeholders in order (entries precompiled to patterns).

        Each match of the same dst entry generates a **unique** replacement marker
        (dst + ordinal); matched records (original text, unique marker) pairs — so
        when a regex entry matches different texts (e.g. arbitrary ``<<<x...>>>``),
        each can be restored individually during restoration.

        Returns:
            (replaced_text, matched_pairs) — each matched_pair is
            (actual matched original text, its unique replacement marker)
        """
        if not self._placeholder_pairs:
            return text, []

        matched: list[tuple[str, str]] = []
        result = text
        for pat, dst in self._placeholder_pairs:
            matches = list(pat.finditer(result))
            if not matches:
                continue
            # replace from the back: earlier match positions are unaffected by length
            # changes from already-replaced markers
            for idx in range(len(matches) - 1, -1, -1):
                m = matches[idx]
                unique_dst = dst if len(matches) == 1 else f"{dst}#{idx}"
                result = result[:m.start()] + unique_dst + result[m.end():]
                matched.append((m.group(0), unique_dst))
            matched.reverse()
        return result, matched

    def _restore_placeholders(self, text: str, placeholders: list[tuple[str, str]]) -> str:
        """Restore replaced placeholders to their original form (reverse order, pure string replacement)."""
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

    # ── Token-level independent scan (v2 core) ──

    def _longest_match(self, patterns: list[re.Pattern], text: str, pos: int):
        """Try each regex anchored at pos, returning the one with the longest matched text.

        - longest match first: resolves prefix conflicts like PRINTFORML / PRINTFORMDL
          (PRINTFORMDL is not truncated by PRINTFORML)
        - ties resolved by list order
        - zero-width matches allowed (e.g. end-of-line $); callers must handle them
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
        """Token-level independent scan: prefix / suffix recognized separately, no pairing required.

        Returns:
            [(kind, matched_text), ...], kind ∈ {'prefix', 'suffix', 'body'}
        """
        if not text:
            return []

        n = len(text)
        tokens: list[tuple[str, str]] = []
        pos = 0

        while pos < n:
            pm = self._longest_match(self._prefix_list, text, pos)
            sm = self._longest_match(self._suffix_list, text, pos)

            # prefix and suffix both match at the same position: longer wins, equal length prefers prefix
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
                    # zero-width match (e.g. end-of-line $): produces no token; ignore and continue plain-text collection
                    chosen = None
                else:
                    tokens.append((kind, text[m.start():m.end()]))
                    pos = m.end()
                    continue

            # plain text: collect up to the next prefix/suffix match with positive length
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
                next_pos = pos + 1  # guard: avoid infinite loop
            tokens.append(('body', text[pos:next_pos]))
            pos = next_pos

        return tokens

    # ── Structure aggregation (v2 core) ──

    def _aggregate_tokens(self, tokens: list[tuple[str, str]]) -> list[SentenceInfo]:
        """Aggregate the token stream into a list of structural segments.

        Shape: consecutive prefix → [prefix segment], body → [body], consecutive suffix → [suffix segment].
        - prefix adjacent to prefix → merged into the same segment's prefix slot
        - body after prefix / after body → merged into the same segment
        - suffix after body / after suffix → merged into the same segment (suffix slot)
        - body after suffix, prefix after body/suffix → close current segment, start a new one
        """
        segments: list[SentenceInfo] = []
        cur: SentenceInfo | None = None
        stage = 0  # 0=no segment, 1=prefix slot, 2=body slot, 3=suffix slot, 4=closed

        for kind, text in tokens:
            if kind == 'prefix':
                if stage in (0, 4):
                    cur = SentenceInfo(prefix=text)
                    segments.append(cur)
                    stage = 1
                elif stage == 1:
                    cur.prefix += text
                else:  # stage in (2, 3): prefix after body/suffix slot → start a new segment
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
                else:  # stage == 3: body after suffix slot → start a new segment
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

    # ── Single-line split (v2 entry) ──

    def _split_text(self, text: str) -> list[SentenceInfo]:
        """Independent recognition + structure aggregation; pure body segments are split by sentence-end punctuation."""
        if not text:
            return []

        tokens = self._scan_tokens(text)
        segments = self._aggregate_tokens(tokens)

        result: list[SentenceInfo] = []
        for seg in segments:
            if seg.prefix == '' and seg.suffix == '':
                # pure body segment: split by sentence-end punctuation (using the stack-based pairing split)
                for sentence in self._split_sentences(seg.body):
                    body_clean, matched = self._replace_placeholders(sentence)
                    result.append(SentenceInfo(
                        body=body_clean, placeholders=matched, is_skip=False,
                    ))
            else:
                # structural segment: body is not split; the whole segment is one SentenceInfo
                body_clean, matched = self._replace_placeholders(seg.body)
                # separator whitespace reassignment: spaces between prefix/suffix and the
                # body (e.g. "PRINTFORML こんにちは") are part of the structure — if left in
                # the body they go into the LLM with the text, and the model's translation
                # lacks that space, so it is lost on merge.
                # leading whitespace goes into prefix, trailing whitespace into suffix
                # (only when the corresponding structure exists; pure body segments have no
                # such issue — leading/trailing whitespace semantics differ).
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

    # ── Core methods ──

    def split(self, text: str) -> list[list[SentenceInfo]]:
        """Split by newlines first (preserving empty lines), then apply rules line by line.

        Returns:
            line_infos: two-dimensional structure, outer level per line. Empty lines are empty lists.
        """
        if not text:
            return []

        lines = text.split('\n')
        line_infos: list[list[SentenceInfo]] = []

        for line in lines:
            # empty lines preserved
            if line == "":
                line_infos.append([])
                continue

            # skip takes priority
            if self._should_skip(line):
                line_infos.append([SentenceInfo(body=line, is_skip=True)])
                continue

            # recognize filter
            if not self._should_recognize(line):
                line_infos.append([SentenceInfo(body=line, is_skip=True)])
                continue

            # normal split
            line_infos.append(self._split_text(line))

        return line_infos

    def merge(self, line_infos: list[list[SentenceInfo]]) -> str:
        """Rebuild by line and join with \\n. Uses each SentenceInfo.body directly.

        Args:
            line_infos: the two-dimensional structure returned by split, with the bodies
                of non-skip sentences already backfilled by the translator.

        Returns:
            the reconstructed full text.
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
