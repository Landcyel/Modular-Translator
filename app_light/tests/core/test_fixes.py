"""回归测试：LlamaTranslator 修复（5-1 / 5-2 / 5-8 / 5-11）。

覆盖（对应 summary/llamatranslator_analysis.md 第 5 章的问题编号）：
- 5-1：后台线程方案下取消/暂停即时生效（close 打断阻塞读取、返回 None）
- 5-2：全 skip 回退 + 空行的译文行对齐；正常路径空行/变量行/结构行不受影响
- 5-8：/tokenize 请求计入全部消息（含 system prompt）
- 5-11：正则占位符多匹配唯一恢复；字面量多匹配兼容；单匹配保持语义标记
"""
import threading
import time

import pytest

from core.executor import LlamaTranslator, Translator
from core.rule_splitter import RuleSplitter


# ═══════════════════════════════════════════════════════════
# 5-2 _merge_chunks 行对齐
# ═══════════════════════════════════════════════════════════

class _T(Translator):
    def _translate(self, *a, **k):
        raise NotImplementedError


def _fallback_splitter():
    """全 skip 规则：任意行都跳过 → _text_split 走整段回退分支。"""
    return RuleSplitter({"prefix": [], "suffix": [], "skip": [{"regex": ".*"}]})


def test_fallback_blank_line_alignment():
    """回退 + 空行，译文保留空行 → 逐行正确对齐（原始 bug 场景）。"""
    s = _fallback_splitter()
    t = _T()
    infos, chunks = t._text_split("A\n\nB", s, {}, {})
    assert chunks == ["A\n\nB"]
    assert t._merge_chunks(s, infos, ["甲\n\n乙"]) == "甲\n\n乙"


def test_fallback_blank_line_translation_without_empty():
    """回退 + 空行，模型合并掉空行（2 行译文）→ 空行保留、译文落到正确行。"""
    s = _fallback_splitter()
    t = _T()
    infos, _ = t._text_split("A\n\nB", s, {}, {})
    assert t._merge_chunks(s, infos, ["甲\n乙"]) == "甲\n\n乙"


def test_fallback_no_blank_line():
    s = _fallback_splitter()
    t = _T()
    infos, _ = t._text_split("A\nB", s, {}, {})
    assert t._merge_chunks(s, infos, ["甲\n乙"]) == "甲\n乙"


def test_normal_blank_line_kept():
    """正常路径：空行为 []，不消耗译文行，保留原文空行。"""
    s = RuleSplitter({"prefix": ["「"], "suffix": ["」"], "skip": [], "placeholder": []})
    t = _T()
    infos, _ = t._text_split("「こんにちは」\n\n「さようなら」", s, {}, {})
    assert t._merge_chunks(s, infos, ["你好\n再见"]) == "「你好」\n\n「再见」"


def test_placeholder_line_consumed():
    """占位符整行替换成标记（body 非空）→ 正常消耗译文行，恢复后还原变量。"""
    s = RuleSplitter({"prefix": [], "suffix": [], "placeholder": [["%VAR%", "<<<V>>>"]]})
    t = _T()
    infos, _ = t._text_split("前文\n%VAR%\n后文", s, {}, {})
    assert t._merge_chunks(s, infos, ["前文訳\n<<<V>>>\n后文訳"]) == "前文訳\n%VAR%\n后文訳"


def test_structural_space_line_kept():
    """全角空格结构行（prefix 非空）→ 消耗译文行并保留前缀结构。"""
    s = RuleSplitter({"prefix": ["「", "\u3000"], "suffix": ["」"], "skip": [], "placeholder": []})
    t = _T()
    infos, _ = t._text_split("「こんにちは」\n\u3000\n「さようなら」", s, {}, {})
    assert t._merge_chunks(s, infos, ["你好\n\n再见"]) == "「你好」\n\u3000\n「再见」"


# ═══════════════════════════════════════════════════════════
# 结构段分隔空白保留（prefix/suffix 与正文间空格）
# ═══════════════════════════════════════════════════════════

_ERB_LIKE = RuleSplitter({
    "prefix": ["PRINTFORML", "PRINTFORMDL", "\t"],
    "suffix": ["」", "』", "）"],
    "placeholder": [],
})


def _translate_line(splitter, text, translation):
    """split 单行后把有正文的 body 替换为译文再 merge（模拟翻译流程）。"""
    infos = splitter.split(text)
    for li in infos:
        for si in li:
            if si.body.strip():
                si.body = translation
    return splitter.merge(infos)


def test_structure_space_before_body_kept():
    """PRINTFORML 与正文间的前导空格翻译后保留。"""
    assert _translate_line(_ERB_LIKE, "PRINTFORML こんにちは", "你好") == "PRINTFORML 你好"


def test_structure_tab_prefix_space_kept():
    """tab 前缀 + PRINTFORML + 空格场景。"""
    assert _translate_line(_ERB_LIKE, "\tPRINTFORML こんにちは", "你好") == "\tPRINTFORML 你好"


def test_structure_space_before_suffix_kept():
    """正文与行尾引号间的空格翻译后保留。"""
    assert _translate_line(_ERB_LIKE, "PRINTFORML こんにちは 」", "你好") == "PRINTFORML 你好 」"


def test_structure_existing_lines_unaffected():
    """既有结构行（全角空格前缀、引号对）往返不受影响。"""
    s = RuleSplitter({"prefix": ["「", "\u3000"], "suffix": ["」"], "skip": [], "placeholder": []})
    infos = s.split("「こんにちは」\n\u3000\n「さようなら」")
    for li in infos:
        for si in li:
            if "こんにちは" in si.body:
                si.body = "你好"
            elif "さようなら" in si.body:
                si.body = "再见"
    assert s.merge(infos) == "「你好」\n\u3000\n「再见」"


# ═══════════════════════════════════════════════════════════
# chunk 左边界：文件头部 skip 行与空行保留
# ═══════════════════════════════════════════════════════════

_ERB_RECOGNIZE = RuleSplitter({
    "prefix": ["PRINTFORML", "PRINTFORMDL", "\t"],
    "suffix": ["」", "』", "）"],
    "recognize": ["PRINTFORML", "PRINTFORMDL"],
    "placeholder": [],
})


def _run_translate_pipeline(splitter, text):
    """完整流程：_text_split 分块（段级分行）→ 模拟翻译（每段加"译"）→ _merge_chunks。"""
    t = _T()
    infos, chunks = t._text_split(text, splitter, {}, {})
    trans = ["\n".join(l + "译" for l in c.split("\n")) for c in chunks]
    return t._merge_chunks(splitter, infos, trans)


def test_file_head_skip_lines_kept():
    """文件头部（第一个可翻译行之前）的注释/代码行翻译后完整保留。"""
    text = (";※コメント\n"
            "@M_KOJO_K1_1\n"
            "CALL M_KOJO_K1_1_1\n"
            "RETURN RESULT\n"
            "PRINTFORML こんにちは\n"
            "PRINTFORMDL さようなら")
    out = _run_translate_pipeline(_ERB_RECOGNIZE, text)
    assert out == (";※コメント\n"
                   "@M_KOJO_K1_1\n"
                   "CALL M_KOJO_K1_1_1\n"
                   "RETURN RESULT\n"
                   "PRINTFORML こんにちは译\n"
                   "PRINTFORMDL さようなら译")


def test_blank_line_in_chunk_range_kept():
    """空行落在 chunk 行范围内保留（不丢失、不新增）。"""
    text = "PRINTFORML 前文\n\nPRINTFORML 後文"
    out = _run_translate_pipeline(_ERB_RECOGNIZE, text)
    assert out == "PRINTFORML 前文译\n\nPRINTFORML 後文译"


def test_no_extra_blank_lines():
    """输出行数与空行位置与原文一致（不添加原文件没有的空行）。"""
    text = "PRINTFORML A\n@CODE\nPRINTFORML B\n\nPRINTFORML C"
    out = _run_translate_pipeline(_ERB_RECOGNIZE, text)
    assert out == "PRINTFORML A译\n@CODE\nPRINTFORML B译\n\nPRINTFORML C译"


def test_multi_si_quote_structure_kept():
    """多 SI 行（PRINTFORML 前缀段 + 「」引号段）翻译后引号结构保留。"""
    text = "PRINTFORML 「……？什么，太近了？」"
    out = _run_translate_pipeline(_ERB_RECOGNIZE, text)
    assert out == "PRINTFORML 「……？什么，太近了？译」"


def test_multi_si_tab_quote_structure_kept():
    """tab 缩进 + PRINTFORML + 引号段：缩进与引号全部保留。"""
    text = "\t\tPRINTFORML 「哈……好热……」"
    out = _run_translate_pipeline(_ERB_RECOGNIZE, text)
    assert out == "\t\tPRINTFORML 「哈……好热……译」"


def _merge_with_trans(splitter, text, trans_lines):
    """完整流程但译文行自定义（含空行/行数漂移）。"""
    t = _T()
    infos, chunks = t._text_split(text, splitter, {}, {})
    return t._merge_chunks(splitter, infos, ["\n".join(trans_lines)])


def test_empty_translation_line_keeps_original():
    """模型某行译空 → 该行回退原文，不产生空内容行/多余空行。"""
    text = "PRINTFORML 前文\nPRINTFORMDL 中段\nPRINTFORML 後文"
    lines = ["前文译", "", "後文译"]  # 第 2 行译空
    out = _merge_with_trans(_ERB_RECOGNIZE, text, lines)
    assert out == "PRINTFORML 前文译\nPRINTFORMDL 中段\nPRINTFORML 後文译"


def test_translation_leading_blank_line_no_empty_content():
    """模型输出开头空行 → 吸收前导空行，译文不错位、不产生空内容行。"""
    text = "PRINTFORML 前文\nPRINTFORML 後文"
    lines = ["", "前文译", "後文译"]  # 多一个开头空行
    out = _merge_with_trans(_ERB_RECOGNIZE, text, lines)
    assert out == "PRINTFORML 前文译\nPRINTFORML 後文译"
    assert "「」" not in out and "\nPRINTFORML \n" not in out and not out.endswith("PRINTFORML ")


def test_translation_fewer_lines_keeps_original():
    """模型合并行（译文行数不足）→ 行数一致、无空内容行、末行回退原文。"""
    text = "PRINTFORML 前文\nPRINTFORMDL 中段\nPRINTFORML 後文"
    lines = ["前文译", "後文译"]  # 少 1 行
    out = _merge_with_trans(_ERB_RECOGNIZE, text, lines)
    assert len(out.split("\n")) == 3
    assert "「」" not in out and "\nPRINTFORML \n" not in out
    assert out.endswith("後文")  # 译文不足，末行回退原文（按序填充语义）


# ═══════════════════════════════════════════════════════════
# 5-1 后台线程：取消 / 暂停
# ═══════════════════════════════════════════════════════════

class _FakeStreamResponse:
    """模拟 SSE 流式响应：产出第一个 token 后阻塞，close() 解除阻塞并结束。"""

    def __init__(self):
        self.started = threading.Event()
        self.released = threading.Event()
        self.closed = False

    def raise_for_status(self):
        pass

    def close(self):
        self.closed = True
        self.released.set()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass  # 不自动 close：让测试能区分"显式 close"（取消触发）与正常完成

    def iter_lines(self, decode_unicode=False):
        self.started.set()
        yield b'data: {"choices":[{"delta":{"content":"\xe7\x94\xb2"}}]}'  # "甲"
        self.released.wait(timeout=5)
        if self.closed:
            return
        yield b"data: [DONE]"
        return


def _make_translator(monkeypatch):
    captured = {"resp_ready": threading.Event()}

    def fake_post(url, json=None, timeout=None, stream=False):
        captured["url"] = url
        captured["body"] = json
        captured["resp"] = _FakeStreamResponse()
        captured["resp_ready"].set()
        return captured["resp"]

    monkeypatch.setattr("core.executor.requests.post", fake_post)
    monkeypatch.setattr(
        "core.executor.LlamaTranslator._wait_for_preparing",
        lambda self, *a, **k: None,
    )
    t = LlamaTranslator({"server_arg": {}})
    return t, captured


_PROMPTS = {"system": "sys", "user_without_glossary": "{ORIGINAL_TEXT}"}


def test_translate_cancel_closes_response(monkeypatch):
    """取消后：响应被 close（打断阻塞读取）、_translate 返回 None。"""
    t, captured = _make_translator(monkeypatch)
    cancel = threading.Event()

    def cancel_later():
        captured["resp_ready"].wait(timeout=3)      # 等 fake_post 创建响应
        captured["resp"].started.wait(timeout=3)    # 等子线程产出第一个 token
        cancel.set()

    threading.Thread(target=cancel_later, daemon=True).start()
    result = t._translate("テキスト", {"request": {}}, _PROMPTS, cancel_event=cancel)

    assert result is None
    assert captured["resp"].closed


def test_translate_pause_resume(monkeypatch):
    """暂停时阻塞等待、不关闭连接；恢复后正常完成并返回译文。"""
    t, captured = _make_translator(monkeypatch)
    pause = threading.Event()
    pause.clear()  # 处于暂停态

    def resume_later():
        captured["resp_ready"].wait(timeout=3)      # 等 fake_post 创建响应
        captured["resp"].started.wait(timeout=3)    # 等子线程产出第一个 token
        time.sleep(0.2)
        pause.set()

    threading.Thread(target=resume_later, daemon=True).start()
    result = t._translate("テキスト", {"request": {}}, _PROMPTS, pause_event=pause)

    assert result == "甲"
    assert not captured["resp"].closed  # 暂停不触发关闭


def test_translate_normal_flow(monkeypatch):
    """正常完成：不取消不暂停，返回拼接译文。"""
    t, captured = _make_translator(monkeypatch)

    class _Immediate(_FakeStreamResponse):
        def iter_lines(self, decode_unicode=False):
            yield b'data: {"choices":[{"delta":{"content":"\xe7\x94\xb2"}}]}'
            yield b"data: [DONE]"
            return

    captured["resp"] = _Immediate()

    def fake_post(url, json=None, timeout=None, stream=False):
        return captured["resp"]

    monkeypatch.setattr("core.executor.requests.post", fake_post)
    result = t._translate("テキスト", {"request": {}}, _PROMPTS)
    assert result == "甲"


# ═══════════════════════════════════════════════════════════
# 5-8 /tokenize 计入全部消息
# ═══════════════════════════════════════════════════════════

def test_token_count_includes_system_prompt(monkeypatch):
    captured = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"tokens": [0] * 10}

    def fake_post(url, json=None, timeout=None, stream=False):
        captured["url"] = url
        captured["body"] = json
        return _FakeResp()

    monkeypatch.setattr("core.executor.requests.post", fake_post)
    monkeypatch.setattr(
        "core.executor.LlamaTranslator._wait_for_preparing",
        lambda self, *a, **k: None,
    )
    t = LlamaTranslator({"server_arg": {}})
    request_body = {
        "messages": [
            {"role": "system", "content": "SYSTEM_PROMPT"},
            {"role": "user", "content": "USER_TEXT"},
        ]
    }
    n = t._get_token_count(request_body)

    assert n == 10
    assert captured["url"].endswith("/tokenize")
    assert "SYSTEM_PROMPT" in captured["body"]["content"]
    assert "USER_TEXT" in captured["body"]["content"]


# ═══════════════════════════════════════════════════════════
# 5-11 占位符唯一恢复
# ═══════════════════════════════════════════════════════════

def test_placeholder_regex_multiple_roundtrip():
    """正则条目匹配不同文本（<<<X>>> / <<<Y>>>）→ 各自唯一还原。"""
    s = RuleSplitter({
        "prefix": [], "suffix": [],
        "placeholder": [[{"regex": "<<<[^>]*>>>"}, "<<<PH>>>"]],
    })
    text = "a<<<X>>>b<<<Y>>>c"
    replaced, matched = s._replace_placeholders(text)
    assert "<<<X>>>" not in replaced and "<<<Y>>>" not in replaced
    assert len(matched) == 2
    assert s._restore_placeholders(replaced, matched) == text


def test_placeholder_literal_multiple_roundtrip():
    """字面量条目多次出现（ERB 兼容）→ 往返一致。"""
    s = RuleSplitter({
        "prefix": [], "suffix": [],
        "placeholder": [["%CALLNAME:MASTER%", "<<<MASTER>>>"]],
    })
    text = "%CALLNAME:MASTER% と %CALLNAME:MASTER%"
    replaced, matched = s._replace_placeholders(text)
    assert s._restore_placeholders(replaced, matched) == text


def test_placeholder_single_keeps_semantic_dst():
    """单匹配：dst 保持语义标记形态（<<<x...>>>），不带序号。"""
    s = RuleSplitter({
        "prefix": [], "suffix": [],
        "placeholder": [["%TEXTTARGET%", "<<<TEXTTARGET>>>"]],
    })
    replaced, _ = s._replace_placeholders("x %TEXTTARGET% y")
    assert replaced == "x <<<TEXTTARGET>>> y"


# ═══════════════════════════════════════════════════════════
# \@ 条件输出结构（ERB 规则多段行）：段级分行回填 + 回退逐句
# ═══════════════════════════════════════════════════════════

_ERB_COND = RuleSplitter({
    "prefix": ["PRINTFORMW", "「", "\\@ ARG == 0 ?", "#"],
    "suffix": ["」", "\\@"],
    "recognize": ["PRINTFORMW"],
    "placeholder": [],
})


class _CondT(Translator):
    """按 plan 依次返回的 mock：第 1 次为整体译文，其后为回退逐句译文。"""

    def __init__(self, plan):
        super().__init__()
        self.plan = list(plan)
        self.calls = []

    def _translate(self, text, *a, **k):
        self.calls.append(text)
        return self.plan.pop(0) if self.plan else None


def test_cond_branch_line_aligned_when_count_matches():
    """\\@ ARG == 0 ? A # B \\@ 条件结构行：译文行数匹配 → 分支各归各位。"""
    text = 'PRINTFORMW 「あら、\\@ ARG == 0 ? A # B \\@…」'
    t = _CondT(['\n'.join(['', '哎呀，', 'A译', 'B译', '……'])])
    out = t.translate(text, {}, {}, splitter=_ERB_COND)
    assert out == 'PRINTFORMW 「哎呀，\\@ ARG == 0 ? A译# B译 \\@……」'
    assert len(t.calls) == 1  # 行数匹配，未回退


def test_cond_branch_line_fallback_per_segment():
    """整体译文行数不足 → 回退逐句，每段单独翻译、各归各位。"""
    text = 'PRINTFORMW 「あら、\\@ ARG == 0 ? A # B \\@…」'
    # 第 1 次调用（整体）返回 1 行合并译文 → 触发回退；其后为逐句译文
    t = _CondT(['哎呀，A译B译……', '哎呀，', 'A译', 'B译', '……'])
    out = t.translate(text, {}, {}, splitter=_ERB_COND)
    assert out == 'PRINTFORMW 「哎呀，\\@ ARG == 0 ? A译# B译 \\@……」'
    assert len(t.calls) == 5  # 1 整体 + 4 逐句（空 body 的 PRINTFORMW 段跳过）


def test_cond_branch_fallback_none_keeps_original():
    """回退逐句中某段返回 None → 该段保留原文 body。"""
    text = 'PRINTFORMW 「あら、\\@ ARG == 0 ? A # B \\@…」'
    t = _CondT(['合并', '哎呀，', None, 'B译', '……'])
    out = t.translate(text, {}, {}, splitter=_ERB_COND)
    assert out == 'PRINTFORMW 「哎呀，\\@ ARG == 0 ? A # B译 \\@……」'


def test_structural_blank_line_roundtrip():
    """空 body 结构段（\u3000 全角空格行）往返不变。"""
    s = RuleSplitter({
        "prefix": ["PRINTFORMW", "「", "\u3000"],
        "suffix": ["」", "\\@"],
        "skip": [],
        "placeholder": [],
    })
    text = 'PRINTFORMW 「こんにちは」\n\u3000\nPRINTFORMW 「さようなら」'
    # 5 个消费单元：PRINTFORMW段('')/「こんにちは」/\u3000段/PRINTFORMW段('')/「さようなら」
    t = _CondT(['\n'.join(['', 'こんにちは译', '', '', 'さようなら译'])])
    out = t.translate(text, {}, {}, splitter=s)
    assert out == 'PRINTFORMW 「こんにちは译」\n\u3000\nPRINTFORMW 「さようなら译」'
