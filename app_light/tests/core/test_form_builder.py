"""form_builder 表单渲染引擎测试 — 构造 flet 控件但不挂载 page。

覆盖：
1. 分组渲染（MOSS 服务参数的段落标题）
2. 各字段类型分支（boolean→Switch / select→Dropdown / secret→密码框 /
   path+browse→TextField+IconButton / list/object→多行框 / number→数字键盘）
3. refs 完整性（每个字段都有控件引用）
4. 错误就地显示（错误文本只出现在对应字段行）
5. values 传入后控件初始值正确（值回环）
6. build_form_rows 兼容入口
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flet as ft

from ui.pages.settings.config_schema import (
    API, LLAMA, PROMPT, RULES, MOSS, MOSS_ARGS, ALL_CONFIG_TYPES,
)
from ui.pages.settings.form_builder import build_form, build_form_rows

ROOT = Path(__file__).resolve().parents[2]


def _find(controls, pred):
    """递归查找第一个满足 pred 的控件。"""
    for c in controls:
        if pred(c):
            return c
        kids = getattr(c, "controls", None) or getattr(c, "content", None)
        if isinstance(kids, list):
            hit = _find(kids, pred)
            if hit is not None:
                return hit
        elif kids is not None and isinstance(kids, ft.Control):
            hit = _find([kids], pred)
            if hit is not None:
                return hit
    return None


def _find_text(controls, value):
    return _find(controls, lambda c: isinstance(c, ft.Text) and c.value == value)


def _textfield_for(controls, key, refs):
    return refs[key]


# ── 1. 分组渲染 ──

def test_group_headers_rendered():
    refs = {}
    controls = build_form(MOSS, refs)
    assert _find_text(controls, "服务参数") is not None
    # 分组标题行不带图标（图标已全局移除）
    row = _find(controls, lambda c: isinstance(c, ft.Row) and any(
        isinstance(k, ft.Text) and k.value == "服务参数"
        for k in (getattr(c, "controls", None) or [])))
    assert row is not None
    assert not any(isinstance(c, ft.Icon) for c in (getattr(row, "controls", None) or []))


def test_flat_fields_no_header_for_unnamed_group():
    refs = {}
    controls = build_form(API, refs)
    # API 无分组 → 不出现任何组标题（全部字段行）
    assert _find_text(controls, "API 配置") is None
    assert len([c for c in controls if isinstance(c, ft.ResponsiveRow)]) == len(API.fields)


# ── 2. 字段类型分支 ──

def test_language_field_is_select_dropdown():
    """MOSS 转写参数「解码方式」为下拉选择：greedy/sample 且默认 greedy。"""
    refs = {}
    build_form(MOSS_ARGS, refs)
    dd = refs["decoding"]
    assert isinstance(dd, ft.Dropdown), "解码方式应为 Dropdown（select 类型）"
    assert [o.key for o in dd.options] == ["greedy", "sample"]
    assert dd.value == "greedy"
    # 初始值有效：sample 保留
    refs2 = {}
    build_form(MOSS_ARGS, refs2, values={"decoding": "sample"})
    assert refs2["decoding"].value == "sample"


def test_secret_field_password_mask():
    refs = {}
    controls = build_form(API, refs)
    tf = _textfield_for(controls, "api_key", refs)
    assert isinstance(tf, ft.TextField)
    assert tf.password is True
    assert tf.can_reveal_password is True


def test_boolean_switch_and_select_dropdown():
    refs = {}
    controls = build_form(MOSS_ARGS, refs)
    sw = refs["single_speaker"]
    assert isinstance(sw, ft.Switch)
    assert sw.value is True  # default "true"
    dd = refs["decoding"]
    assert isinstance(dd, ft.Dropdown)
    assert dd.value == "greedy"


def test_path_browse_has_button():
    refs = {}
    hit = {"called": False}

    def fake_browse(field, ref):
        hit["called"] = True
        assert field.key == "llama_path"

    controls = build_form(LLAMA, refs, on_browse=fake_browse)
    tf = refs["llama_path"]
    assert isinstance(tf, ft.TextField)
    row = _find(controls, lambda c: isinstance(c, ft.Row) and tf in
                (getattr(c, "controls", None) or []))
    assert row is not None
    btn = _find([row], lambda c: isinstance(c, ft.TextButton))
    assert btn is not None
    assert btn.content == "浏览…"
    assert btn.on_click is not None
    btn.on_click(None)
    assert hit["called"]


def test_list_and_object_are_multiline():
    refs = {}
    controls = build_form(RULES, refs)
    tf = refs["prefix"]
    assert isinstance(tf, ft.TextField) and tf.multiline is True
    # PROMPT 移除 name/placeholders 后仅剩 multiline 模板字段
    refs2 = {}
    build_form(PROMPT, refs2)
    assert refs2["system"].multiline is True
    assert set(refs2) == {"system", "user_with_glossary", "user_without_glossary"}


def test_number_keyboard_type():
    refs = {}
    build_form(MOSS_ARGS, refs)
    assert refs["temperature"].keyboard_type == ft.KeyboardType.NUMBER
    assert refs["top_k"].keyboard_type == ft.KeyboardType.NUMBER


# ── 3. refs 完整性 ──

def test_refs_cover_all_fields():
    for ct in ALL_CONFIG_TYPES.values():
        refs = {}
        build_form(ct, refs)
        missing = [f.key for f in ct.fields if f.key not in refs]
        assert not missing, f"{ct.key} 缺失 refs: {missing}"


# ── 4. 错误就地显示 ──

def test_errors_localized_to_field():
    from ui.theme import Palette
    refs = {}
    msg = "「API 地址」为必填项"
    controls = build_form(API, refs, errors={"base_url": [msg]})
    assert _find_text(controls, msg) is not None
    # 错误文本恰好出现一次且颜色为 Palette.ERROR
    n_err = len(_find_all(controls, lambda c: isinstance(c, ft.Text)
                          and c.value == msg and c.color == Palette.ERROR))
    assert n_err == 1


def test_general_errors_appended():
    refs = {}
    controls = build_form(API, refs, errors={"_general": ["整体错误"]})
    assert _find_text(controls, "整体错误") is not None


def _find_all(controls, pred):
    out = []
    for c in controls:
        if pred(c):
            out.append(c)
        kids = getattr(c, "controls", None) or getattr(c, "content", None)
        if isinstance(kids, list):
            out.extend(_find_all(kids, pred))
        elif kids is not None and isinstance(kids, ft.Control):
            out.extend(_find_all([kids], pred))
    return out


# ── 5. 值回环 ──

def test_values_roundtrip_into_controls():
    with open(ROOT / "configs/transcribe/args/default.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    form_values = MOSS_ARGS.to_form_values(data)
    refs = {}
    build_form(MOSS_ARGS, refs, values=form_values)
    assert refs["decoding"].value == "greedy"
    assert refs["single_speaker"].value is True
    assert refs["max_new_tokens"].value == "65536"
    # 收集回环
    values, parse_errors = MOSS_ARGS.collect_values(refs)
    assert not parse_errors
    out = MOSS_ARGS.build_output(values)
    assert out["decoding"] == "greedy"
    assert out["max_new_tokens"] == 65536


# ── 6. 兼容入口 ──

def test_build_form_rows_compat():
    refs = {}
    controls = build_form_rows(API, refs, errors={"model": ["x"]})
    assert _find_text(controls, "x") is not None
    assert refs["model"] is not None


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
