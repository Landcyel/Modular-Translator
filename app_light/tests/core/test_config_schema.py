"""config_schema 声明式引擎测试 — 纯 Python，不依赖 flet 渲染。

覆盖：
1. 单一注册表 CONFIG_GROUPS 派生完整性
2. 模拟控件走完整值管线：to_form_values → collect_values → validate → build_output
3. 与 configs/ 磁盘真实 JSON 的往返一致性（重点：修复过的数据丢失场景）
4. validate 错误精确定位到字段（必填 / 整数 / JSON 解析）
5. Field 扩展属性（group / secret / browse）与 FieldGroup 聚合
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.pages.settings.config_schema import (
    CONFIG_GROUPS, ALL_CONFIG_TYPES, CONFIG_TYPE_LIST,
    LLAMA, API, ARGS, ARGS_API, PROMPT,
    HOTWORDS, GLOSSARY, RULES, OUTPUT,
)

ROOT = Path(__file__).resolve().parents[2]


class _Ctrl:
    """模拟表单控件：只有 value 属性。"""

    def __init__(self, value=None):
        self.value = value


def _read_json(rel: str) -> dict:
    with open(ROOT / rel, "r", encoding="utf-8") as f:
        return json.load(f)


def _pipe(ct, data: dict) -> dict:
    """完整值管线：JSON → 表单值 → 控件 → 类型化值 → 校验 → build_output。"""
    form_values = ct.to_form_values(data)
    refs = {f.key: _Ctrl(form_values[f.key]) for f in ct.fields}
    values, parse_errors = ct.collect_values(refs)
    field_errors, general = ct.validate(values)
    assert not parse_errors, f"{ct.key} parse_errors: {parse_errors}"
    assert not field_errors, f"{ct.key} field_errors: {field_errors}"
    assert not general, f"{ct.key} general: {general}"
    return ct.build_output(values)


# ── 1. 注册表 ──

def test_registry_derivation():
    all_keys = [ct.key for _, items in CONFIG_GROUPS for ct in items]
    assert len(all_keys) == len(set(all_keys)), "CONFIG_GROUPS 内 key 重复"
    assert set(all_keys) == set(ALL_CONFIG_TYPES), "ALL_CONFIG_TYPES 与 CONFIG_GROUPS 不一致"
    assert [ct.key for ct in CONFIG_TYPE_LIST] == all_keys, "CONFIG_TYPE_LIST 顺序与 CONFIG_GROUPS 不一致"
    assert CONFIG_TYPE_LIST[0].key == "translate_default", "默认选中类型应为导航第一项（系统→翻译默认配置）"
    # 导航分组（系统组：翻译/转写/TTS/输出默认配置；转写组：
    # MOSS 转写参数/提示词/热词；语音组：角色配置 + 合成参数两个子条目）
    expect = [("系统", ["translate_default", "transcribe_default", "tts_default",
                        "output"]),
              ("服务", ["llama", "api", "moss", "gsv"]),
              ("翻译", ["prompt", "args", "args_api", "rules", "glossary"]),
              ("转写", ["moss_args", "moss_prompt", "hotwords"]),
              ("语音", ["gsv_role", "gsv_args"])]
    got = [(g, [ct.key for ct in items]) for g, items in CONFIG_GROUPS]
    assert got == expect, f"CONFIG_GROUPS 与预期不符: {got}"


def test_save_dir_matches_config_picker():
    """schema save_dir 与 ui/widgets/config_picker.py 目录映射一致（对外契约）。"""
    from ui.widgets.config_picker import _CONFIG_DIR_MAP

    pairs = [("llama", "llama"), ("api", "api"),
             ("args", "translate_args"), ("args_api", "translate_args_api"),
             ("prompt", "prompts"), ("glossary", "glossary"),
             ("rules", "rules"),
             ("moss", "moss"), ("moss_args", "moss_args"),
             ("hotwords", "hotwords"),
             ("gsv", "gsv"), ("gsv_role", "gsv_role"), ("gsv_args", "gsv_args")]
    for schema_key, map_key in pairs:
        sd = ALL_CONFIG_TYPES[schema_key].save_dir.replace("\\", "/")
        md = _CONFIG_DIR_MAP[map_key].replace("\\", "/")
        assert sd == md, f"{schema_key}: schema={sd} != picker={md}"


# ── 2. 值管线基础 ──

def test_default_pipeline():
    """默认值 → 表单值 → 类型化值 → build_output 可跑通。"""
    for ct in CONFIG_TYPE_LIST:
        refs = {f.key: _Ctrl(f.to_control_value(f.default)) for f in ct.fields}
        values, parse_errors = ct.collect_values(refs)
        assert not parse_errors, f"{ct.key} 默认值管线 parse_errors: {parse_errors}"
        field_errors, general = ct.validate(values)
        assert not field_errors and not general, f"{ct.key} 默认值校验失败: {field_errors} {general}"
        out = ct.build_output(values)
        assert isinstance(out, dict), f"{ct.key} build_output 非 dict"


def test_field_extended_attrs():
    api_key = API.fields[1]
    assert api_key.key == "api_key" and api_key.secret is True
    llama_path = LLAMA.fields[0]
    assert llama_path.browse == "directory"
    model_file = LLAMA.fields[1]
    assert model_file.browse == "file"


# ── 3. 与磁盘真实 JSON 往返一致性 ──

def test_rules_roundtrip_keeps_recognize_skip():
    # 默认规则文件（jp_noval/lrc）不含 recognize/skip → 构建输出空列表;
    # 额外用带 recognize/skip 的内联数据验证往返保留
    for rel in ["configs/translate/rules/jp_noval.json",
                "configs/translate/rules/lrc.json"]:
        data = _read_json(rel)
        out = _pipe(RULES, data)
        assert out["prefix"] == data["prefix"], "prefix 丢失/损坏"
        assert out["suffix"] == data["suffix"], "suffix 丢失/损坏"
        assert out["placeholder"] == data["placeholder"], "placeholder 丢失/损坏"
        assert out["recognize"] == data.get("recognize", []), "recognize 丢失/损坏"
        assert out["skip"] == data.get("skip", []), "skip 丢失/损坏"
    # recognize/skip 非空数据往返保留
    data = {"prefix": ["["], "suffix": ["]"], "placeholder": [],
            "recognize": [{"x": 1}], "skip": ["skip-me"]}
    out = _pipe(RULES, data)
    assert out["recognize"] == data["recognize"]
    assert out["skip"] == data["skip"]


def test_api_roundtrip_matches_disk():
    # 用模板 default.example.json（default.json 含真实密钥，被 .gitignore 不提交）
    data = _read_json("configs/models/API/default.example.json")
    out = _pipe(API, data)
    assert out["base_url"] == data["base_url"]
    assert out["api_key"] == data["api_key"]
    assert out["model"] == data["model"]
    assert out["timeout"] == data["timeout"]  # int


def test_llama_roundtrip_keeps_server_args():
    data = _read_json("configs/models/llama/default.json")
    out = _pipe(LLAMA, data)
    assert out["llama_path"] == data["llama_path"]
    # --keep/-n 已按需求从表单移除：保留键逐一往返一致，且不再输出这两个键
    kept = {"-m", "--host", "--port", "-ngl", "-c"}
    for k, v in data["server_arg"].items():
        if k in kept:
            assert str(out["server_arg"][k]) == str(v), f"server_arg.{k} 丢失/损坏"
    assert not ({"--keep", "-n"} & set(out["server_arg"])), (
        "--keep/-n 不应出现在构建输出中")


def test_args_roundtrip_matches_disk():
    data = _read_json("configs/translate/args_llama/default.json")
    out = _pipe(ARGS, data)
    assert out["max_token_ratio"] == data["max_token_ratio"]
    assert out["max_lines"] == data["max_lines"]
    assert out["request"] == data["request"]


def test_args_api_roundtrip_default_template():
    """API 翻译参数默认模板：model 为 DeepSeek V4 Flash 官方访问名称，
    不含 max_tokens 与 max_token_ratio，max_lines 默认 -1（负值 = 不限制）。"""
    data = _read_json("configs/translate/args_api/default.json")
    out = _pipe(ARGS_API, data)
    assert out["request"]["model"] == "deepseek-v4-flash"
    assert out["request"] == data["request"]
    assert "max_token_ratio" not in data and "max_token_ratio" not in out
    assert out["max_lines"] == -1
    assert "max_tokens" not in out["request"]
    assert "max_tokens" not in data["request"]
    # 与 Llama 词条差异：API 版无 max_tokens/max_token_ratio、max_lines 默认 -1 且无下界
    api_keys = [f.key for f in ARGS_API.fields]
    assert "max_tokens" not in api_keys and "max_token_ratio" not in api_keys
    assert "max_tokens" in [f.key for f in ARGS.fields]
    assert "max_token_ratio" in [f.key for f in ARGS.fields]
    ml = [f for f in ARGS_API.fields if f.key == "max_lines"][0]
    assert ml.default == "-1" and ml.min is None
    assert ARGS_API.fields[1].default == "deepseek-v4-flash"  # model 现为第 2 个字段


def test_max_lines_negative_is_unlimited():
    """max_lines 取负值（-1）时行数限制失效：llama 解析为 None（不限制）。"""
    from core.executor import LlamaTranslator
    dummy = object()  # _resolve_max_lines 仅用参数，不用 self 状态
    assert LlamaTranslator._resolve_max_lines(dummy, None, {"max_lines": -1}) is None
    assert LlamaTranslator._resolve_max_lines(dummy, None, {"max_lines": 0}) is None
    assert LlamaTranslator._resolve_max_lines(dummy, None, {"max_lines": 3}) == 3


def test_prompt_roundtrip():
    data = _read_json("configs/translate/prompts/default.json")
    out = _pipe(PROMPT, data)
    assert out["system"] == data["system"]
    assert out["user_with_glossary"] == data["user_with_glossary"]
    assert out["user_without_glossary"] == data["user_without_glossary"]
    # name/placeholders 已按需求从表单移除：构建输出只含三个模板键
    assert set(out) == {"system", "user_with_glossary", "user_without_glossary"}


def test_glossary_roundtrip():
    data = _read_json("configs/translate/glossary/default.json")
    out = _pipe(GLOSSARY, data)
    # name 已按需求从表单移除：构建输出不含 name
    assert "name" not in out
    assert out["format"] == data["format"]  # format 不可自定义：原值保留
    assert len(out["entries"]) == len(data["entries"])
    assert out["entries"][0] == data["entries"][0]


def test_glossary_new_config_default_format():
    """新建术语表（无 format 键）→ 构建输出默认模板 format（对齐 template.json）。"""
    form = GLOSSARY.to_form_values({"entries": [{"src": "a", "dst": "b"}]})
    refs = {f.key: _Ctrl(form[f.key]) for f in GLOSSARY.fields}
    values, parse_errors = GLOSSARY.collect_values(refs)
    assert not parse_errors
    out = GLOSSARY.build_output(values)
    assert out["format"] == {
        "with_info": "{src}->{dst} #{info}",
        "without_info": "{src}->{dst}",
        "separator": "\n",
    }
    assert out["entries"] == [{"src": "a", "dst": "b"}]


def test_hotwords_roundtrip():
    data = _read_json("configs/transcribe/hotwords/default.json")
    out = _pipe(HOTWORDS, data)
    # description 已按需求从表单移除：构建输出只含 hotwords
    assert set(out) == {"hotwords"}
    assert out["hotwords"] == data["hotwords"]


# ── 4. validate 错误定位 ──

def test_validate_required_field():
    form_values = API.to_form_values({"api_key": "k", "model": "m", "timeout": 120})
    form_values["base_url"] = ""
    refs = {f.key: _Ctrl(form_values[f.key]) for f in API.fields}
    values, parse_errors = API.collect_values(refs)
    assert not parse_errors
    field_errors, general = API.validate(values)
    assert "base_url" in field_errors, field_errors
    assert "「API 地址」为必填项" in field_errors["base_url"]


def test_validate_bad_integer_localized():
    refs = {f.key: _Ctrl(f.to_control_value(f.default)) for f in LLAMA.fields}
    refs["--port"] = _Ctrl("abc")
    values, parse_errors = LLAMA.collect_values(refs)
    assert "--port" in parse_errors
    assert "必须是有效整数" in parse_errors["--port"][0]


def test_validate_bad_json_localized():
    refs = {f.key: _Ctrl(f.to_control_value(f.default)) for f in RULES.fields}
    refs["prefix"] = _Ctrl("{not json")
    values, parse_errors = RULES.collect_values(refs)
    assert "prefix" in parse_errors
    assert "JSON 解析错误" in parse_errors["prefix"][0]


def test_validate_glossary_semantic():
    refs = {f.key: _Ctrl(f.to_control_value(f.default)) for f in GLOSSARY.fields}
    refs["entries_json"] = _Ctrl('[{"src": "先輩"}]')  # 缺 dst
    values, parse_errors = GLOSSARY.collect_values(refs)
    assert not parse_errors  # entries_json 是 multiline，JSON 语义校验在 validate
    field_errors, general = GLOSSARY.validate(values)
    assert "entries_json" in field_errors
    assert "缺少 src 或 dst 字段" in field_errors["entries_json"][0]


def test_validate_llama_port_range():
    refs = {f.key: _Ctrl(f.to_control_value(f.default)) for f in LLAMA.fields}
    refs["--port"] = _Ctrl("99999")
    values, parse_errors = LLAMA.collect_values(refs)
    assert not parse_errors
    field_errors, general = LLAMA.validate(values)
    assert "--port" in field_errors
    assert any("1-65535" in m for m in field_errors["--port"]), field_errors


# ── 5. 状态缓存回环 ──

def test_form_values_state_roundtrip():
    """form_values_from_refs 缓存的控件值可直接 populate_form 恢复（切页状态）。"""
    data = _read_json("configs/translate/rules/jp_noval.json")
    refs1 = {k: _Ctrl(v) for k, v in RULES.to_form_values(data).items()}
    cached = RULES.form_values_from_refs(refs1)
    refs2 = {f.key: _Ctrl(None) for f in RULES.fields}
    RULES.populate_form(refs2, cached)
    values, parse_errors = RULES.collect_values(refs2)
    assert not parse_errors
    assert RULES.build_output(values)["recognize"] == data.get("recognize", [])
    assert RULES.build_output(values)["prefix"] == data["prefix"]


def test_moss_fields_grouped_by_category():
    """MOSS 服务配置词条瘦身：仅服务参数（转写参数/提示词已拆到「转写」组）。"""
    from ui.pages.settings.config_schema import MOSS, MOSS_ARGS, MOSS_PROMPT

    groups = {g.name: {f.key for f in g.fields} for g in MOSS.groups}
    assert groups["服务参数"] == {"model_path", "device", "dtype", "lazy_load"}
    assert len(MOSS.fields) == 4

    # 上游无 beam 解码：修复为 greedy/sample（MOSS 转写参数词条承载）
    decoding = next(f for f in MOSS_ARGS.fields if f.key == "decoding")
    assert decoding.options == ["greedy", "sample"]
    # 留空的 sampling 参数不写入构建输出（executor 仅 sample 时消费）
    refs = {f.key: _Ctrl(f.to_control_value(f.default)) for f in MOSS_ARGS.fields}
    values, parse_errors = MOSS_ARGS.collect_values(refs)
    assert not parse_errors
    out = MOSS_ARGS.build_output(values)
    assert "temperature" not in out and "top_p" not in out and "top_k" not in out
    assert {f.key for f in MOSS_PROMPT.fields} == {"prompt"}


def test_moss_args_roundtrip_matches_disk():
    """MOSS 转写参数词条：configs/transcribe/args/*.json 完整往返。"""
    from ui.pages.settings.config_schema import MOSS_ARGS

    data = _read_json("configs/transcribe/args/default.json")
    out = _pipe(MOSS_ARGS, data)
    assert out["max_new_tokens"] == data["max_new_tokens"]
    assert out["max_len"] == data["max_len"]
    assert out["decoding"] == data["decoding"]
    assert out["single_speaker"] == data["single_speaker"]
    # 长音频切块参数（显存自适应 + 静音边界）完整往返
    for key in ("max_audio_sec", "overlap_sec", "vram_safety_ratio",
                "min_window_sec", "silence_min_sec", "boundary_lookback_sec"):
        assert out[key] == data[key], f"{key} 往返丢失"
    assert out["vram_auto_fit"] == data["vram_auto_fit"]
    assert out["silence_boundary"] == data["silence_boundary"]


def test_moss_service_preserves_prompt_and_args_keys():
    """MOSS 服务配置词条保存时透传非服务参数键（prompt/转写参数不丢）。"""
    from ui.pages.settings.config_schema import MOSS

    data = _read_json("configs/models/moss/default.json")
    refs = {k: _Ctrl(v) for k, v in MOSS.to_form_values(data).items()}
    refs["device"] = _Ctrl("cpu")   # 修改服务参数
    values, parse_errors = MOSS.collect_values(refs)
    assert not parse_errors
    out = MOSS.build_output(values)
    assert out["device"] == "cpu"
    assert out["model_path"] == data["model_path"]
    # 非服务参数字段透传保留
    for k in ("max_new_tokens", "decoding", "single_speaker", "prompt"):
        assert out.get(k) == data.get(k), f"{k} 透传丢失"


def test_moss_prompt_roundtrip_matches_disk():
    """MOSS 提示词词条：configs/transcribe/prompts/*.json 完整往返。"""
    from ui.pages.settings.config_schema import MOSS_PROMPT

    data = _read_json("configs/transcribe/prompts/default.json")
    out = _pipe(MOSS_PROMPT, data)
    assert out == data, f"往返不一致: {out} != {data}"
    assert {f.key for f in MOSS_PROMPT.fields} == {"prompt"}


def test_transcribe_default_moss_only():
    """转写默认配置（ini [transcribe]）仅剩 MOSS 相关字段。"""
    from ui.pages.settings.config_schema import TRANSCRIBE_DEFAULT

    keys = [f.key for f in TRANSCRIBE_DEFAULT.fields]
    assert keys == ["moss_server", "moss_args", "hotwords"]
    assert TRANSCRIBE_DEFAULT.label == "MOSS 转写默认配置"
    # 各字段扫描目录与默认值
    by_key = {f.key: f for f in TRANSCRIBE_DEFAULT.fields}
    assert by_key["moss_server"].scan_config_type == "moss"
    assert by_key["moss_server"].scan_glob == "*.json"
    assert by_key["moss_server"].default == "default.json"
    assert by_key["moss_args"].scan_config_type == "moss_args"
    assert by_key["hotwords"].scan_config_type == "hotwords"


# ── 6. 语音合成（角色配置 / 合成参数）──

def test_gsv_service_roundtrip_matches_disk():
    """GSV 服务配置词条：configs/models/gsv/default.json 完整往返（4 键）。"""
    from ui.pages.settings.config_schema import GSV

    data = _read_json("configs/models/gsv/default.json")
    out = _pipe(GSV, data)
    assert out == data, f"往返不一致: {out} != {data}"
    assert {f.key for f in GSV.fields} == {
        "device", "bert_base_path", "cnhuhbert_base_path", "sv_path"}


def test_gsv_role_roundtrip_matches_disk():
    """角色配置词条：configs/tts/roles/role-*.json 完整往返（6 键，含 mode）。"""
    from ui.pages.settings.config_schema import GSV_ROLE

    data = _read_json("configs/tts/roles/role-ookura-lumine.json")
    out = _pipe(GSV_ROLE, data)
    assert out == data, f"往返不一致: {out} != {data}"
    assert {f.key for f in GSV_ROLE.fields} == {
        "mode", "version", "t2s_weights_path", "vits_weights_path",
        "role_ref_audio", "prompt_text"}


def test_gsv_role_build_omits_empty_optional_fields():
    """GSV 角色配置：留空的参考音频/参考文本不写入构建输出（缺键新建安全）。"""
    from ui.pages.settings.config_schema import GSV_ROLE

    refs = {f.key: _Ctrl(f.to_control_value(f.default)) for f in GSV_ROLE.fields}
    refs["role_ref_audio"] = _Ctrl("")
    refs["prompt_text"] = _Ctrl("")
    values, parse_errors = GSV_ROLE.collect_values(refs)
    assert not parse_errors
    out = GSV_ROLE.build_output(values)
    assert "role_ref_audio" not in out and "prompt_text" not in out
    assert out["version"] == "v2ProPlus"
    assert out["t2s_weights_path"] == refs["t2s_weights_path"].value


def test_gsv_args_roundtrip_matches_disk():
    """合成参数词条：configs/tts/args/*.json 完整往返（适配键透传不丢失）。"""
    from ui.pages.settings.config_schema import GSV_ARGS

    data = _read_json("configs/tts/args/default.json")
    out = _pipe(GSV_ARGS, data)
    assert out == data, f"往返不一致: {out} != {data}"
    # 透传键保留（ref_mode/prompt_lang/text_lang 不在表单字段中）
    for k in ("ref_mode", "prompt_lang", "text_lang"):
        assert out[k] == data[k], f"{k} 透传丢失"


def test_gsv_args_build_omits_empty():
    """合成参数：留空字段不写入；未知键（适配键）经 _preserved 原样回写。"""
    from ui.pages.settings.config_schema import GSV_ARGS

    data = {"speed_factor": 1.2, "seed": 42, "text_split_method": "cut2",
            "ref_mode": "single", "prompt_lang": "en", "text_lang": "zh"}
    form = GSV_ARGS.to_form_values(data)
    assert form["top_k"] == ""  # 数据缺键 → 表单留空
    refs = {f.key: _Ctrl(form[f.key]) for f in GSV_ARGS.fields}
    values, parse_errors = GSV_ARGS.collect_values(refs)
    assert not parse_errors
    out = GSV_ARGS.build_output(values)
    assert out["speed_factor"] == 1.2 and out["seed"] == 42
    assert out["text_split_method"] == "cut2"
    assert "top_k" not in out and "top_p" not in out
    assert out["ref_mode"] == "single" and out["prompt_lang"] == "en"
    assert out["text_lang"] == "zh"


def test_tts_default_three_columns():
    """语音合成默认配置（ini [gsv]）：服务配置 / 角色配置 / 合成参数三栏。"""
    from ui.pages.settings.config_schema import TTS_DEFAULT

    by_key = {f.key: f for f in TTS_DEFAULT.fields}
    assert [f.key for f in TTS_DEFAULT.fields] == ["gsv_service", "gsv_server", "gsv_args"]
    assert by_key["gsv_service"].label == "服务配置"
    assert by_key["gsv_service"].scan_config_type == "gsv"
    assert by_key["gsv_service"].scan_glob == "*.json"
    assert by_key["gsv_service"].default == "default.json"
    assert by_key["gsv_server"].label == "角色配置"
    assert by_key["gsv_server"].scan_config_type == "gsv_role"
    assert by_key["gsv_server"].scan_glob == "*.json"
    assert by_key["gsv_server"].default == "role-ookura-lumine.json"
    assert by_key["gsv_args"].scan_config_type == "gsv_args"
    assert by_key["gsv_args"].default == "default.json"


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
