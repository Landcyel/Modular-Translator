"""Declarative schema for config types — defines each config's fields, defaults, and build/parse rules.

Fully rewritten as a declarative engine:

- ``Field`` supports types: text, multiline, integer, number, boolean, select, path,
  list, object, json; adds grouping (``group``), secret (``secret``), and browse-button
  (``browse``) capabilities; ``width`` takes effect.
- Typed value pipeline: the UI layer interacts with form controls through
  ``read_control`` / ``to_control_value`` / ``from_control_value``; this module does
  **not import flet control classes** — all type dispatch goes through ``Field.type``.
- Intermediate values are unified as typed dicts (boolean→bool, integer→int,
  number→float, list→list, object/json→parsed structures), consumed directly by
  ``build_output``.
- ``validate`` returns ``(field_errors: {key: [msg]}, general: [msg])``; errors are
  located precisely per field, no longer matched by fuzzy strings.
- Single registry ``CONFIG_GROUPS`` (nav groups + all types), with
  ``ALL_CONFIG_TYPES`` / ``CONFIG_TYPE_LIST`` derived from it.
- Each config schema aligns with the real on-disk structure in configs/ (LLAMA's
  --keep/-n removed from the form per requirements; API is base_url/api_key/model/
  timeout; RULES includes recognize/skip; MOSS is split into service config (moss) /
  transcription args (moss_args) / prompts (moss_prompt) three entries; ARGS is
  max_token_ratio/max_lines/request nested).
"""

import json as _json
from pathlib import Path

from app.paths import project_root


def _str_to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() in ("true", "1", "yes")


# ── Field definitions ──

class Field:
    """Description of a single config field — a declarative strongly-typed control.

    Extended attributes:
        group:  group name (None means ungrouped; rendered flat)
        secret: show as secret text (e.g. API Key)
        browse: None | "file" | "directory" — whether path fields get a browse button and its type
    """

    __slots__ = (
        "key", "label", "type", "default", "options",
        "required", "description", "placeholder", "min", "max", "width",
        "group", "secret", "browse", "scan_config_type", "scan_glob",
        "options_provider", "visible_when",
    )

    VALID_TYPES = {
        "text", "multiline", "integer", "number",
        "boolean", "select", "path", "list", "object", "json",
    }

    def __init__(
        self, key: str, label: str, type: str = "text",
        default=None, options: list = None,
        required: bool = False, description: str = "",
        placeholder: str = "", min=None, max=None, width: int | None = None,
        group: str | None = None, secret: bool = False,
        browse: str | None = None,
        scan_config_type: str | None = None,
        scan_glob: str | None = None,
        options_provider=None,
        visible_when: dict | None = None,
    ):
        self.key = key
        self.label = label
        if type not in self.VALID_TYPES:
            raise ValueError(f"Field 类型 '{type}' 不在支持列表中: {self.VALID_TYPES}")
        self.type = type
        self.default = default if default is not None else _default_for_type(type)
        self.options = options or []
        self.required = required
        self.description = description
        self.placeholder = placeholder
        self.min = min
        self.max = max
        self.width = width
        self.group = group
        self.secret = secret
        if browse not in (None, "file", "directory"):
            raise ValueError(f"browse 必须是 None/'file'/'directory'，收到 {browse!r}")
        self.browse = browse
        self.scan_config_type = scan_config_type
        self.scan_glob = scan_glob or "*.json"
        # Lazy options provider: returns a list of str or (key, text) tuples, evaluated
        # when rendering the select (avoids importing the core chain at module top level, slowing startup)
        self.options_provider = options_provider
        # Visibility condition: rendered only when all {field key: value} match (e.g. {"mode": "default"})
        self.visible_when = visible_when

    # ── Value pipeline (this module has zero flet dependency; dispatch goes through Field.type) ──

    def default_typed(self):
        """default (declarative string form) → typed intermediate value."""
        t = self.type
        s = str(self.default).strip()
        if t == "boolean":
            return _str_to_bool(self.default)
        if t == "integer":
            try:
                return int(s) if s else 0
            except ValueError:
                return 0
        if t == "number":
            try:
                return float(s) if s else 0.0
            except ValueError:
                return 0.0
        if t == "list":
            try:
                return _json.loads(s) if s else []
            except _json.JSONDecodeError:
                return []
        if t == "object":
            try:
                return _json.loads(s) if s else {}
            except _json.JSONDecodeError:
                return {}
        if t == "json":
            try:
                return _json.loads(s) if s else None
            except _json.JSONDecodeError:
                return None
        return ""  # text / multiline / path / select

    def read_control(self, ref) -> str | bool:
        """Read the raw value from a form control (duck typing; no dependency on flet types)."""
        raw = getattr(ref, "value", None)
        if self.type == "boolean":
            return bool(raw) if raw is not None else False
        return raw if raw is not None else ""

    def to_control_value(self, value) -> str | bool:
        """Any value (default string / parse output / typed value) → control initial value."""
        if self.type == "boolean":
            return _str_to_bool(value)
        if value is None:
            return "" if self.type in ("text", "multiline", "path", "select") else str(self.default)
        if isinstance(value, (list, dict)):
            return _json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    def from_control_value(self, raw) -> object:
        """Control value → typed intermediate value; JSON parse failure raises ValueError (message points to the field)."""
        t = self.type
        if t == "boolean":
            return bool(raw)
        s = str(raw).strip() if raw is not None else ""
        if t in ("text", "multiline", "path", "select"):
            return s
        if t == "integer":
            if s == "":
                return None
            try:
                return int(s)
            except ValueError:
                raise ValueError(f"「{self.label}」必须是有效整数") from None
        if t == "number":
            if s == "":
                return None
            try:
                return float(s)
            except ValueError:
                raise ValueError(f"「{self.label}」必须是有效数字") from None
        if t in ("list", "object", "json"):
            if s == "":
                return [] if t == "list" else {} if t == "object" else None
            try:
                return _json.loads(s)
            except _json.JSONDecodeError as ex:
                raise ValueError(f"「{self.label}」JSON 解析错误: {ex}") from None
        return s

    def is_empty(self, value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return False


def _default_for_type(type: str):
    if type == "boolean":
        return "false"
    if type in ("integer", "number"):
        return "0"
    if type == "list":
        return "[]"
    if type in ("object", "json"):
        return "{}"
    return ""


# ── Group description ──

class FieldGroup:
    """Container for a field group."""

    __slots__ = ("name", "fields")

    def __init__(self, name: str, fields: list):
        self.name = name
        self.fields = fields


# ── Config type definition ──

class ConfigType:
    """Full description of one config type."""

    __slots__ = ("key", "label", "save_dir", "fields", "build_output",
                 "parse_for_form", "validate_fn", "_form_seed",
                 "ini_section", "name_filter")

    def __init__(self, key: str, label: str, save_dir: str,
                 fields: list, build_output, parse_for_form=None,
                 validate_fn=None, ini_section: str | None = None,
                 name_filter: str = "*.json"):
        self.key = key
        self.label = label
        self.save_dir = save_dir
        self.fields = fields
        self.build_output = build_output
        self.parse_for_form = parse_for_form
        self.validate_fn = validate_fn
        self.name_filter = name_filter
        # A non-empty ini_section means the type is managed by configs/system/default.ini
        # (save/load reads and writes the ini directly, not the save_dir json files)
        self.ini_section = ini_section
        # Full output snapshot from parse: includes hidden _preserved_-prefixed keys (written back as-is on load; not shown in the form)
        self._form_seed: dict | None = None

    @property
    def save_path(self) -> Path:
        return project_root / self.save_dir

    @property
    def groups(self) -> list[FieldGroup]:
        """Aggregate by Field.group; ungrouped fields fall into the unnamed group (name="")."""
        order, seen = [], {}
        for f in self.fields:
            g = f.group or ""
            if g not in seen:
                seen[g] = len(order)
                order.append(g)
        return [
            FieldGroup(
                name=g,
                fields=[f for f in self.fields if (f.group or "") == g],
            )
            for g in order
        ]

    # ── Value pipeline ──

    def to_form_values(self, data: dict) -> dict:
        """Raw JSON → dict[key → control value] (for loading files / tests).

        The full parse output (including _preserved_-prefixed hidden keys) is stored in
        _form_seed, passed through by collect_values and written back verbatim by build_output.
        """
        parsed = self.parse_for_form(data) if self.parse_for_form else data
        self._form_seed = dict(parsed)
        return {f.key: f.to_control_value(parsed.get(f.key, f.default)) for f in self.fields}

    def form_values_from_refs(self, refs: dict) -> dict:
        """Current control values → dict[key → control value] (page state cache)."""
        out = {}
        for f in self.fields:
            ref = refs.get(f.key)
            out[f.key] = f.to_control_value(f.read_control(ref)) if ref is not None else f.default
        return out

    def populate_form(self, refs: dict, form_values: dict):
        """dict[key → control value] → write to controls (form_values comes from to_form_values/form_values_from_refs)."""
        for f in self.fields:
            ref = refs.get(f.key)
            if ref is None:
                continue
            raw = form_values.get(f.key, f.default)
            ref.value = f.to_control_value(raw)

    def collect_values(self, refs: dict) -> tuple[dict, dict]:
        """Controls → (typed intermediate dict, field parse errors {key: [msg]})."""
        values, errors = {}, {}
        # Pass through hidden keys saved at parse (_preserved_ prefix): not shown or validated, written back as-is on build
        if self._form_seed:
            for k, v in self._form_seed.items():
                if k.startswith("_preserved_"):
                    values[k] = v
        for f in self.fields:
            ref = refs.get(f.key)
            if ref is not None:
                raw = f.read_control(ref)
            elif self._form_seed is not None and f.key in self._form_seed:
                raw = self._form_seed[f.key]  # hidden field (visible_when): keep the loaded original value
            else:
                raw = f.default
            try:
                values[f.key] = f.from_control_value(raw)
            except ValueError as ex:
                values[f.key] = f.default_typed()
                errors.setdefault(f.key, []).append(str(ex))
        return values, errors

    def validate(self, values: dict) -> tuple[dict, list]:
        """Typed intermediate values → (field errors {key: [msg]}, general errors [msg])."""
        field_errors, general = {}, []
        for f in self.fields:
            v = values.get(f.key)
            key_errs = []
            if f.required and f.is_empty(v):
                key_errs.append(f"「{f.label}」为必填项")
            if f.type == "integer" and v is not None and not isinstance(v, bool):
                if not isinstance(v, int):
                    key_errs.append(f"「{f.label}」必须是有效整数")
                else:
                    if f.min is not None and v < f.min:
                        key_errs.append(f"「{f.label}」不能小于 {f.min}")
                    if f.max is not None and v > f.max:
                        key_errs.append(f"「{f.label}」不能大于 {f.max}")
            if f.type == "number" and v is not None and not isinstance(v, bool):
                if not isinstance(v, (int, float)):
                    key_errs.append(f"「{f.label}」必须是有效数字")
                else:
                    if f.min is not None and v < f.min:
                        key_errs.append(f"「{f.label}」不能小于 {f.min}")
                    if f.max is not None and v > f.max:
                        key_errs.append(f"「{f.label}」不能大于 {f.max}")
            if key_errs:
                field_errors[f.key] = key_errs
        if self.validate_fn:
            fe, ge = self.validate_fn(values)
            for k, msgs in fe.items():
                field_errors.setdefault(k, []).extend(msgs)
            general.extend(ge)
        return field_errors, general


# ── LLaMA server config ──

_LLAMA_FIELDS = [
    Field("llama_path", "LLaMA 路径", type="path", browse="directory",
          default="dependencies/llama-release",
          description="LLaMA 服务器的可执行文件目录"),
    Field("-m", "模型文件 (-m)", type="path", browse="file",
          default="dependencies/models/sakura/sakura-7b-qwen2.5-v1.0-iq4xs.gguf"),
    Field("--host", "监听地址 (--host)", default="127.0.0.1"),
    Field("--port", "端口 (--port)", type="integer", default="8080", min=1, max=65535,
          width=180),
    Field("-ngl", "GPU 层数 (-ngl)", default="auto", description="可填 auto 或具体整数",
          width=180),
    Field("-c", "上下文长度 (-c)", type="integer", default="2048", min=128, width=180),
]

# Aligns with the server_arg keys in configs/models/llama/default.json (no --keep/-n; removed from the form)
_SERVER_ARG_KEYS = {"-m", "--host", "--port", "-ngl", "-c"}


def _llama_validate(values: dict) -> tuple[dict, list]:
    field_errors, general = {}, []
    port = values.get("--port")
    if port is not None and (
        not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535)
    ):
        field_errors.setdefault("--port", []).append("端口必须在 1-65535 之间")
    c_val = values.get("-c")
    if c_val is not None and (not isinstance(c_val, int) or isinstance(c_val, bool)):
        field_errors.setdefault("-c", []).append("上下文长度 (-c) 必须是有效整数")
    return field_errors, general


def _llama_build(values: dict) -> dict:
    server_arg = {}
    for k in _SERVER_ARG_KEYS:
        v = values.get(k)
        if k == "--port":
            server_arg[k] = int(v) if v else 8080
        elif k == "-c":
            server_arg[k] = int(v) if v else 2048
        else:
            server_arg[k] = v or ""
    return {"llama_path": values.get("llama_path", ""), "server_arg": server_arg}


def _llama_parse(data: dict) -> dict:
    server_arg = data.get("server_arg", {})
    result = {"llama_path": data.get("llama_path", "")}
    for k in _SERVER_ARG_KEYS:
        result[k] = str(server_arg.get(k, ""))
    return result


LLAMA = ConfigType(
    key="llama", label="Llama 服务器配置",
    save_dir="configs/models/llama",
    fields=_LLAMA_FIELDS,
    build_output=_llama_build,
    parse_for_form=_llama_parse,
    validate_fn=_llama_validate,
)

# ── API config (aligned with core/executor.py APITranslator: base_url/api_key/model/timeout) ──

_API_FIELDS = [
    Field("base_url", "API 地址", default="https://api.deepseek.com", required=True,
          description="翻译 API 端点 URL（OpenAI 兼容）"),
    Field("api_key", "API Key", secret=True, default="",
          description="API 密钥（建议使用环境变量）"),
    Field("model", "模型名称", default="deepseek-v4-flash", required=True,
          description="使用的模型 ID"),
    Field("timeout", "超时时间 (秒)", type="integer", default="120", min=1, max=3600,
          width=180,
          description="请求超时秒数"),
]

API = ConfigType(
    key="api", label="API 配置",
    save_dir="configs/models/API",
    fields=_API_FIELDS,
    build_output=lambda d: {f.key: d.get(f.key, f.default_typed()) for f in _API_FIELDS},
    parse_for_form=None,
)

# ── Translation args (aligned with configs/translate/args_llama/*.json: max_token_ratio/max_lines/request) ──

_ARGS_REQUEST_KEYS = (
    "model", "temperature", "top_p",
    "presence_penalty", "frequency_penalty", "repeat_penalty", "max_tokens",
)
_ARGS_API_REQUEST_KEYS = (
    "model", "temperature", "top_p",
    "presence_penalty", "frequency_penalty", "repeat_penalty",
)


def _make_args_fields(model_default: str, max_lines_default: str = "3",
                      max_lines_min=1, include_max_tokens: bool = True,
                      include_max_token_ratio: bool = True) -> list:
    """Generate translation-arg fields — Llama/API entries are isomorphic, differing only in defaults and field set.

    API version: no max_tokens (the request-body whitelist strips None) and no max_token_ratio
    (chunking param; the API backend uses the core default 0.4); max_lines defaults to -1 (unlimited).
    """
    fields = []
    if include_max_token_ratio:
        fields.append(Field("max_token_ratio", "max_token_ratio", type="number",
                            default="0.4", min=0, max=1, width=200,
                            description="分块时按 token 占比估算每块行数"))
    fields += [
        Field("max_lines", "max_lines", type="integer", default=max_lines_default,
              min=max_lines_min, width=160,
              description="每 chunk 的最大行数；取负值时不限制（默认）"),
        Field("model", "模型 (request.model)", default=model_default,
              description="请求体里的 model 字段；API 后端实际以服务配置为准"),
        Field("temperature", "temperature", type="number", default="0.1", min=0, max=2),
        Field("top_p", "top_p", type="number", default="0.3", min=0, max=1),
        Field("presence_penalty", "presence_penalty", type="number", default="0.0",
              min=0, max=2),
        Field("frequency_penalty", "frequency_penalty", type="number", default="0.0",
              min=0, max=2),
        Field("repeat_penalty", "repeat_penalty", type="number", default="1.0",
              min=1, max=2,
              description="仅 Llama 后端生效；API 后端按白名单剥离"),
    ]
    if include_max_tokens:
        fields.append(Field("max_tokens", "max_tokens", type="integer",
                            default="2048", min=1))
    return fields


_ARGS_FIELDS = _make_args_fields("sakura")
_ARGS_API_FIELDS = _make_args_fields("deepseek-v4-flash", max_lines_default="-1",
                                     max_lines_min=None, include_max_tokens=False,
                                     include_max_token_ratio=False)


def _make_args_build(request_keys: tuple, include_max_token_ratio: bool = True) -> object:
    def build(values: dict) -> dict:
        result = {"max_lines": values.get("max_lines")}
        if include_max_token_ratio:
            result["max_token_ratio"] = values.get("max_token_ratio")
        result["request"] = {k: values.get(k) for k in request_keys}
        return result
    return build


def _make_args_parse(request_keys: tuple, include_max_token_ratio: bool = True) -> object:
    def parse(data: dict) -> dict:
        req = data.get("request", {})
        result = {"max_lines": str(data.get("max_lines", 3))}
        if include_max_token_ratio:
            result["max_token_ratio"] = str(data.get("max_token_ratio", 0.4))
        for k in request_keys:
            v = req.get(k)
            result[k] = str(v) if v is not None else ""
        return result
    return parse


ARGS = ConfigType(
    key="args", label="翻译参数 (Llama)",
    save_dir="configs/translate/args_llama",
    fields=_ARGS_FIELDS,
    build_output=_make_args_build(_ARGS_REQUEST_KEYS),
    parse_for_form=_make_args_parse(_ARGS_REQUEST_KEYS),
)

ARGS_API = ConfigType(
    key="args_api", label="翻译参数 (API)",
    save_dir="configs/translate/args_api",
    fields=_ARGS_API_FIELDS,
    build_output=_make_args_build(_ARGS_API_REQUEST_KEYS, include_max_token_ratio=False),
    parse_for_form=_make_args_parse(_ARGS_API_REQUEST_KEYS, include_max_token_ratio=False),
)

# ── Prompt config ──

_PROMPT_FIELDS = [
    Field("system", "System Prompt", type="multiline",
          default="你是一个日文到简体中文的轻小说翻译模型。要求译文流畅自然，保持原文语气与风格，结合上下文正确处理称谓和代词，不擅自补出原文没有的信息。\n\n原文中所有以 <<< 开头、以 >>> 结尾的字符串（如 <<<PH>>>）都是不可变占位符，代表必须原样保留的内容（人名、术语、代码、控制符等）。翻译时：1) 原样保留这些标记本身，不得翻译、改写、删除或拆分；2) 保持它们在译文中的位置和数量与原文一致；3) 不要把其它内容改成标记形式。"),
    Field("user_with_glossary", "User (有术语表)", type="multiline",
          default="根据以下术语表翻译下面的日文文本，只输出译文，不要复述术语表、原文、标题或解释。原文中所有以 <<< 开头、以 >>> 结尾的字符串（如 <<<PH>>>）都是不可变占位符，请原样保留。\n\n术语表：\n{GLOSSARY_TEXT}\n\n待翻译文本：\n{ORIGINAL_TEXT}"),
    Field("user_without_glossary", "User (无术语表)", type="multiline",
          default="原文中所有以 <<< 开头、以 >>> 结尾的字符串（如 <<<PH>>>）都是不可变占位符，请原样保留。将下面的日文文本翻译成简体中文：\n{ORIGINAL_TEXT}"),
]


def _prompt_build(values: dict) -> dict:
    return {f.key: values.get(f.key, "") for f in _PROMPT_FIELDS}


def _prompt_parse(data: dict) -> dict:
    return {f.key: data.get(f.key, f.default) for f in _PROMPT_FIELDS}


PROMPT = ConfigType(
    key="prompt", label="提示词",
    save_dir="configs/translate/prompts",
    fields=_PROMPT_FIELDS,
    build_output=_prompt_build,
    parse_for_form=_prompt_parse,
)

# ── Hotwords config ──

_HOTWORDS_FIELDS = [
    Field("hotwords", "热词列表 (JSON 数组)", type="list",
          default='[\n  \n]',
          description='每项为一个热词字符串，如 ["東京", "大阪"]'),
]


def _hotwords_build(values: dict) -> dict:
    hw = values.get("hotwords")
    return {"hotwords": hw if isinstance(hw, list) else []}


def _hotwords_parse(data: dict) -> dict:
    hw = data.get("hotwords", [])
    return {
        "hotwords": _json.dumps(hw, ensure_ascii=False, indent=2) if hw else "[\n  \n]",
    }


HOTWORDS = ConfigType(
    key="hotwords", label="Hotwords",
    save_dir="configs/transcribe/hotwords",
    fields=_HOTWORDS_FIELDS,
    build_output=_hotwords_build,
    parse_for_form=_hotwords_parse,
)

# ── Glossary ──

# Default format (written back for new configs / missing files; aligned with configs/translate/glossary/template.json)
_DEFAULT_GLOSSARY_FORMAT = {
    "with_info": "{src}->{dst} #{info}",
    "without_info": "{src}->{dst}",
    "separator": "\n",
}

_GLOSSARY_FIELDS = [
    Field("entries_json", "词条 (JSON)", type="multiline",
          default='[{"src":"先輩","dst":"学姐"},\n {"src":"魔導書","dst":"魔导书"}]',
          description="JSON 数组，每项含 src 和 dst 字段"),
]


def _glossary_validate(values: dict) -> tuple[dict, list]:
    field_errors, general = {}, []
    entries_str = values.get("entries_json") or ""
    if str(entries_str).strip():
        try:
            entries = _json.loads(entries_str)
            if not isinstance(entries, list):
                field_errors.setdefault("entries_json", []).append("词条必须是 JSON 数组")
            else:
                for i, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        field_errors.setdefault("entries_json", []).append(
                            f"词条[{i}] 必须是 JSON 对象")
                    elif "src" not in entry or "dst" not in entry:
                        field_errors.setdefault("entries_json", []).append(
                            f"词条[{i}] 缺少 src 或 dst 字段")
        except _json.JSONDecodeError as ex:
            field_errors.setdefault("entries_json", []).append(f"词条 JSON 解析错误: {ex}")
    return field_errors, general


def _glossary_build(values: dict) -> dict:
    # format is not customizable: keep the original value loaded (_preserved_format); use the default template when missing
    fmt = values.get("_preserved_format") or dict(_DEFAULT_GLOSSARY_FORMAT)
    return {
        "format": fmt,
        "entries": _json.loads(values.get("entries_json", "[]") or "[]"),
    }


def _glossary_parse(data: dict) -> dict:
    return {
        # Hidden key: the format dict is kept verbatim (written back on file write; not shown in the form)
        "_preserved_format": data.get("format"),
        "entries_json": _json.dumps(data.get("entries", []), ensure_ascii=False, indent=2),
    }


GLOSSARY = ConfigType(
    key="glossary", label="术语表",
    save_dir="configs/translate/glossary",
    fields=_GLOSSARY_FIELDS,
    build_output=_glossary_build,
    parse_for_form=_glossary_parse,
    validate_fn=_glossary_validate,
)

# ── Rules file (aligned with core/rule_splitter.py: prefix/suffix/placeholder/skip/recognize) ──

_RULES_FIELDS = [
    Field("prefix", "前缀 (JSON 数组)", type="list",
          default='[\n  "「",\n  "『",\n  "PRINTFORMW"\n]',
          description="用于识别段落/行的起始模式（数组，元素为字面量或正则）"),
    Field("suffix", "后缀 (JSON 数组)", type="list",
          default='[\n  "」",\n  "』"\n]',
          description="用于识别段落/行的结束模式（数组）"),
    Field("placeholder", "占位符 (JSON 二维数组)", type="list",
          default='[\n  ["%CALLNAME:MASTER%", "<<<MASTER>>>"],\n  ["%CALLNAME:TARGET%", "<<<TARGET>>>"]\n]',
          description="每项为 [原文匹配, 替换文本]，用于保护不可变标记"),
    Field("recognize", "识别条目 (JSON 数组)", type="list", default="[]",
          description="可选：额外识别条目列表"),
    Field("skip", "跳过条目 (JSON 数组)", type="list", default="[]",
          description="可选：跳过不翻译的条目列表"),
]

_RULES_LIST_KEYS = ("prefix", "suffix", "placeholder", "recognize", "skip")


def _rules_build(values: dict) -> dict:
    result = {}
    for k in _RULES_LIST_KEYS:
        v = values.get(k)
        result[k] = v if isinstance(v, list) else []
    return result


def _rules_parse(data: dict) -> dict:
    def _dump_list(v):
        return _json.dumps(v, ensure_ascii=False, indent=2) if v else "[]"

    result = {}
    for k in _RULES_LIST_KEYS:
        result[k] = _dump_list(data.get(k, []))
    return result


RULES = ConfigType(
    key="rules", label="规则",
    save_dir="configs/translate/rules",
    fields=_RULES_FIELDS,
    build_output=_rules_build,
    parse_for_form=_rules_parse,
)

# ── Output config ──

_OUTPUT_FIELDS = [
    Field("output_dir", "输出目录", type="path", browse="directory", default="output",
          description="翻译/转写结果默认输出目录"),
    Field("auto_export", "自动导出", type="boolean", default="true",
          description="任务完成后自动导出结果文件"),
]

OUTPUT = ConfigType(
    key="output", label="输出默认配置",
    save_dir="configs/output",
    fields=_OUTPUT_FIELDS,
    build_output=lambda d: {f.key: d.get(f.key, f.default_typed()) for f in _OUTPUT_FIELDS},
    parse_for_form=None,
    ini_section="output",
)

# ── Translate / transcribe default configs — manage each page's default selections at app startup ──
#    (field values are .json file names in the corresponding config dirs; options are scanned dynamically when rendering)

_TRANSLATE_DEFAULT_FIELDS = [
    Field("llama_server", "llama 服务配置", type="select",
          scan_config_type="llama", default="default.json"),
    Field("api_server", "api 服务配置", type="select",
          scan_config_type="api", default="default.json"),
    Field("prompt", "提示词", type="select",
          scan_config_type="prompts", default="default.json"),
    Field("translate_args", "翻译参数 (llama)", type="select",
          scan_config_type="translate_args", default="default.json"),
    Field("translate_args_api", "翻译参数 (api)", type="select",
          scan_config_type="translate_args_api", default="default.json"),
    Field("rule", "规则", type="select", options=["无"],
          scan_config_type="rules", default="jp_noval.json"),
    Field("glossary", "术语表", type="select", options=["无"],
          scan_config_type="glossary", default="default.json"),
]

TRANSLATE_DEFAULT = ConfigType(
    key="translate_default", label="翻译默认配置",
    save_dir="configs/defaults/translate",
    fields=_TRANSLATE_DEFAULT_FIELDS,
    build_output=lambda d: {f.key: d.get(f.key, f.default_typed()) for f in _TRANSLATE_DEFAULT_FIELDS},
    parse_for_form=None,
    ini_section="translate",
)

_TRANSCRIBE_DEFAULT_FIELDS = [
    Field("moss_server", "MOSS 服务配置", type="select",
          scan_config_type="moss", scan_glob="*.json", default="default.json"),
    Field("moss_args", "MOSS 转写参数", type="select", options=["无"],
          scan_config_type="moss_args", default="default.json"),
    Field("hotwords", "Hotwords", type="select", options=["无"],
          scan_config_type="hotwords", default="default.json"),
]

# ── GSV service config / MOSS service config / TTS default config ──
#    (configs/models has one subdirectory per service; service configs are uniformly default.json)

# ── GSV service config (configs/models/gsv/*.json — engine-level: device + model dirs) ──
#    (role-related: S1/S2 weights, reference audio/text — see the "GSV Role Config" entry)

_GSV_SERVICE_FIELDS = [
    Field("device", "推理设备", type="select", options=["auto", "cuda", "cpu"], default="auto"),
    Field("bert_base_path", "BERT 目录", type="path", browse="directory",
          default="dependencies/models/v4/chinese-roberta-wwm-ext-large"),
    Field("cnhuhbert_base_path", "CNHuBERT 目录", type="path", browse="directory",
          default="dependencies/models/v4/chinese-hubert-base"),
    Field("sv_path", "SV 权重", type="path", browse="file",
          default="dependencies/models/gsv/sv/pretrained_eres2netv2w24s4ep4.ckpt"),
]


def _gsv_service_parse(data: dict) -> dict:
    """gsv/default.json → form values (flattened at top level; unknown keys passed through as _preserved_gsv)."""
    out = {}
    for f in _GSV_SERVICE_FIELDS:
        v = data.get(f.key)
        out[f.key] = "" if v is None else str(v)
    out["_preserved_gsv"] = _json.dumps(
        {k: v for k, v in data.items() if k not in {f.key for f in _GSV_SERVICE_FIELDS}},
        ensure_ascii=False)
    return out


def _gsv_service_build(values: dict) -> dict:
    """Form values → gsv/default.json structure (empty fields omitted; unknown keys written back)."""
    out = {}
    for f in _GSV_SERVICE_FIELDS:
        v = values.get(f.key)
        if v not in (None, ""):
            out[f.key] = v
    try:
        extra = _json.loads(values.get("_preserved_gsv") or "{}")
    except Exception:
        extra = {}
    if isinstance(extra, dict):
        out.update(extra)
    return out


GSV = ConfigType(
    key="gsv", label="GSV 服务配置",
    save_dir="configs/models/gsv", name_filter="*.json",
    fields=_GSV_SERVICE_FIELDS,
    build_output=_gsv_service_build,
    parse_for_form=_gsv_service_parse,
)

# ── GSV role config (configs/tts/roles/role-*.json — role model assets: S1/S2 weights +
#    reference audio/text; merged with the "GSV service config" before loading the engine) ──

_GSV_ROLE_FIELDS = [
    Field("mode", "模式", type="select", options=["default", "aux", "dual"], default="default",
          description="default=单参考(参考音频+参考文本) · aux/dual=情绪+角色参考"),
    Field("version", "模型版本", type="select", options=["v2ProPlus"], default="v2ProPlus",
          description="当前仅 v2ProPlus 可用；v4 需 dependencies/models/v4/gsv-v4-pretrained/ 权重"),
    Field("t2s_weights_path", "S1(GPT) 权重", type="path", browse="file", required=True,
          default="dependencies/models/v4/s1v3.ckpt",
          description="角色微调 ckpt（characters/<角色>/…-eXX.ckpt）"),
    Field("vits_weights_path", "S2(SoVITS) 权重", type="path", browse="file", required=True,
          default="dependencies/models/gsv/v2proplus/s2Gv2ProPlus.pth",
          description="角色微调 pth（characters/<角色>/…_eX_sXXX.pth）"),
    Field("role_ref_audio", "参考音频", type="path", browse="file",
          description="角色固有干声参考（3~10s）；可选，任务级可覆盖"),
    Field("prompt_text", "参考文本", type="multiline",
          visible_when={"mode": "default"},
          description="与参考音频逐字一致（仅 default 模式显示；aux/dual 用情绪参考文本）"),
]


def _gsv_role_parse(data: dict) -> dict:
    """role-*.json → form values (flattened at top level; unknown keys passed through as _preserved_role)."""
    out = {}
    for f in _GSV_ROLE_FIELDS:
        v = data.get(f.key)
        out[f.key] = "" if v is None else str(v)
    if not out.get("mode"):
        out["mode"] = "default"  # old role JSON lacks the mode key → treat as default
    out["_preserved_role"] = _json.dumps(
        {k: v for k, v in data.items() if k not in {f.key for f in _GSV_ROLE_FIELDS}},
        ensure_ascii=False)
    return out


def _gsv_role_build(values: dict) -> dict:
    """Form values → role-*.json structure (empty fields omitted; unknown keys written back)."""
    out = {}
    for f in _GSV_ROLE_FIELDS:
        v = values.get(f.key)
        if v not in (None, ""):
            out[f.key] = v
    try:
        extra = _json.loads(values.get("_preserved_role") or "{}")
    except Exception:
        extra = {}
    if isinstance(extra, dict):
        out.update(extra)
    return out


GSV_ROLE = ConfigType(
    key="gsv_role", label="角色配置",
    save_dir="configs/tts/roles", name_filter="*.json",
    fields=_GSV_ROLE_FIELDS,
    build_output=_gsv_role_build,
    parse_for_form=_gsv_role_parse,
)

_MOSS_PROMPT_DEFAULT = ("请将音频转写为文本，每一段需以起始时间戳和说话人编号"
                        "（[S01]、[S02]…）开头，正文为对应的语音内容，并在段末标注"
                        "结束时间戳，以清晰标明该段语音范围。")

# The MOSS service-config entry only carries service params; the transcription-args/prompt
# keys in moss.json are passed back through _preserved_moss (edited via the "MOSS Transcription
# Args" and "MOSS Prompt" entries)
_MOSS_FIELDS = [
    Field("model_path", "模型目录", type="path", browse="directory",
          default="dependencies/models/moss", required=True,
          group="服务参数"),
    Field("device", "推理设备", type="select", options=["auto", "cuda", "cpu"],
          default="auto", group="服务参数"),
    Field("dtype", "精度", type="select", options=["bf16", "fp16", "fp32"],
          default="bf16", group="服务参数"),
    Field("lazy_load", "启动即加载模型（关闭懒加载）", type="boolean",
          default="true", group="服务参数",
          description="true=首个转写任务时才加载模型（默认）；false=服务启动时即加载，"
                      "加载耗时前移到 start()，设备/显存占用提前确定"),
]


def _moss_parse(data: dict) -> dict:
    """moss.json → form values; non-service-param fields passed through (_preserved_moss)."""
    out = {}
    for f in _MOSS_FIELDS:
        v = data.get(f.key)
        if f.type == "boolean":
            out[f.key] = "true" if v else "false"
        elif v is None:
            out[f.key] = ""
        else:
            out[f.key] = str(v)
    out["_preserved_moss"] = _json.dumps(
        {k: v for k, v in data.items() if k not in {f.key for f in _MOSS_FIELDS}},
        ensure_ascii=False)
    return out


def _moss_build(values: dict) -> dict:
    """Form values → moss.json structure (service params + unknown keys written back; transcription args/prompt are not lost)."""
    out = {}
    for k in ("model_path", "device", "dtype"):
        v = values.get(k)
        if v not in (None, ""):
            out[k] = v
    if values.get("lazy_load") not in (None, ""):
        out["lazy_load"] = values.get("lazy_load") == "true"
    try:
        extra = _json.loads(values.get("_preserved_moss") or "{}")
    except Exception:
        extra = {}
    if isinstance(extra, dict):
        out.update(extra)
    return out


MOSS = ConfigType(
    key="moss", label="MOSS 服务配置",
    save_dir="configs/models/moss", name_filter="*.json",
    fields=_MOSS_FIELDS,
    build_output=_moss_build,
    parse_for_form=_moss_parse,
)

# ── MOSS transcription args (task-level configs/transcribe/args/*.json; service defaults in moss/default.json) ──

_MOSS_ARGS_FIELDS = [
    Field("max_new_tokens", "最大生成 token", type="integer", default="65536", min=1),
    Field("max_len", "最大长度", type="integer", default="131072", min=1),
    Field("decoding", "解码方式", type="select", options=["greedy", "sample"],
          default="greedy",
          description="greedy=贪心；sample=采样（temperature/top_p/top_k 仅 sample 生效）"),
    Field("temperature", "temperature", type="number", default="",
          description="采样温度（留空不设置；仅 sample 生效）"),
    Field("top_p", "top_p", type="number", default="",
          description="核采样（留空不设置；仅 sample 生效）"),
    Field("top_k", "top_k", type="integer", default="",
          description="top-k 采样（留空不设置；仅 sample 生效）"),
    Field("single_speaker", "单说话人归一", type="boolean", default="true",
          description="prompt 抑制 + 结果侧 force_single_speaker 双保险"),
    Field("max_audio_sec", "长音频分段上限（秒）", type="number", default="180",
          description="单窗时长的硬上限：超过该时长的音频按滑动窗口分段转写"
                      "（Qwen3 全注意力显存随音频长度平方增长）；0=关闭分段"),
    Field("overlap_sec", "分段重叠（秒）", type="number", default="10",
          description="相邻窗口重叠时长，用于跨边界句子续接（自动 ≤ 上限的一半）"),
    Field("vram_auto_fit", "按显存自适应分段", type="boolean", default="true",
          description="按空闲显存预算自动收敛单窗时长（不超过「分段上限」）；"
                      "关闭则固定用上限值"),
    Field("vram_safety_ratio", "显存安全系数", type="number", default="0.7",
          description="空闲显存中留给单个窗口峰值使用的份额（0.1-0.95）"),
    Field("min_window_sec", "最小单窗（秒）", type="number", default="60",
          description="显存自适应时的窗口下限（不建议低于 45）"),
    Field("silence_boundary", "静音切分边界", type="boolean", default="true",
          description="在候选切点前的回看范围内寻找静音点切分，避免截断正常说话"),
    Field("silence_min_sec", "静音判定时长（秒）", type="number", default="0.35",
          description="连续静音达到该时长才可作为切点"),
    Field("boundary_lookback_sec", "边界回看范围（秒）", type="number", default="30",
          description="在目标切点前该范围内寻找最佳静音点（窗口只缩不长，不超显存预算）"),
]


def _moss_args_parse(data: dict) -> dict:
    out = {}
    for f in _MOSS_ARGS_FIELDS:
        v = data.get(f.key)
        if f.type == "boolean":
            out[f.key] = "true" if v else "false"
        elif v is None:
            out[f.key] = ""
        else:
            out[f.key] = str(v)
    return out


def _moss_args_build(values: dict) -> dict:
    out = {}
    decoding = values.get("decoding")
    if decoding not in (None, ""):
        out["decoding"] = decoding
    out["max_new_tokens"] = int(values.get("max_new_tokens") or 65536)
    out["max_len"] = int(values.get("max_len") or 131072)
    # sampling params: omitted when empty (executor only consumes them when decoding="sample")
    for k, cast in (("temperature", float), ("top_p", float), ("top_k", int)):
        v = values.get(k)
        if v not in (None, ""):
            out[k] = cast(v)
    out["single_speaker"] = bool(values.get("single_speaker", True))
    for k in ("max_audio_sec", "overlap_sec", "vram_safety_ratio",
              "min_window_sec", "silence_min_sec", "boundary_lookback_sec"):
        v = values.get(k)
        if v not in (None, ""):
            out[k] = float(v)
    out["vram_auto_fit"] = bool(values.get("vram_auto_fit", True))
    out["silence_boundary"] = bool(values.get("silence_boundary", True))
    return out


MOSS_ARGS = ConfigType(
    key="moss_args", label="MOSS 转写参数",
    save_dir="configs/transcribe/args",
    fields=_MOSS_ARGS_FIELDS,
    build_output=_moss_args_build,
    parse_for_form=_moss_args_parse,
)

# ── MOSS prompt (configs/transcribe/prompts/*.json — task-level prompt config;
#    service-level default in the models/moss/default.json prompt key) ──

_MOSS_PROMPT_FIELDS = [
    Field("prompt", "转写提示词", type="multiline", default=_MOSS_PROMPT_DEFAULT,
          description="热词通过「热词提示：…」附加到提示词末尾（转写页选择热词文件）"),
]


def _moss_prompt_parse(data: dict) -> dict:
    """moss.json → form values; non-prompt keys passed through (_preserved_moss)."""
    out = {"prompt": str(data.get("prompt") or "")}
    out["_preserved_moss"] = _json.dumps(
        {k: v for k, v in data.items() if k != "prompt"},
        ensure_ascii=False)
    return out


def _moss_prompt_build(values: dict) -> dict:
    """Form values → moss.json structure (prompt + unknown keys written back; service params not lost)."""
    out = {}
    if values.get("prompt"):
        out["prompt"] = values["prompt"]
    try:
        extra = _json.loads(values.get("_preserved_moss") or "{}")
    except Exception:
        extra = {}
    if isinstance(extra, dict):
        out.update(extra)
    return out


MOSS_PROMPT = ConfigType(
    key="moss_prompt", label="MOSS 提示词",
    save_dir="configs/transcribe/prompts", name_filter="*.json",
    fields=_MOSS_PROMPT_FIELDS,
    build_output=_moss_prompt_build,
    parse_for_form=_moss_prompt_parse,
)

_TTS_DEFAULT_FIELDS = [
    Field("gsv_service", "服务配置", type="select",
          scan_config_type="gsv", scan_glob="*.json", default="default.json",
          description="语音合成页启动时默认选中的 GSV 服务配置（models/gsv/*.json）"),
    Field("gsv_server", "角色配置", type="select",
          scan_config_type="gsv_role", scan_glob="*.json", default="role-ookura-lumine.json",
          description="语音合成页启动时默认选中的 GSV 角色配置（角色模型）"),
    Field("gsv_args", "合成参数", type="select", options=["无"],
          scan_config_type="gsv_args", default="default.json",
          description="语音合成页启动时默认选中的「合成参数」模板（configs/tts/args/*.json）"),
]

TTS_DEFAULT = ConfigType(
    key="tts_default", label="语音合成默认配置",
    save_dir="configs/defaults/tts",
    fields=_TTS_DEFAULT_FIELDS,
    build_output=lambda d: {f.key: d.get(f.key, f.default_typed()) for f in _TTS_DEFAULT_FIELDS},
    parse_for_form=None,
    ini_section="gsv",
)

# ── TTS args (configs/tts/args/*.json — "TTS Args" template on the TTS page; adapter keys
#    like ref_mode/prompt_lang/text_lang pass through _preserved_args and are written back
#    verbatim on save, not lost by editing this entry) ──

_GSV_ARGS_FIELDS = [
    Field("speed_factor", "语速", type="number", default="1.0", min=0.1, max=3),
    Field("text_split_method", "切分方式", type="select",
          options=["cut1", "cut0", "cut2", "cut3", "cut4", "cut5"], default="cut1"),
    Field("seed", "seed（-1 随机）", type="integer", default="-1", min=-1),
    Field("top_k", "top_k", type="integer", default="15", min=1),
    Field("top_p", "top_p", type="number", default="1.0", min=0, max=1),
    Field("temperature", "temperature", type="number", default="1.0", min=0),
    Field("repetition_penalty", "重复惩罚", type="number", default="1.35", min=0),
]


def _gsv_args_parse(data: dict) -> dict:
    """configs/tts/args/*.json → form values; non-TTS-arg fields passed through (_preserved_args)."""
    out = {}
    field_keys = {f.key for f in _GSV_ARGS_FIELDS}
    for f in _GSV_ARGS_FIELDS:
        v = data.get(f.key)
        if f.type == "boolean":
            out[f.key] = "true" if v else "false"
        elif v is None:
            out[f.key] = ""
        else:
            out[f.key] = str(v)
    out["_preserved_args"] = _json.dumps(
        {k: v for k, v in data.items() if k not in field_keys},
        ensure_ascii=False)
    return out


def _gsv_args_build(values: dict) -> dict:
    """Form values → configs/tts/args/*.json structure (empty fields omitted; unknown keys written back)."""
    out = {}
    for f in _GSV_ARGS_FIELDS:
        v = values.get(f.key)
        if v in (None, ""):
            continue
        if f.type == "integer":
            out[f.key] = int(v)
        elif f.type == "number":
            out[f.key] = float(v)
        elif f.type == "boolean":
            out[f.key] = bool(v)
        else:
            out[f.key] = v
    try:
        extra = _json.loads(values.get("_preserved_args") or "{}")
    except Exception:
        extra = {}
    if isinstance(extra, dict):
        out.update(extra)
    return out


GSV_ARGS = ConfigType(
    key="gsv_args", label="合成参数",
    save_dir="configs/tts/args",
    fields=_GSV_ARGS_FIELDS,
    build_output=_gsv_args_build,
    parse_for_form=_gsv_args_parse,
)

TRANSCRIBE_DEFAULT = ConfigType(
    key="transcribe_default", label="MOSS 转写默认配置",
    save_dir="configs/defaults/transcribe",
    fields=_TRANSCRIBE_DEFAULT_FIELDS,
    build_output=lambda d: {f.key: d.get(f.key, f.default_typed()) for f in _TRANSCRIBE_DEFAULT_FIELDS},
    parse_for_form=None,
    ini_section="transcribe",
)

# ── Single registry — nav groups + all types (order = left-side nav display order) ──

CONFIG_GROUPS: list[tuple[str, list[ConfigType]]] = [
    ("系统", [TRANSLATE_DEFAULT, TRANSCRIBE_DEFAULT, TTS_DEFAULT, OUTPUT]),
    ("服务", [LLAMA, API, MOSS, GSV]),
    ("翻译", [PROMPT, ARGS, ARGS_API, RULES, GLOSSARY]),
    ("转写", [MOSS_ARGS, MOSS_PROMPT, HOTWORDS]),
    ("语音", [GSV_ROLE, GSV_ARGS]),
]

ALL_CONFIG_TYPES: dict[str, ConfigType] = {
    ct.key: ct for _, items in CONFIG_GROUPS for ct in items
}

CONFIG_TYPE_LIST: list[ConfigType] = [
    ct for _, items in CONFIG_GROUPS for ct in items
]
