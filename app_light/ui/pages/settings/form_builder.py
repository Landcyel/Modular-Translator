"""设置页表单渲染引擎 — 基于声明式 schema 渲染分组强类型表单。

- 按 ``ConfigType.groups`` 渲染分组标题。
- 每类字段一个渲染分支：
    boolean → Switch；select → Dropdown；secret → 可揭示密码框；
    browse=file/directory → TextField + 浏览按钮（``on_browse(field, ref)`` 注入）；
    multiline / list / object / json → 多行框；integer / number → 数字键盘；其余单行。
- 错误就地显示：错误文本直接放在字段行内（不重建整棵树、不丢焦点）。
- ``field.width`` 显式指定时生效（否则 expand 撑满输入列）。
"""

import flet as ft

from ui.theme import Layout, Palette, Radius
from ui.pages.settings.config_schema import ConfigType, Field, FieldGroup
from ui.widgets.config_picker import _scan_config_dir, _option_for

# ── 响应式列宽 ──
LABEL_COL = {"sm": 12, "md": 3}
INPUT_COL = {"sm": 12, "md": 9}


def _str_to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")


def _error_text(msg: str) -> ft.Text:
    return ft.Text(msg, size=11, color=Palette.ERROR, italic=True)


def _field_label(field: Field) -> ft.Text:
    return ft.Text(field.label, size=12, weight=ft.FontWeight.W_500,
                   color=Palette.SUBTEXT)


def _textfield(field: Field, multiline: bool = False, value=None) -> ft.TextField:
    return ft.TextField(
        value=field.default if value is None else value,
        dense=True,
        text_style=ft.TextStyle(size=13, color=Palette.TEXT),
        border_color=Palette.BORDER,
        border_radius=8,
        bgcolor=Palette.BG,
        hint_text=field.placeholder or "",
        hint_style=ft.TextStyle(size=12, color=Palette.SUBTEXT),
        tooltip=field.description or field.label,
        multiline=multiline,
        min_lines=3 if multiline else 1,
        max_lines=8 if multiline else 1,
        password=field.secret,
        can_reveal_password=field.secret,
        width=field.width,
        expand=field.width is None,
    )


def _field_row(field: Field, input_widgets: list, center: bool = True) -> ft.ResponsiveRow:
    return ft.ResponsiveRow([
        ft.Column([_field_label(field)], col=LABEL_COL),
        ft.Column(input_widgets, col=INPUT_COL),
    ], spacing=Layout.COLUMN_SPACING,
       vertical_alignment=ft.CrossAxisAlignment.CENTER if center
       else ft.CrossAxisAlignment.START)


def _group_header(group: FieldGroup, is_first: bool) -> ft.Column:
    return ft.Column([
        ft.Container(height=Layout.COLUMN_SPACING),
        ft.Row([
            ft.Text(group.name, size=13, weight=ft.FontWeight.BOLD,
                    color=Palette.TEXT_SECOND),
            ft.Container(expand=True),
        ], spacing=8),
        ft.Divider(height=1, thickness=1, color=Palette.BORDER_SUBTLE),
    ], spacing=8)


def _visible(field: Field, values: dict | None) -> bool:
    """按 ``field.visible_when``（{key: 值} 全匹配）判断字段是否渲染。"""
    if not field.visible_when:
        return True
    for k, v in field.visible_when.items():
        cur = values.get(k) if values else field.default
        if str(cur) != str(v):
            return False
    return True


def _build_field(field: Field, refs: dict, errors: list,
                 values: dict | None, on_browse, on_change=None) -> ft.Control:
    err_widgets = [_error_text(m) for m in errors]
    init = values.get(field.key) if values and field.key in values else field.default

    # ── boolean: Switch ──
    if field.type == "boolean":
        sw = ft.Switch(
            value=_str_to_bool(init),
            active_color=Palette.PRIMARY,
            tooltip=field.description or field.label,
        )
        refs[field.key] = sw
        return _field_row(field, [ft.Row([sw] + err_widgets, spacing=8)], center=True)

    # ── select: Dropdown ──
    if field.type == "select":
        common = dict(
            dense=True,
            text_style=ft.TextStyle(size=13, color=Palette.TEXT),
            border_color=Palette.BORDER,
            border_radius=8,
            bgcolor=Palette.BG,
            tooltip=field.description or field.label,
            width=field.width,
            expand=field.width is None,
        )
        # 选项来源：field.options（str 或 (key,text) 元组）+ 惰性 options_provider
        #           + scan_config_type 目录扫描（"默认加载项"选择）
        def _mk_option(o):
            """str → Option(key=text)；两元组 → 显示名与值分离（如「自动检测（留空）」）。"""
            if isinstance(o, (tuple, list)) and len(o) == 2:
                return ft.dropdown.Option(key=str(o[0]), text=str(o[1]))
            return ft.dropdown.Option(str(o))

        opts = [_mk_option(o) for o in field.options]
        if field.options_provider:
            opts += [_mk_option(o) for o in field.options_provider()]
        if field.scan_config_type:
            opts += [_option_for(p.name) for p in _scan_config_dir(
                field.scan_config_type, getattr(field, "scan_glob", "*.json"))]
        init_val = str(init) if init is not None else ""
        # 初始值不在选项中时回退首项（如旧配置语言代码不在列表 → 回退「自动检测」）
        if init_val and init_val not in [o.key for o in opts]:
            init_val = opts[0].key if opts else None
        dd = ft.Dropdown(value=init_val, options=opts, data=field.key,
                         on_select=on_change, **common)
        refs[field.key] = dd
        return _field_row(field, [dd] + err_widgets, center=True)

    # ── multiline / list / object / json: 多行 TextField ──
    if field.type in ("multiline", "list", "object", "json"):
        ctrl = _textfield(field, multiline=True, value=init)
        refs[field.key] = ctrl
        return _field_row(field, [ctrl] + err_widgets, center=False)

    # ── path + browse: TextField + 浏览按钮 ──
    if field.type == "path" and field.browse:
        ctrl = _textfield(field, value=init)
        refs[field.key] = ctrl
        browse_btn = ft.TextButton(
            "浏览…",
            style=ft.ButtonStyle(color=Palette.PRIMARY),
            on_click=(lambda e: on_browse(field, ctrl)) if on_browse else None,
        )
        return _field_row(field, [ft.Row([ctrl, browse_btn], spacing=4)] + err_widgets,
                          center=False)

    # ── text / secret / integer / number: 单行 TextField ──
    ctrl = _textfield(field, value=init)
    if field.type in ("integer", "number"):
        ctrl.keyboard_type = ft.KeyboardType.NUMBER
    refs[field.key] = ctrl
    return _field_row(field, [ctrl] + err_widgets, center=True)


def build_form(ct: ConfigType, refs: dict, values: dict | None = None,
               errors: dict | None = None, on_browse=None,
               on_change=None) -> list:
    """渲染整棵表单控件树。

    values:   dict[key → 控件值]（None 时用字段默认值，来自 to_form_values/
              form_values_from_refs）
    errors:   dict[key → [msg]]；含 "_general" 时在表单末尾显示通用错误
    on_browse: callable(field, ref) — path 字段浏览按钮回调（由页面层注入
              FilePicker 逻辑；须为同步函数，内部自行调度异步浏览）
    on_change: callable(e) — select 字段 on_change（如模式联动重渲染；由页面层注入）
    """
    refs.clear()
    err_dict = errors or {}
    controls = []
    for i, group in enumerate(ct.groups):
        if group.name:
            controls.append(_group_header(group, is_first=(i == 0)))
        for field in group.fields:
            if not _visible(field, values):
                continue  # visible_when 不满足 → 隐藏该字段
            controls.append(_build_field(
                field, refs, err_dict.get(field.key, []), values, on_browse,
                on_change=on_change))
    general_msgs = err_dict.get("_general", [])
    if general_msgs:
        controls.append(ft.Column([_error_text(m) for m in general_msgs], spacing=2))
    return controls


def build_form_rows(ct: ConfigType, refs: dict, errors: dict | None = None) -> list:
    """兼容入口（页面层旧调用；新代码请走 build_form）。"""
    return build_form(ct, refs, values=None, errors=errors)
