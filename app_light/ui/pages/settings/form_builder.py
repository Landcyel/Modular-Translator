"""Settings page form rendering engine — renders grouped strongly-typed forms from a declarative schema.

- Renders group titles from ``ConfigType.groups``.
- One render branch per field type:
    boolean → Switch; select → Dropdown; secret → revealable password field;
    browse=file/directory → TextField + browse button (``on_browse(field, ref)`` injected);
    multiline / list / object / json → multiline field; integer / number → numeric keyboard; others single-line.
- Errors shown in place: error text sits directly in the field row (no full-tree rebuild, no focus loss).
- ``field.width`` takes effect when explicitly given (otherwise expand fills the input column).
"""

import flet as ft

from ui.theme import Layout, Palette, Radius
from ui.pages.settings.config_schema import ConfigType, Field, FieldGroup
from ui.widgets.config_picker import _scan_config_dir, _option_for

# ── Responsive column widths ──
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
    """Determine whether a field renders per ``field.visible_when`` (all {key: value} must match)."""
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
        # Option sources: field.options (str or (key,text) tuples) + lazy options_provider
        #                 + scan_config_type directory scan (for "default selection" pickers)
        def _mk_option(o):
            """str → Option(key=text); two-tuple → display name and value separated (e.g. "Auto-detect (blank)")."""
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
        # If the initial value is not in the options, fall back to the first (e.g. an old
        # config's language code not in the list → fall back to "Auto-detect")
        if init_val and init_val not in [o.key for o in opts]:
            init_val = opts[0].key if opts else None
        dd = ft.Dropdown(value=init_val, options=opts, data=field.key,
                         on_select=on_change, **common)
        refs[field.key] = dd
        return _field_row(field, [dd] + err_widgets, center=True)

    # ── multiline / list / object / json: multiline TextField ──
    if field.type in ("multiline", "list", "object", "json"):
        ctrl = _textfield(field, multiline=True, value=init)
        refs[field.key] = ctrl
        return _field_row(field, [ctrl] + err_widgets, center=False)

    # ── path + browse: TextField + browse button ──
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

    # ── text / secret / integer / number: single-line TextField ──
    ctrl = _textfield(field, value=init)
    if field.type in ("integer", "number"):
        ctrl.keyboard_type = ft.KeyboardType.NUMBER
    refs[field.key] = ctrl
    return _field_row(field, [ctrl] + err_widgets, center=True)


def build_form(ct: ConfigType, refs: dict, values: dict | None = None,
               errors: dict | None = None, on_browse=None,
               on_change=None) -> list:
    """Render the whole form control tree.

    values:   dict[key → control value] (None means field defaults; comes from to_form_values /
              form_values_from_refs)
    errors:   dict[key → [msg]]; a "_general" key shows general errors at the end of the form
    on_browse: callable(field, ref) — browse-button callback for path fields (injected with
              FilePicker logic by the page layer; must be a synchronous function that
              dispatches the async browse itself)
    on_change: callable(e) — on_change for select fields (e.g. mode-linked re-render; injected by the page layer)
    """
    refs.clear()
    err_dict = errors or {}
    controls = []
    for i, group in enumerate(ct.groups):
        if group.name:
            controls.append(_group_header(group, is_first=(i == 0)))
        for field in group.fields:
            if not _visible(field, values):
                continue  # visible_when not satisfied → hide this field
            controls.append(_build_field(
                field, refs, err_dict.get(field.key, []), values, on_browse,
                on_change=on_change))
    general_msgs = err_dict.get("_general", [])
    if general_msgs:
        controls.append(ft.Column([_error_text(m) for m in general_msgs], spacing=2))
    return controls


def build_form_rows(ct: ConfigType, refs: dict, errors: dict | None = None) -> list:
    """Compatibility entry (old page-layer call; new code should use build_form)."""
    return build_form(ct, refs, values=None, errors=errors)
