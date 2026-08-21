"""Reusable "config picker" component — a dropdown that re-scans its directory on focus; get_value returns a Path."""

import flet as ft
from pathlib import Path

from app.paths import project_root
from ui.theme import Palette, Radius, Typography
from ui.components import _text

# config_type → relative directory path map (aligned with the real configs/ structure; keys unambiguous)
_CONFIG_DIR_MAP = {
    "llama":           "configs/models/llama",
    "api":             "configs/models/API",
    "translate_args":  "configs/translate/args_llama",
    "translate_args_api": "configs/translate/args_api",
    "prompts":         "configs/translate/prompts",
    "glossary":        "configs/translate/glossary",
    "rules":           "configs/translate/rules",
    "hotwords":        "configs/transcribe/hotwords",
    # service configs (per-service subdirectories under configs/models)
    "gsv":             "configs/models/gsv",
    "moss":            "configs/models/moss",
    # task-level templates
    "gsv_args":        "configs/tts/args",
    "moss_args":       "configs/transcribe/args",
    # GSV role configs (configs/tts/roles/role-*.json — role model assets)
    "gsv_role":        "configs/tts/roles",
}


def _scan_config_dir(config_type: str, glob_filter: str = "*.json") -> list[Path]:
    """Scan the config directory and return Paths of .json config files matching *glob_filter*
    (absolute paths, sorted by file name)."""
    if not config_type:
        return []
    rel_dir = _CONFIG_DIR_MAP.get(config_type)
    if not rel_dir:
        return []
    scan_dir = project_root / rel_dir
    if not scan_dir.exists():
        return []
    return sorted(scan_dir.glob(glob_filter or "*.json"), key=lambda p: p.name)


def _to_key(value) -> str:
    """str / Path → internal key (Path uses .name; str as-is)."""
    return value.name if isinstance(value, Path) else value


def _display_name(key: str) -> str:
    """Display name: strips the .json suffix (act01.json → act01); no suffix → as-is (e.g. "None")."""
    return key[:-5] if key.lower().endswith(".json") else key


def _option_for(key: str) -> ft.dropdown.Option:
    """Build an option from the internal key: text is the suffix-stripped display name, key keeps the full file name (for get_value to build the path)."""
    return ft.dropdown.Option(key=key, text=_display_name(key))


def _merged_names(fixed: list, scanned_paths: list) -> list:
    """Merge fixed items with directory scan results: fixed items first, deduplicated."""
    names = [n for n in fixed]
    for p in scanned_paths:
        if p.name not in names:
            names.append(p.name)
    return names


def config_picker(
    label: str,
    options: list = None,          # fixed extra items: str (e.g. "None") or Path; merged with scan results
    config_type: str = "",
    width: int = 200,
    value=None,                    # initial selection: str file name or Path; None → default to the first option
    glob_filter: str = "*.json",   # directory scan filter (e.g. "gsv*.json"; configs/models mixes multiple config kinds)
) -> tuple:
    """Config picker: a Dropdown listing the existing configs.

    - On focus, it re-scans the *config_type* directory and appends newly found .json file
      names matching *glob_filter* to the options (keeping the options fixed items,
      deduplicated, and preserving the current selection).
    - Option display names contain only the file name without the .json suffix; internal
      values keep the full file name.
    - The component has no outer frame (just label + Dropdown; spacing is provided by the
      surrounding container).
    - ``get_value()`` returns a :class:`Path` for the selected config file; selecting a fixed
      item (e.g. "None") or nothing returns ``None``.
    - ``get_value.set_config_type(ctype)`` externally changes the target directory (e.g.
      called by a button on backend switch); it re-scans and refreshes the options immediately.

    Returns (container, get_value) — container is the control, get_value is a callback reading the current selection.
    """
    fixed = [_to_key(o) for o in (options or [])]
    state = {"config_type": config_type}
    scanned_paths = _scan_config_dir(state["config_type"], glob_filter)

    def _refresh(_e=None):
        """Re-scan the directory and refresh the options (keeping the current selection)."""
        keys = _merged_names(fixed, _scan_config_dir(state["config_type"], glob_filter))
        dropdown.options = [_option_for(k) for k in keys]
        if dropdown.value is not None and dropdown.value not in keys:
            dropdown.value = None
        try:
            dropdown.update()
        except RuntimeError:
            pass  # skip the refresh push when not mounted to a page (e.g. unit verification)

    initial_keys = _merged_names(fixed, scanned_paths)
    initial_value = _to_key(value) if value is not None else (initial_keys[0] if initial_keys else None)

    dropdown = ft.Dropdown(
        label_style=ft.TextStyle(size=Typography.SMALL, color=Palette.TEXT_MUTED),
        text_style=ft.TextStyle(size=Typography.BODY, color=Palette.TEXT),
        bgcolor=Palette.SURFACE2,
        border_color=Palette.BORDER,
        border_radius=Radius.MD,
        filled=True,
        dense=True,
        expand=True,
        options=[_option_for(k) for k in initial_keys],
        value=initial_value,
        on_focus=_refresh,
    )

    def get_value() -> Path | None:
        """Return the selected config file's Path; None for a fixed item (e.g. "None") or no selection."""
        name = dropdown.value
        rel_dir = _CONFIG_DIR_MAP.get(state["config_type"], "")
        if not name or not rel_dir:
            return None
        p = project_root / rel_dir / name
        return p if p.is_file() else None

    def set_config_type(ctype: str) -> None:
        """Externally change config_type (target directory) and refresh the options immediately."""
        state["config_type"] = ctype
        _refresh()

    def set_value(name: str | None) -> None:
        """Externally set the selection (key=file name; None clears) and refresh the options.

        Used by default-config linkage: on backend switch/start, set the dropdown selection to
        the file specified by the default config. If the target is not in the current options,
        _refresh clears it automatically (falls back to no selection).
        """
        dropdown.value = name
        _refresh()

    get_value.set_config_type = set_config_type
    get_value.set_value = set_value

    container = ft.Container(
        content=ft.Column([
            ft.Row([
                _text(label, Typography.SMALL, "normal", Palette.TEXT_MUTED),
            ]),
            ft.Row([
                dropdown,
            ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ], spacing=4),
        width=width,
    )

    return container, get_value
