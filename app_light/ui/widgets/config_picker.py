"""可复用的"配置选择器"组件 — 下拉选择，点击时动态扫描目录，get_value 返回 Path。"""

import flet as ft
from pathlib import Path

from app.paths import project_root
from ui.theme import Palette, Radius, Typography
from ui.components import _text

# config_type → 相对目录路径映射（对齐 configs/ 实际结构，键无歧义）
_CONFIG_DIR_MAP = {
    "llama":           "configs/models/llama",
    "api":             "configs/models/API",
    "translate_args":  "configs/translate/args_llama",
    "translate_args_api": "configs/translate/args_api",
    "prompts":         "configs/translate/prompts",
    "glossary":        "configs/translate/glossary",
    "rules":           "configs/translate/rules",
    "hotwords":        "configs/transcribe/hotwords",
    # 服务配置（configs/models 下按服务分子目录）
    "gsv":             "configs/models/gsv",
    "moss":            "configs/models/moss",
    # 任务级模板
    "gsv_args":        "configs/tts/args",
    "moss_args":       "configs/transcribe/args",
    # GSV 角色配置（configs/tts/roles/role-*.json — 角色模型资产）
    "gsv_role":        "configs/tts/roles",
}


def _scan_config_dir(config_type: str, glob_filter: str = "*.json") -> list[Path]:
    """扫描配置目录，返回匹配 *glob_filter* 的 .json 配置文件的 Path
    （绝对路径，按文件名排序）。"""
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
    """str / Path → 内部 key（Path 取 .name；str 原样）。"""
    return value.name if isinstance(value, Path) else value


def _display_name(key: str) -> str:
    """显示名：去掉 .json 后缀（act01.json → act01）；无后缀原样（如"无"）。"""
    return key[:-5] if key.lower().endswith(".json") else key


def _option_for(key: str) -> ft.dropdown.Option:
    """按内部 key 构建选项：text 为去后缀显示名，key 保持完整文件名（供 get_value 拼路径）。"""
    return ft.dropdown.Option(key=key, text=_display_name(key))


def _merged_names(fixed: list, scanned_paths: list) -> list:
    """合并固定项与目录扫描结果：固定项在前，去重。"""
    names = [n for n in fixed]
    for p in scanned_paths:
        if p.name not in names:
            names.append(p.name)
    return names


def config_picker(
    label: str,
    options: list = None,          # 固定附加项：str（如"无"）或 Path；与目录扫描结果合并
    config_type: str = "",
    width: int = 200,
    value=None,                    # 初始选中：str 文件名或 Path；None 时默认选中首个选项
    glob_filter: str = "*.json",   # 目录扫描过滤（如 "gsv*.json"；configs/models 混放多类配置）
) -> tuple:
    """配置选择器：Dropdown 展示已有配置。

    - 点击（聚焦）时自动重新扫描 *config_type* 对应目录，把目录下新增的
      匹配 *glob_filter* 的 .json 文件名追加到选项（保留 options 固定项、
      去重、保留当前选中）。
    - 选项显示名只含文件名、不带 .json 后缀；内部值保持完整文件名。
    - 组件无外框（仅 label + Dropdown，由外层容器提供间距）。
    - ``get_value()`` 返回选中配置文件的 :class:`Path` 对象；选中固定项
      （如"无"）或未选中时返回 ``None``。
    - ``get_value.set_config_type(ctype)`` 可从外部修改目标目录（如后端
      切换时由按钮调用），修改后立即重新扫描并刷新选项。

    返回 (container, get_value) — container 是控件，get_value 是读取当前选中值的回调。
    """
    fixed = [_to_key(o) for o in (options or [])]
    state = {"config_type": config_type}
    scanned_paths = _scan_config_dir(state["config_type"], glob_filter)

    def _refresh(_e=None):
        """重新扫描目录并刷新选项（保留当前选中）。"""
        keys = _merged_names(fixed, _scan_config_dir(state["config_type"], glob_filter))
        dropdown.options = [_option_for(k) for k in keys]
        if dropdown.value is not None and dropdown.value not in keys:
            dropdown.value = None
        try:
            dropdown.update()
        except RuntimeError:
            pass  # 未挂载 page（如单元验证）时跳过刷新推送

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
        """返回选中配置文件的 Path；选中固定项（如"无"）或未选中返回 None。"""
        name = dropdown.value
        rel_dir = _CONFIG_DIR_MAP.get(state["config_type"], "")
        if not name or not rel_dir:
            return None
        p = project_root / rel_dir / name
        return p if p.is_file() else None

    def set_config_type(ctype: str) -> None:
        """外部修改 config_type（目标目录）并立即刷新选项。"""
        state["config_type"] = ctype
        _refresh()

    def set_value(name: str | None) -> None:
        """外部设置选中项（key=文件名，None 清空）并刷新选项。

        供默认配置联动使用：后端切换/启动时按默认配置指定文件设置下拉选中。
        目标不在当前选项中时由 _refresh 自动清空（回退未选中）。
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
