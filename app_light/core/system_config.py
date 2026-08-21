"""系统默认配置 — configs/system/default.ini 统一读写。

三个"默认配置"（翻译/转写/输出）的全部默认设置存放在单一 INI 文件：
- [translate]  翻译页默认加载项（llama_server/api_server/prompt/translate_args/translate_args_api/rule/glossary）
- [transcribe] 转写页默认加载项（moss_server/moss_args/hotwords）
- [output]     已完成页默认输出（output_dir/auto_export）

本模块零 flet 依赖，使用标准库 configparser（utf-8）。
"""

import configparser

from app.paths import project_root

# <应用根>/configs/system/default.ini（project_root：开发=项目根 / 冻结=产物根）
DEFAULT_INI_PATH = project_root / "configs" / "system" / "default.ini"


def load_section(section: str, default: dict | None = None) -> dict:
    """读取 ini 中某 section 的全部键值（字符串形式）。

    缺失 section / 文件缺失或解析失败时返回 ``default``（或空 dict），不抛错。
    """
    cp = configparser.ConfigParser()
    try:
        cp.read(DEFAULT_INI_PATH, encoding="utf-8")
    except Exception:
        return dict(default or {})
    if cp.has_section(section):
        return dict(cp.items(section))
    return dict(default or {})


def save_section(section: str, data: dict) -> None:
    """写入 ini 某 section，保留其它 section；文件缺失时创建。

    ``data`` 全部键值转为字符串写入；非字符串（如 bool）按 str() 规范化。
    """
    cp = configparser.ConfigParser()
    if DEFAULT_INI_PATH.exists():
        try:
            cp.read(DEFAULT_INI_PATH, encoding="utf-8")
        except Exception:
            cp = configparser.ConfigParser()
    if not cp.has_section(section):
        cp.add_section(section)
    for key, value in data.items():
        cp.set(section, key, str(value))
    DEFAULT_INI_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_INI_PATH, "w", encoding="utf-8") as f:
        cp.write(f)
