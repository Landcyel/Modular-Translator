"""System default configuration — unified read/write of configs/system/default.ini.

All default settings for the three "default configs" (translate / transcribe / output)
live in a single INI file:
- [translate]   defaults loaded by the translate page (llama_server/api_server/prompt/translate_args/translate_args_api/rule/glossary)
- [transcribe]  defaults loaded by the transcribe page (moss_server/moss_args/hotwords)
- [output]      defaults output by the finished page (output_dir/auto_export)

This module has zero flet dependencies; it uses the stdlib configparser (utf-8).
"""

import configparser

from app.paths import project_root

# <app root>/configs/system/default.ini (project_root: dev=project root / frozen=artifact root)
DEFAULT_INI_PATH = project_root / "configs" / "system" / "default.ini"


def load_section(section: str, default: dict | None = None) -> dict:
    """Read all key-values of a section in the ini (as strings).

    Returns ``default`` (or an empty dict) when the section is missing / the file is
    missing or fails to parse; never raises.
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
    """Write a section to the ini, preserving other sections; creates the file if missing.

    All ``data`` key-values are written as strings; non-strings (e.g. bool) are normalized via str().
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
