"""App root path resolution — the unified baseline for "source root / PyInstaller output root".

Modules (service / system_config / config_picker / config_schema, etc.) previously
derived the project root via ``Path(__file__)``; after PyInstaller packaging,
``__file__`` points at the compiled artifact inside ``_internal/``, so the derived
value was misaligned to ``_internal/`` while ``dependencies/`` and ``configs/``
live at the output root (siblings of ``_internal/``).

This module provides the single baseline: under a PyInstaller frozen runtime it
takes ``sys.executable``'s directory (the onedir output root); in dev runs it
uses the project root derived from ``__file__``. Every module relying on
``dependencies/`` / ``configs/`` relative paths should use ``project_root``.
"""

import os
import sys
from pathlib import Path

__all__ = ["app_root", "project_root", "is_frozen"]

# This file lives in the <root>/app/ package; dirname is app/, so one level up is the project root.
_DEV_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """Whether running from a packaged build (PyInstaller onedir)."""
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    """Return the application root directory (Path).

    - Frozen packaged runtime (PyInstaller onedir): the directory of
      ``sys.executable``, i.e. the output root (sibling of ``_internal/``,
      containing ``dependencies/`` and ``configs/``).
    - Dev/source run: the project root.
    """
    if is_frozen():
        # Under onedir, sys.executable is <output root>/ModularTranslator.exe
        return Path(sys.executable).resolve().parent
    return _DEV_ROOT


# Module-level constant: modules reuse it directly via ``from app.paths import project_root``.
project_root: Path = app_root()
