"""应用根路径定位 — 统一「源码根 / PyInstaller 产物根」基准。

各模块（service / system_config / config_picker /
config_schema 等）此前用 ``Path(__file__)`` 推导项目根；PyInstaller 打包后
``__file__`` 指向 ``_internal/`` 内编译产物，推导值错位到 ``_internal/``，
而 ``dependencies/`` 与 ``configs/`` 都在产物根（``_internal/`` 同级）。

本模块提供唯一基准：PyInstaller 冻结运行时取 ``sys.executable`` 所在目录
（onedir 产物根），开发运行时取 ``__file__`` 推导的项目根。所有依赖
``dependencies/`` / ``configs/`` 相对路径的模块都应改用 ``project_root``。
"""

import os
import sys
from pathlib import Path

__all__ = ["app_root", "project_root", "is_frozen"]

# 本文件位于 <根>/app/ 包内，dirname 是 app/，向上取一级得到项目根。
_DEV_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """是否运行于打包产物（PyInstaller onedir）。"""
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    """返回应用根目录（Path）。

    - 打包冻结运行时（PyInstaller onedir）：``sys.executable``
      所在目录，即产物根（与 ``_internal/`` 同级，含 ``dependencies/``、``configs/``）。
    - 开发/源码运行：项目根。
    """
    if is_frozen():
        # sys.executable 在 onedir 下即 <产物根>/ModularTranslator.exe
        return Path(sys.executable).resolve().parent
    return _DEV_ROOT


# 模块级常量：各模块直接 ``from app.paths import project_root`` 复用。
project_root: Path = app_root()
