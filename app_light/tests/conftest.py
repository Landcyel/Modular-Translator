"""pytest 全局初始化（tests/ 顶层 conftest）。

测试代码 import core 会触发 core 顶层 import app.torch_runtime：本文件在
tests 收集时最先加载，确保任何 import core 之前完成 torch 运行时选择。

（原 app/conftest.py 同职责，迁移后保留在 tests/app/conftest.py；
本文件兜底覆盖 tests/core 与 tests/scripts 下的用例。）
"""

from app import torch_runtime  # noqa: F401  # 先选 torch 运行时（dependencies 外挂）

# ── tmp_path fixture（等价替换 pytest 内建版，pytest.ini 已禁用 tmpdir 插件）──
# Windows/Python 3.14 下 pytest 以 mode=0o700 创建 basetemp，受限沙箱中目录
# ACL 不可枚举（PermissionError: [WinError 5]）。此处用项目本地 temp 目录、
# 不带 mode 参数创建，每个用例独立、结束后清理。
import os  # noqa: E402
import shutil  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def tmp_path() -> Path:
    # 每次运行使用全新 base（历史运行的 0o700 目录在沙箱中不可枚举/写入）
    base = (Path(__file__).resolve().parents[1] / "temp"
            / f"pytest-tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    os.makedirs(base, exist_ok=True)
    path = base / f"test-{uuid.uuid4().hex[:12]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(base, ignore_errors=True)
