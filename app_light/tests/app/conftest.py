"""pytest 全局初始化。

测试代码 import core 会触发 core 顶层 import app.torch_runtime：本文件在
tests 收集时最先加载，确保任何 import core 之前完成 torch 运行时选择。
"""

from app import torch_runtime  # noqa: F401  # 先选 torch 运行时（dependencies 外挂）
