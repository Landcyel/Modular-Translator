"""GsvEngine —— GPT-SoVITS 推理引擎（复刻 inference_webui_fast.py GUI 推理流程）。

- 源码 vendored 于 vendor/（从 GPT-SoVITS-main 复制，零修改）
- 重型权重经 Windows junction 指向 models/（见 vendor_links.py）
- 仅实现推理，不依赖 core/ 框架
"""

from .engine import GsvEngine

__all__ = ["GsvEngine"]
__version__ = "0.1.0"
