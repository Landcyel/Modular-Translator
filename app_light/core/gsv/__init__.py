"""GsvEngine — GPT-SoVITS inference engine (replicates the inference_webui_fast.py GUI inference flow).

- Source is vendored under vendor/ (copied from GPT-SoVITS-main, zero modification)
- Heavy weights point to models/ via Windows junctions (see vendor_links.py)
- Inference only; does not depend on the core/ framework
"""

from .engine import GsvEngine

__all__ = ["GsvEngine"]
__version__ = "0.1.0"
