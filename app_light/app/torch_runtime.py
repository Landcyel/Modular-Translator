"""torch_runtime — pluggable PyTorch runtime selection (CPU baseline + CUDA from dependencies/).

Background: PyTorch's CUDA support is compiled into the wheel — a CPU-only torch
returns False from ``torch.backends.cuda.is_built()`` even if cuBLAS/cuDNN DLLs
are loaded. So GPU acceleration for GSV / MOSS requires attaching the **complete
CUDA torch packages** (``torch/ + torchaudio/``), not loose DLLs.

Before any ``import torch`` this module does three things (imported first at the
top of APP.py):
1. Choose the runtime directory (priority order in :func:`setup`);
2. Install a custom finder placed at the head of ``sys.meta_path`` that redirects
   imports of ``torch`` / ``torchaudio`` (and torchgen/functorch) to the chosen
   directory — **only torch-family packages are hijacked**; other packages keep
   the normal system/venv search path, avoiding version pollution from inserting
   the whole directory at the head of ``sys.path``;
3. ``os.add_dll_directory`` registers ``torch/lib`` / ``torchaudio/lib`` under
   the chosen directory for .pyd dependency resolution.

Selection priority (``TRANSLATOR_TORCH_RUNTIME=auto|cuda|cpu``, default auto):
- auto:
  1. ``dependencies/runtime/torch-cuda/`` complete and a GPU detected via ``cudart64_*.dll``
  2. ``dependencies/runtime/torch-cpu/`` complete
  3. ``dependencies/runtime/torch-cuda/`` complete but **no GPU detected** (and no
     torch-cpu fallback) → activate the CUDA slot degraded, running in CPU mode
     (``torch.cuda.is_available()`` = False; the app layer automatically uses CPU
     device); CUDA builds by default ship only the torch-cuda slot
  4. Dev-mode-only fallback: ``dependencies/venv/Lib/site-packages`` (if it is a
     CUDA build and a GPU is detected; the repo's venv is the cu128 environment)
  5. System default search path (dev machine's global CPU torch / root .venv CPU torch)
- cuda / cpu: force-select the corresponding slot only; an incomplete cuda slot
  logs an error and falls back to system; a complete cuda slot without a GPU also
  degrades to CPU mode.

This module itself **never imports torch**: the GPU precheck loads the chosen
directory's ``cudart64_*.dll`` via ctypes and calls ``cudaGetDeviceCount``, so a
failed torch import is never left in ``sys.modules``, keeping the CPU fallback a
clean new path.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

from app.log import log
from app.paths import is_frozen, project_root

__all__ = [
    "setup",
    "runtime_kind",
    "runtime_root",
    "describe",
    "ensure_available",
    "TORCH_RUNTIME_ENV",
]

TORCH_RUNTIME_ENV = "TRANSLATOR_TORCH_RUNTIME"

# Runtime directories (frozen and dev share the same base: dependencies/ under the
# output or project root)
RUNTIME_DIR = Path(project_root) / "dependencies" / "runtime"
CUDA_SLOT = RUNTIME_DIR / "torch-cuda"
CPU_SLOT = RUNTIME_DIR / "torch-cpu"

# Dev-only: the repo's bundled dependencies/venv (cu128 torch + full GSV/MOSS deps).
# Frozen builds do not have this directory, so the packaged path is unaffected.
DEV_VENV_SITE = Path(project_root) / "dependencies" / "venv" / "Lib" / "site-packages"

# Packages to redirect to the external runtime (torch's companion packages included).
_REDIRECT_TOP = ("torch", "torchaudio", "torchgen", "functorch")

_state: dict | None = None


class _TorchFinder(MetaPathFinder):
    """meta_path finder that only claims torch-family packages, pointing imports at the external runtime directory."""

    def __init__(self, root: Path):
        self.root = str(root)

    def find_spec(self, fullname: str, path=None, target=None):
        # Only claim the 4 top-level package names; submodules (torch.nn…) return
        # None and are resolved by the default PathFinder against the parent
        # package's __path__ (already pointing at the external torch/ directory).
        # Delegating submodules to PathFinder with path=[root] too would resolve
        # dotted names against the wrong base (e.g. resolving
        # torch.nn.modules.distance to a top-level distance package).
        if fullname not in _REDIRECT_TOP:
            return None
        from importlib.machinery import PathFinder

        return PathFinder.find_spec(fullname, path=[self.root])

    def __repr__(self) -> str:  # pragma: no cover - debug display
        return f"<TorchRuntimeFinder root={self.root}>"


# ── Detection helpers ──


def _is_complete(root: Path) -> bool:
    """Minimal completeness check for the torch package to be importable (no import)."""
    return (root / "torch" / "__init__.py").is_file() and (
        root / "torch" / "lib"
    ).is_dir()


def _is_cuda_build(root: Path) -> bool:
    """Marker of a CUDA torch build: torch/lib/torch_cuda.dll exists."""
    return (root / "torch" / "lib" / "torch_cuda.dll").is_file()


def _cuda_device_count(root: Path) -> int:
    """Precheck the GPU using the chosen directory's cudart64_*.dll (no torch import).

    Returns the number of visible devices; load failure / call failure / 0
    devices all return 0.
    """
    lib = root / "torch" / "lib"
    if not lib.is_dir():
        return 0
    for dll in sorted(lib.glob("cudart64_*.dll")):
        try:
            handle = ctypes.CDLL(str(dll))
        except OSError:
            continue
        try:
            count = ctypes.c_int(-1)
            if handle.cudaGetDeviceCount(ctypes.byref(count)) != 0:
                return 0
            return max(0, count.value)
        except Exception:
            return 0
    return 0


def _activate(root: Path, label: str) -> str:
    """Install the external runtime: meta_path finder + DLL directories. Returns the normalized kind."""
    global _state
    kind = "cuda" if _is_cuda_build(root) else "cpu"
    finder = _TorchFinder(root)
    # Prevent duplicate activation (remove the old finder first on idempotent re-entry)
    for mf in list(sys.meta_path):
        if isinstance(mf, _TorchFinder):
            sys.meta_path.remove(mf)
    sys.meta_path.insert(0, finder)
    for dll_dir in (root / "torch" / "lib", root / "torchaudio" / "lib"):
        if dll_dir.is_dir():
            try:
                os.add_dll_directory(str(dll_dir))
            except (OSError, AttributeError) as exc:
                log.record("warn", f"torch_runtime: 注册 DLL 目录失败 {dll_dir}: {exc}")
    _state = {"kind": kind, "root": root, "label": label}
    return kind


def _select(forced: str) -> tuple[str, Path | None, str]:
    """Select (kind, root, label) by priority; kind='system' when nothing is selected.

    auto priority: GPU-capable torch-cuda > torch-cpu slot > GPU-less torch-cuda
    degraded (only when there is no torch-cpu slot, activated in CPU mode).
    """
    cuda_candidates: list[tuple[Path, str]] = [(CUDA_SLOT, "dependencies/runtime/torch-cuda")]
    if not is_frozen() and DEV_VENV_SITE.is_dir():
        cuda_candidates.append((DEV_VENV_SITE, "dependencies/venv"))

    degraded: tuple[Path, str] | None = None  # degraded candidate: CUDA slot present but no GPU
    if forced in ("auto", "cuda"):
        for root, label in cuda_candidates:
            if not root.is_dir():
                continue  # not attached, normal state, no warning
            if not _is_complete(root):
                log.record(
                    "warn",
                    f"torch_runtime: CUDA 运行时目录不完整，跳过: {root}",
                )
                continue
            devices = _cuda_device_count(root)
            if devices > 0:
                return "cuda", root, label
            # No GPU: don't skip, keep as degraded candidate (activated in CPU
            # mode when there is no torch-cpu slot)
            log.record(
                "warn",
                f"torch_runtime: {label} 未探测到可用 GPU（设备数={devices}）",
            )
            if degraded is None:
                degraded = (root, label)
        if forced == "cuda":
            if degraded is not None:
                log.record("warn", "torch_runtime: 强制 CUDA 但未探测到 GPU，将以 CPU 模式运行")
                return "cuda", *degraded
            log.record("error", "torch_runtime: 强制 CUDA 但未找到完整且可用的 CUDA 运行时")
            return "system", None, "system"

    if forced in ("auto", "cpu"):
        if _is_complete(CPU_SLOT):
            return "cpu", CPU_SLOT, "dependencies/runtime/torch-cpu"
        if forced == "cpu":
            log.record("warn", "torch_runtime: 强制 CPU 但未找到 dependencies/runtime/torch-cpu，使用系统 torch")
            return "system", None, "system"

    if forced == "auto" and degraded is not None:
        log.record(
            "warn",
            "torch_runtime: 未探测到 GPU 且无 torch-cpu 回退，使用 CUDA 运行时以 CPU 模式运行",
        )
        return "cuda", *degraded
    return "system", None, "system"


def setup() -> str:
    """Perform runtime selection (idempotent). Returns 'cuda' / 'cpu' / 'system'."""
    import time

    global _state
    if _state is not None:
        return _state["kind"]
    _t0 = time.perf_counter()

    if "torch" in sys.modules:
        # Indicates this module was imported too late (APP.py imports it first to
        # guarantee ordering); the finder cannot replace an already-loaded torch,
        # so only log it for troubleshooting.
        log.record(
            "warn",
            "torch_runtime: torch 已在本模块之前导入，运行时选择无法替换已加载实例",
        )

    forced = os.environ.get(TORCH_RUNTIME_ENV, "auto").strip().lower()
    if forced not in ("auto", "cuda", "cpu"):
        log.record("warn", f"torch_runtime: 未知环境变量值 {forced!r}，按 auto 处理")
        forced = "auto"

    kind, root, label = _select(forced)
    if root is not None:
        kind = _activate(root, label)
        log.record("info", f"torch_runtime: 已选择 {kind.upper()} 运行时 [{label}] -> {root}")
    else:
        _state = {"kind": "system", "root": None, "label": "system"}
        spec = importlib.util.find_spec("torch")
        src = "未发现" if spec is None else getattr(spec, "origin", "?")
        log.record(
            "info" if spec is not None else "warn",
            f"torch_runtime: 使用系统默认 torch（{src}）",
        )
    log.record(
        "info",
        f"torch_runtime: setup 用时 {time.perf_counter() - _t0:.1f}s（kind={_state['kind']}，"
        f"含 GPU 探测 + DLL 注册 + 外挂目录首访）",
    )
    return _state["kind"]


# ── Query interface ──


def runtime_kind() -> str:
    """The currently selected runtime: 'cuda' / 'cpu' / 'system'."""
    setup()
    return _state["kind"]


def runtime_root() -> Path | None:
    """External runtime root directory; None in system mode."""
    setup()
    return _state["root"]


def describe() -> str:
    """Human-readable runtime description (for service startup logs)."""
    setup()
    if _state["kind"] == "system":
        return "system"
    return f"{_state['kind']}: {_state['label']} ({_state['root']})"


def ensure_available() -> str:
    """Call before starting heavy services: confirm torch can be found, otherwise raise an error with install instructions."""
    setup()
    spec = importlib.util.find_spec("torch")
    if spec is None:
        raise RuntimeError(
            "未找到 torch 运行时。请将完整的 torch 包解压到 "
            f"dependencies/runtime/torch-cpu（或 CUDA 版 torch-cuda），"
            "或在本环境执行: pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
    return describe()


# Module import completes selection (must be imported at the top of APP.py, before
# any import core/flet).
setup()
