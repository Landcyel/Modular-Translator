"""torch_runtime — 可插拔 PyTorch 运行时选择（CPU 基线 + dependencies/ 外挂 CUDA）。

背景：PyTorch 的 CUDA 是编译进 wheel 的能力——CPU 版 torch 即使加载了 cuBLAS/cuDNN DLL，
``torch.backends.cuda.is_built()`` 仍为 False。因此 GSV / MOSS 的 GPU 加速
需要外挂**完整的 CUDA 版 torch 包**（``torch/ + torchaudio/``），而不是散装 DLL。

本模块在**任何 ``import torch`` 之前**完成三件事（APP.py 顶部最先导入）：
1. 选择运行时目录（优先级见 :func:`setup`）；
2. 安装一个置于 ``sys.meta_path`` 首位的自定义 finder，把 ``torch`` /
   ``torchaudio``（及 torchgen/functorch）的导入重定向到所选目录——
   **只劫持 torch 系包**，其余包仍走系统/venv 的正常搜索路径，避免
   整目录插到 ``sys.path`` 头部造成的版本污染；
3. ``os.add_dll_directory`` 注册所选目录下的 ``torch/lib`` / ``torchaudio/lib``，
   供 .pyd 依赖解析。

选择优先级（``TRANSLATOR_TORCH_RUNTIME=auto|cuda|cpu``，默认 auto）：
- auto:
  1. ``dependencies/runtime/torch-cuda/`` 完整 + ``cudart64_*.dll`` 探测到 GPU
  2. ``dependencies/runtime/torch-cpu/`` 完整
  3. ``dependencies/runtime/torch-cuda/`` 完整但**未探测到 GPU**（且无 torch-cpu
     回退）→ 降级激活该 CUDA 槽，以 CPU 模式运行（``torch.cuda.is_available()``
     = False，应用层自动走 CPU device）；CUDA 版打包默认只带 torch-cuda 槽
  4. 开发模式专用回退：``dependencies/venv/Lib/site-packages``（若为 CUDA 构建
     且探测到 GPU；当前仓库该 venv 即 cu128 环境）
  5. 系统默认搜索路径（开发机全局 CPU torch / 根 .venv 的 CPU torch）
- cuda / cpu：只按对应槽位强制选择；cuda 槽不完整时记录 error 并退回系统；
  cuda 槽完整但无 GPU 时同样降级为 CPU 模式。

本模块自身**绝不 import torch**：GPU 预检通过直接 ctypes 加载所选目录的
``cudart64_*.dll`` 并调用 ``cudaGetDeviceCount`` 完成，避免把失败的 torch
导入留在 ``sys.modules`` 里，保证 CPU 回退是干净的新路径。
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

# 运行时目录（冻结与开发同基准：产物根或项目根下的 dependencies/）
RUNTIME_DIR = Path(project_root) / "dependencies" / "runtime"
CUDA_SLOT = RUNTIME_DIR / "torch-cuda"
CPU_SLOT = RUNTIME_DIR / "torch-cpu"

# 开发模式专用：仓库自带 dependencies/venv（cu128 torch + GSV/MOSS 全套依赖）。
# 冻结产物不存在该目录，不影响打包路径。
DEV_VENV_SITE = Path(project_root) / "dependencies" / "venv" / "Lib" / "site-packages"

# 需要重定向到外部运行时的包（torch 的伴随包一并处理）。
_REDIRECT_TOP = ("torch", "torchaudio", "torchgen", "functorch")

_state: dict | None = None


class _TorchFinder(MetaPathFinder):
    """仅认领 torch 系包的 meta_path finder，把导入指向外部运行时目录。"""

    def __init__(self, root: Path):
        self.root = str(root)

    def find_spec(self, fullname: str, path=None, target=None):
        # 只认领 4 个顶层包名；子模块（torch.nn…）返回 None，交给默认
        # PathFinder 按父包 __path__（已指向外部 torch/ 目录）解析。
        # 若对子模块也用 path=[root] 委托 PathFinder，dotted name 会按
        # 错误基准解析（如把 torch.nn.modules.distance 解析到顶层 distance 包）。
        if fullname not in _REDIRECT_TOP:
            return None
        from importlib.machinery import PathFinder

        return PathFinder.find_spec(fullname, path=[self.root])

    def __repr__(self) -> str:  # pragma: no cover - 调试展示
        return f"<TorchRuntimeFinder root={self.root}>"


# ---------------------------------------------------------------- 探测辅助


def _is_complete(root: Path) -> bool:
    """torch 包可导入的最小完整性检查（不 import）。"""
    return (root / "torch" / "__init__.py").is_file() and (
        root / "torch" / "lib"
    ).is_dir()


def _is_cuda_build(root: Path) -> bool:
    """CUDA 版 torch 的标志：torch/lib/torch_cuda.dll 存在。"""
    return (root / "torch" / "lib" / "torch_cuda.dll").is_file()


def _cuda_device_count(root: Path) -> int:
    """用所选目录的 cudart64_*.dll 预检 GPU（不 import torch）。

    返回可见设备数；加载失败 / 调用失败 / 0 设备均返回 0。
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
    """安装外部运行时：meta_path finder + DLL 目录。返回归一化 kind。"""
    global _state
    kind = "cuda" if _is_cuda_build(root) else "cpu"
    finder = _TorchFinder(root)
    # 防重复激活（幂等重入时先移除旧 finder）
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
    """按优先级选出 (kind, root, label)；未选中时 kind='system'。

    auto 优先级：GPU 可用的 torch-cuda > torch-cpu 槽 > 无 GPU 的 torch-cuda
    降级（仅当无 torch-cpu 槽，以 CPU 模式激活）。
    """
    cuda_candidates: list[tuple[Path, str]] = [(CUDA_SLOT, "dependencies/runtime/torch-cuda")]
    if not is_frozen() and DEV_VENV_SITE.is_dir():
        cuda_candidates.append((DEV_VENV_SITE, "dependencies/venv"))

    degraded: tuple[Path, str] | None = None  # CUDA 槽存在但无 GPU 的降级候选
    if forced in ("auto", "cuda"):
        for root, label in cuda_candidates:
            if not root.is_dir():
                continue  # 未外挂，属正常状态，不告警
            if not _is_complete(root):
                log.record(
                    "warn",
                    f"torch_runtime: CUDA 运行时目录不完整，跳过: {root}",
                )
                continue
            devices = _cuda_device_count(root)
            if devices > 0:
                return "cuda", root, label
            # 无 GPU：不跳过，暂存为降级候选（无 torch-cpu 槽时以 CPU 模式激活）
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
    """执行运行时选择（幂等）。返回 'cuda' / 'cpu' / 'system'。"""
    import time

    global _state
    if _state is not None:
        return _state["kind"]
    _t0 = time.perf_counter()

    if "torch" in sys.modules:
        # 说明本模块被 import 得太晚（应有 APP.py 顶部导入保证顺序），
        # finder 无法替换已加载的 torch，仅记录便于排查。
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


# ---------------------------------------------------------------- 查询接口


def runtime_kind() -> str:
    """当前选择的运行时：'cuda' / 'cpu' / 'system'。"""
    setup()
    return _state["kind"]


def runtime_root() -> Path | None:
    """外部运行时根目录；system 模式为 None。"""
    setup()
    return _state["root"]


def describe() -> str:
    """人类可读的运行时描述（供服务启动日志）。"""
    setup()
    if _state["kind"] == "system":
        return "system"
    return f"{_state['kind']}: {_state['label']} ({_state['root']})"


def ensure_available() -> str:
    """在重服务启动前调用：确认 torch 可被找到，否则抛带安装指引的错误。"""
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


# 模块导入即完成选择（须在 APP.py 顶部、任何 import core/flet 之前导入）。
setup()
