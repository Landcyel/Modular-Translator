"""Modular Translator CUDA packaging script (PyInstaller onedir portable directory).

Must run with the project .venv's Python (interpreter pinning prevents a global
Python from mixing in old dependencies):

  .venv\\Scripts\\python.exe build/build_cuda.py                  # default output D:\\ModularTranslator_cuda
  .venv\\Scripts\\python.exe build/build_cuda.py --verify --smoke
  .venv\\Scripts\\python.exe build/build_cuda.py --with-cpu-fallback   # also bundles the torch-cpu fallback
  .venv\\Scripts\\python.exe build/build_cuda.py --with-whisper-cuda  # also bundles fasterwisper-cuda DLLs

CUDA build contents: torch-cuda runtime (the only torch slot by default, auto-
degrading to CPU mode without a GPU, see app/torch_runtime.py) + full llama-release
(CPU+CUDA); fasterwisper-cuda (~1.3GB) stays external by default, optionally
bundled via --with-whisper-cuda; --with-cpu-fallback optionally bundles the
torch-cpu fallback (+504MB, preferred when no GPU). Models and character assets
stay external (multi-volume archives made by tools/make_model_volumes.py).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Modular Translator CUDA 版打包（PyInstaller onedir，v3）"
    )
    common.add_common_args(ap)
    ap.add_argument(
        "--with-cpu-fallback",
        action="store_true",
        help="额外拷贝 torch-cpu 回退运行时（+504MB；无 GPU 时优先使用，默认省略——无 GPU 会自动降级用 torch-cuda 以 CPU 模式运行）",
    )
    ap.add_argument(
        "--with-whisper-cuda",
        action="store_true",
        help="额外把 dependencies/fasterwisper-cuda/（约 1.3GB）打入产物，whisper 可用 GPU 加速",
    )
    args = ap.parse_args()

    common.build(
        mode="cuda",
        default_output=r"D:\ModularTranslator_cuda",
        embed_engine=not args.no_engine,
        with_characters=args.with_characters,
        include_cpu_fallback=args.with_cpu_fallback,
        with_whisper_cuda=args.with_whisper_cuda,
        extra_paths=args.extra_paths,
        verify=args.verify,
        smoke=args.smoke,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
