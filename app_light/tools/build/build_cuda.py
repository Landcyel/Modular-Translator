"""Modular Translator CUDA 版打包脚本（PyInstaller onedir 绿色目录）。

必须使用项目 .venv 的 Python 运行（解释器锁定，防止全局 Python 混入旧版依赖）：

  .venv\\Scripts\\python.exe tools/build/build_cuda.py                  # 默认输出 D:\\ModularTranslator_cuda
  .venv\\Scripts\\python.exe tools/build/build_cuda.py --verify --smoke
  .venv\\Scripts\\python.exe tools/build/build_cuda.py --with-cpu-fallback   # 额外带回 torch-cpu 回退
  .venv\\Scripts\\python.exe tools/build/build_cuda.py --with-whisper-cuda  # 额外打入 fasterwisper-cuda DLL

CUDA 版内容：torch-cuda 运行时（默认唯一 torch 槽位，无 GPU 时自动降级为 CPU
模式运行，见 app/torch_runtime.py）+ llama-release 完整版（CPU+CUDA）；
fasterwisper-cuda（约 1.3GB）默认外置，--with-whisper-cuda 可选打入；
--with-cpu-fallback 可额外打入 torch-cpu 回退（+504MB，无 GPU 时优先使用）。
模型与角色资产外置（由 tools/make_model_volumes.py 制作分卷）。
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
