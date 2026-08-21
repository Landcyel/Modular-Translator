"""Modular Translator CPU 版打包脚本（PyInstaller onedir 绿色目录）。

必须使用项目 .venv 的 Python 运行（解释器锁定，防止全局 Python 混入旧版依赖）：

  .venv\\Scripts\\python.exe build/build_cpu.py                  # 默认输出 D:\\ModularTranslator_cpu
  .venv\\Scripts\\python.exe build/build_cpu.py --verify --smoke  # 构建后校验 + 启动冒烟
  .venv\\Scripts\\python.exe build/build_cpu.py --no-engine --output-dir D:\\MT-cpu

CPU 版内容：torch-cpu 运行时 + llama-release CPU 子集 + FFmpeg(ffmpeg/ffprobe)，
模型与角色资产外置（由 tools/make_model_volumes.py 制作分卷）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Modular Translator CPU 版打包（PyInstaller onedir，v3）"
    )
    common.add_common_args(ap)
    args = ap.parse_args()

    common.build(
        mode="cpu",
        default_output=r"D:\ModularTranslator_cpu",
        embed_engine=not args.no_engine,
        with_characters=args.with_characters,
        include_cpu_fallback=False,
        with_whisper_cuda=False,
        extra_paths=args.extra_paths,
        verify=args.verify,
        smoke=args.smoke,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
