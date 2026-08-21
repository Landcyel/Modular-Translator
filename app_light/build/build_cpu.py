"""Modular Translator CPU packaging script (PyInstaller onedir portable directory).

Must run with the project .venv's Python (interpreter pinning prevents a global
Python from mixing in old dependencies):

  .venv\\Scripts\\python.exe build/build_cpu.py                  # default output D:\\ModularTranslator_cpu
  .venv\\Scripts\\python.exe build/build_cpu.py --verify --smoke  # verify after build + startup smoke test
  .venv\\Scripts\\python.exe build/build_cpu.py --no-engine --output-dir D:\\MT-cpu

CPU build contents: torch-cpu runtime + llama-release CPU subset + FFmpeg
(ffmpeg/ffprobe); models and character assets stay external (multi-volume
archives made by tools/make_model_volumes.py).
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
