"""CLI smoke-test entry: python -m core.gsv.cli synth ...

Example::

    .venv/Scripts/python -m core.gsv.cli synth \\
        --text "你好，这是一段测试。" --text-lang zh \\
        --ref "characters/ookura_lumine[v2ProPlus]/[normal1]助けてって言われても、服飾部門の受験者を集めてもらわないと、.wav" \\
        --prompt-text "[normal1]助けてって言われても、服飾部門の受験者を集めてもらわないと、" \\
        --prompt-lang ja --out output/gsv/cli.wav
"""

from __future__ import annotations

import argparse
import time


def _synth(args) -> int:
    from .engine import GsvEngine

    config = {
        "version": args.version,
        "device": args.device,
    }
    if args.s1:
        config["t2s_weights_path"] = args.s1
    if args.s2:
        config["vits_weights_path"] = args.s2

    t0 = time.time()
    engine = GsvEngine(config)
    print(f"[gsv] 引擎加载完成 ({time.time()-t0:.1f}s) version={engine.version} device={engine.device}")

    t0 = time.time()
    result = engine.synth_to_file(
        text=args.text,
        out_path=args.out,
        text_lang=args.text_lang,
        ref_audio_path=args.ref,
        prompt_text=args.prompt_text,
        prompt_lang=args.prompt_lang,
        speed_factor=args.speed,
        seed=args.seed,
    )
    print(
        f"[gsv] 合成完成 ({time.time()-t0:.1f}s): {result['audio_path']} "
        f"{result['sample_rate']}Hz {result['duration']:.2f}s"
    )
    engine.release()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="core.gsv.cli", description="GPT-SoVITS 推理引擎 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="文本合成语音")
    p.add_argument("--text", required=True, help="目标文本")
    p.add_argument("--text-lang", default="zh", help="目标语种 (zh/ja/en/yue/ko/auto)")
    p.add_argument("--ref", default=None, help="参考音频 wav 路径")
    p.add_argument("--prompt-text", default="", help="参考音频文本")
    p.add_argument("--prompt-lang", default="ja", help="参考音频语种")
    p.add_argument("--out", default="output/gsv/cli.wav", help="输出 wav 路径")
    p.add_argument("--version", default="v2ProPlus", help="模型版本 (v2ProPlus/v4/...)")
    p.add_argument("--device", default="auto", help="auto/cuda/cpu")
    p.add_argument("--s1", default=None, help="S1(GPT) 权重路径")
    p.add_argument("--s2", default=None, help="S2(SoVITS) 权重路径")
    p.add_argument("--speed", type=float, default=1.0, help="语速系数")
    p.add_argument("--seed", type=int, default=-1, help="随机种子 (-1=随机)")
    p.set_defaults(func=_synth)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
