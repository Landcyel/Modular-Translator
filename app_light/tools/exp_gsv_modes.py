"""实验：对比 dual/aux/single 各模式合成时长与 prompt 语义长度，定位"目标文本后半段消失"根因。

用法（venv CUDA 运行时）::

    dependencies/venv/Scripts/python tools/exp_gsv_modes.py

每个模式输出到 output/gsv/exp_<label>.wav，并打印:
    prompt_semantic 长度（S1 参考语义 token 数）· 合成时长 · 输出文件
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.service import GsvService  # noqa: E402

CONFIG = ROOT / "configs" / "models" / "gsv-est-gealach-arnotts.json"
ROLE_DIR = ROOT / "characters" / "est_gealach_arnotts[v2ProPlus]"
NORMAL_REF = next(
    p for p in sorted(ROLE_DIR.glob("*.wav")) if p.name.startswith("[normal]")
)
EMOTION_REF = ROOT / "test_data" / "audioA_01.wav"
EMO_TEXT = (ROOT / "test_data" / "audioA_01.txt").read_text(encoding="utf-8").strip()
ROLE_TEXT = "[normal]あなたを気に入ったことも理由の一つだし、助けてもらったお礼もあるから。"
OTHER_TEXT = "今天天气很好，我们一起去公园散步吧。"
HAPPY_REF = next(
    p for p in sorted(ROLE_DIR.glob("*.wav")) if p.name.startswith("[happy2]")
)
HAPPY_TEXT = "[happy2]ありがとう。今日は目覚めてからずっと気分がいいの"


def main() -> int:
    svc = GsvService(str(CONFIG))
    svc.start()
    engine = svc.engine
    tts = engine._tts
    print(f"engine={engine} 加载完成", flush=True)

    def run(label: str, primary_ref, fn) -> None:
        # 显式预置主参考，保证 prompt_semantic 为该参考音频的语义
        engine.set_ref_audio(str(primary_ref))
        t0 = time.time()
        chunks, sr = [], None
        for sr, frag in fn():
            chunks.append(frag)
        audio = np.concatenate(chunks)
        dur = len(audio) / sr
        out = ROOT / "output" / "gsv" / f"exp_{label}.wav"
        sf.write(str(out), audio, sr, subtype="PCM_16")
        ps = tts.prompt_cache["prompt_semantic"]
        print(
            f"[{label}] prompt_len={ps.shape[0] if ps is not None else 'N/A'} "
            f"dur={dur:.2f}s out={out.name} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    # 1. dual 同文本（复现问题）
    run("dual_same", EMOTION_REF, lambda: engine.synth_cross_speaker(
        EMO_TEXT, "zh", str(EMOTION_REF), EMO_TEXT, "zh", str(NORMAL_REF)))
    # 2. dual 换目标文本（测试"目标==参考"是否是诱因）
    run("dual_other", EMOTION_REF, lambda: engine.synth_cross_speaker(
        OTHER_TEXT, "zh", str(EMOTION_REF), EMO_TEXT, "zh", str(NORMAL_REF)))
    # 3. aux 同文本（GSV 原生多参考机制）
    run("aux_same", EMOTION_REF, lambda: engine.synth_stream(
        EMO_TEXT, "zh", str(EMOTION_REF), EMO_TEXT, "zh",
        aux_ref_audio_paths=[str(NORMAL_REF)]))
    # 4. single 角色日语参考（模型基线）
    run("single_role", NORMAL_REF, lambda: engine.synth_stream(
        ROLE_TEXT, "ja", str(NORMAL_REF), ROLE_TEXT, "ja"))
    # 5. single 中文情绪参考（无角色音频参与）
    run("single_emo", EMOTION_REF, lambda: engine.synth_stream(
        EMO_TEXT, "zh", str(EMOTION_REF), EMO_TEXT, "zh"))
    # 6. single 日语 prompt + 中文目标（区分"中文 prompt"与"中文目标"哪个是诱因）
    run("single_role_zh", NORMAL_REF, lambda: engine.synth_stream(
        OTHER_TEXT, "zh", str(NORMAL_REF), ROLE_TEXT, "ja"))
    # 7. dual 角色自身日语情绪参考（[happy2]）+ 中文目标（标准多参考用法）
    run("dual_role_emo", HAPPY_REF, lambda: engine.synth_cross_speaker(
        EMO_TEXT, "zh", str(HAPPY_REF), HAPPY_TEXT, "ja", str(NORMAL_REF)))
    # 8. 同文本 + repetition_penalty=1.0（验证 rep_penalty 是提前 EOS 的推手）
    run("emo_rp1", EMOTION_REF, lambda: engine.synth_stream(
        EMO_TEXT, "zh", str(EMOTION_REF), EMO_TEXT, "zh", repetition_penalty=1.0))
    # 9. 同文本 + top_k=100 + rp=1.0（放宽采样再验证）
    run("emo_tk100", EMOTION_REF, lambda: engine.synth_stream(
        EMO_TEXT, "zh", str(EMOTION_REF), EMO_TEXT, "zh",
        top_k=100, repetition_penalty=1.0))

    svc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
