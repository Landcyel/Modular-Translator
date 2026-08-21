"""GSV 双参考（dual）合成冒烟测试 — est_gealach_arnotts 微调角色 + normal 角色参考 + audioA_01 情绪参考。

验证内容:
  1. 角色微调权重确实被加载（characters/est_gealach_arnotts[v2ProPlus]/
     est_gealach_arnotts-e32.ckpt + est_gealach_arnotts_e16_s1344.pth）
  2. 角色参考音频使用 [normal]（音色锚定）
  3. 情绪参考音频 = test_data/audioA_01.wav
  4. 参考文本与目标文本 = test_data/audioA_01.txt 内容

走与 APP 完全相同的代码路径:
    GsvService(config_path) → start() → GsvTTSExecutor.execute(task)（ref_mode=dual）

用法（须在项目根目录、CUDA 运行时环境执行）::

    dependencies/venv/Scripts/python tools/test_gsv_dual_ref.py [--out 输出路径]

退出码: 0=成功, 1=失败。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROLE_NAME = "est_gealach_arnotts[v2ProPlus]"
ROLE_DIR = ROOT / "characters" / ROLE_NAME
CONFIG_PATH = ROOT / "configs" / "models" / "gsv-est-gealach-arnotts.json"
EMOTION_REF = ROOT / "test_data" / "audioA_01.wav"
TEXT_FILE = ROOT / "test_data" / "audioA_01.txt"


def find_normal_ref() -> Path:
    """角色目录下选择 [normal] 参考音频（glob 的 [] 是字符集，须过滤匹配）。"""
    wavs = sorted(p for p in ROLE_DIR.glob("*.wav") if p.name.startswith("[normal]"))
    if not wavs:
        raise FileNotFoundError(f"角色目录下未找到 [normal]*.wav: {ROLE_DIR}")
    exact = next((p for p in wavs if p.name.startswith("[normal].")), wavs[0])
    return exact


def verify_loaded_weights(engine) -> None:
    """打印引擎实际加载的权重路径（vendored TTS_Config 属性，含回退判定后结果）。"""
    tts = getattr(engine, "_tts", None)
    cfg = getattr(tts, "configs", None)
    if cfg is None:
        print("  [warn] 无法访问 _tts.configs，跳过实际加载权重核验")
        return
    print("  S1 实际加载: {}".format(getattr(cfg, "t2s_weights_path", "?")))
    print("  S2 实际加载: {}".format(getattr(cfg, "vits_weights_path", "?")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GSV dual 双参考合成冒烟测试")
    ap.add_argument("--out", default=str(ROOT / "output" / "gsv" / "test_est_normal.wav"))
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="S1 repetition_penalty。注意：目标文本与参考文本相同时须用 1.0，"
                         "默认 1.35 会压制与参考重复的语义 token 导致提前 EOS、后半段消失")
    args = ap.parse_args(argv)

    # ── 0. 输入核验 ────────────────────────────────────────────────
    for label, p in (("角色目录", ROLE_DIR), ("角色配置", Path(args.config)),
                     ("情绪参考", EMOTION_REF), ("文本文件", TEXT_FILE)):
        if not p.exists():
            print(f"[错误] {label}不存在: {p}")
            return 1
    role_ref = find_normal_ref()
    text = TEXT_FILE.read_text(encoding="utf-8").strip()
    print("=== GSV dual 双参考合成测试 ===")
    print(f"角色目录     : {ROLE_DIR}")
    print(f"角色参考音频 : {role_ref.name}")
    print(f"情绪参考音频 : {EMOTION_REF} ("
          f"{EMOTION_REF.stat().st_size/1024:.0f}KB)")
    print(f"参考/目标文本: {text!r}")

    import soundfile as sf

    for label, p in (("角色参考", role_ref), ("情绪参考", EMOTION_REF)):
        info = sf.info(str(p))
        print(f"{label}时长     : {info.frames/info.samplerate:.2f}s "
              f"(须 3~10s)")
        if not (3.0 <= info.frames / info.samplerate <= 10.0):
            print(f"[错误] {label}时长超出 3~10s 范围")
            return 1

    # ── 1. 引擎加载（APP 同款路径）────────────────────────────────
    from core.service import GsvService

    t0 = time.time()
    svc = GsvService(args.config)
    svc.start()
    print(f"引擎加载     : {time.time()-t0:.1f}s "
          f"device={svc.engine.device} version={svc.engine.version}")
    print("--- 权重核验（resolve_config 绝对路径）---")
    print("  配置 S1: {}".format(svc.engine.config["t2s_weights_path"]))
    print("  配置 S2: {}".format(svc.engine.config["vits_weights_path"]))
    verify_loaded_weights(svc.engine)

    # ── 2. 构造任务（dual: 情绪 → S1 语义, 角色 → S2 谱/SV 音色锚定）──
    from core.contracts import Task

    text_lang = "zh"
    prompt_lang = "zh"  # audioA_01.txt 为中文参考文本
    task = Task(
        task_type="gsv",
        file_path=str(TEXT_FILE),           # 执行器自动读取 .txt 内容为目标文本
        file_name="test_est_normal.txt",
        configs={"args": {
            "ref_mode": "dual",
            "ref_audio_path": str(EMOTION_REF),
            "role_ref_audio": str(role_ref),
            "prompt_text": text,
            "prompt_lang": prompt_lang,
            "text_lang": text_lang,
            "repetition_penalty": args.rep_penalty,
        }},
        id="est_normal_test",
    )

    # ── 3. 执行合成 ────────────────────────────────────────────────
    executor = svc.get_executor()
    n_frag = 0

    def _progress(done, total, *rest):
        nonlocal n_frag
        n_frag += 1
        print(f"  片段 {n_frag}/{total} ({done}/{total})", flush=True)

    t0 = time.time()
    print("--- 开始合成（dual 双参考）---")
    try:
        result = executor.execute(task, progress_callback=_progress)
    finally:
        svc.stop()  # 释放引擎（del 模型 + empty_cache）
    elapsed = time.time() - t0

    # ── 4. 结果核验 ────────────────────────────────────────────────
    print("--- 结果 ---")
    print(f"输出音频     : {result['audio_path']}")
    print(f"时长         : {result['duration']:.2f}s · "
          f"{result['sample_rate']}Hz · 片段 {result['info']['fragments']} · "
          f"seed {result['info']['seed']} · 合成耗时 {result['info']['elapsed_sec']}s")
    out_path = Path(result["audio_path"])
    if not out_path.is_file() or out_path.stat().st_size == 0:
        print("[错误] 输出音频缺失或为空")
        return 1
    print(f"输出文件     : {out_path.stat().st_size/1024:.0f}KB")
    print(f"总计         : 加载+合成 {elapsed:.1f}s")
    print("=== 测试通过 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
