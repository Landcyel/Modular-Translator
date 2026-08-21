"""GSV / MOSS CPU vs GPU 速度对比基准（短输入，单进程单引擎单设备）。

用法（venv 解释器）::

    dependencies/venv/Scripts/python tools/bench_gsv_moss_cpu_gpu.py gsv cuda
    dependencies/venv/Scripts/python tools/bench_gsv_moss_cpu_gpu.py moss cpu

输出：每段耗时 + 汇总 JSON 行（engine/device/load_sec/run_sec/...）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── GSV 固定输入：单句中文（≈28 字），dual 模式，ookura 角色 ──
GSV_TEXT = "你好，这是一段速度测试文本。我们来看一下合成需要多长时间。"
GSV_ROLE_CFG = ROOT / "configs" / "models" / "gsv-ookura-lumine.json"
GSV_EMO_REF = ROOT / "output" / "gsv-test" / "_refs" / "neutral_seg0005.wav"

# ── MOSS 固定输入：4.46s 短音频 ──
MOSS_WAV = ROOT / "test_data" / "audioA_01.wav"
MOSS_CFG = ROOT / "configs" / "models" / "moss.json"


def bench_gsv(device: str) -> int:
    from core.contracts import Task
    from core.service import GsvService

    cfg = json.loads(GSV_ROLE_CFG.read_text(encoding="utf-8"))
    cfg["device"] = device
    # 角色配置 defaults 提供 role_ref_audio/prompt_text/prompt_lang/text_lang，
    # 情绪参考随任务传入；seed 固定便于复现
    defaults = cfg.get("defaults", {})
    args = {
        "ref_mode": "dual",
        "ref_audio_path": str(GSV_EMO_REF),
        "role_ref_audio": defaults.get("role_ref_audio"),
        "prompt_text": defaults.get("prompt_text", ""),
        "prompt_lang": defaults.get("prompt_lang", "ja"),
        "text_lang": defaults.get("text_lang", "zh"),
        "top_k": 15, "top_p": 1.0, "temperature": 1.0,
        "repetition_penalty": 1.35, "speed_factor": 1.0,
        "text_split_method": "cut1", "seed": 42,
    }

    svc = GsvService(cfg)
    t0 = time.time()
    print(f"[gsv:{device}] 加载引擎 ...", flush=True)
    svc.start()
    load_sec = time.time() - t0
    print(f"[gsv:{device}] 引擎加载完成 {load_sec:.1f}s", flush=True)

    executor = svc.get_executor()

    def _run_once(tag: str) -> dict:
        task = Task(
            task_type="gsv",
            file_path=GSV_TEXT,
            file_name="bench_gsv.txt",
            configs={"args": args},
            id=f"bench-gsv-{tag}",
        )
        t0 = time.time()
        result = executor.execute(task)
        wall_sec = time.time() - t0
        info = result.get("info", {}) or {}
        return {
            "engine": "gsv", "device": device, "tag": tag, "version": info.get("version"),
            "load_sec": round(load_sec, 2), "execute_sec": round(wall_sec, 2),
            "synth_elapsed_sec": info.get("elapsed_sec"),
            "audio_duration": result.get("duration"),
            "text_chars": len(GSV_TEXT), "fragments": info.get("fragments"),
            "ref_mode": info.get("ref_mode"),
        }

    warm = _run_once("warm")   # 预热：含 CUDA 上下文/cudnn autotune 等一次性开销
    print("[gsv-warm] " + json.dumps(warm, ensure_ascii=False), flush=True)
    out = _run_once("steady")
    print("[gsv-result] " + json.dumps(out, ensure_ascii=False), flush=True)
    svc.stop()
    return 0


def bench_moss(device: str) -> int:
    from core.contracts import Task
    from core.moss.moss_service import MossService

    cfg = json.loads(MOSS_CFG.read_text(encoding="utf-8"))
    cfg["device"] = device
    print(f"[moss:{device}] 启动服务（模型懒加载）...", flush=True)
    svc = MossService(cfg)
    t0 = time.time()
    svc.start()
    start_sec = time.time() - t0
    print(f"[moss:{device}] start() 返回 {start_sec:.1f}s（实际加载在首次转写内）", flush=True)

    executor = svc.get_executor()
    progress = {}

    def on_progress(pos, total, speed=None, payload=None):
        if isinstance(payload, dict):
            progress.update(payload)

    def _run_once(tag: str) -> dict:
        task = Task(
            task_type="transcribe",
            file_path=MOSS_WAV,
            file_name="audioA_01.wav",
            configs={"args": {}},
            id=f"bench-moss-{tag}",
        )
        t0 = time.time()
        result = executor.execute(task, progress_callback=on_progress)
        wall_sec = time.time() - t0
        segs = result.get("segments") or []
        info = result.get("info", {}) or {}
        return {
            "engine": "moss", "device": device, "tag": tag,
            "start_sec": round(start_sec, 2), "execute_sec": round(wall_sec, 2),
            "audio_duration": 4.457234,
            "segments": len(segs),
            "speakers": sorted({s.get("speaker") for s in segs if s.get("speaker")}),
            "tokens": progress.get("generated_tokens"),
            "elapsed_sec": info.get("elapsed_sec"),
        }

    warm = _run_once("warm")    # 预热：含首次模型加载（MOSS 懒加载）
    print("[moss-warm] " + json.dumps(warm, ensure_ascii=False), flush=True)
    out = _run_once("steady")   # 稳态：模型已驻留，纯转写耗时
    print("[moss-result] " + json.dumps(out, ensure_ascii=False), flush=True)
    svc.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GSV/MOSS CPU vs GPU 基准")
    ap.add_argument("engine", choices=("gsv", "moss"))
    ap.add_argument("device", choices=("cuda", "cpu"))
    args = ap.parse_args()

    import torch
    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"threads={torch.get_num_threads()} devices={torch.cuda.device_count()}", flush=True)

    if args.engine == "gsv":
        if not GSV_EMO_REF.is_file():
            print(f"[error] 情绪参考缺失: {GSV_EMO_REF}")
            return 2
        return bench_gsv(args.device)
    return bench_moss(args.device)


if __name__ == "__main__":
    sys.exit(main())
