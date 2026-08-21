"""快速转写 test_data 下的音频，验证 audioA_01.wav 内容是否与其 .txt 配对。

用法（主环境执行，MOSS CPU 推理）::

    python tools/exp_transcribe_wav.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from core.moss.moss_service import MossService  # noqa: E402

WAV = ROOT / "test_data" / "audioA_01.wav"
CFG = json.loads((ROOT / "configs" / "models" / "moss.json").read_text(encoding="utf-8"))


def main() -> int:
    svc = MossService(CFG)
    t0 = time.time()
    svc.start()
    print(f"模型加载 {time.time()-t0:.1f}s", flush=True)
    executor = svc.get_executor()

    from core.contracts import Task

    task = Task(
        task_type="transcribe",
        file_path=WAV,
        file_name="audioA_01.wav",
        configs={"args": {}},
        id="probe",
    )
    t0 = time.time()
    result = executor.execute(task)
    segs = result.get("segments") or []
    print(f"转写 {time.time()-t0:.1f}s, {len(segs)} 段:")
    for s in segs:
        print(f"  [{s.get('start', 0):6.2f} -> {s.get('end', 0):6.2f}] {s.get('text', '')}")
    svc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
