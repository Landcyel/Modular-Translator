"""一次性冒烟：MOSS eager 加载（lazy_load=false）验证。"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.moss.moss_service import MossService  # noqa: E402


def main() -> int:
    cfg = json.loads((ROOT / "configs" / "models" / "moss.json").read_text(encoding="utf-8"))
    print(f"[smoke] lazy_load = {cfg.get('lazy_load')}", flush=True)
    svc = MossService(cfg)
    t0 = time.time()
    svc.start()
    print(f"[smoke] start() 完成 {time.time() - t0:.1f}s", flush=True)
    print(f"[smoke] is_loaded = {svc._runner.is_loaded}", flush=True)
    print(f"[smoke] runner._device = {svc._runner._device} | service.device = {svc.device}", flush=True)
    assert svc._runner.is_loaded, "eager load 未生效"
    assert svc.device in ("cuda", "cpu"), f"device 异常: {svc.device}"
    svc.stop()
    print("[smoke] SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
