"""解析 PML 调用栈：定位 slow/fast 中 nvrtc / cudnn runtime compiled / torch_cuda 读取的调用者。"""
import sys
from collections import Counter

from procmon_parser import read_all_events_from_pml

KEYWORDS = ("nvrtc64_120_0.dll", "cudnn_engines_runtime_compiled64_9.dll",
            "torch_cuda.dll", "cublasLt64_12.dll", "llvmlite.dll", "nvcuda64.dll", "nvcuda.dll")

def analyze(path, label):
    print(f"===== {label}: {path} =====", flush=True)
    events = read_all_events_from_pml(path)
    print(f"事件总数: {len(events)}", flush=True)
    # 目标事件收集
    targets = {k: [] for k in KEYWORDS}
    for ev in events:
        proc = getattr(ev.process, "process_name", "") or ""
        if proc.lower() != "modulartranslator.exe":
            continue
        op = ev.operation or ""
        p = ev.path or ""
        if op != "ReadFile":
            continue
        for kw in KEYWORDS:
            if kw.lower() in p.lower():
                targets[kw].append(ev)
                break
    for kw in KEYWORDS:
        evs = targets[kw]
        print(f"\n-- {kw}: {len(evs)} 个 ReadFile 事件 --", flush=True)
        if not evs:
            continue
        # 栈帧模块序列统计（取前 8 帧）
        stack_mods = Counter()
        sample = None
        for ev in evs:
            st = getattr(ev, "stacktrace", None)
            if st:
                frames = []
                for fr in st:
                    mod = getattr(fr, "module", None)
                    if mod is None and isinstance(fr, (tuple, list)):
                        mod = fr[0] if fr else None
                    name = getattr(mod, "name", None) or (mod if isinstance(mod, str) else str(mod))
                    frames.append(name)
                key = " <- ".join(frames[:8])
                stack_mods[key] += 1
                if sample is None:
                    sample = frames
        print(f"  不同调用栈数: {len(stack_mods)}")
        for s, n in stack_mods.most_common(12):
            print(f"    [{n}] {s}")
        if sample:
            print(f"  示例栈帧({len(sample)}): {sample[:10]}")
        ev0 = evs[0]
        print(f"  示例事件: op={ev0.operation} path={ev0.path[:100]} result={ev0.result} tid={ev0.tid}")
        st0 = getattr(ev0, "stacktrace", None)
        print(f"  stacktrace 类型: {type(st0)} 长度: {len(st0) if st0 else 0}")
        if st0:
            fr0 = st0[0]
            print(f"  帧0 类型: {type(fr0)} 内容: {fr0}")
    return targets

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Cyel\Desktop\slow_Log.PML"
    label = sys.argv[2] if len(sys.argv) > 2 else "SLOW"
    analyze(path, label)
