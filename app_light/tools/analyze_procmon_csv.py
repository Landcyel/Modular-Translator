"""分析 Process Monitor CSV：对比快（cmd /c 启动）与慢（explorer 启动）实例。

对比项：
1. ModularTranslator.exe 的 Load Image 集合差异（DLL 加载差异）
2. Process Start 行（父进程/命令行/当前目录）
3. 关键阶段时间线（进程启动 → 模型加载 → 转写）
"""
import csv
import sys
from collections import Counter
from pathlib import Path


def load_events(path: str):
    events = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            time, proc, pid, op, target, result, detail = [c.strip() for c in row[:7]]
            events.append((time, proc, pid, op, target, result, detail))
    return events


def main():
    fast_path = r"C:\Users\Cyel\Desktop\fast_Log.CSV"
    slow_path = r"C:\Users\Cyel\Desktop\slow_Log.CSV"
    print("== 解析 fast ...", flush=True)
    fast = load_events(fast_path)
    print(f"   fast 事件数: {len(fast)}", flush=True)
    print("== 解析 slow ...", flush=True)
    slow = load_events(slow_path)
    print(f"   slow 事件数: {len(slow)}", flush=True)

    def extract(events, name):
        """按进程名提取 Load Image 路径与 Process Start。"""
        loads = set()
        starts = []
        for time, proc, pid, op, target, result, detail in events:
            if proc.lower() == name.lower():
                if op == "Load Image" and target:
                    loads.add(target)
                elif op == "Process Start":
                    starts.append((time, pid, detail))
        return loads, starts

    print("\n== 1. ModularTranslator.exe Load Image 差异 ==")
    fl, fs = extract(fast, "ModularTranslator.exe")
    sl, ss = extract(slow, "ModularTranslator.exe")
    only_fast = sorted(fl - sl)
    only_slow = sorted(sl - fl)
    print(f"fast 加载 {len(fl)} 个映像，slow 加载 {len(sl)} 个映像")
    print(f"\n-- 仅 fast 加载 ({len(only_fast)}): --")
    for p in only_fast:
        print("  ", p)
    print(f"\n-- 仅 slow 加载 ({len(only_slow)}): --")
    for p in only_slow:
        print("  ", p)

    print("\n== 2. Process Start（fast）==")
    for t, pid, d in fs:
        # 只打印关键字段（父PID/命令行/当前目录），跳过完整环境
        head = d[:400]
        print(f"  [{t}] pid={pid}: {head}...")
    print("\n== 2. Process Start（slow）==")
    for t, pid, d in ss:
        head = d[:400]
        print(f"  [{t}] pid={pid}: {head}...")

    print("\n== 3. 时间线（ModularTranslator.exe 关键事件）==")
    for label, events in (("FAST", fast), ("SLOW", slow)):
        print(f"-- {label} --")
        counts = Counter()
        first_load = None
        last_ts = None
        for time, proc, pid, op, target, result, detail in events:
            if proc.lower() != "modulartranslator.exe":
                continue
            counts[op] += 1
            if first_load is None and op == "Load Image":
                first_load = time
            last_ts = time
        print(f"   Process Start@{counts.get('Process Start', 0)} 首 Load Image@{first_load} 最后事件@{last_ts}")
        for op, n in counts.most_common(8):
            print(f"     {op}: {n}")


if __name__ == "__main__":
    main()
