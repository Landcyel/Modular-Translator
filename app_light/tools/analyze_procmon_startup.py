"""提取启动阶段事件：Process Start → 第一个模型 safetensors 读取。"""
import csv

def startup_events(path, start_after="", stop_at="safetensors"):
    """返回 Process Start 后到首次读取模型文件前的事件（操作+目标，聚合计数）。"""
    from collections import Counter, defaultdict
    started = False
    op_target = Counter()
    ops = Counter()
    times = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            time, proc, pid, op, target, result = [x.strip() for x in (row[0], row[1], row[2], row[3], row[4], row[5])]
            if proc.lower() != "modulartranslator.exe":
                continue
            if op == "Process Start":
                started = True
                continue
            if not started:
                continue
            if stop_at in target and op == "ReadFile":
                times.append(time)
                break
            ops[op] += 1
            if target:
                key = f"{op}: {target}"
                op_target[key] += 1
    print(f"首读模型时间: {times[0] if times else 'N/A'}")
    print("操作分布:")
    for op, n in ops.most_common(15):
        print(f"  {op}: {n}")
    print("高频 (操作+目标) top 25:")
    for k, n in op_target.most_common(25):
        print(f"  {n:6d}  {k[:150]}")

print("======== SLOW 启动阶段（21:21:04 → 首读模型）========")
startup_events(r"C:\Users\Cyel\Desktop\slow_Log.CSV")
print("\n======== FAST 启动阶段（21:30:53 → 首读模型）========")
startup_events(r"C:\Users\Cyel\Desktop\fast_Log.CSV")
