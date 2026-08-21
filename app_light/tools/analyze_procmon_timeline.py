"""分析 slow/fast 中关键大文件的 ReadFile 时间分布（验证工作集修剪假说）。"""
import csv
from collections import Counter

TARGETS = [
    "cublasLt64_12.dll",
    "torch_cuda.dll",
    "torch_cpu.dll",
    "nvrtc64_120_0.dll",
    "llvmlite.dll",
    "model-00000-of-00001.safetensors",
    "Audio02.mp3",
]

def timeline(path):
    """返回 {目标关键词: [(时间, 次数)]}，按时间聚合到秒。"""
    buckets = {}
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            time, proc, op, target = [x.strip() for x in (row[0], row[1], row[3], row[4])]
            if proc.lower() != "modulartranslator.exe" or op != "ReadFile" or not target:
                continue
            for key in TARGETS:
                if key in target:
                    sec = time.split(".")[0]  # HH:MM:SS
                    buckets.setdefault(key, Counter())[sec] += 1
                    break
    return buckets

for label, path in (("FAST", r"C:\Users\Cyel\Desktop\fast_Log.CSV"),
                    ("SLOW", r"C:\Users\Cyel\Desktop\slow_Log.CSV")):
    print(f"\n===== {label} =====")
    buckets = timeline(path)
    for key in TARGETS:
        if key not in buckets:
            continue
        items = sorted(buckets[key].items())
        # 聚合为时间段（每 10 秒一个桶）
        from collections import defaultdict
        agg = defaultdict(int)
        for sec, n in items:
            hh, mm, ss = sec.split(":")
            bucket = int(ss) // 10 * 10
            agg[f"{hh}:{mm}:{bucket:02d}"] += n
        total = sum(buckets[key].values())
        peak = max(agg.items(), key=lambda kv: kv[1])
        print(f"{key}: 总读={total} 峰值时段={peak[0]}（{peak[1]}次） 时段分布=")
        for k in sorted(agg):
            bar = "#" * min(agg[k] // 50, 40)
            print(f"    {k} {agg[k]:6d} {bar}")
