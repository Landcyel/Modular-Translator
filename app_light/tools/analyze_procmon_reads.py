"""ProcMon CSV：对比 fast/slow 的 ReadFile 目标分布。"""
import csv
import sys
from collections import Counter

def read_counts(path, op):
    c = Counter()
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            proc, o, target = [x.strip() for x in (row[1], row[3], row[4])]
            if proc.lower() == "modulartranslator.exe" and o == op and target:
                c[target] += 1
    return c

fast_r = read_counts(r"C:\Users\Cyel\Desktop\fast_Log.CSV", "ReadFile")
slow_r = read_counts(r"C:\Users\Cyel\Desktop\slow_Log.CSV", "ReadFile")

print("== ReadFile 目标 top 30（slow）==")
for path, n in slow_r.most_common(30):
    f = fast_r.get(path, 0)
    print(f"  slow={n:6d} fast={f:6d}  {path}")

print("\n== slow 比 fast 多读 >500 次的目标 ==")
for path, n in slow_r.most_common(80):
    f = fast_r.get(path, 0)
    if n - f > 500:
        print(f"  slow={n:6d} fast={f:6d}  diff={n-f:6d}  {path}")

print("\n== fast 比 slow 多读 >500 次的目标 ==")
for path, n in fast_r.most_common(80):
    s = slow_r.get(path, 0)
    if n - s > 500:
        print(f"  fast={n:6d} slow={s:6d}  diff={n-s:6d}  {path}")
