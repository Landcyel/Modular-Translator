"""受控实验：同一 CUDA 基准在不同启动/句柄方式下的耗时差异。

模拟 exe 被不同方式启动时的进程环境差异：
- 直接运行（继承调用方句柄/控制台）
- 重定向 stdout/stderr（nul）
- 独立会话

用法（分别执行后对比耗时）:
    python tools/bench_cuda_handle.py
"""
import os
import sys
import time

import torch

print(f"[env] stderr={sys.stderr!r} stdout={sys.stdout!r} cwd={os.getcwd()}")
print(f"[torch] {torch.__version__} threads={torch.get_num_threads()}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {dev}: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'n/a'}")

# 模拟 MOSS 推理的典型张量形状（bf16 大矩阵乘 + attention 形状）
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)

# 预热（CUDA 上下文 + cuBLAS workspace 建立）
for _ in range(3):
    c = a @ b
if dev == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
N = 30
for _ in range(N):
    c = a @ b
if dev == "cuda":
    torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"[result] matmul 4096x4096 x{N}: {dt:.3f}s = {dt / N * 1000:.1f}ms/次")

# 模拟自回归 decode 的逐 token 小 GEMM（batch=1）
x = torch.randn(1, 1, 4096, device=dev, dtype=torch.bfloat16)
w = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
t0 = time.perf_counter()
for _ in range(500):
    y = x @ w
if dev == "cuda":
    torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"[result] decode GEMM 1x4096x4096 x500: {dt:.3f}s = {dt / 500 * 1000:.3f}ms/次")
