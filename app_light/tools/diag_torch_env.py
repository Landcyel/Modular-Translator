"""诊断 torch 运行时环境（dev/app 共用同一加载机制，结果代表两者）。

在 import torch 之前先激活 app.torch_runtime（与 APP.py 同顺序），
打印与 GPU 利用率/速度相关的关键状态：运行时选择、线程数、cuDNN、
SDPA kernel 开关、显存探测值（_resolve_window_sec 依赖 mem_get_info）。

用法::
    python tools/diag_torch_env.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.torch_runtime import setup, describe  # noqa: E402  必须在 import torch 之前

kind = setup()
print(f"[runtime] kind={kind} describe={describe()}")

import torch  # noqa: E402

print(f"[torch] version={torch.__version__} cuda_built={torch.version.cuda}")

# 线程（CPU 侧瓶颈排查：冻结下 OMP 异常会压成 1 线程）
print(f"[threads] intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
print(f"[omp] KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')} "
      f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

# CUDA 可用性 / 设备
avail = torch.cuda.is_available()
print(f"[cuda] is_available={avail}")
if avail:
    print(f"[cuda] device={torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"[cuda] vram free={free / 2 ** 30:.1f}GiB total={total / 2 ** 30:.1f}GiB "
          f"(free*0.85={free * 0.85 / 2 ** 30:.1f}GiB)")

# cuDNN（kernel 降级排查）
print(f"[cudnn] is_available={torch.backends.cudnn.is_available()} "
      f"version={torch.backends.cudnn.version()} benchmark={torch.backends.cudnn.benchmark}")

# SDPA kernel 开关（回退 math 会慢且利用率低）
print(f"[sdp] flash={torch.backends.cuda.flash_sdp_enabled()} "
      f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
      f"math={torch.backends.cuda.math_sdp_enabled()}")

# 与 MOSS 相同的 autocast dtype 验证
print(f"[amp] cuda bf16 supported={torch.cuda.is_bf16_supported() if avail else 'N/A'}")
