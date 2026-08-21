"""对照实验 v6：模拟 app —— torch_runtime 激活 + 长序列逐 token 推理。

ShellExecuteEx vs CreateProcess 各 2 次（先预热一次再计时，消除冷缓存）。
"""
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time

RESULT = r"C:\Users\Cyel\AppData\Local\Temp\infer_result.txt"
LOAD_PY = r"C:\Users\Cyel\AppData\Local\Temp\infer_task.py"

with open(LOAD_PY, "w") as f:
    f.write(
        "import sys, time, traceback\n"
        "sys.path.insert(0, r'D:\\ReasonixProjects\\TestFletApp\\Reasonix_code')\n"
        "try:\n"
        "    from app.torch_runtime import setup\n"
        "    kind = setup()\n"
        "    import torch\n"
        "    torch.manual_seed(0)\n"
        "    # 预热（CUDA 上下文 + cuBLAS workspace）\n"
        "    a=torch.randn(1024,1024,device='cuda',dtype=torch.bfloat16)\n"
        "    _=a@a; torch.cuda.synchronize()\n"
        "    # 长序列逐 token 推理模拟（batch=1 自回归）\n"
        "    ctx=torch.randn(1,4096,512,device='cuda',dtype=torch.bfloat16)\n"
        "    w=torch.randn(512,512,device='cuda',dtype=torch.bfloat16)\n"
        "    t0=time.perf_counter()\n"
        "    for i in range(2000):\n"
        "        ctx=ctx@w\n"
        "        if i % 100 == 0:\n"
        "            torch.cuda.synchronize()\n"
        "    torch.cuda.synchronize()\n"
        "    dt=time.perf_counter()-t0\n"
        f"    open({RESULT!r},'a').write(f'{{dt:.3f}}s\\n')\n"
        "except Exception:\n"
        f"    open({RESULT!r},'a').write('ERR '+traceback.format_exc().splitlines()[-1]+'\\n')\n"
    )

py = r"D:\ReasonixProjects\TestFletApp\Reasonix_code\dependencies\venv\Scripts\python.exe"

shell32 = ctypes.WinDLL("shell32", use_last_error=True)


class SEI(ctypes.Structure):
    _fields_ = [
        ("cbSize", wt.DWORD), ("fMask", wt.ULONG), ("hwnd", wt.HANDLE),
        ("lpVerb", wt.LPCWSTR), ("lpFile", wt.LPCWSTR), ("lpParameters", wt.LPCWSTR),
        ("lpDirectory", wt.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wt.HINSTANCE),
        ("lpIDList", ctypes.c_void_p), ("lpClass", wt.LPCWSTR), ("hkeyClass", wt.HANDLE),
        ("dwHotKey", wt.DWORD), ("hIcon", wt.HANDLE), ("hProcess", wt.HANDLE),
    ]


def shell_execute(file, params):
    sei = SEI()
    sei.cbSize = ctypes.sizeof(sei)
    sei.fMask = 0x40
    sei.lpVerb = "open"
    sei.lpFile = file
    sei.lpParameters = params
    sei.lpDirectory = r"D:\ModularTranslator_cuda"
    sei.nShow = 1
    if not shell32.ShellExecuteExW(ctypes.byref(sei)):
        print("ShellExecuteEx 失败:", ctypes.get_last_error())


with open(RESULT, "w") as f:
    f.write("")

print("== 方式A: ShellExecuteEx（先预热1次，再计时2次）==", flush=True)
shell_execute(py, f'"{LOAD_PY}"')
time.sleep(25)
shell_execute(py, f'"{LOAD_PY}"')
time.sleep(25)
shell_execute(py, f'"{LOAD_PY}"')
time.sleep(25)

print("== 方式B: CreateProcess（先预热1次，再计时2次）==", flush=True)
subprocess.Popen([py, LOAD_PY])
time.sleep(25)
subprocess.Popen([py, LOAD_PY])
time.sleep(25)
subprocess.Popen([py, LOAD_PY])
time.sleep(25)

print("等待完成...", flush=True)
time.sleep(10)
with open(RESULT) as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"结果（{len(lines)} 行）：")
for i, l in enumerate(lines):
    print(f"  [{i}] {l}")
