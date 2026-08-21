"""对照实验 v4：ShellExecuteEx（explorer 双击核心）vs CreateProcess（cmd/start 核心）。"""
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time

RESULT = r"C:\Users\Cyel\AppData\Local\Temp\load_result.txt"
LOAD_PY = r"C:\Users\Cyel\AppData\Local\Temp\load_task.py"

with open(LOAD_PY, "w") as f:
    f.write(
        "import sys, traceback\n"
        "try:\n"
        "    import time, numpy as np\n"
        "    t0=time.perf_counter()\n"
        "    a=np.random.rand(2048,2048).astype(np.float64)\n"
        "    [a@a for _ in range(15)]\n"
        "    dt=time.perf_counter()-t0\n"
        f"    open({RESULT!r},'a').write(f'{{dt:.3f}}s\\n')\n"
        "except Exception:\n"
        f"    open({os.path.splitext(RESULT)[0] + '_err.txt'!r},'w').write(traceback.format_exc())\n"
    )

py = r"D:\ReasonixProjects\TestFletApp\Reasonix_code\.venv\Scripts\python.exe"

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

print("== 方式A: ShellExecuteEx x3 ==", flush=True)
for i in range(3):
    shell_execute(py, f'"{LOAD_PY}"')
    time.sleep(6)

print("== 方式B: CreateProcess x3 ==", flush=True)
for i in range(3):
    subprocess.Popen([py, LOAD_PY])
    time.sleep(6)

print("等待完成...", flush=True)
time.sleep(25)
with open(RESULT) as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"结果（{len(lines)} 次）：")
for i, l in enumerate(lines):
    print(f"  [{i}] {l}")
if len(lines) >= 4:
    a = [float(x) for x in lines[:3]]
    b = [float(x) for x in lines[3:6]]
    if len(a) == 3 and len(b) == 3:
        print(f"方式A(ShellExecuteEx) 平均: {sum(a)/3:.3f}s")
        print(f"方式B(CreateProcess)  平均: {sum(b)/3:.3f}s")
        print(f"比值 A/B: {sum(a)/sum(b):.2f}")
