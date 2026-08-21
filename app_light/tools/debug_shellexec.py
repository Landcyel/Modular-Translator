"""Debug: ShellExecuteEx 跑 numpy 负载，捕获 stderr。"""
import ctypes
import ctypes.wintypes as wt
import os
import time

shell32 = ctypes.WinDLL("shell32", use_last_error=True)


class SEI(ctypes.Structure):
    _fields_ = [
        ("cbSize", wt.DWORD), ("fMask", wt.ULONG), ("hwnd", wt.HANDLE),
        ("lpVerb", wt.LPCWSTR), ("lpFile", wt.LPCWSTR), ("lpParameters", wt.LPCWSTR),
        ("lpDirectory", wt.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wt.HINSTANCE),
        ("lpIDList", ctypes.c_void_p), ("lpClass", wt.LPCWSTR), ("hkeyClass", wt.HANDLE),
        ("dwHotKey", wt.DWORD), ("hIcon", wt.HANDLE), ("hProcess", wt.HANDLE),
    ]


py = r"D:\ReasonixProjects\TestFletApp\Reasonix_code\.venv\Scripts\python.exe"
out = r"C:\Users\Cyel\AppData\Local\Temp\load_result.txt"
errf = r"C:\Users\Cyel\AppData\Local\Temp\load_err.txt"
loader = r"C:\Users\Cyel\AppData\Local\Temp\load_task.py"
with open(loader, "w") as f:
    f.write(
        "import sys, traceback\n"
        "try:\n"
        "    import time, numpy as np\n"
        "    t0=time.perf_counter()\n"
        "    a=np.random.rand(2048,2048).astype(np.float64)\n"
        "    [a@a for _ in range(15)]\n"
        "    dt=time.perf_counter()-t0\n"
        f"    open({out!r},'a').write(f'{{dt:.3f}}s\\n')\n"
        "except Exception:\n"
        f"    open({errf!r},'w').write(traceback.format_exc())\n"
    )

with open(out, "w") as f:
    f.write("")

sei = SEI()
sei.cbSize = ctypes.sizeof(sei)
sei.fMask = 0x40
sei.lpVerb = "open"
sei.lpFile = py
sei.lpParameters = f'"{loader}"'
sei.lpDirectory = r"D:\ModularTranslator_cuda"
sei.nShow = 1
ok = shell32.ShellExecuteExW(ctypes.byref(sei))
print("ShellExecuteEx ok:", bool(ok), "err:", ctypes.get_last_error())
time.sleep(20)
print("结果:", open(out).read() if os.path.exists(out) else "<无>")
print("错误:", open(errf).read()[:500] if os.path.exists(errf) else "<无>")
