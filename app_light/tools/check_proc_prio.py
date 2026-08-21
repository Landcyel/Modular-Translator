"""对比两种启动方式的 IoPriority（NtQueryInformationProcess ProcessIoPriority=0x21）。"""
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys
import time

k = ctypes.WinDLL("kernel32", use_last_error=True)
nt = ctypes.WinDLL("ntdll", use_last_error=True)
k.GetProcessId.restype = ctypes.c_ulong

PY = r"D:\ReasonixProjects\TestFletApp\Reasonix_code\.venv\Scripts\python.exe"
SLEEP = "import time; time.sleep(300)"


def probe(pid, label):
    h = k.OpenProcess(0x0400, False, pid)
    if not h:
        print(f"{label} pid={pid}: OpenProcess失败 err={ctypes.get_last_error()}")
        return
    # NtQueryInformationProcess ProcessIoPriority = 0x21
    val = ctypes.c_uint(0)
    ret = nt.NtQueryInformationProcess(h, 0x21, ctypes.byref(val), 4, None)
    if ret == 0:
        print(f"{label} pid={pid}: IoPriority={val.value}（正常=5，后台=1-2）")
    else:
        print(f"{label} pid={pid}: NtQuery IoPriority err=0x{ret & 0xFFFFFFFF:x}")
    # 也查 ProcessPriorityClass(0x1B?) 与 CPU 亲和性
    mask = ctypes.c_size_t(0)
    sysmask = ctypes.c_size_t(0)
    k.GetProcessAffinityMask(h, ctypes.byref(mask), ctypes.byref(sysmask))
    print(f"   亲和性: 0x{mask.value:x}（{bin(mask.value).count('1')} 核）")
    k.CloseHandle(h)


def shell_execute(file, params):
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)

    class SEI(ctypes.Structure):
        _fields_ = [
            ("cbSize", wt.DWORD), ("fMask", wt.ULONG), ("hwnd", wt.HANDLE),
            ("lpVerb", wt.LPCWSTR), ("lpFile", wt.LPCWSTR), ("lpParameters", wt.LPCWSTR),
            ("lpDirectory", wt.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wt.HINSTANCE),
            ("lpIDList", ctypes.c_void_p), ("lpClass", wt.LPCWSTR), ("hkeyClass", wt.HANDLE),
            ("dwHotKey", wt.DWORD), ("hIcon", wt.HANDLE), ("hProcess", wt.HANDLE),
        ]

    sei = SEI()
    sei.cbSize = ctypes.sizeof(sei)
    sei.fMask = 0x40
    sei.lpVerb = "open"
    sei.lpFile = file
    sei.lpParameters = params
    sei.lpDirectory = r"D:\ModularTranslator_cuda"
    sei.nShow = 1
    ok = shell32.ShellExecuteExW(ctypes.byref(sei))
    if not ok:
        print("ShellExecuteEx 失败:", ctypes.get_last_error())
        return None
    return k.GetProcessId(sei.hProcess)


p2 = subprocess.Popen([PY, "-c", SLEEP])
time.sleep(2)
probe(p2.pid, "CreateProcess(模拟cmd/start)")

pid1 = shell_execute(PY, f'-c "{SLEEP}"')
time.sleep(3)
if pid1:
    probe(pid1, "ShellExecuteEx(模拟双击)")
