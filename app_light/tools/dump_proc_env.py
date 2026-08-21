"""读取指定进程的环境块（PEB 解析，普通权限可读同用户进程）。

用法::
    python tools/dump_proc_env.py <pid> [--save <file>]

输出进程环境变量（按名称排序）与进程树（父进程链）。
用于对比"快实例（cmd /c 启动）"与"慢实例（explorer 双击）"的环境差异。
"""
import ctypes
import ctypes.wintypes as wt
import sys
from pathlib import Path

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


class PBI(ctypes.Structure):
    """PROCESS_BASIC_INFORMATION（64 位布局，ctypes 自动对齐）。"""
    _fields_ = [
        ("ExitStatus", wt.LONG),
        ("PebBaseAddress", ctypes.c_void_p),
        ("AffinityMask", ctypes.c_void_p),
        ("BasePriority", wt.LONG),
        ("UniqueProcessId", ctypes.c_void_p),
        ("InheritedFromUniqueProcessId", ctypes.c_void_p),
    ]


class UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", wt.USHORT), ("MaximumLength", wt.USHORT),
                ("Buffer", ctypes.c_void_p)]


class RTL_USER_PROCESS_PARAMETERS(ctypes.Structure):
    _fields_ = [
        ("Reserved1", ctypes.c_ubyte * 16),
        ("Reserved2", ctypes.c_void_p * 10),
        ("ImagePathName", UNICODE_STRING),
        ("CommandLine", UNICODE_STRING),
        ("Environment", ctypes.c_void_p),
    ]


def read_mem(proc, addr, size):
    buf = ctypes.create_string_buffer(size)
    read = ctypes.c_size_t(0)
    if not kernel32.ReadProcessMemory(proc, ctypes.c_void_p(addr), buf, size, ctypes.byref(read)):
        return None
    return buf.raw[:read.value]


def read_wide(proc, us: UNICODE_STRING):
    if not us.Buffer or not us.MaximumLength:
        return ""
    data = read_mem(proc, us.Buffer, us.MaximumLength)
    if not data:
        return ""
    return data.decode("utf-16-le", errors="replace").split("\x00", 1)[0]


def get_process_params(pid):
    proc = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not proc:
        return None, f"OpenProcess 失败 (error={ctypes.get_last_error()})"
    try:
        pbi = PBI()
        size = ctypes.c_ulong(0)
        ret = ntdll.NtQueryInformationProcess(
            proc, 0, ctypes.byref(pbi), ctypes.sizeof(pbi), ctypes.byref(size))
        if ret != 0:
            return None, f"NtQueryInformationProcess 失败 (0x{ret & 0xFFFFFFFF:x})"
        peb_raw = read_mem(proc, pbi.PebBaseAddress, 0x400)
        if not peb_raw:
            return None, "读取 PEB 失败"
        # PEB.ProcessParameters 偏移 0x20（64 位）
        params_addr = int.from_bytes(peb_raw[0x20:0x28], "little")
        params_raw = read_mem(proc, params_addr, 0x400)
        if not params_raw:
            return None, "读取 ProcessParameters 失败"
        # Environment 偏移 0x80（64 位）
        env_addr = int.from_bytes(params_raw[0x80:0x88], "little")
        cmd_us = UNICODE_STRING()
        cmd_us.Length = int.from_bytes(params_raw[0x48:0x4A], "little")
        cmd_us.MaximumLength = int.from_bytes(params_raw[0x4A:0x4C], "little")
        cmd_us.Buffer = int.from_bytes(params_raw[0x50:0x58], "little")
        img_us = UNICODE_STRING()
        img_us.Length = int.from_bytes(params_raw[0x38:0x3A], "little")
        img_us.MaximumLength = int.from_bytes(params_raw[0x3A:0x3C], "little")
        img_us.Buffer = int.from_bytes(params_raw[0x40:0x48], "little")
        env_raw = read_mem(proc, env_addr, 65536) or b""
        envs = {}
        for part in env_raw.decode("utf-16-le", errors="replace").split("\x00"):
            if "=" in part:
                k, v = part.split("=", 1)
                envs[k] = v
        return {
            "env": envs,
            "cmdline": read_wide(proc, cmd_us),
            "image": read_wide(proc, img_us),
            "ppid": pbi.InheritedFromUniqueProcessId,
        }, None
    finally:
        kernel32.CloseHandle(proc)


def main():
    pid = int(sys.argv[1])
    save = None
    if "--save" in sys.argv:
        save = sys.argv[sys.argv.index("--save") + 1]
    data, err = get_process_params(pid)
    if err:
        print(f"[error] {err}")
        sys.exit(1)
    lines = [f"PID={pid}", f"PPID={data['ppid']}", f"Image={data['image']}",
             f"CommandLine={data['cmdline']}", ""]
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
             f"$t=@(); while($p){{ $t+=$p.Name+'('+$p.ProcessId+')'; "
             f"$p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$p.ParentProcessId) }}; "
             f"$t -join ' <- '"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        lines.append(f"ProcessTree={out}")
    except Exception as e:
        lines.append(f"ProcessTree=<err {e}>")
    lines.append("")
    for k in sorted(data["env"]):
        lines.append(f"{k}={data['env'][k]}")
    text = "\n".join(lines)
    if save:
        Path(save).write_text(text, encoding="utf-8")
        print(f"saved -> {save}")
    else:
        print(text)


import subprocess  # noqa: E402

if __name__ == "__main__":
    main()
