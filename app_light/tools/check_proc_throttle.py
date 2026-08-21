"""查询进程的 PowerThrottling（ExecutionSpeed 节流）状态 + CPU 频率。

用法::
    python tools/check_proc_throttle.py <pid>

输出：
- PROCESS_POWER_THROTTLING_STATE（ControlMask/StateMask）
- StateMask & 0x1 = ExecutionSpeed 节流生效（CPU 限频）
- 进程 CPU 亲和性/优先级（对照）
"""
import ctypes
import ctypes.wintypes as wt
import sys

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

PROCESS_QUERY_INFORMATION = 0x0400
ProcessPowerThrottling = 0x4D  # PROCESSINFOCLASS 枚举值
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4


class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [
        ("Version", wt.ULONG),
        ("ControlMask", wt.ULONG),
        ("StateMask", wt.ULONG),
    ]


def main():
    pid = int(sys.argv[1])
    proc = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
    if not proc:
        print(f"OpenProcess 失败 error={ctypes.get_last_error()}")
        sys.exit(1)
    try:
        state = PROCESS_POWER_THROTTLING_STATE()
        size = ctypes.c_ulong(ctypes.sizeof(state))
        ok = kernel32.GetProcessInformation(
            proc, ProcessPowerThrottling, ctypes.byref(state), size)
        if not ok:
            print(f"GetProcessInformation(ProcessPowerThrottling) 失败 error={ctypes.get_last_error()}")
        else:
            print(f"PID {pid}: Version={state.Version} ControlMask=0x{state.ControlMask:x} "
                  f"StateMask=0x{state.StateMask:x}")
            if state.StateMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED:
                print("  >>> ExecutionSpeed 节流已启用（CPU 限频中）<<<")
            else:
                print("  无 ExecutionSpeed 节流")
            if state.StateMask & PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION:
                print("  忽略定时器分辨率（已设）")
        # 亲和性/优先级对照
        mask = ctypes.c_size_t(0)
        kernel32.GetProcessAffinityMask(proc, ctypes.byref(mask), ctypes.c_size_t())
        print(f"CPU 亲和性掩码: 0x{mask.value:x}（{bin(mask.value).count('1')} 核）")
        priority = kernel32.GetPriorityClass(proc)
        names = {0x100: "IDLE", 0x4000: "BELOW_NORMAL", 0x20: "NORMAL",
                 0x8000: "ABOVE_NORMAL", 0x80: "HIGH", 0x10000: "REALTIME"}
        print(f"优先级类: {names.get(priority, hex(priority))}")
    finally:
        kernel32.CloseHandle(proc)


if __name__ == "__main__":
    main()
