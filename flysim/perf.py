"""Process-level performance settings."""

import sys


def tune_gil_switching():
    """Several threads share the GIL (physics, brain engine, streaming). With
    the default 5 ms switch interval, a thread that only needs the GIL briefly
    -- e.g. the GPU brain engine between kernel launches -- can wait up to 5 ms
    each time; 0.5 ms keeps hand-offs quick."""
    sys.setswitchinterval(0.0005)


def disable_windows_power_throttling():
    """Windows 11 schedules background processes on efficiency cores, which runs
    the physics and brain ~4x slower on hybrid CPUs. Opt this process out of it."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        ]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
    ]
    PROCESS_POWER_THROTTLING = 4
    EXECUTION_SPEED = 0x1
    state = PowerThrottlingState(1, EXECUTION_SPEED, 0)
    kernel32.SetProcessInformation(
        kernel32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
        ctypes.byref(state), ctypes.sizeof(state),
    )
