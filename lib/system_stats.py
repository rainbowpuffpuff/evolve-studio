"""Lightweight CPU / RAM stats without requiring psutil."""
from __future__ import annotations

import os
import time
from typing import Any


_last_cpu: tuple[float, float] | None = None  # (idle, total)


def _read_proc_stat() -> tuple[float, float]:
    with open("/proc/stat", "r", encoding="utf-8") as f:
        line = f.readline()
    parts = line.split()
    # cpu user nice system idle iowait irq softirq steal ...
    nums = [float(x) for x in parts[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0.0)
    total = sum(nums)
    return idle, total


def cpu_percent() -> float:
    global _last_cpu
    idle, total = _read_proc_stat()
    if _last_cpu is None:
        _last_cpu = (idle, total)
        time.sleep(0.05)
        idle2, total2 = _read_proc_stat()
    else:
        idle2, total2 = idle, total
        idle, total = _last_cpu
        _last_cpu = (idle2, total2)
    didle = idle2 - idle
    dtotal = total2 - total
    if dtotal <= 0:
        return 0.0
    return round(max(0.0, min(100.0, (1.0 - didle / dtotal) * 100.0)), 1)


def mem_info() -> dict[str, Any]:
    info: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            # values in kB
            info[k.strip()] = int(v.strip().split()[0]) * 1024
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", info.get("MemFree", 0))
    used = max(0, total - available)
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": used,
        "used_pct": round(100.0 * used / total, 1) if total else 0.0,
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "available_gb": round(available / (1024**3), 2),
    }


def load_avg() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except Exception:
        return [0.0, 0.0, 0.0]


def snapshot() -> dict[str, Any]:
    return {
        "cpu_percent": cpu_percent(),
        "load_avg": load_avg(),
        "memory": mem_info(),
        "cpu_count": os.cpu_count() or 1,
        "pid": os.getpid(),
    }
