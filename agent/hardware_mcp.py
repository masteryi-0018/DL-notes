#!/usr/bin/env python3
"""获取电脑硬件信息的 MCP Server（FastMCP 版，stdio 通信）。"""
import platform, subprocess
from typing import Any, Dict, List

import psutil
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hardware-info")


def sh(cmd: str) -> str:
    """执行 shell 命令，返回 stdout（失败/超时返回空串）。"""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


@mcp.tool()
def get_cpu_info() -> Dict[str, Any]:
    """获取 CPU 信息：型号、架构、核心数、频率。"""
    f = psutil.cpu_freq()
    # 当前频率：WSL2 下 psutil.current 准确
    cur_mhz = round(f.current, 1) if f and f.current else None
    # 最大频率：依次尝试 /proc/cpuinfo 的 cpu max MHz、psutil；
    #         WSL2 下这些常缺失或返回 0，统一归为 None，避免显示假的 0.0
    max_mhz = None
    max_raw = sh("grep -m1 'cpu max MHz' /proc/cpuinfo | cut -d: -f2").strip()
    if max_raw:
        try:
            max_mhz = round(float(max_raw), 1)
        except ValueError:
            pass
    if max_mhz is None and f and f.max:
        max_mhz = round(f.max, 1)
    return {
        "型号": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip() or platform.processor() or "未知",
        "架构": platform.machine(),
        "物理核心": psutil.cpu_count(False),
        "逻辑核心": psutil.cpu_count(True),
        "当前频率MHz": cur_mhz,
        "最大频率MHz": max_mhz,
    }


@mcp.tool()
def get_memory_info() -> Dict[str, Any]:
    """获取内存与 Swap 使用情况。"""
    m, s = psutil.virtual_memory(), psutil.swap_memory()
    return {
        "总内存GB": round(m.total / 1024**3, 2),
        "已用GB": round(m.used / 1024**3, 2),
        "可用GB": round(m.available / 1024**3, 2),
        "使用率%": m.percent,
        "Swap总GB": round(s.total / 1024**3, 2),
        "Swap使用率%": s.percent,
    }


@mcp.tool()
def get_disk_info() -> List[Dict[str, Any]]:
    """获取磁盘分区与容量信息。"""
    out = []
    for p in psutil.disk_partitions():
        try:
            u = psutil.disk_usage(p.mountpoint)
            out.append({
                "设备": p.device, "挂载点": p.mountpoint, "文件系统": p.fstype,
                "总GB": round(u.total / 1024**3, 2),
                "已用GB": round(u.used / 1024**3, 2),
                "使用率%": u.percent,
            })
        except PermissionError:
            continue
    return out


@mcp.tool()
def get_network_info() -> Dict[str, Any]:
    """获取网络接口与流量信息。"""
    ifaces, stats = {}, psutil.net_if_stats()
    for name, addrs in psutil.net_if_addrs().items():
        if name == "lo":
            continue
        d = {"状态": "UP" if stats.get(name) and stats[name].isup else "DOWN"}
        for a in addrs:
            if a.family.name == "AF_INET":
                d["IPv4"] = a.address
            elif a.family.name == "AF_INET6":
                d["IPv6"] = a.address
        ifaces[name] = d
    io = psutil.net_io_counters()
    return {"接口": ifaces, "累计发送MB": round(io.bytes_sent / 1024**2, 2),
            "累计接收MB": round(io.bytes_recv / 1024**2, 2)}


@mcp.tool()
def get_gpu_info() -> List[Dict[str, Any]]:
    """获取 GPU 显卡信息。"""
    gpus = []
    for line in sh("lspci | grep -i 'vga\\|3d\\|display'").splitlines():
        if ":" in line:
            gpus.append({"设备": line.split(":", 1)[1].strip()})
    smi = sh("nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits")
    for i, line in enumerate(smi.splitlines()):
        if line and i < len(gpus):
            name, mem, util = [x.strip() for x in line.split(",")]
            gpus[i].update({"显存MB": mem, "GPU使用率%": util})
    return gpus or [{"信息": "未检测到 GPU（WSL2 通常无 GPU 透传）"}]


@mcp.tool()
def get_system_info() -> Dict[str, Any]:
    """获取操作系统信息。"""
    return {
        "系统": platform.system(), "版本": platform.version(),
        "内核": platform.release(), "主机名": platform.node(),
        "Python": platform.python_version(), "开机时间": psutil.boot_time(),
    }


@mcp.tool()
def get_temperature() -> Dict[str, Any]:
    """获取硬件温度信息。"""
    try:
        return {n: [{"标签": e.label or "N/A", "当前℃": e.current,
                     "最高℃": e.high, "临界℃": e.critical} for e in entries]
                for n, entries in psutil.sensors_temperatures().items()}
    except Exception:
        return {"信息": "温度传感器不可用"}


@mcp.tool()
def get_all_hardware() -> Dict[str, Any]:
    """获取全部硬件信息（CPU/内存/磁盘/网络/GPU/系统/温度）。"""
    return {
        "cpu": get_cpu_info(), "memory": get_memory_info(), "disk": get_disk_info(),
        "network": get_network_info(), "gpu": get_gpu_info(),
        "system": get_system_info(), "temperature": get_temperature(),
    }


if __name__ == "__main__":
    mcp.run()
