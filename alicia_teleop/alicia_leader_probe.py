#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""示教臂串口/按键自检（对应官方 Demo 01 的轻量版，不带 matplotlib 绘图）。

官方文档建议：上 MuJoCo 仿真之前，先确认「串口读得到关节角、按键状态正常」。
本脚本就是干这件事的：连上示教臂，把 6 轴角度、夹爪值、状态字、死人开关/锁定键
实时打印出来。只读，不下发任何运动指令。

用法
----
    python alicia_leader_probe.py                      # 自动搜索串口
    python alicia_leader_probe.py --port COM5
    python alicia_leader_probe.py --seconds 30 --fps 20
    python alicia_leader_probe.py --disable-torque     # 顺便关力矩（可用手拖动）
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

from alicia_virtual_teleop import (  # noqa: E402
    LeaderState,
    create_leader,
    leader_collector,
    list_serial_ports,
    try_disable_torque,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="示教臂只读自检：确认串口读取与按键状态",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=str, default="", help="串口号，留空=自动搜索")
    parser.add_argument("--variant", type=str, default="leader", help="示教臂机型变体")
    parser.add_argument("--gripper-type", type=str, default="50mm", help="夹爪规格")
    parser.add_argument("--fps", type=float, default=50.0, help="采集频率 Hz")
    parser.add_argument("--seconds", type=float, default=15.0, help="运行时长（秒）")
    parser.add_argument("--interval", type=float, default=0.2, help="终端刷新间隔（秒）")
    parser.add_argument("--disable-torque", action="store_true",
                        help="启动时关闭力矩（用手拖动示教臂时才需要）")
    parser.add_argument("--list-ports", action="store_true", help="只列出串口后退出")
    parser.add_argument("--debug", action="store_true", help="打印采集异常")
    args = parser.parse_args(argv)

    if args.list_ports:
        list_serial_ports()
        return 0

    robot = create_leader(args.port, args.variant, args.gripper_type, args.debug)
    if robot is None:
        print("\n[ERROR] 示教臂连接失败，常见原因：")
        print("  1) 串口被占用（关掉 Synria Desk 上位机 / 串口助手 / 其他脚本）")
        print("  2) 串口号不对：用 --list-ports 查看")
        print("  3) 数据线只供电不通信，换一根 USB 线")
        return 1

    if args.disable_torque:
        print("[WARN] 将关闭示教臂力矩，请用手扶住机械臂！")
        try_disable_torque(robot)

    state = LeaderState()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=leader_collector, args=(robot, state, stop_event, args.fps, args.debug), daemon=True
    )
    worker.start()

    print(f"[OK] 开始采集 {args.seconds:.0f} s（Ctrl+C 或关窗口可提前结束）")
    print(f"{'t':>6} {'status':>12} {'grip':>6} {'deadman':>8} {'lock':>5} {'fps':>5}  关节(deg)")
    deadline = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < deadline:
            time.sleep(args.interval)
            snapshot = state.snapshot()
            joints = " ".join(f"{math.degrees(a):+7.2f}" for a in snapshot["joint_angles"])
            grip = snapshot["gripper_value"]
            grip_text = f"{grip:6.0f}" if isinstance(grip, (int, float)) else "    --"
            print(f"{time.perf_counter() - (deadline - args.seconds):6.1f} "
                  f"{str(snapshot['run_status_text']):>12} {grip_text} "
                  f"{'PRESSED' if snapshot['button1'] else 'released':>8} "
                  f"{'PRESSED' if snapshot['button2'] else 'released':>5} "
                  f"{snapshot['fps']:5.1f}  {joints}")
    except KeyboardInterrupt:
        print("\n[INFO] 手动中断")
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        robot.disconnect()
        print("[OK] 示教臂已断开（全程未下发任何运动指令）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
