#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""示教臂（手摇跟随）链路：真机 Leader --(串口)--> 本界面里的 Alicia follower。

为什么要单独一个模块
--------------------
* 界面（``alicia_libero_app``）只该关心「6 个关节角 + 夹爪开口」两件事，不该知道
  SDK / 串口 / 线程的细节；
* 链路本身在上一个工程 ``..\\alicia_teleop\\alicia_virtual_teleop.py`` 里**已经跑通**
  （示教臂真机 → MuJoCo follower，含 SDK bug 规避与官方按键解码），这里**复用**它的
  ``LeaderState`` / ``create_leader`` / ``leader_collector`` / ``try_disable_torque`` /
  ``VirtualLeader``，不重复实现；
* 但那边是「独立跑一个 MuJoCo viewer」的脚本，这里是「嵌进 PySide6 主循环」，
  所以要一层薄适配：连/断、线程安全快照、看门狗，以及"示教臂量 → 界面量"的映射。

两种数据源
----------
=======  ==================================================================
leader   真机示教臂（USB 串口 + ``alicia_d_sdk``）
virtual  **模拟示教臂**：不接任何硬件，用代码/键盘给角度（界面自测 + 回归测试）
=======  ==================================================================

映射约定（与遥操作脚本 §5.3 同一套）
------------------------------------
* 关节：``target = leader_angles * signs + offsets``；**相对模式**（默认）再减掉
  "使能那一刻"的偏置 → 使能瞬间**不跳变**；绝对模式 = 完全镜像示教臂姿态；
* 夹爪：SDK ``0`` = 闭合 / ``1000`` = 张开 → 界面 ``gripper`` ``0`` = 闭合 / ``1`` = 张开；
* 死人开关：真机**左键**（状态字 bit4 / ``sync*``）**或**界面上的「使能跟随」勾选框；
* 速率限制：每帧最多动 ``max_step_deg``（默认 8°，与界面里 IK 的限速同量级），
  绝对模式首次使能也不会"甩"过去。

安全约定
--------
* **未使能 → 什么都不下发**（机械臂保持当前目标，鼠标/键盘/滑块照常可用）；
* 真机超过 ``STALE_SECONDS`` 没有新快照 → 状态行报警（串口被上位机占用 / 线松了）；
* ``disable_torque=True``（默认）→ 连接后关掉示教臂力矩，**手摇必须开这个**。
"""

from __future__ import annotations

import functools
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TELEOP_DIR = HERE.parent / "alicia_teleop"        # 上一个工程：已经跑通的示教臂链路
TELEOP_FILE = TELEOP_DIR / "alicia_virtual_teleop.py"

SOURCES = ("leader", "virtual")
SOURCE_LABELS = ("真机示教臂（串口）", "模拟（无硬件自测）")
SOURCE_HINTS = {
    "leader": "真机：选串口 → 点「连接示教臂」→ 勾「使能跟随」（或按住示教臂左键）",
    "virtual": "模拟：不接硬件；连接后用 1..6 / , . / o c 摇虚拟示教臂（走同一条映射路径）",
}
VARIANTS = ("leader", "leader_ur", "gripper_50mm", "gripper_100mm", "vertical_50mm")

GRIPPER_SDK_MAX = 1000.0        # SDK 夹爪行程：0 = 闭合，1000 = 张开（与遥操作脚本一致）
TRIGGER_THRESHOLD = 900.0       # 夹爪值 < 该值 ⇒ 扳机按下（官方判据）
DEFAULT_FPS = 50.0              # 示教臂轮询频率（官方同款；串口带宽有限，别盲目调高）
STALE_SECONDS = 2.0             # 真机多久没有新快照就在状态行报警
DEFAULT_MAX_STEP_DEG = 8.0      # 每帧最多动多少度（= 界面里 IK 的限速，防"甩臂"）

# 链路没连上时给界面用的空快照（字段与 teleop.LeaderState.FIELDS 一致）
EMPTY_SNAPSHOT = {
    "joint_angles": [0.0] * 6,
    "gripper_value": GRIPPER_SDK_MAX,
    "trigger": False,
    "button1": False,
    "button2": False,
    "run_status_text": "unavailable",
    "run_status_raw": None,
    "fps": 0.0,
    "source": "unavailable",
}


class LeaderError(RuntimeError):
    """示教臂连接 / 校准出错（消息直接显示在界面上，中文）。"""


@functools.lru_cache(maxsize=1)
def _load_teleop():
    """导入上一个工程的遥操作脚本（**第一次连接时才导入**：缺 SDK/缺目录都不影响界面启动）。"""
    if not TELEOP_FILE.is_file():
        raise LeaderError(f"找不到示教臂链路脚本：{TELEOP_FILE}")
    if str(TELEOP_DIR) not in sys.path:
        sys.path.insert(0, str(TELEOP_DIR))
    try:
        import alicia_virtual_teleop as teleop  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise LeaderError(f"导入 alicia_virtual_teleop 失败：{exc}") from exc
    return teleop


def list_ports() -> list[tuple[str, str]]:
    """枚举本机串口 → ``[(设备名, 描述)]``；没装 pyserial 时抛 :class:`LeaderError`。"""
    try:
        from serial.tools import list_ports as serial_ports  # noqa: PLC0415
    except ImportError as exc:
        raise LeaderError("未安装 pyserial，无法枚举串口（pip install pyserial）") from exc
    return [(port.device, port.description or "") for port in serial_ports.comports()]


def parse_six(text: str, name: str) -> np.ndarray:
    """把 ``"1,1,1,1,1,1"``（也认空格/中文逗号）解析成 6 个数，供方向系数与偏置用。"""
    cleaned = str(text).replace(" ", "").replace("，", ",")
    try:
        values = [float(v) for v in cleaned.split(",") if v != ""]
    except ValueError as exc:
        raise LeaderError(f"{name} 解析失败（要 6 个逗号分隔的数）：{text}") from exc
    if len(values) != 6:
        raise LeaderError(f"{name} 要 6 个数（例 1,1,1,1,1,1），实际 {len(values)} 个：{text}")
    return np.array(values, dtype=float)


def gripper_sdk_to_unit(value) -> float:
    """SDK 夹爪值（0 = 闭合 / 1000 = 张开）→ 界面 ``gripper``（0 = 闭合 / 1 = 张开）。

    ⚠ 注意别照抄遥操作脚本里的 ``1 - sdk/1000``：那条公式算的是 **MuJoCo 原始滑轨位移**
    （``left_finger`` qpos 的 0 = 两指分开 = 张开，见 `alicia_ik.finger_targets`），
    在**界面这一层** ``session.gripper`` 表示的是"张开度"（1 = 张开），
    所以这里是正比关系：``sdk 1000 → 1.0（张开）``、``sdk 0 → 0.0（闭合）``。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0                       # 读不到就按"张开"兜底（与遥操作脚本一致）
    return min(max(float(value), 0.0), GRIPPER_SDK_MAX) / GRIPPER_SDK_MAX


def describe_snapshot(snapshot: dict) -> str:
    """一行摘要（关节角 / 夹爪 / 按键），状态行与日志都用它。"""
    angles = " ".join(f"{math.degrees(a):+6.1f}" for a in snapshot.get("joint_angles", [])[:6])
    grip = snapshot.get("gripper_value")
    grip_text = "张开" if gripper_sdk_to_unit(grip) > 0.5 else "闭合"
    if isinstance(grip, bool) or not isinstance(grip, (int, float)):
        grip_text = "未知"
    return (f"关节(deg) {angles} · 夹爪 {grip_text}"
            f" · 左键 {'按下' if snapshot.get('button1') else '松开'}"
            f" · 扳机 {'按下' if snapshot.get('trigger') else '松开'}")

@dataclass
class LeaderCommand:
    """本帧要下发给仿真的目标（``enabled=False`` 表示未使能，界面**不要**下发）。"""

    target: np.ndarray          # 6 关节目标角（rad，已做每帧限速）
    gripper: float              # 夹爪开口：0 = 闭合 / 1 = 张开
    enabled: bool               # 本帧是否使能（真机左键 or 界面勾选）
    just_enabled: bool          # 这一帧刚使能（界面用它去暂停自动执行）
    frozen: bool                # 未使能 → 机械臂保持当前目标


def _make_state(teleop):
    """造一个带 ``last_rx`` 时间戳的 ``LeaderState`` 子类（看门狗用）。

    ``LeaderState.update()`` 只认 ``FIELDS`` 里的键，往快照里塞新字段会被静默丢掉，
    所以这里**继承**它再加一层打点，而不是改快照内容。
    """

    class StampedState(teleop.LeaderState):  # type: ignore[misc]
        def __init__(self, source: str = "unknown") -> None:
            super().__init__(source)
            self.last_rx = 0.0

        def update(self, **kwargs) -> None:
            super().update(**kwargs)
            self.last_rx = time.perf_counter()

    return StampedState


class LeaderLink:
    """一条示教臂链路：连接 / 断开 / 读快照（真机或模拟）。

    真机走 ``alicia_teleop`` 的 SDK 轮询线程，模拟源用它的 ``VirtualLeader``，
    两条路都写进同一个线程安全的 ``LeaderState``，所以界面这一层完全无感。
    """

    def __init__(self, source: str = "leader", port: str = "", variant: str = "leader",
                 gripper_type: str = "50mm", fps: float = DEFAULT_FPS,
                 disable_torque: bool = True, debug: bool = False) -> None:
        if source not in SOURCES:
            raise LeaderError(f"未知数据源：{source}（可用：{'/'.join(SOURCES)}）")
        self.source = source
        self.port = str(port or "").strip()
        self.variant = variant
        self.gripper_type = gripper_type
        self.fps = float(fps)
        self.disable_torque = bool(disable_torque)
        self.debug = bool(debug)

        self.state = None                  # teleop.LeaderState 子类（带 last_rx）
        self.robot = None                  # SDK 机器人对象（只有真机才有）
        self.virtual = None                # teleop.VirtualLeader（只有模拟源才有）
        self.connected = False
        self.note = ""                     # 连接时的一句补充说明（关力矩 / 模拟源等）
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    # ── 连接 / 断开 ──
    def connect(self) -> None:
        """建立链路；失败抛 :class:`LeaderError`（消息可直接显示给用户）。"""
        if self.connected:
            return
        teleop = _load_teleop()
        state = _make_state(teleop)()
        if self.source == "virtual":
            self.virtual = teleop.VirtualLeader(state)
            state.update(source="virtual (mock)")
            self.note = "模拟示教臂（无硬件）：用 1..6 / , . / o c 摇它"
        else:
            robot = teleop.create_leader(self.port, self.variant, self.gripper_type, self.debug)
            if robot is None:
                raise LeaderError(self.connect_help())
            if self.disable_torque:
                teleop.try_disable_torque(robot)
                self.note = "已关闭示教臂力矩（可用手拖动）"
            else:
                self.note = "未关力矩：手动示教前请确认示教臂可以被拖动"
            self.robot = robot
            self._stop = threading.Event()
            self._worker = threading.Thread(
                target=teleop.leader_collector,
                args=(robot, state, self._stop, self.fps, self.debug),
                daemon=True, name="leader-collector")
            self._worker.start()
            state.update(source=f"leader ({self.port or 'auto'})")
        self.state = state
        self.connected = True

    def close(self) -> None:
        """停轮询线程 + 断开真机（幂等；关窗口 / 换数据源 / 手动断开都调它）。"""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self.robot = None
        self.virtual = None
        self.state = None
        self.connected = False

    def connect_help(self) -> str:
        """连不上时给用户的排查清单（与 ``alicia_leader_probe.py`` 同一套）。"""
        return ("示教臂连接失败：① 串口被占用（关掉 Synria Desk 上位机 / 串口助手 / 别的脚本）"
                "② 串口号不对（点「刷新」看列表；本机有两个 CH343 时务必分清哪个是示教臂）"
                "③ USB 线只供电不通信（换一根）④ 没装 alicia-d-sdk。"
                "只想先验证界面链路：数据源选「模拟（无硬件自测）」")

    # ── 读取 ──
    def snapshot(self) -> dict:
        """当前快照（没连上就返回空快照，界面不用判空）。"""
        return self.state.snapshot() if self.state is not None else dict(EMPTY_SNAPSHOT)

    def age(self) -> float:
        """距上一次读到快照的秒数（真机掉线看门狗用；模拟源恒为 0）。"""
        if self.source == "virtual" or self.state is None:
            return 0.0
        return max(time.perf_counter() - float(getattr(self.state, "last_rx", 0.0)), 0.0)

    def stale(self) -> bool:
        """真机超过 :data:`STALE_SECONDS` 没有新数据（串口被抢 / 线松了）。"""
        return self.source == "leader" and self.connected and self.age() > STALE_SECONDS

    def describe(self) -> str:
        where = "模拟示教臂" if self.source == "virtual" else f"串口 {self.port or '自动'}"
        return f"{where} · {self.variant} · {self.fps:.0f} Hz"

    # ── 模拟源驱动（界面按键 / 回归测试都走这里）──
    def virtual_key(self, key: str) -> bool:
        """把按键转给虚拟示教臂（只有模拟源会消费）；返回是否被消费。"""
        if self.virtual is None:
            return False
        return bool(self.virtual.handle_key(key))

    def push_virtual(self, angles=None, gripper=None, enabled=None, buttons=None) -> None:
        """程序化驱动模拟示教臂（界面自测 / 回归测试用）：设好值后立刻发布一帧快照。

        ``buttons``：直接注入按键状态，例如 ``{"button1": True}``（真机死人开关的等价物）。
        """
        if self.virtual is None:
            raise LeaderError("当前数据源不是模拟示教臂，不能程序化驱动")
        if angles is not None:
            self.virtual.joint_angles = [float(a) for a in angles]
        if gripper is not None:
            self.virtual.gripper = float(gripper)
        if enabled is not None:
            self.virtual.enabled = bool(enabled)
        self.virtual.push()
        if buttons:
            self.state.update(**buttons)


class LeaderFollowMapper:
    """示教臂快照 → 本帧要下发的（6 关节目标角, 夹爪开口, 是否使能）。

    映射逻辑与遥操作脚本主循环**逐字对应**（同一套约定，见模块开头），区别只有两点：
    ① 多了每帧 ``max_step_deg`` 的限速（绝对模式首次使能也不会"甩臂"）；
    ② 把"刚使能"当成一个事件报给界面（界面用它去暂停自动执行 / 连跑）。
    """

    def __init__(self, signs=None, offsets_deg=None, mode: str = "relative",
                 max_step_deg: float = DEFAULT_MAX_STEP_DEG) -> None:
        self.signs = np.array([1.0] * 6 if signs is None else signs, dtype=float)
        self.offsets = np.radians(np.zeros(6) if offsets_deg is None
                                  else np.asarray(offsets_deg, dtype=float))
        self.max_step_deg = float(max_step_deg)
        self.mode = "relative"
        self.rel_offset = np.zeros(6)
        self.frozen_q = np.zeros(6)
        self.frozen_grip = 1.0
        self.was_enabled = False
        self.set_mode(mode)

    # ── 设置 ──
    def set_mode(self, mode: str) -> None:
        """``relative``（只镜像增量：使能瞬间不跳变，默认）/ ``absolute``（完全镜像姿态）。"""
        if mode not in ("relative", "absolute"):
            raise LeaderError(f"未知同步方式：{mode}（可用 relative / absolute）")
        self.mode = mode

    def set_calibration(self, signs_text, offsets_text) -> None:
        """从界面文本框更新 6 轴方向系数（±1）与角度偏置（度）。"""
        self.signs = parse_six(signs_text, "方向系数")
        self.offsets = np.radians(parse_six(offsets_text, "角度偏置"))

    def limit_rad(self, limit_rad: float | None = None) -> float:
        """每帧允许的最大角度变化（rad）。"""
        return math.radians(self.max_step_deg) if limit_rad is None else float(limit_rad)

    # ── 对齐 / 复位 ──
    def align(self, snapshot: dict, follower_rad) -> np.ndarray:
        """把"此刻"当零点：相对模式记下偏置（机械臂原地不动）；绝对模式=目标即示教臂姿态。"""
        leader = self.leader_target(snapshot)
        self.frozen_q = np.asarray(follower_rad, dtype=float).copy()
        self.frozen_grip = gripper_sdk_to_unit(snapshot.get("gripper_value"))
        self.rel_offset = leader - self.frozen_q if self.mode == "relative" else np.zeros(6)
        return self.rel_offset.copy()

    def reset(self, follower_rad=None) -> None:
        """断开 / 换源时清状态（丢掉上一次的偏置与冻结目标）。"""
        self.rel_offset = np.zeros(6)
        self.frozen_q = (np.zeros(6) if follower_rad is None
                         else np.asarray(follower_rad, dtype=float).copy())
        self.frozen_grip = 1.0
        self.was_enabled = False

    def leader_target(self, snapshot: dict) -> np.ndarray:
        """示教臂关节角 → 带方向系数与偏置的关节目标角（rad）。"""
        angles = np.asarray(snapshot.get("joint_angles", [0.0] * 6), dtype=float).ravel()[:6]
        if angles.size < 6:                       # 真机偶尔少读几轴：缺的按 0 补
            angles = np.concatenate([angles, np.zeros(6 - angles.size)])
        return angles * self.signs + self.offsets

    # ── 每帧映射 ──
    def step(self, snapshot: dict, follower_rad, manual_enable: bool = False,
             limit_rad: float | None = None) -> LeaderCommand:
        """算本帧目标；未使能时返回 ``enabled=False``（界面**不要**下发）。"""
        enabled = bool(manual_enable) or bool(snapshot.get("button1"))
        just_enabled = enabled and not self.was_enabled
        if just_enabled:
            # 使能瞬间以**机械臂当前目标**为基准：相对模式记偏置、绝对模式从这里限速爬升
            self.frozen_q = np.asarray(follower_rad, dtype=float).copy()
            if self.mode == "relative":
                self.rel_offset = self.leader_target(snapshot) - self.frozen_q
        if enabled:
            target = self.leader_target(snapshot)
            if self.mode == "relative":
                target = target - self.rel_offset
            limit = self.limit_rad(limit_rad)
            self.frozen_q = np.clip(target, self.frozen_q - limit, self.frozen_q + limit)
            self.frozen_grip = gripper_sdk_to_unit(snapshot.get("gripper_value"))
        self.was_enabled = enabled
        return LeaderCommand(target=self.frozen_q.copy(), gripper=float(self.frozen_grip),
                             enabled=enabled, just_enabled=just_enabled, frozen=not enabled)


