#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alicia-D 虚拟遥操作：示教臂(Leader) → MuJoCo 里的虚拟操作臂(Follower)。

对应官方文档
------------
https://docs.sparklingrobo.com/docs/alicia-d-series/leader-force-control/doc_05_mujoco_leader
官方实现是 Teleoperation-SDK 里的 ``02_demo_mujoco_follower.py``；本脚本沿用它的
「示教臂真机 --(串口)--> MuJoCo 虚拟操作臂」链路与交互约定，并做三点增强：

1. ``--source virtual``：不接任何硬件也能跑，用键盘当“虚拟示教臂”，先在 PC 上把
   仿真链路、模型、夹爪映射跑通，再上真机联调
2. actuator 自适应：位置伺服模型直接写目标角度；纯力矩模型由脚本内部做 PD 闭环，
   所以 Synria-Robot-Descriptions 里的任意变体（50/100mm、leader、ARX、vertical）都能用
3. ``--record``：把每次采样写成 CSV，方便事后画图核对跟随效果

模型准备（只需一次）
--------------------
    python make_follower_xml.py        # 生成 *_teleop.xml（带位置伺服 actuator）

用法
----
    python alicia_virtual_teleop.py --source virtual           # 无硬件自测
    python alicia_virtual_teleop.py --source leader --port COM5
    python alicia_virtual_teleop.py --source leader --port COM5 --disable-torque
    python alicia_virtual_teleop.py --source auto --port COM5  # 连不上自动退回 virtual

按键（焦点在 MuJoCo 窗口里按；MuJoCo 自身占用了 空格 [ ] - = ，已避开）
---------------------------------------------------------------------
    t  使能 / 断开遥操作（等价于官方“按住示教臂左键”的死人开关，真机模式下
       直接按住真机左键也可以）
    a  对齐：把当前示教臂姿态设为虚拟机械臂的目标（relative 模式=重置偏置）
    p  在终端打印一次状态
    r  重新加载 XML（改完模型不用重启程序）
    v  开关腕部相机预览（模型里要有 <camera>，且需要 opencv）
    q  退出（直接关窗口也一样）

    ``--source virtual`` 时额外：
    1..6  选择要操作的关节（终端会提示当前选中关节）
    , .   选中关节 -2° / +2°
    o c   夹爪 张开 / 闭合
    0     虚拟示教臂复位到零位
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

for _stream in (sys.stdout, sys.stderr):  # 避免中文终端里出现无法编码的字符时报错
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

HERE = Path(__file__).resolve().parent
MUJOCO_ROOT = HERE.parent
REPO_MJCF = MUJOCO_ROOT / "Synria-Robot-Descriptions-main" / "synriard" / "mjcf"
DEFAULT_XML = REPO_MJCF / "Alicia_D_v5_6" / "Alicia_D_v5_6_gripper_50mm_teleop.xml"

TRIGGER_THRESHOLD = 900.0    # 示教臂夹爪值 < 该阈值 ⇒ 扳机按下
GRIPPER_SDK_MAX = 1000.0     # SDK 夹爪行程：0 = 完全闭合，1000 = 完全张开
GRIPPER_M_FALLBACK = 0.025   # 模型里读不到滑轨行程时的兜底值（50mm 夹爪）
ARM_JOINTS = [f"Joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ("left_finger", "right_finger")
VIEWER_FPS = 60.0
VIRTUAL_STEP_RAD = math.radians(2.0)   # 虚拟示教臂每按一次 ,/. 的步长

# 力矩型模型（如 Synria-Robot-Descriptions 原始 MJCF）没有关节阻尼/armature，
# 舵机反射惯量只有 1e-4 量级，脚本内做 PD 会立刻发散。
# 这里沿用官方遥操作模型的取值，加载时自动补齐（真机与仿真都稳）。
DEFAULT_JOINT_ARMATURE = 0.02
DEFAULT_JOINT_DAMPING = 1.0


def precise_sleep(seconds: float, spin_threshold: float = 0.002, sleep_margin: float = 0.001) -> None:
    """比 time.sleep 更准的等待（优先用 SDK 里的实现，保持一致）。"""
    if seconds <= 0:
        return
    try:
        from alicia_d_sdk.utils import precise_sleep as sdk_sleep  # noqa: PLC0415

        sdk_sleep(seconds, spin_threshold=spin_threshold, sleep_margin=sleep_margin)
        return
    except Exception:  # noqa: BLE001  - SDK 没装时用本地实现
        pass

    end_time = time.perf_counter() + seconds
    while True:
        remaining = end_time - time.perf_counter()
        if remaining <= 0:
            break
        if remaining > spin_threshold:
            time.sleep(max(remaining - sleep_margin, 0))


class LeaderState:
    """线程安全的示教臂状态快照容器（无论真机还是虚拟输入都用它）。"""

    FIELDS = {
        "joint_angles": [0.0] * 6,
        "gripper_value": GRIPPER_SDK_MAX,   # 1000 = 张开
        "trigger": False,
        "button1": False,                   # 左键 = 死人开关
        "button2": False,                   # 右键 = 锁定示教臂
        "run_status_text": "unknown",
        "run_status_raw": None,
        "fps": 0.0,
        "source": "unknown",
    }

    def __init__(self, source: str = "unknown") -> None:
        self._lock = threading.Lock()
        self._values = dict(self.FIELDS)
        self._values["source"] = source

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if key in self._values:
                    self._values[key] = value

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._values)


# ══════════════════ 真机输入：SDK 轮询线程 ══════════════════
def try_disable_torque(robot) -> None:
    """按 SDK 版本差异尝试关闭示教臂力矩（方便用手拖拽）。"""
    for method_name, method_args in (
        ("torque_control", ("off",)),
        ("disable_torque", ()),
        ("torque_enable", (False,)),
    ):
        fn = getattr(robot, method_name, None)
        if callable(fn):
            fn(*method_args)
            print(f"[OK] 已调用 {method_name}{method_args} 关闭力矩")
            return
    print("[WARN] 未找到可用的关闭力矩接口，请手动确认示教臂可以拖动")


def decode_handle_inputs(raw_status, run_status_text, gripper_value=None) -> dict:
    """把示教臂的状态字解码成三个遥操作开关。

    * 左键(死人开关)：run_status_text == "sync" / "sync_locked"，或状态字 bit4 (0x10)
    * 右键(锁定)    ：run_status_text == "locked" / "sync_locked"，或状态字 bit0 (0x01)
    * 扳机          ：夹爪值 < TRIGGER_THRESHOLD
    """
    left_button = False
    right_button = False
    trigger = gripper_value is not None and gripper_value < TRIGGER_THRESHOLD

    if isinstance(run_status_text, str):
        if run_status_text == "sync":
            left_button = True
        elif run_status_text == "locked":
            right_button = True
        elif run_status_text == "sync_locked":
            left_button = True
            right_button = True

    if isinstance(raw_status, int):
        left_button = left_button or bool(raw_status & 0x10)
        right_button = right_button or bool(raw_status & 0x01)

    return {"trigger": trigger, "button1": left_button, "button2": right_button}


def leader_collector(robot, state: LeaderState, stop_event: threading.Event, fps: float, debug: bool) -> None:
    """后台线程：以固定频率读取示教臂，写入 LeaderState。"""
    interval = 1.0 / fps
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            raw = robot.get_robot_state("joint_gripper")
            if raw is not None:
                angles = list(raw.angles)[:6]
                if len(angles) < 6:
                    angles.extend([0.0] * (6 - len(angles)))

                run_text = getattr(raw, "run_status_text", "unknown")
                run_raw = getattr(robot.data_parser, "_run_status", None)
                grip_val = getattr(raw, "gripper", None)
                decoded = decode_handle_inputs(run_raw, run_text, gripper_value=grip_val)

                dt_loop = time.perf_counter() - t0
                state.update(
                    joint_angles=angles,
                    gripper_value=grip_val,
                    trigger=decoded["trigger"],
                    button1=decoded["button1"],
                    button2=decoded["button2"],
                    run_status_text=run_text,
                    run_status_raw=run_raw,
                    fps=1.0 / dt_loop if dt_loop > 0 else 0.0,
                )
        except Exception as exc:  # noqa: BLE001
            if debug:
                print(f"[collector] {exc}")

        dt = time.perf_counter() - t0
        if dt < interval:
            precise_sleep(interval - dt)


# ══════════════════ 虚拟输入：键盘当示教臂 ══════════════════
class VirtualLeader:
    """没有硬件时的替身示教臂：用键盘产生与真机同构的状态。

    真机上「左键」是按住生效的死人开关；键盘没有按住/抬起的区分
    （MuJoCo 只把按键按下事件回调给 Python），所以这里用 ``t`` 做成开关。
    """

    def __init__(self, state: LeaderState) -> None:
        self.state = state
        self.joint_angles = [0.0] * 6
        self.gripper = GRIPPER_SDK_MAX
        self.enabled = False
        self.selected = 0
        self._last_push = 0.0
        self.push(force=True)

    def push(self, force: bool = False) -> None:
        """把虚拟示教臂的当前状态发布到 LeaderState。"""
        now = time.perf_counter()
        if force:
            fps = 0.0
        else:
            dt = now - self._last_push
            fps = 1.0 / dt if dt > 0 else 0.0
        self._last_push = now
        self.state.update(
            joint_angles=list(self.joint_angles),
            gripper_value=self.gripper,
            trigger=self.gripper < TRIGGER_THRESHOLD,
            button1=self.enabled,
            button2=False,
            run_status_text="sync" if self.enabled else "idle",
            run_status_raw=0x10 if self.enabled else 0x00,
            fps=fps,
        )

    def handle_key(self, key: str) -> bool:
        """处理按键，返回 True 表示该键被虚拟示教臂消费掉了。"""
        if key in "123456":
            self.selected = int(key) - 1
            print(f"[VIRTUAL] 选中 Joint{self.selected + 1}")
        elif key == ",":
            self.joint_angles[self.selected] -= VIRTUAL_STEP_RAD
        elif key == ".":
            self.joint_angles[self.selected] += VIRTUAL_STEP_RAD
        elif key == "o":
            self.gripper = min(GRIPPER_SDK_MAX, self.gripper + 50.0)
        elif key == "c":
            self.gripper = max(0.0, self.gripper - 50.0)
        elif key == "0":
            self.joint_angles = [0.0] * 6
            self.gripper = GRIPPER_SDK_MAX
            print("[VIRTUAL] 虚拟示教臂已复位")
        elif key == "t":
            self.enabled = not self.enabled
        else:
            return False

        # 关节角限幅到模型关节范围之外不必要，MuJoCo 的 position actuator 会自动限位
        self.joint_angles[self.selected] = float(
            np.clip(self.joint_angles[self.selected], -math.pi, math.pi)
        )
        self.push()
        return True

    def describe(self) -> str:
        joints = " ".join(f"{math.degrees(a):+7.2f}" for a in self.joint_angles)
        return (f"[VIRTUAL] 使能={'ON ' if self.enabled else 'OFF'} "
                f"选中=Joint{self.selected + 1} 关节(deg)={joints} "
                f"夹爪={self.gripper:.0f}")


# ══════════════════ MuJoCo 模型侧：模型解析与指令下发 ══════════════════
class Channel:
    """一路执行通道：某个关节 ↔ 驱动它的 actuator。"""

    __slots__ = ("joint", "act_id", "mode", "qadr", "vadr", "ctrl_lo", "ctrl_hi", "sign")

    def __init__(self, joint, act_id, mode, qadr, vadr, ctrl_lo, ctrl_hi, sign=1.0):
        self.joint = joint
        self.act_id = act_id
        self.mode = mode          # "pos" = ctrl 是目标位置；"frc" = ctrl 是力矩
        self.qadr = qadr          # qpos 下标
        self.vadr = vadr          # qvel 下标
        self.ctrl_lo = ctrl_lo
        self.ctrl_hi = ctrl_hi
        self.sign = sign


class FollowerBundle:
    """加载 follower 模型，并把它解析成 6 路臂关节 + 2 路夹爪的通道。"""

    def __init__(self, xml_path: Path, kp: float = 400.0, kd: float = 10.0) -> None:
        self.xml_path = Path(xml_path)
        self.kp = kp
        self.kd = kd
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        missing = [j for j in ARM_JOINTS
                   if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) < 0]
        if missing:
            raise RuntimeError(
                f"模型里缺少关节 {missing}；请确认 --xml 指向的是 Alicia follower 模型"
            )

        self.arm = [self._make_channel(j) for j in ARM_JOINTS]
        self.fingers = [self._make_channel(j) for j in FINGER_JOINTS
                        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0]
        self.grip_max = self._finger_travel()

        if not self.fingers:
            print("[WARN] 模型里没有夹爪滑轨关节，夹爪通道将被忽略")

        self._stabilise_torque_channels()

    def _stabilise_torque_channels(self) -> None:
        """给「力矩型」通道补关节阻尼/armature 并把积分器切到 implicit。

        Synria-Robot-Descriptions 的原始 MJCF 没有 <default>，关节的 armature 与
        damping 都是 0；此时脚本内做 PD 会瞬间发散（实测 QACC 爆表）。
        官方遥操作模型正是靠 ``<option integrator="implicit">`` +
        ``<joint damping="1.0" armature="0.02"/>`` 才稳的，这里等价补齐。
        """
        torque_channels = [c for c in self.arm + self.fingers if c.mode == "frc"]
        if not torque_channels:
            return

        model = self.model
        patched = False
        for channel in torque_channels:
            dof = channel.vadr
            if model.dof_armature[dof] < 1e-6:
                model.dof_armature[dof] = DEFAULT_JOINT_ARMATURE
                patched = True
            if model.dof_damping[dof] < 1e-6:
                model.dof_damping[dof] = DEFAULT_JOINT_DAMPING
                patched = True

        if patched:
            model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICIT)
            print(
                f"[WARN] 该模型用的是力矩 actuator 且缺少关节阻尼/armature，"
                f"已自动补 armature={DEFAULT_JOINT_ARMATURE} damping={DEFAULT_JOINT_DAMPING} "
                "并切换 implicit 积分器（与官方遥操作模型一致）；"
                "想更接近官方效果，建议用 make_follower_xml.py 生成位置伺服版"
            )

    # ── 解析辅助 ──
    def _make_channel(self, joint_name: str) -> Channel:
        model = self.model
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        act_id = next((aid for aid in range(model.nu)
                       if model.actuator_trnid[aid, 0] == jid), -1)
        if act_id < 0:
            raise RuntimeError(
                f"关节 '{joint_name}' 没有对应的 actuator。"
                "如果用的是 Synria-Robot-Descriptions 原始模型，"
                "请先运行 make_follower_xml.py 生成 *_teleop.xml"
            )

        bias = int(model.actuator_biastype[act_id])
        bias_prm = model.actuator_biasprm[act_id]
        if bias == int(mujoco.mjtBias.mjBIAS_AFFINE) and bias_prm[1] < 0:
            mode = "pos"      # <position> 或 <general biastype=affine ...>：ctrl = 目标位置
        elif bias == int(mujoco.mjtBias.mjBIAS_AFFINE) and bias_prm[1] == 0:
            raise RuntimeError(f"actuator '{joint_name}' 是速度型（velocity）伺服，暂不支持")
        else:
            mode = "frc"      # <motor> / general dyntype=none：ctrl = 力矩，脚本内做 PD

        ctrl_lo, ctrl_hi = model.actuator_ctrlrange[act_id]
        if ctrl_lo >= ctrl_hi:
            ctrl_lo, ctrl_hi = -1e6, 1e6

        lo, hi = model.jnt_range[jid]
        sign = 1.0 if hi > 0 else -1.0        # 夹爪两侧滑轨方向相反
        return Channel(joint_name, act_id, mode, model.jnt_qposadr[jid],
                       model.jnt_dofadr[jid], float(ctrl_lo), float(ctrl_hi), sign)

    def _finger_travel(self) -> float:
        """从滑轨关节 range 读夹爪单侧行程（50mm→0.025，100mm→0.05）。"""
        if not self.fingers:
            return GRIPPER_M_FALLBACK
        channel = self.fingers[0]
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, channel.joint)
        lo, hi = self.model.jnt_range[jid]
        travel = max(abs(float(lo)), abs(float(hi)))
        return travel if travel > 0 else GRIPPER_M_FALLBACK

    # ── 指令下发 ──
    def _write(self, channel: Channel, target: float) -> None:
        if channel.mode == "pos":
            self.data.ctrl[channel.act_id] = target
            return
        q = self.data.qpos[channel.qadr]
        v = self.data.qvel[channel.vadr]
        torque = self.kp * (target - q) - self.kd * v
        self.data.ctrl[channel.act_id] = float(np.clip(torque, channel.ctrl_lo, channel.ctrl_hi))

    def command(self, arm_target, grip_target: float) -> None:
        """写入 6 轴目标角度(rad) + 夹爪目标行程(m)。"""
        for channel, target in zip(self.arm, arm_target):
            self._write(channel, float(target))
        for channel in self.fingers:
            self._write(channel, channel.sign * grip_target)

    def current_arm_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[c.qadr] for c in self.arm], dtype=float)

    def current_gripper_gap(self) -> float:
        if not self.fingers:
            return 0.0
        return float(sum(abs(self.data.qpos[c.qadr]) for c in self.fingers))

    def describe(self) -> str:
        modes = {c.mode for c in self.arm}
        mode_text = "position 位置伺服（直接写目标角）" if modes == {"pos"} else \
            f"力矩 PD 闭环（kp={self.kp:g}, kd={self.kd:g}）"
        return (f"模型: {self.xml_path.name} | nq={self.model.nq} nu={self.model.nu} | "
                f"控制: {mode_text} | 夹爪单侧行程: {self.grip_max * 1000:.1f} mm")


def gripper_sdk_to_mujoco(sdk_val: float, travel: float) -> float:
    """SDK 夹爪值 (0=闭合, 1000=张开) → MuJoCo 滑轨目标行程(m)。"""
    clamped = max(0.0, min(GRIPPER_SDK_MAX, float(sdk_val)))
    return (1.0 - clamped / GRIPPER_SDK_MAX) * travel


# ══════════════════ HUD / 录制 / 相机预览 ══════════════════
def draw_hud(viewer, snap: dict, enabled: bool, grip_m: float, footer: str) -> None:
    """在 MuJoCo 窗口左上角画状态面板（只用 ASCII，MuJoCo 内置字体不支持中文）。

    注意：mujoco >= 3.13 的 viewer 改成 ``Handle.set_texts()``；
    3.12 及更早是 ``mjr_overlay(viewer.viewport, ..., viewer.ctx)``。这里两种都兼容。
    """
    gv = snap["gripper_value"]
    rows = [
        ("SOURCE", snap["source"]),
        ("Leader FPS", f"{snap['fps']:.1f}"),
        ("Deadman (left btn)", "PRESSED" if snap["button1"] else "released"),
        ("Right btn (lock)", "PRESSED" if snap["button2"] else "released"),
        ("Trigger", "ON" if snap["trigger"] else "OFF"),
        ("Gripper SDK", f"{gv:.0f} / 1000" if isinstance(gv, (int, float)) else "--"),
        ("Gripper MuJoCo", f"{grip_m * 1000:.1f} mm"),
        ("Run status", str(snap["run_status_text"])),
        ("TELEOP", "ACTIVE" if enabled else "DISABLED"),
    ]
    for idx, angle in enumerate(snap["joint_angles"]):
        rows.append((f"J{idx + 1}", f"{math.degrees(angle):+7.2f} deg"))

    left = "\n".join(key for key, _ in rows)
    right = "\n".join(value for _, value in rows)

    set_texts = getattr(viewer, "set_texts", None)
    if callable(set_texts):                      # mujoco >= 3.13
        texts = [
            (mujoco.mjtFontScale.mjFONTSCALE_150,
             mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
        ]
        if footer:
            texts.append((mujoco.mjtFontScale.mjFONTSCALE_100,
                          mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, footer, ""))
        set_texts(texts)
        return

    mjr_overlay = getattr(mujoco, "mjr_overlay", None)   # mujoco <= 3.12
    if mjr_overlay is not None and hasattr(viewer, "ctx"):
        mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    viewer.viewport, left, right, viewer.ctx)
        if footer:
            mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                        viewer.viewport, footer, "", viewer.ctx)



class CsvRecorder:
    """把遥操作过程写成 CSV，方便事后画图核对跟随效果。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(
            ["t", "enabled"]
            + [f"leader_j{i + 1}" for i in range(6)]
            + ["leader_gripper_sdk"]
            + [f"follower_j{i + 1}" for i in range(6)]
            + ["follower_gripper_m"]
        )

    def write(self, t: float, enabled: bool, leader_q, grip_sdk, follower_q, grip_m: float) -> None:
        self._writer.writerow(
            [f"{t:.4f}", int(enabled)]
            + [f"{v:.6f}" for v in leader_q]
            + [f"{grip_sdk:.1f}" if isinstance(grip_sdk, (int, float)) else ""]
            + [f"{v:.6f}" for v in follower_q]
            + [f"{grip_m:.6f}"]
        )

    def close(self) -> None:
        try:
            self._handle.close()
            print(f"[OK] 已保存遥操作记录: {self.path}")
        except Exception:  # noqa: BLE001
            pass


class CameraPreview:
    """可选的腕部相机预览（模型里有 <camera> 且装了 opencv 才有用）。"""

    def __init__(self, model, camera_name: str, width: int = 640, height: int = 480) -> None:
        self.enabled = False
        self._cv2 = None
        try:
            import cv2  # noqa: PLC0415

            self._cv2 = cv2
        except ImportError:
            print("[WARN] 未安装 opencv，无法显示相机预览")
            return

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            print(f"[WARN] 模型里没有名为 '{camera_name}' 的 camera，预览不可用")
            return

        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._window = f"wrist camera: {camera_name}"
        self.enabled = True

    def show(self, data) -> None:
        if not self.enabled:
            return
        self._renderer.update_scene(data, camera=None)
        frame = self._renderer.render()
        self._cv2.imshow(self._window, self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))
        self._cv2.waitKey(1)

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self._cv2.destroyWindow(self._window)
        except Exception:  # noqa: BLE001
            pass
        close_fn = getattr(self._renderer, "close", None)
        if callable(close_fn):
            close_fn()


def list_serial_ports() -> None:
    """列出本机串口，方便确定 --port。"""
    try:
        from serial.tools import list_ports  # noqa: PLC0415
    except ImportError:
        print("[ERROR] 未安装 pyserial，无法枚举串口（pip install pyserial）")
        return

    ports = list(list_ports.comports())
    if not ports:
        print("没有检测到任何串口设备。")
        return
    print(f"检测到 {len(ports)} 个串口：")
    for port in ports:
        print(f"  {port.device:8s} {port.description}")


# ══════════════════ main ══════════════════
def parse_six(text: str, name: str) -> list[float]:
    try:
        values = [float(v) for v in str(text).replace(" ", "").split(",") if v != ""]
    except ValueError as exc:
        raise SystemExit(f"[ERROR] --{name} 解析失败: {exc}") from exc
    if len(values) != 6:
        raise SystemExit(f"[ERROR] --{name} 需要 6 个逗号分隔的数，例如 1,1,-1,1,1,1")
    return values


def create_leader(port: str, variant: str, gripper_type: str, debug: bool):
    """创建并连接示教臂；失败返回 None（不抛异常，方便 --source auto 回退）。

    注意：部分版本的 alicia_d_sdk（实测 6.1.0rc4）在 ``debug_mode=True`` 时会抛
    ``'SerialComm' object has no attribute '_print_hex_frame'``，导致握手失败。
    这里会在 debug 模式失败后自动用 ``debug_mode=False`` 重试一次。
    """
    try:
        import alicia_d_sdk  # noqa: PLC0415
    except ImportError as exc:
        print(f"[WARN] 无法导入 alicia_d_sdk：{exc}")
        return None

    for attempt_debug in ([True, False] if debug else [False]):
        robot = None
        try:
            robot = alicia_d_sdk.create_robot(
                port=port, variant=variant, gripper_type=gripper_type, debug_mode=attempt_debug
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 连接示教臂时出错：{exc}")

        if robot is not None and robot.is_connected():
            if debug and not attempt_debug:
                print("[WARN] 已改用 debug_mode=False（SDK 的 debug 模式在本机版本有 bug）")
            return robot

        if robot is not None:
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if attempt_debug:
            print("[WARN] SDK debug 模式连接失败（已知问题：alicia_d_sdk 6.1.0rc4 缺 _print_hex_frame），"
                  "改用 debug_mode=False 重试 ……")

    print("[WARN] 串口未打开或握手失败，示教臂不可用")
    return None


def print_controls(source: str) -> None:
    print("-" * 66)
    print(f"  输入源: {source}")
    print("  t  使能/断开遥操作        a  对齐/清零        p  打印状态")
    print("  r  重载 XML               v  相机预览        q  退出（关窗口也可以）")
    if source.startswith("virtual"):
        print("  虚拟示教臂: 1..6 选关节   , .  关节 -/+ 2    o/c 夹爪开/合   0 复位")
    print("-" * 66)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alicia-D 虚拟遥操作：示教臂 → MuJoCo follower",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml", type=str, default=str(DEFAULT_XML),
                        help="follower 模型（*.xml）")
    parser.add_argument("--source", choices=["leader", "virtual", "auto"], default="leader",
                        help="输入源：真机示教臂 / 键盘虚拟示教臂 / 连不上自动回退")
    parser.add_argument("--port", type=str, default="",
                        help="示教臂串口，留空=自动搜索（Windows 例：COM5）")
    parser.add_argument("--variant", type=str, default="leader",
                        choices=["leader", "leader_ur", "gripper_50mm", "gripper_100mm", "vertical_50mm"],
                        help="示教臂机型变体")
    parser.add_argument("--gripper-type", type=str, default="50mm",
                        help="夹爪规格（仅在 --variant 缺省时生效）")
    parser.add_argument("--fps", type=float, default=50.0, help="示教臂轮询频率 Hz")
    parser.add_argument("--mode", choices=["absolute", "relative"], default="absolute",
                        help="absolute=完全镜像示教臂姿态；relative=只镜像增量（使能瞬间不跳变）")
    parser.add_argument("--signs", type=str, default="1,1,1,1,1,1",
                        help="6 轴方向系数，姿态方向不对时改这里的正负号")
    parser.add_argument("--offsets", type=str, default="0,0,0,0,0,0",
                        help="6 轴角度偏置（deg），零点不一致时用来粗校准")
    parser.add_argument("--kp", type=float, default=400.0,
                        help="力矩型模型内部 PD 的比例增益（位置伺服模型用不到）")
    parser.add_argument("--kd", type=float, default=10.0,
                        help="力矩型模型内部 PD 的微分增益")
    parser.add_argument("--disable-torque", action="store_true",
                        help="启动时关闭示教臂力矩（需要用手拖着示教臂时加这个）")
    parser.add_argument("--record", type=str, default="",
                        help="把每次采样写成 CSV，例如 logs/teleop.csv")
    parser.add_argument("--camera", type=str, default="",
                        help="腕部相机名（模型里有 <camera> 才能用），留空=不显示")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行 N 秒后自动退出；0 = 一直运行（自动化测试用）")
    parser.add_argument("--list-ports", action="store_true", help="列出串口后退出")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_ports:
        list_serial_ports()
        return 0

    xml_path = Path(args.xml).expanduser()
    if not xml_path.is_file():
        print(f"[ERROR] 找不到模型文件: {xml_path}")
        print("        先执行一次:  python make_follower_xml.py")
        return 1

    signs = np.array(parse_six(args.signs, "signs"))
    offsets = np.radians(np.array(parse_six(args.offsets, "offsets")))

    try:
        bundle = FollowerBundle(xml_path, kp=args.kp, kd=args.kd)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 加载模型失败: {exc}")
        return 1
    print(f"[OK] {bundle.describe()}")

    # ── 输入源：真机示教臂 / 键盘虚拟示教臂 ──
    state = LeaderState()
    stop_event = threading.Event()
    worker = None
    robot = None
    virtual = None

    if args.source in ("leader", "auto"):
        robot = create_leader(args.port, args.variant, args.gripper_type, args.debug)

    if robot is not None:
        state.update(source=f"leader ({args.port or 'auto'})")
        print(f"[OK] 示教臂已连接 | variant={args.variant}")
        if args.disable_torque:
            print("[WARN] 启动即关闭示教臂力矩，请用手扶住机械臂！")
            try_disable_torque(robot)
        worker = threading.Thread(
            target=leader_collector,
            args=(robot, state, stop_event, args.fps, args.debug),
            daemon=True,
        )
        worker.start()
        print(f"[OK] 示教臂轮询线程已启动 @ {args.fps:.0f} Hz")
    else:
        if args.source == "leader":
            print("[ERROR] 无法连接示教臂。")
            print("        * 用 --list-ports 查看串口名，再加 --port COMx")
            print("        * 只想先验证仿真链路：--source virtual")
            return 1
        print("[WARN] 示教臂不可用，自动切换到 virtual（键盘模拟示教臂）")
        virtual = VirtualLeader(state)
        state.update(source="virtual (keyboard)")

    recorder = CsvRecorder(Path(args.record)) if args.record else None
    preview = CameraPreview(bundle.model, args.camera) if args.camera else None
    keys: deque = deque()
    print_controls(state.snapshot()["source"])

    reload_requested = True
    quit_requested = False
    exit_code = 0

    while reload_requested:
        reload_requested = False
        try:  # 每轮都重新读 XML，配合 r 键就能热改模型
            bundle = FollowerBundle(xml_path, kp=args.kp, kd=args.kd)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] 加载模型失败: {exc}")
            exit_code = 1
            break

        model, data = bundle.model, bundle.data
        frozen_q = np.zeros(6)      # 遥操作断开时冻结的目标角
        frozen_grip = 0.0           # 0 = 夹爪张开
        rel_offset = np.zeros(6)
        manual_enable = False
        preview_on = bool(preview is not None and preview.enabled)
        was_enabled = False
        bundle.command(frozen_q, frozen_grip)

        n_substeps = max(1, round(1.0 / (VIEWER_FPS * model.opt.timestep)))
        render_interval = n_substeps * model.opt.timestep

        def key_callback(keycode):
            """注意：该回调由 MuJoCo 的界面线程调用，只往队列里丢按键，别在这做重活。"""
            try:
                keys.append(chr(keycode).lower())
            except (ValueError, OverflowError):
                pass

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            print(f"[OK] MuJoCo 窗口已打开（{n_substeps} substeps/帧，仿真时间≈真实时间）")
            print("[OK] " + ("按 t 使能，再用键盘驱动虚拟示教臂"
                             if virtual else "等待示教臂左键（死人开关）或按 t 使能"))
            t_start = time.perf_counter()

            while viewer.is_running():
                step_start = time.perf_counter()

                # ── 按键处理 ──
                while keys:
                    key = keys.popleft()
                    if virtual is not None and virtual.handle_key(key):
                        if key == "t":
                            print(f"[TELEOP] 使能 = {'ON' if virtual.enabled else 'OFF'}")
                        continue
                    if key == "t":
                        manual_enable = not manual_enable
                        print(f"[TELEOP] 手动使能 = {'ON' if manual_enable else 'OFF'}")
                    elif key == "p":
                        snapshot = state.snapshot()
                        print(f"[STATE] {virtual.describe() if virtual else 'leader'} | "
                              f"enabled={bool(snapshot['button1']) or manual_enable} | "
                              f"leader_fps={snapshot['fps']:.1f} | follower(deg)="
                              f"{np.round(np.degrees(bundle.current_arm_positions()), 2).tolist()}")
                    elif key == "a":
                        snapshot = state.snapshot()
                        raw_now = np.array(snapshot["joint_angles"], dtype=float) * signs + offsets
                        if args.mode == "relative":
                            rel_offset = raw_now - frozen_q
                            print("[TELEOP] relative 模式：偏置已清零 "
                                  f"{np.round(np.degrees(rel_offset), 1).tolist()}")
                        else:
                            frozen_q = raw_now.copy()
                            print("[TELEOP] absolute 模式：目标已对齐到示教臂当前姿态")
                    elif key == "r":
                        print("[INFO] 重新加载 XML ……")
                        reload_requested = True
                        break
                    elif key == "v":
                        if preview is None or not preview.enabled:
                            print("[WARN] 相机预览不可用（模型里没有该 camera，或未装 opencv）")
                        else:
                            preview_on = not preview_on
                            print(f"[INFO] 相机预览 {'开启' if preview_on else '关闭'}")
                    elif key in ("q", "\x1b"):
                        print("[INFO] 收到退出按键")
                        quit_requested = True
                        break

                if reload_requested or quit_requested:
                    break

                # ── 遥操作映射 ──
                if virtual is not None:
                    virtual.push()
                snapshot = state.snapshot()
                enabled = bool(snapshot["button1"]) or manual_enable
                grip_sdk = snapshot["gripper_value"]
                if not isinstance(grip_sdk, (int, float)):
                    grip_sdk = GRIPPER_SDK_MAX
                leader_target = np.array(snapshot["joint_angles"], dtype=float) * signs + offsets

                if enabled and not was_enabled and args.mode == "relative":
                    rel_offset = leader_target - frozen_q
                    print("[TELEOP] relative 初始偏置 "
                          f"{np.round(np.degrees(rel_offset), 1).tolist()} deg")

                if enabled:
                    target = leader_target - rel_offset if args.mode == "relative" else leader_target
                    grip = gripper_sdk_to_mujoco(grip_sdk, bundle.grip_max)
                    frozen_q = target.copy()
                    frozen_grip = grip
                    if not was_enabled:
                        print("[TELEOP] ON  - follower 开始跟随示教臂")
                else:
                    target, grip = frozen_q, frozen_grip
                    if was_enabled:
                        print("[TELEOP] OFF - follower 冻结在当前姿态")
                was_enabled = enabled

                # ── 下发目标并推进仿真（按 ~1/60 s 走 n_substeps 步）──
                bundle.command(target, grip)
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()

                # ── HUD / CSV / 相机 ──
                gap = bundle.current_gripper_gap()
                draw_hud(
                    viewer, snapshot, enabled, gap,
                    "[TELEOP ACTIVE] follower tracking leader" if enabled
                    else "[TELEOP DISABLED] press deadman switch (or 't') to enable",
                )
                if recorder is not None:
                    recorder.write(time.perf_counter() - t_start, enabled,
                                   snapshot["joint_angles"], grip_sdk,
                                   bundle.current_arm_positions(), gap)
                if preview_on:
                    preview.show(data)

                elapsed = time.perf_counter() - step_start
                if elapsed < render_interval:
                    precise_sleep(render_interval - elapsed)

                if args.duration and (time.perf_counter() - t_start) >= args.duration:
                    print(f"[INFO] 已运行 {args.duration:.1f}s（--duration），自动退出")
                    quit_requested = True
                    break

        if quit_requested:
            break

    # ── 收尾 ──
    if preview is not None:
        preview.close()
    stop_event.set()
    if worker is not None:
        worker.join(timeout=2.0)
    if robot is not None:
        try:
            robot.disconnect()
            print("[OK] 示教臂已断开")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 断开示教臂时出错: {exc}")
    if recorder is not None:
        recorder.close()
    print("再见！")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())






