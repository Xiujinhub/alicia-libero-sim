#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭环技能库：用**实测接触 / 跟随误差**决定动作时刻，而不是预先算死的绝对高度。

为什么要这样写
--------------
上一版（见归档 ``MOTION_PLAN.md`` §9）失败的根因几乎都是同一类：
"在**规划时**用一个算出来的绝对高度决定**执行时**的动作时刻"。

* t5：把木托盘侧壁顶面当成"内底" → 瓶底离底 55mm 就松手；
* t9：只用**目标**容器算搬运高度 → 横移时刮到**出发**容器（托盘）的沿；
* t8：目标是桌面标记区却按"盘面"算释放高度 → 从 21mm 高处掉下来；
* t6：推书任务被算成了"夹取动作点"。

本模块的做法：

1. **物体位姿每帧现读**（``data.xpos`` + catalog 的 AABB 盒），所有高度都表示成
   "相对当前物体"的偏移，物体被碰动了会自己跟上；
2. **接触即事实**：夹取/松手的时刻由 ``data.contact`` 里的真实接触对决定
   （手指—物体、物体—目标容器/桌面），不再依赖"内底/盘面"这类推断量；
3. **跟随即判据**：夹住了吗？抬起来量"物体有没有跟着 TCP 走"；放开了吗？
   抬起来量"物体有没有留在原地"；
4. 每条技能都是**生成器**，每帧 ``yield`` 一次状态文字，主循环一帧推进一步
   （``SkillRunner``），因此天然支持暂停 / 单步 / 超时 / 失败原因上报。

坐标与接口约定
--------------
* 只通过 ``SimSession`` 现有接口动机械臂：``set_ee_target()``（内部做 IK）+
  ``gripper``（0=闭合, 1=张开, 开口 ≈ 50mm × gripper）+ ``step()``；
* ``session.tcp_position()`` 是 ``tool0_site``（≈ 两指中点）；
* 物体 body 名 = 任务 ``objects`` 里的 ``name``（短名，如 ``ketchup``）；
* 本模块**不做任何写死的绝对高度**，只用 catalog 提供的"相对物体本体"的几何量
  （窄带中点 ``grasp_tcp_offset``、``size`` 等），并用**实时**物体位姿换算。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from libero_catalog import catalog_object, normalize_quat, quat_to_mat
from libero_scene import TABLE_TOP_Z
from libero_tasks import check_success, object_world_aabb

# ───────────────────────── 调参常量（全部有实测依据，改动请写入注释） ─────────────────────────
FRAME_Z = 0.002 * 8          # 主循环一帧 = 8×2ms = 16ms 仿真时间（文档用，代码里不再引用）
APPROACH_H = 0.060           # 抓取前"停到物体顶面上方"的高度（60mm，留出手腕/指爪空间）
ALIGN_H = 0.028              # 横向对齐高度：顶面上方 28mm（指尖≈20mm，刚好悬在顶面上方）
DESCEND_STEP = 0.0035        # 下降每帧位移 3.5mm ≈ 0.22 m/s
LIFT_STEP = 0.0040           # 抬升每帧位移 4.0mm ≈ 0.25 m/s
MOVE_STEP = 0.0045           # 横移每帧位移 4.5mm ≈ 0.28 m/s（分小步走，避免大位移"下沉"）
CLOSE_STEP = 0.06            # 夹爪每帧闭合量（gripper 0~1，6%/帧 ≈ 0.3s 合拢）
OPEN_STEP = 0.10             # 夹爪每帧张开量
GRIP_FORCE_STOP = 6.0        # 夹爪位置伺服力超过该值 = 夹到东西了（N）
GRIP_STALL_FRAMES = 6        # 连续这么多帧"开口几乎不变"也算夹住
GRIP_STALL_EPS = 0.0002      # 开口变化小于 0.2mm/帧 视为停滞
GRASP_FOLLOW_TOL = 0.012     # 抬起 25mm 时，物体应跟着走，允许 12mm 误差
RELEASE_STAY_TOL = 0.012     # 松手后抬起 40mm，物体留在原地，允许 12mm
GRASP_BAND_LADDER = (0.020, 0.000, 0.045, 0.075)
"""抓取高度档位：TCP 相对"窄带中点"的偏移（米），从高到低逐档试。

⚠ 为什么要逐档试（实测踩坑）：``grasp_tcp_offset`` 是物体的"窄带中点"，而**夹持面在
TCP 上方 5~70mm、掌底只比 TCP 低约 4mm**（直接用模型量出来的）。矮物体（黄油 40mm 高）
的窄带贴在桌面附近，要让夹持面套住它，TCP 必须压到很接近桌面 —— 所以第一档就按窄带中点走，
夹不住再往上抬。实测：黄油在 TCP≈0.81 时夹持面正好套住它上半段，能夹稳。"""
MIN_TCP_Z = TABLE_TOP_Z + 0.004
"""TCP 允许的最低高度：掌底只比 TCP 低 4mm，再低就是掌底压进桌面了。"""
JAW_OPEN = 0.050
"""爪口最大开口（模型里两指滑轨各 25mm）。判断"套得下去吗"必须拿它比**投影宽**。"""

CONTACT_STOP_TASKS = {"t3_pudding_into_ramekin"}
"""下降阶段"**手指一碰到物体就停、就地合爪**"的任务（其余任务保持原来的"停滞才停"）。

⚠ 为什么只给 t3 开（实测记录）：布丁盒 27.4×46.3mm 的**矩形**截面几乎塞满爪口 ——
沿闭合轴的投影宽 = 27.4·cosθ + 46.3·sinθ，而 IK 姿态是**软约束**，实测命令 yaw=0° 时
真实闭合轴与窄边差 **θ=40°** → 投影 **51.4mm ≈ 爪口 50mm（间隙为负）**，
手指根本套不下去，只能压在盒盖/顶沿上。
旧代码要等"连续 14 帧不动"才停，而命令点期间仍以 3.5mm/帧 往下走（≈49mm 过行程）——
等于用伺服力把盒子**拧了 25°**（下降前后按高度切片对比：沿闭合轴投影 51.4→39.9mm，
垂直轴投影 52.7→52.0mm），这就是用户看到的"夹爪下去就把物品撞歪"。
现在：①先扫候选朝向挑**投影最窄**的（``plan_clear_yaw``）；②一接触就停（≤1 帧、≤1.75mm
过行程）；③触到就跳过低位横向微调（避免侧推）。"""

PUSH_STOP_GAP = 0.003
"""推滑"停手间隙"：被推物体的前沿距**目标物体**前沿还剩这么多就收手（撤力 → 等待 → 判分）。

⚠ 为什么必须"碰到之前就停"（t6 实测，脚本 ``E:\\deepenv\\_tools\\diag_t6d.py``）：
书的投影半宽就有 84mm，而旧判据是"书心进盘心 85mm 内"——等于要求书**爬到盘子上**（盘高 19mm）。
实测书爬不上去（盘的碰撞体是 10 个台阶式方块、没有斜坡），而机械臂推力远大于盘子摩擦：
把盘子加重到 **920g**、只要继续顶，盘子照样被推走 100mm（三种质量 11.5/69/276g × 两种推速
都是一样）。改成"前沿快到盘沿就停"后，四种间隙（0/2/5/10mm）下**盘子位移全是 0.0mm**。
取 3mm 留一点安全余量；停手时书心在 (117,-13)mm 附近，重复性 ±1mm。"""

PUSH_TARGET_SHIFT_TOL = 0.012
"""推滑过程中目标物体被推走的允许上限（12mm）；超了直接报失败，
免得"书到了判据点、盘子却被顶跑了"这种情况被当成通过。"""



class SkillError(RuntimeError):
    """技能执行失败（超时/没夹住/没找到物体…），带可读原因。"""


@dataclass
class Step:
    """一步技能的记录，便于在界面上显示与事后分析。"""
    label: str
    ok: bool
    detail: str = ""


# ────────────────────────────── 感知：一切判据的来源 ──────────────────────────────

def _body_id(sess, name: str) -> int:
    bid = sess.model.body(name).id
    if bid < 0:
        raise SkillError(f"模型里没有物体 body '{name}'")
    return bid


def aabb(sess, name: str):
    """物体在世界系下的 AABB（凸分解盒 + 实时位姿），返回 (lo, hi)。"""
    return object_world_aabb(sess.model, sess.data, sess.catalog, name)


def center(sess, name: str) -> np.ndarray:
    lo, hi = aabb(sess, name)
    return (lo + hi) / 2.0


def bottom_z(sess, name: str) -> float:
    return float(aabb(sess, name)[0][2])


def top_z(sess, name: str) -> float:
    return float(aabb(sess, name)[1][2])


def width_along(sess, name: str, axis: str) -> float:
    """物体 AABB 在 x/y 方向的宽度（实时，物体被转过也能跟上）。"""
    lo, hi = aabb(sess, name)
    return float(hi["xy".index(axis)] - lo["xy".index(axis)])


def world_box_corners(sess, name: str) -> np.ndarray:
    """物体**每个碰撞盒在世界系下的 8 个角点**（和 ``object_world_aabb`` 用同一套几何）。"""
    obj = catalog_object(sess.catalog, name)
    bid = _body_id(sess, name)
    rot = quat_to_mat(sess.data.xquat[bid])
    org = sess.data.xpos[bid]
    points = []
    for bpos, bquat, bsize in obj["boxes"]:
        box_rot = quat_to_mat(normalize_quat(bquat))
        half = np.asarray(bsize, dtype=float)
        for sign in itertools.product((-1.0, 1.0), repeat=3):
            local = np.asarray(bpos, dtype=float) + box_rot @ (half * np.asarray(sign))
            points.append(org + rot @ local)
    return np.asarray(points)


def gap_along(sess, name_a: str, name_b: str, direction) -> float:
    """沿 ``direction`` 量：物体 A 的前沿到物体 B 的前沿还剩多少**几何间隙**（米）。

    ⚠ 为什么不用 AABB（t6 实测）：书在被推的过程中会转，AABB 从 134×110mm 涨到 **172×164mm**，
    用 AABB 投影估间隙会低估 20mm 以上（"以为还差 3mm，其实已经顶上盘子了"）。
    这里用碰撞盒角点在方向上的投影（支持函数），量到的是真间隙。
    """
    pa, pb = world_box_corners(sess, name_a), world_box_corners(sess, name_b)
    center_a = (pa.min(axis=0) + pa.max(axis=0)) / 2.0
    center_b = (pb.min(axis=0) + pb.max(axis=0)) / 2.0
    lead_a = float(np.max((pa[:, :2] - center_a[:2]) @ direction))
    lead_b = float(np.max((pb[:, :2] - center_b[:2]) @ (-direction)))
    return float(np.linalg.norm(center_b[:2] - center_a[:2]) - lead_a - lead_b)


def scene_rim_z(sess, exclude: str) -> float:
    """**现场实测**：除被抓物以外，其它物体（含容器沿）的最高点。

    这是修掉旧版 t9 事故的关键——搬运高度由"当前场景里所有东西的最高点"决定，
    所以"从托盘里取物"会自动把出发托盘的沿算进去，不需要任务里再写"源容器"字段。
    """
    names = [item.get("name", item["key"].split("/")[-1]) for item in sess.task["objects"]]
    rim = -np.inf
    for name in names:
        if name == exclude:
            continue
        rim = max(rim, top_z(sess, name))
    return float(rim if np.isfinite(rim) else 0.0)


def finger_gap(sess) -> float:
    """两指开口（米）：模型里 left/right_finger 滑轨各 25mm，开口 = 50mm × gripper。"""
    travel = float(sess.model.jnt_range[sess.model.joint("left_finger").id][1])
    q = float(sess.data.qpos[sess.model.jnt_qposadr[sess.model.joint("left_finger").id]])
    return float(2.0 * (travel - q))


def gripper_force(sess) -> float:
    """夹爪位置伺服的出力（N），用来判断"夹到东西了"。"""
    return float(max(abs(sess.data.actuator_force[act]) for act in sess.grip_act))


def tcp(sess) -> np.ndarray:
    return sess.tcp_position()


def body_names_in_contact(sess, names: set[str]) -> set[str]:
    """返回与 ``names`` 中任一 body 有接触的 **其它 body 名**（只看有没有接触）。"""
    model, data = sess.model, sess.data
    hit: set[str] = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        n1 = model.body(b1).name if b1 > 0 else "world"
        n2 = model.body(b2).name if b2 > 0 else "world"
        if n1 in names:
            hit.add(n2)
        if n2 in names:
            hit.add(n1)
    return hit


def finger_bodies(sess) -> set[str]:
    """夹爪手指所在的 **body** 名。

    ⚠ 踩坑 5：模型里 ``left_finger``/``right_finger`` 是**关节(joint)**名，
    手指的 body 其实叫 ``left_gripper``/``right_gripper``。先前按 joint 名去查 body
    永远查不到 → "手指碰到物体就停"的判据恒为假、t6 的"卡住"计数也从未启用，
    表现为手指一路钻到底把物体推来推去。这里直接从模型反查，避免再写错名字。
    """
    names: set[str] = set()
    for joint in ("left_finger", "right_finger"):
        try:
            jid = sess.model.joint(joint).id
        except Exception:  # noqa: BLE001  模型里没有这个名字
            continue
        names.add(sess.model.body(sess.model.jnt_bodyid[jid]).name)
    for extra in ("left_gripper", "right_gripper"):
        if sess.model.body(extra).id >= 0:
            names.add(extra)
    return names


def touched_by_fingers(sess, obj: str) -> bool:
    """手指是否碰到目标物体（用于"降到能夹的高度"的停止判据）。"""
    return obj in body_names_in_contact(sess, finger_bodies(sess))


def live_grasp_axis(sess, obj: str) -> str:
    """**实时**判断夹爪该沿哪个轴闭合：取物体 AABB 较短的那条水平边。"""
    return "x" if width_along(sess, obj, "x") <= width_along(sess, obj, "y") else "y"


def yaw_for_axis(sess, obj: str) -> float:
    """把"沿哪条轴闭合"换算成 ``down_orientation`` 需要的偏航角（度）。

    约定（见 ``SimSession._grasp_yaw`` 与 ``AliciaIK`` 的零位推导）：
    偏航 0° 时闭合轴沿 X；要沿 Y 闭合则 +90°。另外叠加物体自身的实时偏航。
    """
    bid = _body_id(sess, obj)
    rot = quat_to_mat(sess.data.xquat[bid])
    obj_yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return obj_yaw + (0.0 if live_grasp_axis(sess, obj) == "x" else 90.0)


def grasp_offset_now(sess, obj: str) -> float:
    """夹取时 TCP 应停的高度（相对**实时**物体中心）。

    直接用 catalog 的窄带中点 ``grasp_tcp_offset``：它是"相对物体本体"的几何量，
    物体动/转都不会失效（旧版是把这一刻的高度算成绝对值写到计划里才出的问题）。
    """
    return float(catalog_object(sess.catalog, obj)["grasp_tcp_offset"])



# ────────────────────────────── 基本技能（生成器：每帧 yield 一次） ──────────────────────────────

def ramp(sess, goal_fn, yaw=None, step=MOVE_STEP, tol=0.004, stop=None,
         frames=400, label="移动"):
    """闭环限速移动：维护一个**命令点** ``cmd``，每帧朝 ``goal_fn(cmd)`` 推进一步。

    为什么不能拿"实际 TCP"当基准（踩过的坑）
    ----------------------------------------
    最初写成 ``nxt = 实际TCP + step``：实测第一次抬臂时 IK 解在关节空间跳 40°
    （零位姿态下工具是斜的，-Z 斜向上 45°，要"竖直向下"必须大幅改姿态），
    而位置伺服有 5~30° 的滞后 → 实际 TCP 先掉 30mm，命令点又被"实际 TCP"带下去，
    于是**目标越走越低**，手指导着桌面蹭。

    现在：命令点只朝目标单调前进（像一个轨迹发生器），实际 TCP 只用于"是否到位"判定
    和 ``stop`` 判据；关节层面的"甩臂"由 ``set_ee_target`` 里的限速兜住。
    """
    cmd = tcp(sess).copy()
    for _ in range(frames):
        if stop is not None and stop():
            return True
        goal = np.asarray(goal_fn(cmd), dtype=float)
        delta = goal - cmd
        dist = float(np.linalg.norm(delta))
        if dist < tol and float(np.linalg.norm(goal - tcp(sess))) < max(tol, 0.008):
            return True
        cmd = goal if dist <= step else cmd + delta / dist * step
        sess.set_ee_target(cmd, yaw)
        yield label
    return False


def goto(sess, target, yaw=None, step=MOVE_STEP, tol=0.004, frames=400, label="移动"):
    """走到固定目标点（目标写死在外面，不随物体变）。"""
    target = np.asarray(target, dtype=float)
    return (yield from ramp(sess, lambda _cur: target, yaw=yaw, step=step, tol=tol,
                            frames=frames, label=label))


def close_gripper(sess, frames=140, force_stop=GRIP_FORCE_STOP):
    """闭合夹爪直到"夹住东西"：力超阈值 or 开口停滞；返回**实测**开口（米）。

    ⚠ 夹爪开口 = 50mm × ``gripper``（模型里每指滑轨 25mm）。夹 37mm 的瓶子时开口会停
    在 37mm 附近，而空夹会一路合到 0mm —— 这就是"夹住了没有"的物理依据，
    比旧版"按计划高度到没到"可靠得多。
    """
    prev = finger_gap(sess)
    still = 0
    for _ in range(frames):
        if sess.gripper > 0.0:
            sess.gripper = max(0.0, sess.gripper - CLOSE_STEP)
        yield f"合夹爪（开口 {finger_gap(sess) * 1000:.0f}mm）"
        gap = finger_gap(sess)
        still = still + 1 if abs(gap - prev) < GRIP_STALL_EPS else 0
        prev = gap
        if gripper_force(sess) > force_stop or still >= GRIP_STALL_FRAMES:
            return gap
        if sess.gripper <= 0.0 and still >= 2:
            return gap
    return prev


def open_gripper(sess, frames=40) -> None:
    """张开夹爪到最大（分帧，避免一秒弹开把物体弹飞）。"""
    for _ in range(frames):
        if sess.gripper < 1.0:
            sess.gripper = min(1.0, sess.gripper + OPEN_STEP)
        yield f"松夹爪（开口 {finger_gap(sess) * 1000:.0f}mm）"
        if sess.gripper >= 1.0:
            return

def held_offset(sess, obj: str) -> float:
    """当前 TCP 与物体底面的高度差（抓住之后量一次，整段搬运都用它）。"""
    return float(tcp(sess)[2] - bottom_z(sess, obj))


def align_xy(sess, goal_xy, yaw: float, z: float, rounds: int = 3,
             tol: float = 0.003, hold: int = 26, max_shift: float = 0.012,
             mask=(1.0, 1.0)):
    """闭环把末端 **xy 对准目标**：逐轮把"指令点"反向偏置，抵消位置伺服的静态误差。

    ⚠ 为什么必须这么做（实测）：低位姿态下位置伺服的 xy 有 6~7mm 静态误差
    （指令 y=-0.1398，实测 -0.1466）。直接把指令点当成"末端实际位置"，会让一只手指
    偏进物体侧面 2.7mm —— 指尖压住物体顶面、下降被卡死，夹持面永远到不了物体高度
    （t2 黄油连续 4 档全失败的根因）。这里量出偏差、反向偏置指令，直到实测到位。

    ``mask``：逐轴开关（1 = 允许修正）。**低位微调时要把"夹紧轴"关掉**——
    夹紧方向上手指就贴着物体两侧，横move 会把轻物体推倒（实测布丁被推倒、AABB 宽度
    从 27mm 变成 49mm，t3 因此失败）；沿"物体长轴"方向才有自由空间。

    ``max_shift`` 限制总偏置量，避免在物体旁边"横扫"把它推走。返回是否对准。
    """
    goal = np.asarray(goal_xy, dtype=float)
    mask = np.asarray(mask, dtype=float)
    bias = np.zeros(2)
    for _ in range(rounds):
        err = (goal - tcp(sess)[:2]) * mask
        if float(np.linalg.norm(err)) < tol:
            return True
        bias = np.clip(bias + err, -max_shift, max_shift)
        cmd = np.array([goal[0] + bias[0], goal[1] + bias[1], float(z)])

        def _hold(_c, cmd=cmd):
            return cmd

        yield from ramp(sess, _hold, yaw=yaw, frames=hold, tol=0.0, step=MOVE_STEP,
                        label="xy 闭环微调")
    return float(np.linalg.norm((goal - tcp(sess)[:2]) * mask)) < 0.006


YAW_CALIBRATED_TASKS = {"t2_butter_onto_plate"}
"""需要"**实测校准**夹爪闭合轴"的任务；其余任务保持原 ``yaw_for_axis`` 约定不动。

⚠ 为什么只给 t2 开（实测记录，见下）：``yaw`` 与"真实闭合轴"**不是**注释里写的那种
简单对应关系 —— ``AliciaIK`` 把姿态当**软约束**（位置优先 + 零空间修正），实际姿态会
偏离命令值。实测黄油（世界 AABB 76×17×40mm，薄边沿 Y）：

* ``yaw=+90°``（原约定的答案）→ 实测闭合轴 (0.94, 0.15, 0.30)，与薄边夹角 **81°**：
  手指是"横着"扎进 76mm 长的瓶身侧面 —— 用户观察到的"撞到好几次夹不住"就是这个；
* ``yaw=-90°`` → 实测闭合轴 (0.11, 0.99, -0.12)，夹角 **9.5°** ✓ 正确夹住 17mm 薄边。

所以这里改成：**把末端摆到物体上方，用两指 body 连线当"真实闭合轴"，逐个候选角实测、
挑夹角最小的那个**（闭环，不猜）。其它任务按原约定已经能过，就先不动它们。"""


def thin_axis_world(sess, obj: str) -> np.ndarray:
    """物体"薄边"在世界水平面上的方向（单位向量，只取 x/y）。"""
    ext = aabb(sess, obj)[1] - aabb(sess, obj)[0]
    return np.array([1.0, 0.0]) if ext[0] <= ext[1] else np.array([0.0, 1.0])


def measured_closing_axis(sess) -> np.ndarray:
    """实测闭合轴：两个手指 body 原点连线（比"按约定推"可靠）。"""
    pl = sess.data.xpos[sess.model.body("left_gripper").id]
    pr = sess.data.xpos[sess.model.body("right_gripper").id]
    vec = np.asarray(pr) - np.asarray(pl)
    return vec / (np.linalg.norm(vec) + 1e-12)


def calibrated_yaw(sess, obj: str, yaw0: float | None = None,
                   probe_frames: int = 70) -> float:
    """实测挑选偏航角（生成器）：对候选角逐个摆好、量真实闭合轴，返回与薄边最平行者。

    摆动发生在"物体上方 28mm"处（``ALIGN_H``），此时指尖还悬在物体上方，不会撞到它。
    ``yaw0`` 默认用物体当前偏航；候选 = yaw0 + {0, ±90, 180}。
    """
    thin2 = thin_axis_world(sess, obj)
    thin3 = np.array([thin2[0], thin2[1], 0.0])
    base = yaw_for_axis(sess, obj) if yaw0 is None else yaw0
    best_yaw, best_ang = base, 999.0
    for cand in (base, base - 90.0, base + 180.0, base + 90.0):
        c = center(sess, obj)
        target = np.array([c[0], c[1], top_z(sess, obj) + ALIGN_H])

        def _hold(_c, t=target):
            return t

        yield from ramp(sess, _hold, yaw=cand, frames=probe_frames, tol=0.0,
                        label=f"校准闭合轴 {cand:.0f}°")
        axis = measured_closing_axis(sess)
        ang = float(np.degrees(np.arccos(np.clip(abs(axis @ thin3), -1.0, 1.0))))
        if ang < best_ang:
            best_yaw, best_ang = cand, ang
        if ang < 12.0:                      # 已经足够正，不用再试
            break
    return best_yaw


STABLE_GRASP_WIDTH_TASKS = {"t3_pudding_into_ramekin"}
"""开口判据改用 catalog 里**建场景时量好的** ``grasp_width``，而不是"实时 AABB 宽度"。

⚠ 为什么只给 t3 开（实测记录）：夹取过程里"实时宽度"会被自己的手指污染 ——
布丁盒 27.4mm 宽，指尖一压就歪，实测闭合那一刻的实时宽度依次是 **65.7 / 56.0 / 52.9mm**，
而**开口**却是 26.4 / 31.9 / 32.1mm、**夹持力** 2.8~3.2N，且抬起后物体底分别跟随了
+19.8 / +17.5 / +16.8mm（= 真的夹住了、也抬起来了）。可判据拿"失真的宽度"当参照，
于是把三次成功全判成失败 → 主动在空中松爪把盒子摔回桌面，越摔越歪，第 4 次只能在
896mm 高处空夹（开口 -0.2mm）——用户看到的"夹住了几次又松开"就是这个。

改用 catalog 的 ``grasp_width``（27.4mm，与三次实测开口 26.4/31.9/32.1 都吻合，而与
空夹的 -0.2mm 差 27mm）后，四次判定全部正确。其它任务按实时宽度已能过，就不动它们。"""


def jaws_clearance(sess, obj: str, axis) -> tuple[float, float]:
    """沿"实测闭合轴"算物体的**投影宽度**与**单边间隙**（米），用于判断"套得下去吗"。

    ⚠ 不能拿"爪口 50mm vs 盒宽 27.4mm"直接比：矩形截面的投影宽 = 27.4·cosθ + 46.3·sinθ，
    θ 是闭合轴与盒子窄边的夹角（IK 姿态是软约束，实测 t3 的 θ≈40° → 投影 51.4mm，
    比爪口还宽）。物体按世界轴对齐（本项目的物体 yaw 都是 0）。
    """
    ext = aabb(sess, obj)[1] - aabb(sess, obj)[0]
    short, long_ = sorted((float(ext[0]), float(ext[1])))
    u = np.asarray(axis, dtype=float).copy()
    u[2] = 0.0
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return -1.0, 0.0
    u = u / n
    ang = math.asin(min(1.0, abs(float(u[1]))))          # 世界 x 与闭合轴的夹角
    width = short * math.cos(ang) + long_ * math.sin(ang)
    return (JAW_OPEN - width) / 2.0, width


def plan_clear_yaw(sess, obj: str, offsets=(0.0, 90.0, 180.0),
                   settle: int = 40, label=True):
    """候选偏航角列表 + 每个朝向**实测**的"预计单边间隙"（米），供明细显示与排序参考。

    在物体上方 ``ALIGN_H`` 处逐个摆好、用两指 body 连线量真实闭合轴（见
    ``CONTACT_STOP_TASKS`` 的实测说明）。返回 ``[(yaw, 单边间隙, 投影宽), ...]``。
    """
    base = yaw_for_axis(sess, obj)
    plan: list[tuple[float, float, float]] = []
    for off in offsets:
        cand = base + off
        c = center(sess, obj)
        target = np.array([c[0], c[1], top_z(sess, obj) + ALIGN_H])

        def _hold(_c, t=target):
            return t

        yield from ramp(sess, _hold, yaw=cand, frames=settle, tol=0.0,
                        label=(f"试朝向 {cand:.0f}°" if label else f"到 {obj} 上方"))
        gap_side, width = jaws_clearance(sess, obj, measured_closing_axis(sess))
        plan.append((cand, gap_side, width))
    return plan


def grasp_object(sess, obj: str) -> tuple[bool, str]:
    """抓取：接近 → 对齐 → **竖直下降（xy 冻结）** → 闭合 → 抬起验证"物体是否跟着走"。

    判据全部来自实测：闭合后的**开口**要落在物体宽度附近；抬起 25mm 后物体的底面要跟着
    升高 —— 两个都对才算抓住（旧版只看"高度到位了没"）。一档不行就换下一档
    （``GRASP_BAND_LADDER``）。物体"几乎塞满爪口"的任务（``CONTACT_STOP_TASKS``）还会
    先量出各候选朝向的实际间隙（``plan_clear_yaw``），并且**手指一碰到就停**（避免推歪）。
    """
    task_id = sess.task.get("id")
    watch = task_id in CONTACT_STOP_TASKS
    #   t3 实测：dz=0 那一档才能夹到盒子的窄边（开口 26mm），dz=+20mm 只会卡在盒顶
    #   （开口停在全开 51mm = 空夹）—— 所以"几乎塞满爪口"的任务把 dz=0 提到第一档。
    ladder = (GRASP_BAND_LADDER[1], GRASP_BAND_LADDER[0]) if watch else GRASP_BAND_LADDER
    descend_step = DESCEND_STEP * 0.5 if watch else DESCEND_STEP
    last = "未尝试"
    yaw_fixed = None
    if task_id in YAW_CALIBRATED_TASKS:
        # 先用实测挑出"真能夹住"的偏航角（见 YAW_CALIBRATED_TASKS 的说明）
        yaw_fixed = yield from calibrated_yaw(sess, obj)
    yaw_seq: list = [yaw_fixed]
    yaw_clear: dict = {}
    if watch:
        # 扫 0°/90°/180°：既把末端摆到对齐高度，又量出每个朝向的"预计单边间隙"
        # （实测 t3：0° 真实闭合轴与盒子窄边差 40° → 沿轴投影 51.4mm ≈ 爪口 50mm）
        plan = yield from plan_clear_yaw(sess, obj)
        yaw_seq = [cand for cand, _, _ in plan]
        yaw_clear = {cand: gap for cand, gap, _ in plan}
    for yaw_sel in yaw_seq:
        for attempt, dz in enumerate(ladder, start=1):
            yaw = yaw_sel if yaw_sel is not None else yaw_for_axis(sess, obj)
            tag = f"（朝向 {yaw:.0f}°）" if watch else ""

            # ① 到物体正上方（高处，保证不会碰到任何东西）
            def above(_c):
                c = center(sess, obj)
                return np.array([c[0], c[1], top_z(sess, obj) + APPROACH_H])

            yield from ramp(sess, above, yaw=yaw, frames=300,
                            label=f"接近 {obj}（第 {attempt} 次）")

            # ② 对齐到"顶面上方 28mm"（指尖刚好停在物体顶面之上），横向校正都在这一步做完
            def align(_c):
                c = center(sess, obj)
                return np.array([c[0], c[1], top_z(sess, obj) + ALIGN_H])

            yield from ramp(sess, align, yaw=yaw, frames=200, label=f"对齐 {obj} 正上方")
            c0 = center(sess, obj)
            yield from align_xy(sess, c0[:2], yaw, top_z(sess, obj) + ALIGN_H)

            # ③ **竖直下降：xy 冻结**，不再每帧追物体。
            #    ⚠ 踩坑 3：先前下降时 xy 还在追"物体实时中心"，而手指只有 50mm 间距、
            #    黄油/布丁这类轻物体（素材密度 100 ≈ 泡沫）一碰就跑 —— 实测手指把 18mm 厚的
            #    黄油块撞倒，实时 AABB 宽度从 18mm 变成 47mm，再合爪就夹空（t2/t3 失败根因）。
            #    ⚠ 踩坑 6：停止判据**不能用"手指一碰就停"**——掌底只比 TCP 低 4mm，
            #    矮物体（黄油 40mm 高）下降时掌底先碰到它的顶面，一停就落在"夹持面还在物体
            #    上方"的位置，合爪只能夹到顶上一条边（实测开口停在 19mm 但抬起就滑脱）。
            #    现在改成：降到目标高度为止，若**末端停滞**（被物体/桌面顶住）也停。
            #    ⚠ 例外（``CONTACT_STOP_TASKS``，仅 t3）：布丁盒几乎塞满爪口，手指必然先压到
            #    盒盖，而"停滞 14 帧"期间命令点还会多走 49mm、把盒子拧歪 25°。所以这里
            #    **一接触就停**（≤1 帧、≤1.75mm 过行程），就地合爪。
            aligned = tcp(sess)
            xy0 = aligned[:2].copy()
            stalled = _stall_watch(tol=0.0004, need=14)
            flags = {"touch": False}

            def band(_c):
                c = center(sess, obj)
                return np.array([xy0[0], xy0[1],
                                 max(c[2] + grasp_offset_now(sess, obj) + dz, MIN_TCP_Z)])

            def _stop(flags=flags, stalled=stalled):
                if watch and touched_by_fingers(sess, obj):
                    flags["touch"] = True
                    return True
                return stalled(tcp(sess))

            yield from ramp(sess, band, yaw=yaw, frames=200, step=descend_step,
                            stop=_stop,
                            label=f"竖直下降对准 {obj}（档 {attempt}）{tag}")
            # ③.5 低位再微调一次 xy：姿态变了，伺服的静态误差也变了（实测低位会漂 6~7mm）。
            #      但**只允许沿物体长轴修**——夹紧轴方向手指正贴着物体两侧，
            #      横move 会把轻物体推倒（实测布丁 AABB 宽度 27→49mm，t3 就是这么挂的）。
            #      手指已经搭在物体上时直接跳过（任何横move 都是侧推）。
            c1 = center(sess, obj)
            long_axis = (1.0, 0.0) if live_grasp_axis(sess, obj) == "y" else (0.0, 1.0)
            if not flags["touch"]:
                yield from align_xy(sess, c1[:2], yaw, float(tcp(sess)[2]),
                                    rounds=2, tol=0.0025, hold=22, max_shift=0.006,
                                    mask=long_axis)
            axis = live_grasp_axis(sess, obj)
            if task_id in STABLE_GRASP_WIDTH_TASKS:
                # 见 STABLE_GRASP_WIDTH_TASKS 的说明：实时宽度会被手指压歪而失真
                expect = float(catalog_object(sess.catalog, obj)["grasp_width"])
            else:
                expect = width_along(sess, obj, axis)
            sess.grasp_yaw = yaw
            gap = yield from close_gripper(sess)
            z0_obj, z0_tcp = bottom_z(sess, obj), float(tcp(sess)[2])
            yield from ramp(sess,
                            lambda c: np.array([c[0], c[1], z0_tcp + 0.025]),
                            yaw=yaw, step=LIFT_STEP, frames=60, label=f"抬起验证 {obj}")
            dz_obj = bottom_z(sess, obj) - z0_obj
            dz_tcp = float(tcp(sess)[2]) - z0_tcp
            following = abs(dz_obj - dz_tcp) < GRASP_FOLLOW_TOL and dz_obj > 0.008
            gap_ok = abs(gap - expect) < 0.014
            gap_side = yaw_clear.get(yaw, 0.0)
            if following and gap_ok:
                extra = "（触到盒顶即夹）" if flags["touch"] else ""
                return True, (f"夹住 {obj}：开口 {gap * 1000:.0f}mm"
                              f"（物体宽 {expect * 1000:.0f}mm），"
                              f"抬起跟随 {dz_obj * 1000:.0f}mm{extra}")
            last = (f"第 {attempt} 次未夹住{tag}：开口 {gap * 1000:.0f}mm"
                    f"（物体宽 {expect * 1000:.0f}mm），抬起后物体只动 {dz_obj * 1000:.0f}mm")
            if watch:
                enough = "够" if gap_side > 0.001 else "不足"
                last += (f"；该朝向预计单边间隙 {gap_side * 1000:+.1f}mm（{enough}）")
            if flags["touch"]:
                last += "；下降时手指已触到物体"
            yield from open_gripper(sess)
            yield from ramp(sess,
                            lambda c: np.array([c[0], c[1], top_z(sess, obj) + APPROACH_H]),
                            yaw=yaw, step=LIFT_STEP, frames=80, label="退回重试")
    return False, last


def release_object(sess, obj: str) -> tuple[bool, str]:
    """松手并验证：抬离 40mm 后物体应留在原地（没被带回 = 真放下了）。"""
    yield from open_gripper(sess)
    z0_obj, z0_tcp = bottom_z(sess, obj), float(tcp(sess)[2])
    yield from ramp(sess,
                    lambda c: np.array([c[0], c[1], z0_tcp + 0.040]),
                    step=LIFT_STEP, frames=80, label=f"抬离 {obj}")
    dz_obj = bottom_z(sess, obj) - z0_obj
    if abs(dz_obj) < RELEASE_STAY_TOL:
        return True, f"已放下 {obj}（抬离 40mm，物体只动 {dz_obj * 1000:+.0f}mm）"
    return False, f"松手后物体仍跟着走 {dz_obj * 1000:+.0f}mm（可能卡在爪里）"


def _stall_watch(tol: float = 0.0006, need: int = 15):
    """返回一个"末端/物体连续若干帧不动"的判据闭包（用于"卡住了"的兜底停止）。"""
    state = {"ref": None, "n": 0}

    def check(value) -> bool:
        ref = state["ref"]
        if ref is None:
            state["ref"] = np.asarray(value, dtype=float)
            return False
        moved = float(np.linalg.norm(np.asarray(value, dtype=float) - ref))
        state["n"] = state["n"] + 1 if moved < tol else 0
        state["ref"] = np.asarray(value, dtype=float)
        return state["n"] >= need

    return check


def place_object(sess, obj: str, target: str | None) -> tuple[bool, str]:
    """放置：抬到"现场最高沿 + 30mm" → 移到目标正上方 → 下降到**接触**为止 → 松手。

    ``target=None`` 表示"放到桌面/标记区"（t8）——接触判据自动退化成"碰到世界/桌子"，
    不再需要任务里填一个假的 ``target_object``（这正是旧版 t8 从 21mm 高处掉下来的原因）。
    ``target`` 存在时，判据是"**被夹物**碰到目标物"（物体—容器/物体—罐顶），
    不依赖任何"内底/盘面"的推算（这正是旧版 t5 离底 55mm 松手的原因）。
    """
    def safe_z():
        return scene_rim_z(sess, exclude=obj) + 0.03 + held_offset(sess, obj)

    yield from ramp(sess,
                    lambda c: np.array([c[0], c[1], safe_z()]),
                    yaw=sess.grasp_yaw, step=LIFT_STEP, frames=200, label="抬到安全高度")

    def above(_cur):
        c = center(sess, target) if target else np.array([*sess.task["success"]["xy"], 0.0])
        return np.array([c[0], c[1], safe_z()])

    yield from ramp(sess, above, yaw=sess.grasp_yaw, frames=520, label="移到目标上方")
    # 目标上方做一次 xy 闭环对准（此时高悬、没有可撞的东西，最安全）
    tg = center(sess, target) if target else np.array([*sess.task["success"]["xy"], 0.0])
    yield from align_xy(sess, tg[:2], sess.grasp_yaw, float(tcp(sess)[2]), rounds=2)
    watch = {obj}

    def touched() -> bool:
        others = body_names_in_contact(sess, watch)
        if target is not None and target in others:
            return True
        return "world" in others

    stalled = _stall_watch(tol=0.0008, need=20)
    how = {"why": "超时"}

    def stop() -> bool:
        if touched():
            how["why"] = "接触"
            return True
        if stalled(center(sess, obj)):
            how["why"] = "卡住（物体不再下降）"
            return True
        return False

    def down(_cur):
        c = center(sess, target) if target else np.array([*sess.task["success"]["xy"], 0.0])
        return np.array([c[0], c[1], 0.80 + 0.005 + held_offset(sess, obj)])

    yield from ramp(sess, down, yaw=sess.grasp_yaw, step=DESCEND_STEP, frames=300,
                    stop=stop, label="下降放置")
    for _ in range(12):                     # 放稳再松手（不额外下压，避免把容器压翻）
        yield f"放稳（{how['why']}）"
    ok, detail = yield from release_object(sess, obj)
    where = target or "桌面/标记区"
    return ok, f"{how['why']}于 {where}；{detail}"


def push_object(sess, obj: str, speed: float = 0.0025) -> tuple[bool, str]:
    """推滑（t6 专用）：**高处绕到物体后方 → 竖直下降 → 推到目标物体前沿前停住**。

    判据就是任务的 ``check_success``（书心落在"盘沿前"的判据点上）——不依赖任何绝对高度，
    也不用"夹取高度"那套（旧版把推任务算成夹取动作，对 134mm 宽的书毫无意义）。
    推进方向每帧朝**目标物体当前中心**重新对准；到前沿前 ``PUSH_STOP_GAP`` 就撤力停手，
    并全程守着"目标物体不许被推走"（``PUSH_TARGET_SHIFT_TOL``）。
    """
    target = sess.task.get("target_object")
    spec_xy = np.asarray(sess.task["success"]["xy"], dtype=float)

    def state():
        """(物体中心, 朝目标物体中心的单位方向, 到目标的距离)。"""
        c = center(sess, obj)
        aim = center(sess, target)[:2] if target else spec_xy[:2]
        d = aim - c[:2]
        n = float(np.linalg.norm(d))
        return c, (d / n if n > 1e-6 else np.array([1.0, 0.0])), n

    c, d, _ = state()
    half = (abs(d[0]) * width_along(sess, obj, "x") / 2
            + abs(d[1]) * width_along(sess, obj, "y") / 2)
    # ⚠ 推的 TCP 高度要压到"物体中部"：夹持面在 TCP 上方 10~50mm，停在物体顶面上方
    #   10mm 时指尖只能压住物体**顶面**（往下压而非往前推），实测 214 帧后"贴上了但推不动"。
    down_z = max(MIN_TCP_Z, top_z(sess, obj) - 0.025)
    back = np.array([c[0] - d[0] * (half + 0.05), c[1] - d[1] * (half + 0.05), down_z])
    yaw = math.degrees(math.atan2(d[1], d[0]))    # 闭合轴沿推进方向：后指推、前指让位

    # ⚠ 踩坑 7（t6 实测）：原先是**一步** ``goto(back)`` 直接到"后方 804mm"，等于贴着桌面
    #   横穿过去 —— 实测路上先把书撞了 20.7mm，而且指尖一路贴着书，推的时候很容易被
    #   "卡住"判据提前收场，表现就是用户看到的"书本没有推动"。
    #   现在改成：先在**高处**平移到位（现场最高沿/物体顶面之上），再**竖直下降**。
    travel_z = max(scene_rim_z(sess, exclude=obj) + 0.05, top_z(sess, obj) + 0.05)
    yield from goto(sess, np.array([back[0], back[1], travel_z]), yaw=yaw,
                    frames=420, label=f"高处绕到 {obj} 后方")
    yield from goto(sess, back, yaw=yaw, step=DESCEND_STEP, frames=200,
                    label=f"竖直降到 {obj} 后方")

    anchor = center(sess, target)[:2].copy() if target else None
    last = center(sess, obj)[:2].copy()
    stuck = 0
    touched = False
    gap = gap_along(sess, obj, target, d) if target else 0.0
    # ⚠ 踩坑 9（t6 实测）：判据从"书心压到盘心 85mm 内"（= 要求书爬上盘子，物理上做不到）
    #   改成"推到盘沿前停住"，所以停手条件是**几何间隙**，不是"check_success 通过"。
    for _ in range(900):
        c, d, dist = state()
        if target is not None:
            gap = gap_along(sess, obj, target, d)
            shoved = float(np.linalg.norm(center(sess, target)[:2] - anchor))
            if shoved > PUSH_TARGET_SHIFT_TOL:
                return False, f"{target} 被推走了 {shoved * 1000:.0f}mm（推力远大于它的摩擦）"
            if gap <= PUSH_STOP_GAP:
                break                            # 到前沿了：撤力、等待、判分
        touched = touched or touched_by_fingers(sess, obj)
        cur = tcp(sess)
        step = d * speed
        sess.set_ee_target(np.array([cur[0] + step[0], cur[1] + step[1], cur[2]]), yaw)
        yield f"推 {obj}（距目标 {dist * 1000:.0f}mm，前沿间隙 {gap * 1000:.0f}mm）"
        now = center(sess, obj)[:2]
        # ⚠ 踩坑 4："卡住"必须在**指尖已经贴上物体之后**才开始计数：
        #   先前从第一帧就数"物体没动"，而接近阶段（要走 50mm 才碰到书）物体本来就不动，
        #   于是 60 帧就误判"推不动了"直接收场（t6 实测 169 帧结束）。
        stuck = stuck + 1 if (touched and float(np.linalg.norm(now - last)) < 0.0004) else 0
        last = now
        if stuck >= 60:
            break
    else:
        return False, "推进超时"

    # 撤力：目标往回退 3mm 卸掉推力，再等物体停稳（贴着盘沿停住时，残余推力会把盘子顶走）
    for _ in range(10):
        cur = tcp(sess)
        sess.set_ee_target(np.array([cur[0] - d[0] * 0.003, cur[1] - d[1] * 0.003, cur[2]]), yaw)
        yield "收手（撤掉推力）"
    for _ in range(45):
        yield "等待稳定（物体滑停在盘沿前）"

    if target is not None:
        _, d_end, _ = state()
        gap = gap_along(sess, obj, target, d_end)
        shoved = float(np.linalg.norm(center(sess, target)[:2] - anchor))
        if shoved > PUSH_TARGET_SHIFT_TOL:
            return False, f"{target} 被推走了 {shoved * 1000:.0f}mm（推力远大于它的摩擦）"
    ok, msg = check_success(sess.task, sess.model, sess.data, sess.catalog)
    if not ok:
        why = "指尖已贴上但推不动" if stuck >= 60 else "停在了目标前沿前"
        return False, f"{why}，且没进判据（{msg}）"
    where = f"{target} 前沿前 {gap * 1000:.0f}mm" if target else "判据点"
    return True, f"已把 {obj} 推到 {where} 停住（{msg}）"


# ────────────────────────────── 每关的动作骨架 ──────────────────────────────

def script_for(sess):
    """按任务的 ``kind`` 组装一条技能流水线（生成器，返回值 = 每步结果列表）。

    ``into/onto/stack/from/region`` → 抓取 + 放置；``push`` → 闭环推滑。
    ``region``（t8）没有容器，放置判据自动退化成"碰到桌面"。
    """
    task = sess.task
    obj = task["grasp_object"]
    kind = task.get("kind", "onto")
    steps: list[Step] = []
    sess.gripper = 1.0

    # 0) 先抬到"全场最高点之上"再横移：开局/复位后 TCP 可能贴着桌面，直接横扫会撞东西
    yield from ramp(sess,
                    lambda c: np.array([c[0], c[1],
                                        scene_rim_z(sess, exclude=obj) + 0.06]),
                    yaw=sess.grasp_yaw, step=LIFT_STEP, frames=200, label="抬臂准备")
    if kind == "push":
        ok, detail = yield from push_object(sess, obj)
        steps.append(Step("推滑", ok, detail))
    else:
        ok, detail = yield from grasp_object(sess, obj)
        steps.append(Step("抓取", ok, detail))
        if ok:
            target = task.get("target_object") if kind != "region" else None
            ok2, detail2 = yield from place_object(sess, obj, target)
            steps.append(Step("放置", ok2, detail2))
        else:
            steps.append(Step("放置", False, "未夹住，跳过放置"))
    for _ in range(45):                     # 让物体静置（判分要的是稳定状态）
        yield "等待稳定"
    ok, msg = check_success(task, sess.model, sess.data, sess.catalog)
    steps.append(Step("判分", ok, msg))
    return steps


class SkillRunner:
    """把一条技能生成器挂到主循环上：每帧 ``step()`` 推进一步，可暂停 / 单步。"""

    def __init__(self, sess) -> None:
        self.sess = sess
        self.gen = None
        self.active = False
        self.paused = False
        self.finished = False
        self.status = "就绪"
        self.error = ""
        self.steps: list[Step] = []

    # ── 控制 ──
    def start(self) -> None:
        self.gen = script_for(self.sess)
        self.active, self.paused, self.finished = True, False, False
        self.error, self.steps, self.status = "", [], "开始"

    def pause(self) -> None:
        if self.active:
            self.paused = True
            self.status = "已暂停（" + self.status + "）"

    def resume(self) -> None:
        if self.active:
            self.paused = False

    def stop(self) -> None:
        self.active = self.paused = False
        self.status = "已停止"

    def step(self) -> str:
        """推进一帧；返回当前状态文字。"""
        if not self.active or self.paused or self.gen is None:
            return self.status
        try:
            label = next(self.gen)
            if isinstance(label, str):
                self.status = label
        except StopIteration as exc:
            self.active, self.finished = False, True
            self.steps = list(exc.value or [])
            ok = bool(self.steps) and self.steps[-1].ok
            self.status = "完成 ✅" if ok else "完成 ❌（见下方明细）"
        except SkillError as exc:
            self.active, self.finished, self.error = False, True, str(exc)
            self.status = f"技能失败：{exc}"
        except Exception as exc:  # noqa: BLE001  （界面不能因为规划异常整块崩掉）
            import traceback

            self.active, self.finished = False, True
            self.error = traceback.format_exc()
            self.status = f"异常：{exc}"
        return self.status

    # ── 展示 ──
    @property
    def ok(self) -> bool:
        return bool(self.steps) and self.steps[-1].ok

    def summary(self) -> str:
        if self.error:
            return "❌ " + self.error.strip().splitlines()[-1]
        if not self.steps:
            return "（没有记录）"
        lines = [f"{'✅' if s.ok else '❌'} {s.label}：{s.detail}" for s in self.steps]
        lines.append("✅ 任务完成（判分通过）" if self.ok else "❌ 任务未完成（判分未过）")
        return "\n".join(lines)

