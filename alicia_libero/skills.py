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
LYING_GRASP_BASE = 0.010
"""**平放物体**的抓取高度：TCP 停在"物体底面 + 10mm"（米）。

⚠ 立着的物体用的是"中心 + ``grasp_tcp_offset``"，那个偏移是沿物体**局部 Z** 量的；
瓶子平放后局部 Z 变成水平，"加高度"就没有意义了。平放时指面（在 TCP 之上 0~73mm）
只要贴着底面开始往上盖就能把躺着的瓶子整段包住（实测躺姿高 56.2mm < 指面 74mm）。
底面 + 10mm 同时保证掌底（TCP−4mm）离桌面还有 6mm，不会蹭桌。"""

CONTAINER_DROP_CLEAR = 0.015
"""**平放/斜躺**被夹物放置时，在"容器口上方"这么多米处**凌空松手**（米）。

⚠ 为什么要跟立姿物区别对待（实测 §8.9）：平放的瓶子直直降到"碰篮子沿"再松手时，
有 1/6 会顶在沿上不倒进去（真实最低点 896~913mm，判分失败）；而在容器口上方 15mm
凌空松手，落体测试 9/9 全部滑到容器内底（最低点 817~827mm）。"""

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

GRASP_BAND_LADDER_TASKS: dict[str, tuple[float, ...]] = {
    # t1：把夹取高度**往下 10mm**（TCP 相对"窄带中点"再低 10mm），指面多咬住瓶子约 10mm
    "t1_ketchup_into_basket": (0.010, 0.020, 0.000, 0.045, 0.075),
    # t2：**dz=0 提到第一档**。黄油只有 17~40mm 厚，而指面全在 TCP 之上 ~73mm ——
    #     dz=+20 时指面整段都停在黄油顶面之上，只咬住顶角（浅握），抬起 25mm 的校验能过，
    #     但**搬运途中会滑落**（实测平放局因此整局失败：判分水平偏 292mm、黄油躺在桌面上）。
    #     dz=0 时指面盖住黄油 32mm（807~839mm 对 800~839.5mm），握得实。
    "t2_butter_onto_plate": (0.000, 0.020, 0.045, 0.075),
    # t9：和 t1 是**同一支瓶子**（ketchup，56.2×36.8×145.6mm、grasp_tcp_offset=+39.8mm），
    #     所以直接沿用 t1 实测出来的那档：dz=+10mm 让指面多咬住 34.5~35.5mm（+20mm 只咬 25mm，
    #     是"浅握"）。随机化之后这一档更关键 —— 瓶子在托盘里的位置/朝向每局都不同，
    #     浅握会在抬起 25mm 的"跟随校验"里**刚好过关**、却在搬运途中滑落（实测：判分偏 409mm、
    #     瓶子又坐回托盘里，而"抓取/放置"两步都报 ✓）。
    "t9_ketchup_out_of_tray": (0.010, 0.020, 0.000, 0.045, 0.075),
}
"""逐任务的抓取高度档位覆盖（覆盖 ``GRASP_BAND_LADDER``；列表顺序 = 尝试顺序）。

⚠ 为什么 t1 要单独往下压（实测，脚本都在 ``E:\\deepenv\\_tools\\``：``diag_t1_depth.py``、
``diag_t1_depth2.py``、``diag_t1_sweep.py``、``diag_t1_quality.py``、``diag_t1_align.py``）：

夹爪**指面只有 74mm 长，而且全在 TCP 之上**（指面 z ≈ TCP+0 ~ +73mm），而 ketchup 的
``grasp_tcp_offset=+39.8mm`` 把 TCP 停在 912.8mm（瓶子 803~962mm）—— 指面只"咬"住瓶子
**顶部 21~25mm**，就是用户说的"夹得浅、夹得不稳"。

但"往下压"有硬限制（这张表是逐档量出来的，不是猜的）：

| dz | 指面盖住瓶子 | 抓取时 TCP | 合爪质量（开口−物体宽 / 两指接触） | 判定 |
| --- | --- | --- | --- | --- |
| +20mm（原第一档） | 25mm | 932.5mm | +1.3mm / 1+1 指 | 稳，但太浅 |
| **+10mm（新档）** | **34.5~35.5mm** | **922.5mm** | **+2.9~3.1mm / 两指都接触** | ✅ 深了一点、质量不掉 |
| 0mm | 41~45mm | 915.9mm | +8.9mm / 只有 1 指 ✗ | 远端会"虚夹" |
| -10mm | 52mm | 905.4mm | +11.9mm / 只有 1 指 ✗ | 远端更差 |
| -20mm | 60mm | 897.4mm | 近处 3+2 指✓，远端 +7.1mm ✗ | 近处很好、远端没准头 |
| -35mm 及更深 | 62mm | 892.9mm | 下探时把瓶子带歪（实时宽 37.5→67.9mm ✗） | 远端会碰倒 |

根因（``diag_t1_align.py`` 实测）：对齐阶段结束时 TCP 与物体中心的 xy 误差在所有摆位都只有
**0.9~1.8mm**，但**低位姿态下位置伺服还有 6~9mm 静态漂移**（0.4~0.5m 处更明显），
而爪口对 37mm 瓶子的单边间隙只有 ~5mm —— 走得越深，手指就越容易蹭到瓶身
（实测远端"开口−物体宽"从 +1.3mm 恶化到 +11.9mm，甚至把瓶子顶歪）。
**这是本机械臂在远端的物理限制，不是调参能绕过的**，所以取"两个极端摆位都不失手"的最深档
（+10mm）；原来那四档全部保留在后面当回退 —— 最坏情况与改动前**完全一致**。"""


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

HOME_LIFT_EXTRA = 0.060
"""回程第一步"原地抬到安全高度"的余量：抬到**当前场景所有物体的最高点**之上 60mm。

与开局 ``抬臂准备`` 用同一个判据（那里的 0.06 是排除被抓物来算的，这里算上全部物体）。"""
HOME_TOL = 0.026
"""回程到位的允许误差（26mm）。实测 9 关回零位后 TCP 离零位点只有 **1~2mm**
（低位姿态的位置伺服静态误差），判定放到 26mm 是给不同任务/抖动留余量。"""
HOME_SETTLE = 24
"""回程最后静置帧数：让位置伺服把静态误差收干净（不再发新目标）。"""



class SkillError(RuntimeError):
    """技能执行失败（超时/没夹住/没找到物体…），带可读原因。"""


@dataclass
class Step:
    """一步技能的记录，便于在界面上显示与事后分析。"""
    label: str
    ok: bool
    detail: str = ""


JUDGE_LABEL = "判分"
"""判定任务成败的那一步的 ``label``。``回程`` 是判分之后追加的收尾动作，不参与成败。"""


def task_ok(steps: list[Step]) -> bool:
    """任务是否通过：只看**判分**那一步（回程成不成不影响"这关过没过"）。"""
    return any(s.label == JUDGE_LABEL and s.ok for s in steps)


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


def scene_rim_z(sess, exclude: str | None = None) -> float:
    """**现场实测**：其它物体（含容器沿）的最高点。

    这是修掉旧版 t9 事故的关键——搬运高度由"当前场景里所有东西的最高点"决定，
    所以"从托盘里取物"会自动把出发托盘的沿算进去，不需要任务里再写"源容器"字段。

    ``exclude=None``（回程用）= 连被抓物一起算，取**全部**物体的最高点。
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


def home_tcp(sess) -> np.ndarray:
    """**零位**（``reset()`` 之后、6 个关节目标全 0）下的末端位置，回程的终点。

    ``SimSession.reset()`` 会把当时的 TCP 记进 ``sess.home_tcp``；取不到该属性时
    退化成"当前 TCP"（回程就变成原地停住，不会报错、也不会乱动）。
    """
    return np.asarray(getattr(sess, "home_tcp", tcp(sess)), dtype=float).copy()


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


def grasp_axes_world(sess, obj: str):
    """物体**薄轴 / 长轴**在世界系下的方向（单位向量），按**实时姿态**解析算出。

    薄轴 = **大致水平**的局部轴里**最短**的那条（平行夹爪只能水平开合），长轴 = 最长的
    那条。姿态随机之后（§8.9 瓶子可能平放、还斜 45°）**不能**再用"世界 AABB 谁短"来定
    闭合方向：斜躺时 AABB 退化成一个近似正方形（实测 45° 时 129×129mm），按它猜有 50%
    概率夹错方向，手指会横着扎进瓶身。

    * 立着的瓶子/黄油（局部 Y 最薄且水平）→ 薄轴 = 局部 Y，与旧版逐字一致；
    * 平放的瓶子（三条轴都水平）→ 仍是最短的局部 Y，也不变；
    * 平放的**黄油**（t2，局部 Y 被转成竖直）→ 退而选次短的局部 Z（39.5mm，正是它横躺时
      那对侧面）；旧版这时只能回退到"世界 AABB 短边"，斜 45° 时 81.8×81.8mm 猜不准。

    返回 ``(薄轴, 长轴)``；三条局部轴里**没有**水平的（不可能发生）时返回
    ``(None, 长轴)``，调用方回退到旧逻辑。
    """
    obj_cat = catalog_object(sess.catalog, obj)
    local = np.asarray(obj_cat["size"], dtype=float)
    rot = quat_to_mat(sess.data.xquat[_body_id(sess, obj)])
    axes = rot.T                       # ⚠ axes[i] = 旋转矩阵的**第 i 列** = 局部轴 i 在世界系的方向
    long_w = np.asarray(axes[int(np.argmax(local))], dtype=float)
    horiz = [i for i in range(3) if abs(float(axes[i][2])) < 0.9]
    if not horiz:
        return None, long_w
    thin_w = np.asarray(axes[min(horiz, key=lambda i: local[i])], dtype=float)
    return thin_w, long_w


def long_axis_dir(sess, obj: str) -> tuple[float, float]:
    """物体**长轴**在世界水平面上的方向（单位向量）—— 低位微调只允许沿它修 xy。

    长轴接近竖直时（立着的瓶子）退回"薄轴的垂线"，也就是旧版 ``mask`` 的效果
    （夹紧轴方向不许动、另一个水平方向可以动）。
    """
    _, long_w = grasp_axes_world(sess, obj)
    flat = np.asarray(long_w[:2], dtype=float)
    norm = float(np.linalg.norm(flat))
    if norm > 0.2:
        return (float(flat[0] / norm), float(flat[1] / norm))
    thin = thin_axis_world(sess, obj)
    return (float(-thin[1]), float(thin[0]))


def width_along_dir(sess, obj: str, direction) -> float:
    """物体沿**任意水平方向**的投影宽度（米）—— 用碰撞盒角点投影，斜放也算得准。

    ``width_along`` 只给得出世界 X/Y 的宽度，斜躺的物体用它量"沿闭合轴的宽度"
    会得到 129mm（AABB 的对角），而真实闭合宽只有 36.8mm —— 合爪判据会误判。
    """
    pts = world_box_corners(sess, obj)[:, :2]
    d = np.asarray(direction, dtype=float)[:2]
    d = d / (np.linalg.norm(d) + 1e-12)
    proj = pts @ d
    return float(proj.max() - proj.min())


def yaw_for_axis(sess, obj: str) -> float:
    """把"沿哪条轴闭合"换算成 ``down_orientation`` 需要的偏航角（度）。

    约定（见 ``SimSession._grasp_yaw`` 与 ``AliciaIK`` 的零位推导）：
    偏航 0° 时闭合轴沿 X；要沿 Y 闭合则 +90°。这里直接取**薄轴的水平方向角**，
    所以物体立着、平放、斜 45° 都成立（立着时与旧算法一致：薄轴是局部 Y → 90°）；
    平放/斜躺时再折算到"负角"那半圈（``reachable_yaw``：躺姿只有负角够得着）。
    """
    thin = thin_axis_world(sess, obj)
    yaw = math.degrees(math.atan2(thin[1], thin[0]))
    return yaw if is_upright(sess, obj) else reachable_yaw(yaw)


def grasp_tcp_z(sess, obj: str) -> float:
    """抓取时 TCP 该停的**高度**（米）—— 随物体姿态换算法。

    * **立着**（物体的局部 Z 竖直）：``物体中心 + catalog 的 grasp_tcp_offset``，
      与旧版逐字一致；
    * **平放/斜躺**：改成 ``物体底面 + LYING_GRASP_BASE``（局部 Z 已经水平，
      沿它加高度没有意义；详见 ``LYING_GRASP_BASE`` 的说明）。
    """
    if is_upright(sess, obj):
        return float(center(sess, obj)[2] + grasp_offset_now(sess, obj))
    return float(bottom_z(sess, obj) + LYING_GRASP_BASE)

def reachable_yaw(yaw_deg: float) -> float:
    """把闭合轴偏航角折算成"**躺姿也够得着**"的那个等价代表（度）。

    ⚠ 为什么可以随便换 ±180°：平行夹爪是沿一条**直线**闭合的 —— yaw 和 yaw±180° 只是
    把两个手指对调，物理上完全一样。但机械臂的姿态不同、够得着的范围也不同：实测躺姿
    抓取（TCP 要压到 810~830mm）在 **负角** 那半圈才够得着（−45°/−135° 各 6/6），
    正角那半圈压不到（+45° 2/6、+135° 1/6，手指会偏进瓶身）。所以躺姿算出来的闭合角
    一律折算到 ``(−180°, 0°]``；立姿不折算（+90° 立姿夹得好好的，别动它）。
    """
    y = float(yaw_deg) % 360.0
    if y > 180.0:
        y -= 360.0
    elif y > 0.0:
        y -= 180.0
    return y


def is_upright(sess, obj: str) -> bool:
    """物体是否**立着**（局部 Z 大致竖直）。斜躺/平放时抓取高度、接近高度都要换算法。"""
    rot = quat_to_mat(sess.data.xquat[_body_id(sess, obj)])
    return abs(float((rot @ np.array([0.0, 0.0, 1.0]))[2])) > 0.9


def approach_z(sess, obj: str) -> float:
    """抓取前"高悬点位"的高度（物体顶面之上 ``APPROACH_H``）。

    ⚠ 平放/斜躺时物体顶面只有 56mm 高（实测躺着的瓶子顶面 861mm），"顶面 + 60mm"
    比篮子沿（941mm）还低 —— 横移过去的那段路会从篮子身上擦过去。所以非立姿时改成
    "顶面与现场最高沿取大者 + ``APPROACH_H``"。立姿时与旧版逐字一致（t1 立着的瓶子
    945.6mm 本来就比篮子高），其它任务全部走立姿分支，行为不变。
    """
    top = top_z(sess, obj)
    if is_upright(sess, obj):
        return top + APPROACH_H
    return max(top, scene_rim_z(sess, exclude=obj)) + APPROACH_H


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
             mask=(1.0, 1.0), direction=None, ref=None):
    """闭环把末端 **xy 对准目标**：逐轮把"指令点"反向偏置，抵消位置伺服的静态误差。

    ⚠ 为什么必须这么做（实测）：低位姿态下位置伺服的 xy 有 6~7mm 静态误差
    （指令 y=-0.1398，实测 -0.1466）。直接把指令点当成"末端实际位置"，会让一只手指
    偏进物体侧面 2.7mm —— 指尖压住物体顶面、下降被卡死，夹持面永远到不了物体高度
    （t2 黄油连续 4 档全失败的根因）。这里量出偏差、反向偏置指令，直到实测到位。

    ``mask``：逐轴开关（1 = 允许修正）。**低位微调时要把"夹紧轴"关掉**——
    夹紧方向上手指就贴着物体两侧，横move 会把轻物体推倒（实测布丁被推倒、AABB 宽度
    从 27mm 变成 49mm，t3 因此失败）；沿"物体长轴"方向才有自由空间。

    ``direction``：给一个**单位方向**时改成"只沿该方向修"（误差投影到它上面）。
    物理上比逐轴 mask 更对：平放并斜 45° 的瓶子，长轴不与世界轴平行，
    逐轴 mask 表达不出"只沿瓶身长轴微调"（§8.9）；轴对齐时两者结果完全一致。

    ``ref``：**被对准的那个点**（默认末端 TCP）。搬着东西对准容器时要用"被夹物的中心"
    （``ref=lambda s: center(s, obj)[:2]``）—— 物体相对 TCP 可能偏十几毫米（夹得偏一点、
    抬起时再滑一点），只对 TCP 会让瓶子重心落在篮子边沿上（§8.9 平放掉不进去的根因）。

    ``max_shift`` 限制总偏置量，避免在物体旁边"横扫"把它推走。返回是否对准。
    """
    goal = np.asarray(goal_xy, dtype=float)
    mask = np.asarray(mask, dtype=float)

    def where() -> np.ndarray:
        return np.asarray(ref(sess)[:2], dtype=float) if ref is not None else tcp(sess)[:2]

    dirv = None
    if direction is not None:
        dirv = np.asarray(direction, dtype=float)[:2]
        n = float(np.linalg.norm(dirv))
        dirv = dirv / n if n > 1e-9 else None
    bias = np.zeros(2)
    for _ in range(rounds):
        delta = goal - where()
        err = (delta @ dirv) * dirv if dirv is not None else delta * mask
        if float(np.linalg.norm(err)) < tol:
            return True
        bias = np.clip(bias + err, -max_shift, max_shift)
        cmd = np.array([goal[0] + bias[0], goal[1] + bias[1], float(z)])

        def _hold(_c, cmd=cmd):
            return cmd

        yield from ramp(sess, _hold, yaw=yaw, frames=hold, tol=0.0, step=MOVE_STEP,
                        label="xy 闭环微调")
    delta = goal - where()
    err = (delta @ dirv) * dirv if dirv is not None else delta * mask
    return float(np.linalg.norm(err)) < 0.006


YAW_CALIBRATED_TASKS = {"t2_butter_onto_plate", "t7_stack_pudding_on_can",
                        "t9_ketchup_out_of_tray"}
"""需要"**实测校准**夹爪闭合轴"的任务；其余任务保持原 ``yaw_for_axis`` 约定不动。

⚠ 为什么只给 t2 开（实测记录，见下）：``yaw`` 与"真实闭合轴"**不是**注释里写的那种
简单对应关系 —— ``AliciaIK`` 把姿态当**软约束**（位置优先 + 零空间修正），实际姿态会
偏离命令值。实测黄油（世界 AABB 76×17×40mm，薄边沿 Y）：

* ``yaw=+90°``（原约定的答案）→ 实测闭合轴 (0.94, 0.15, 0.30)，与薄边夹角 **81°**：
  手指是"横着"扎进 76mm 长的瓶身侧面 —— 用户观察到的"撞到好几次夹不住"就是这个；
* ``yaw=-90°`` → 实测闭合轴 (0.11, 0.99, -0.12)，夹角 **9.5°** ✓ 正确夹住 17mm 薄边。

所以这里改成：**把末端摆到物体上方，用两指 body 连线当"真实闭合轴"，逐个候选角实测、
挑夹角最小的那个**（闭环，不猜）。其它任务按原约定已经能过，就先不动它们。

⚠ 为什么又给 t7 开（实测记录）：布丁盒 27.4×46.3×**80.2mm**（又高又轻），
不标定时实测闭合轴是 **135.7°/115.5°/128.9°** —— 沿闭合轴的投影宽
= 27.4·cosθ + 46.3·sinθ 涨到 **70~81mm**，比爪口 50mm 还宽：手指是**夹着布丁的对角线**
（两个角点接触），布丁在爪里能自由旋转。后果是抬起/搬运中布丁歪掉
（AABB 高度 80.2 → 87.3 → **95.2mm**，歪 ~35°），落到罐顶时只有一个角支着、随后滑落到
罐旁边的桌面 —— 判分"水平偏 84mm、低于罐口 76mm"。标定后闭合轴 −0.0°、
投影宽 27.4mm（正好夹住薄边），抬起阶段最大倾斜 **+0.3mm**，判分偏 **9mm**。"""


def thin_axis_world(sess, obj: str) -> np.ndarray:
    """物体"薄边"在世界水平面上的方向（单位向量，只取 x/y）。

    旧版是"AABB 短边取世界 X 或 Y"—— **只在物体轴对齐时**成立；平放且斜 45° 时
    AABB 退化成一个近似正方形（实测 129×129mm），按它猜有 50% 概率夹错方向。
    现在按"局部薄轴 + 实时姿态"解析算（见 ``grasp_axes_world``），轴对齐时结果与旧版一致。
    """
    thin, _ = grasp_axes_world(sess, obj)
    if thin is None:                                  # 薄轴竖直：回退到旧逻辑
        ext = aabb(sess, obj)[1] - aabb(sess, obj)[0]
        return np.array([1.0, 0.0]) if ext[0] <= ext[1] else np.array([0.0, 1.0])
    flat = np.asarray(thin[:2], dtype=float)
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 1e-6 else np.array([1.0, 0.0])


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

    ⚠ 候选只差 90°，所以残余最多还有 ~45° —— 对"几乎塞满爪口"的物体不够用（实测平放的
    黄油 39.5mm 侧面 / 爪口 50mm：残余 31.5° 时投影宽 39.5·cos31.5 + 76.2·sin31.5 = 72mm，
    手指直接夹空）。试过"带符号闭环微调"（按实测误差反向补命令角）但**方向关系不稳定**
    （IK 的偏航误差随姿态变、还随 yaw 换分支），实测把本来能过的朝向也带坏了，故不做；
    改成**按实测筛朝向**（§8.9 t2：8 个朝向里只有 yaw≈−90°/−135° 那 4 个夹得住，
    于是 `lying_yaws` 只保留对应的朝向）。
    """
    thin2 = thin_axis_world(sess, obj)
    thin3 = np.array([thin2[0], thin2[1], 0.0])
    base = yaw_for_axis(sess, obj) if yaw0 is None else yaw0
    c = center(sess, obj)
    target = np.array([c[0], c[1], top_z(sess, obj) + ALIGN_H])
    best_yaw, best_ang = base, 999.0
    for cand in (base, base - 90.0, base + 180.0, base + 90.0):

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


WIDE_YAW_SCAN_TASKS = {"t3_pudding_into_ramekin"}
"""候选闭合朝向**按实测单边间隙排序、不够时再补扫 ±45° 家族**的任务（只给 t3 开）。

⚠ 为什么（实测，脚本 ``_tools\\diag_t3_poses.py`` / ``diag_t3_flow.py``）：布丁盒
27.4×46.3mm 的矩形截面几乎塞满爪口（沿闭合轴投影 = 27.4·cosθ + 46.3·sinθ，θ=40° 时
51.4mm > 爪口 50mm），而 IK 姿态是**软约束**、实测闭合轴偏航误差 0~40° **随臂的姿态变**
—— 同一个朝向换个摆位、同一个摆位换个朝向，误差都不一样。于是"只扫 0°/90°/180° 三个候选、
按固定顺序试"会漏掉唯一能夹住的那个分支。隔离实验（同一位置 (48,−190) 换朝向、
同一朝向 138° 换位置）：

| 配置 | 只扫 {0,+90,+180}（旧） | 排序 + 补扫 ±45°（新） |
| --- | --- | --- |
| θ=138° @ (48,−190) | **0/2 ✗**（三个候选都夹空，布丁留在桌上） | **2/2 ✓** |
| θ=0° / θ=228° @ (48,−190) | 2/2 ✓ / 2/2 ✓ | 2/2 ✓ / 2/2 ✓ |
| θ=138° @ (30,−120) | 2/2 ✓ | 2/2 ✓ |

实现：先用 ``plan_clear_yaw`` 把三个 90° 家族候选都量一遍并按**实测单边间隙**从大到小排序
（同一把爪子的真实投影宽，最"套得进去"的先试）；只有当三个都量到 **≤0**（都套不进去）时，
才再花 4 次探测补扫 ±45° 家族 —— 所以 90° 家族够用的局不会多付帧数。

其它任务保持原来的 3 个候选 + 原顺序，行为一字不改。"""
WIDE_YAW_SCAN_OFFSETS = (0.0, 90.0, 180.0)
"""t3 先扫的候选偏航偏移（相对 ``yaw_for_axis``）。"""
WIDE_YAW_SCAN_FALLBACK = (45.0, -45.0, 135.0, -135.0)
"""三个 90° 家族候选实测都"套不进去"时，t3 再补扫的 ±45° 家族。"""


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
    #  逐任务的档位覆盖（见 GRASP_BAND_LADDER_TASKS：t1 需要夹得更深）
    base_ladder = GRASP_BAND_LADDER_TASKS.get(task_id, GRASP_BAND_LADDER)
    #   t3 实测：dz=0 那一档才能夹到盒子的窄边（开口 26mm），dz=+20mm 只会卡在盒顶
    #   （开口停在全开 51mm = 空夹）—— 所以"几乎塞满爪口"的任务把 dz=0 提到第一档。
    ladder = (base_ladder[1], base_ladder[0]) if watch else base_ladder
    descend_step = DESCEND_STEP * 0.5 if watch else DESCEND_STEP
    last = "未尝试"
    yaw_fixed = None
    if task_id in YAW_CALIBRATED_TASKS:
        # 先用实测挑出"真能夹住"的偏航角（见 YAW_CALIBRATED_TASKS 的说明）
        yaw_fixed = yield from calibrated_yaw(sess, obj)
    yaw_seq: list = [yaw_fixed]
    yaw_clear: dict = {}
    if watch:
        # 扫 0°/90°/180°（t3 见 WIDE_YAW_SCAN_TASKS：按实测间隙排序、不够再补 ±45° 家族）：
        # 既把末端摆到对齐高度，又量出每个朝向的"预计单边间隙"
        # （实测 t3：0° 真实闭合轴与盒子窄边差 40° → 沿轴投影 51.4mm ≈ 爪口 50mm）
        wide = task_id in WIDE_YAW_SCAN_TASKS
        plan = yield from plan_clear_yaw(
            sess, obj, offsets=(WIDE_YAW_SCAN_OFFSETS if wide else (0.0, 90.0, 180.0)))
        if wide:
            plan.sort(key=lambda item: -item[1])       # 实测间隙大的先试
            if max(gap for _, gap, _ in plan) <= 0.0:  # 三个都套不进去 → 补扫 ±45° 家族
                plan += (yield from plan_clear_yaw(sess, obj,
                                                   offsets=WIDE_YAW_SCAN_FALLBACK))
                plan.sort(key=lambda item: -item[1])
        yaw_seq = [cand for cand, _, _ in plan]
        yaw_clear = {cand: gap for cand, gap, _ in plan}
    for yaw_sel in yaw_seq:
        for attempt, dz in enumerate(ladder, start=1):
            yaw = yaw_sel if yaw_sel is not None else yaw_for_axis(sess, obj)
            tag = f"（朝向 {yaw:.0f}°）" if watch else ""

            # ① 到物体正上方（高处，保证不会碰到任何东西）
            def above(_c):
                c = center(sess, obj)
                return np.array([c[0], c[1], approach_z(sess, obj)])

            yield from ramp(sess, above, yaw=yaw, frames=300,
                            label=f"接近 {obj}（第 {attempt} 次）")

            # ② 对齐到"顶面上方 28mm"（指尖刚好停在物体顶面之上），横向校正都在这一步做完
            def align(_c):
                c = center(sess, obj)
                return np.array([c[0], c[1], top_z(sess, obj) + ALIGN_H])

            yield from ramp(sess, align, yaw=yaw, frames=200, label=f"对齐 {obj} 正上方")
            c0 = center(sess, obj)
            yield from align_xy(sess, c0[:2], yaw, top_z(sess, obj) + ALIGN_H)
            # ②.5 平放/斜躺的物体还要再"贴近顶面"补一次完整 xy 对准：
            #     躺姿的抓取位比立姿低 ~90mm（810~830mm），低位伺服的静态漂移会把手指
            #     送到物体侧面上（实测远端漂 5~20mm，直接把瓶子撞走 → 整局失败）。
            #     在"顶面上方 5mm"处对准时指面还悬在物体上方（掌底 = TCP−4mm ≈ 物体顶面），
            #     横move 不会碰到它；立姿不走进这个分支（行为与旧版一致）。
            if not is_upright(sess, obj):
                yield from align_xy(sess, c0[:2], yaw, top_z(sess, obj) + 0.005, rounds=2)

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
                return np.array([xy0[0], xy0[1],
                                 max(grasp_tcp_z(sess, obj) + dz, MIN_TCP_Z)])

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
            long_axis = long_axis_dir(sess, obj)
            if not flags["touch"]:
                yield from align_xy(sess, c1[:2], yaw, float(tcp(sess)[2]),
                                    rounds=2, tol=0.0025, hold=22, max_shift=0.006,
                                    mask=long_axis, direction=long_axis)
            if task_id in STABLE_GRASP_WIDTH_TASKS:
                # 见 STABLE_GRASP_WIDTH_TASKS 的说明：实时宽度会被手指压歪而失真
                expect = float(catalog_object(sess.catalog, obj)["grasp_width"])
            else:
                # ⚠ 用**沿实测闭合轴的投影宽**（``width_along`` 只认世界 X/Y，斜躺的瓶子
                #   会被量成 AABB 对角线 129mm，合爪判据直接失效）
                expect = width_along_dir(sess, obj, thin_axis_world(sess, obj))
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
                            lambda c: np.array([c[0], c[1], approach_z(sess, obj)]),
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


def return_home(sess) -> tuple[bool, str]:
    """**回程**：张开夹爪 → 原地抬到安全高度 → 横移到零位上方 → 落到零位。返回 (是否到位, 明细)。

    ⚠ 为什么要有这一步：判分一旦完成，整条臂就停在物体上方几十毫米处（``放稳`` 那一步），
    既挡住相机视野，也没法直接接着跑下一关 —— 用户看到的就是"任务做完了，臂却杵在那里"。

    ⚠ 为什么三步的顺序不能变（实测，脚本 ``E:\\deepenv\\_tools\\diag_home.py``）：
    结束时末端下方就是刚摆好的物体，**任何横move 都会扫到它**；先**原地竖直**抬到"当前
    场景全部物体的最高点之上 60mm"（与开局 ``抬臂准备`` 同一判据）再横移就不会碰。
    实测各关物体到"零位 TCP 点"最近也有 91mm（t9 木托盘），指尖总宽约 50mm，落下去是安全的。

    ⚠ 为什么最后一步**不能**用关节空间直接收敛到零位：实测从零位 xy 起做关节空间归零，
    t5 的 TCP 会下沉到 880.9mm（原 908.2）并**刮到木托盘**、t7 刮到布丁盒/番茄酱罐 ——
    关节空间直线在笛卡尔空间是条弧，指尖会摆出去。所以全程走笛卡尔 + IK（``ramp``），
    单帧关节变化仍由 ``set_ee_target`` 里的 8° 限速兜住。

    代价（实测，脚本 ``E:\\deepenv\\_tools\\diag_home_pose2.py``）：**位置**回到零位点
    （9 关 TCP 差 **1~2mm**），但**姿态**与零位差 **24°（t1）/ 45°（t7）** —— 零位的工具是
    斜向上 45°，而 IK 按"接近轴竖直向下"出姿态（``down_orientation``，全任务都这么用），
    要一模一样只能走关节空间，而那条路会刮到物体。所以"回程"= **回到零位点上方的待命姿态**。
    """
    yield from open_gripper(sess)
    safe = scene_rim_z(sess, None) + HOME_LIFT_EXTRA
    yield from ramp(sess, lambda c: np.array([c[0], c[1], safe]),
                    yaw=sess.grasp_yaw, step=LIFT_STEP, frames=320, label="回程：抬到安全高度")

    fingers = finger_bodies(sess)

    def bumped() -> bool:
        """指尖碰到外部东西（含桌面）→ 立刻停止回程，别硬挤。"""
        return bool(body_names_in_contact(sess, fingers) - fingers)

    home = home_tcp(sess)
    yield from ramp(sess, lambda c: np.array([home[0], home[1], c[2]]),
                    yaw=sess.grasp_yaw, step=MOVE_STEP, frames=620, stop=bumped,
                    label="回程：横移到零位上方")
    yield from ramp(sess, lambda _c: home, yaw=sess.grasp_yaw, step=LIFT_STEP, frames=320,
                    stop=bumped, label="回程：落到零位")
    for _ in range(HOME_SETTLE):
        yield "回程：等伺服收稳"

    err = float(np.linalg.norm(tcp(sess) - home))
    if bumped():
        return False, f"回程途中指尖碰到外部物体，停在 {np.round(tcp(sess) * 1000).astype(int)}mm"
    ok = err < HOME_TOL
    return ok, (f"{'已' if ok else '未完全'}回到零位：TCP "
                f"{np.round(tcp(sess) * 1000).astype(int)}mm，离零位 {err * 1000:.0f}mm")


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

    ⚠ 例外（**平放/斜躺**的被夹物，实测 §8.9）：直直降到"碰到容器沿"再松手，细长物体
    会顶在沿上不倒进去（实测 t1 平放的瓶子 1/6 卡在沿口，真实最低点 896~913mm）。
    改成**在容器口上方 ``CONTAINER_DROP_CLEAR`` 处凌空松手** —— 落体测试里同样高度
    凌空松手 9/9 全部滑到容器内底（最低点 817~827mm）。立姿物仍走原来的"接触即停"，
    其它任务一字不变。
    """
    def safe_z():
        return scene_rim_z(sess, exclude=obj) + 0.03 + held_offset(sess, obj)

    lying = not is_upright(sess, obj)          # 抓住之后物体的姿态仍是平躺的
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
        z = 0.80 + 0.005 + held_offset(sess, obj)          # 物体底面落到桌面（旧行为）
        if lying and target is not None:                   # 平放物：在容器口上方凌空松手
            z = scene_rim_z(sess, exclude=obj) + CONTAINER_DROP_CLEAR + held_offset(sess, obj)
        return np.array([c[0], c[1], z])

    yield from ramp(sess, down, yaw=sess.grasp_yaw, step=DESCEND_STEP, frames=300,
                    stop=stop, label="下降放置")
    if lying and target is not None:
        # ⚠ 平放物松手前要**拿"被夹物的中心"再对准一次篮子中心**：一是低位伺服有 5~15mm
        # 静态漂移，二是物体本身相对 TCP 就可能偏十几毫米（夹得偏一点、抬起时再滑一点）。
        # 而"瓶子重心偏离篮子中心"正是它架在沿口、判分失败的**直接原因**（实测四个朝向
        # 20 局里坏 4 局，坏的那几局判分水平偏差都只有 14~16mm —— 就差这几毫米倒不进去）。
        # 此时瓶子悬在篮口上方 15mm，横move 不会碰到篮子，align_xy 自己还限了 12mm 偏置量。
        yield from align_xy(sess, tg[:2], sess.grasp_yaw, float(tcp(sess)[2]), rounds=3,
                            ref=lambda s: center(s, obj)[:2])
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

LIFT_OVER_GRASP_TASKS = {"t9_ketchup_out_of_tray"}
"""开局"抬臂准备"的高度**连被抓物一起算**（``max(其它物体最高点, 被抓物顶面) + 60mm``）的任务。

⚠ 为什么只给 t9 开（实测，脚本 ``_tools\\diag_t9_grasp.py`` 逐帧追踪）：开局/复位后 TCP 停在
零位 ``(42, 0, 908)mm``，而"抬臂准备"的高度原本只按**其它物体**算（``scene_rim_z(exclude=obj)``），
t9 里就是木托盘沿 **881mm** → 抬到 941mm 就开始横移。可 t9 的番茄酱瓶顶在 **953mm**
（抓取物自己才是现场最高的东西），于是张开的手指（指尖比 TCP 低 ~35mm、两指各在 TCP 两侧
25~35mm）正好在瓶身高度段（808~953mm）里横扫过去：

| 配置 | 旧行为（抬到 941mm 再横移） | 抬高到瓶顶之上（1013mm） |
| --- | --- | --- |
| 瓶在托盘中心、托 (25,5)、瓶朝向 45° | 第 1 帧还在 (68.8, 20.2, 881) 立着，**第 15 帧就变成躺着 (112.4, −25.1, 864)** —— 被撞倒、还被拖出托盘 88mm ✗ | 瓶子原地不动 ✓ |

撞倒之后：抓取 4 档全空（"物体宽"被量成 141mm = 高度），瓶子最后躺在托盘外，判分偏 261~579mm。
抬高之后横移发生在 1013mm（= ``approach_z``，瓶顶之上 60mm），指尖 978mm > 瓶顶 953mm ✓ 全清。
其它任务不列进这个集合，"抬臂准备"逐字不变。"""


def script_for(sess):
    """按任务的 ``kind`` 组装一条技能流水线（生成器，返回值 = 每步结果列表）。

    ``into/onto/stack/from/region`` → 抓取 + 放置；``push`` → 闭环推滑。
    ``region``（t8）没有容器，放置判据自动退化成"碰到桌面"。
    判分之后追加**回程**（``return_home``）：把臂收回零位，免得任务做完后臂停在物体上方。
    """
    task = sess.task
    obj = task["grasp_object"]
    kind = task.get("kind", "onto")
    steps: list[Step] = []
    sess.gripper = 1.0

    # 0) 先抬到"全场最高点之上"再横移：开局/复位后 TCP 可能贴着桌面，直接横扫会撞东西
    #    ⚠ 高度默认只算**其它物体**（``exclude=obj``）；t9 的瓶子本身就是现场最高的东西，
    #    必须连它一起算 —— 否则横移时手指从瓶身里扫过、把瓶子撞倒拖出托盘
    #    （见 ``LIFT_OVER_GRASP_TASKS``，只对该集合里的任务生效，其它关卡逐字不变）。
    over_grasp = task.get("id") in LIFT_OVER_GRASP_TASKS

    def prep_z():
        rim = scene_rim_z(sess, exclude=obj)
        return (max(rim, top_z(sess, obj)) if over_grasp else rim) + 0.06

    yield from ramp(sess,
                    lambda c: np.array([c[0], c[1], prep_z()]),
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
    steps.append(Step(JUDGE_LABEL, ok, msg))
    # 判分之后**回程**（用户报的"任务做完了臂还杵在物体上方"）：收尾动作，不参与成败判定
    ok_home, detail_home = yield from return_home(sess)
    steps.append(Step("回程", ok_home, detail_home))
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
            ok = task_ok(self.steps)          # 只看"判分"那一步（回程不参与成败）
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
        return task_ok(self.steps)

    def summary(self) -> str:
        if self.error:
            return "❌ " + self.error.strip().splitlines()[-1]
        if not self.steps:
            return "（没有记录）"
        lines = [f"{'✅' if s.ok else '❌'} {s.label}：{s.detail}" for s in self.steps]
        lines.append("✅ 任务完成（判分通过）" if self.ok else "❌ 任务未完成（判分未过）")
        return "\n".join(lines)

