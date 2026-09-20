#!/usr/bin/env python3
"""RevA1-sim 的控制核心（不依赖任何界面框架）。

分工
----
* **本模块**：相机、运动插值、IK、仿真推进、到点误差回报——纯逻辑，可以被 ``--selftest``
  直接驱动（不用开窗口）；
* ``revA1_gui.py``：PySide6 界面（主推），只做"控件事件 → 方法调用"和"状态 → 控件"；
* ``interactive_control_tk.py``：旧的 tkinter 界面（保留可跑，逻辑同源）。

三个核心对象
------------
``Camera``  轨道相机（azimuth / elevation / distance / lookat），语义与 MuJoCo 一致；
``Motion``  一次运动：关节空间 smoothstep 插值；给了 ``waypoints`` 就沿着它们（笛卡尔直线路点）走；
``ArmSim``  模型 + 数据 + 6 个位置伺服 + 指令来源（单关节 / 关节 P2P / 笛卡尔直线 / IK）。

为什么"点到点"要改成笛卡尔直线
------------------------------
旧版点到点是**关节空间**插值（6 个关节各自 smoothstep），工具尖走的是一条弧线，
肉眼看不出"从 A 到 B 的直线"，所以很不直观。现在默认走 :meth:`ArmSim.plan_line`：

    沿"当前工具尖 → 目标点"的**直线**每 1 cm 解一次 IK，得到一串关节路点，
    再用 smoothstep 在时间上回放（首尾速度 0）。

于是画面里工具尖就是沿直线过去的（与真机 MoveL 语义一致），而且**求解过程可以量**：
直线度、每点 IK 残差、到位误差都能报出来（见 ``--selftest``）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl()
import mujoco  # noqa: E402  (必须在设置 MUJOCO_GL 之后)

import revA1_spec as spec  # noqa: E402
import sim_ik as ik  # noqa: E402

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 判定阈值 / 步长
# IK 判定：位置 5 mm、工具轴 1.5° 以内就算"解出来了"。
# （残差来自数值迭代 + 关节限位；实测 home/前伸/侧向/低位四个预设都能到 0.1 mm、0.00°，
#   留这么宽的阈值只是为了容忍极端姿态。）
IK_TOL_POS_M = 5e-3
IK_TOL_AXIS_RAD = np.radians(1.5)
LINE_STEP_M = 0.01          # 笛卡尔直线：每 1 cm 解一个路点
LINE_MIN_POINTS = 9         # 路点下限（目标很近时也要有中间点，否则看不出直线）
LINE_MAX_POINTS = 400       # 路点上限（防手快输个 10 m 把界面卡死）
LINE_MAX_ERR_M = 3e-3       # 单个路点 IK 残差超过它就算这一步"没解出来"

# 机体尺度（用于"目标是否在工作空间附近"的粗筛，不是硬约束）
REACH_MIN_M = 0.05
REACH_MAX_M = 1.05          # J2+J3 全伸 ≈ 0.83 m，再算上腕部 0.1 m 左右

FLOOR_CLEAR_M = 0.005       # 目标 z 低于地面 + 这个值就提醒（机械臂会砸地）


# ---------------------------------------------------------------- 小工具
def unit(v: np.ndarray) -> np.ndarray:
    """归一化（零向量返回原样，避免 nan）。"""
    v = np.asarray(v, dtype=float).ravel()
    n = float(np.linalg.norm(v))
    return v.copy() if n < 1e-12 else v / n


def frame_from_z(z) -> np.ndarray:
    """构造正交基：第 3 列（z 轴）= 给定方向。用于画工具轴指示箭头。"""
    z = unit(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = unit(np.cross(helper, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def interp_axis(a0, a1, t: float) -> np.ndarray:
    """两个方向之间的最短弧插值（t=0 → a0，t=1 → a1）。

    直线运动时工具轴要跟着一起转，用四元数/旋转矩阵插值才不会让姿态在中途抖。
    两向量反向（夹角 180°）时叉积退化，这里随便挑一根垂直轴绕过去。
    """
    u, v = unit(a0), unit(a1)
    d = float(np.clip(float(u @ v), -1.0, 1.0))
    if d > 1.0 - 1e-12:
        return v
    if d < -1.0 + 1e-9:
        helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        w = unit(np.cross(helper, u))
    else:
        w = unit(np.cross(u, v))
    ang = float(np.arccos(d)) * float(t)
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    R = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)
    return unit(R @ u)


def add_marker(scene, gtype: int, size, pos, mat, rgba) -> bool:
    """往 MjvScene 里塞一个自定义几何（画面标记用，不参与物理/碰撞）。

    用法：``mujoco.Renderer.update_scene()`` 之后调用，把 ``renderer.scene`` 传进来。
    几何数超出 maxgeom 时返回 False（静默忽略，不会崩）。
    """
    if scene is None or scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, int(gtype),
        np.asarray(size, dtype=np.float64).reshape(3),
        np.asarray(pos, dtype=np.float64).reshape(3),
        np.asarray(mat, dtype=np.float64).reshape(9),
        np.asarray(rgba, dtype=np.float32).reshape(4),
    )
    scene.ngeom += 1
    return True


# ---------------------------------------------------------------- IK 结果
@dataclass
class IKResult:
    """一次 IK 的结果（界面直接拿它显示，日志也用它格式化）。"""

    q: np.ndarray
    pos_err_m: float
    axis_err_rad: float
    ms: float = 0.0
    ok: bool = False
    reason: str = ""
    target: np.ndarray | None = None
    axis: np.ndarray | None = None
    attempts: int = 1          # 用了几组初值（>1 说明首选初值没解出来，换了备选）

    @property
    def pos_err_mm(self) -> float:
        return float(self.pos_err_m) * 1000.0

    @property
    def axis_err_deg(self) -> float:
        return float(np.degrees(self.axis_err_rad))

    def summary(self) -> str:
        txt = (f"位置误差 {self.pos_err_mm:.2f} mm ｜ 姿态误差 {self.axis_err_deg:.2f}°"
               f" ｜ {self.ms:.0f} ms ｜ {self.attempts} 组初值")
        if self.ok:
            return "IK 可解：" + txt
        return f"IK 不可用（{self.reason}）：" + txt


# ---------------------------------------------------------------- 相机
# 四个常用视角（界面上的「视角」按钮就是切这几个）。默认斜视 = 与旧版一致。
VIEW_PRESETS: dict[str, dict] = {
    "斜视": dict(distance=2.2, azimuth=135.0, elevation=-18.0, lookat=(0.30, 0.0, 0.32)),
    "俯视": dict(distance=2.6, azimuth=90.0, elevation=-68.0, lookat=(0.30, 0.0, 0.20)),
    "侧视": dict(distance=2.4, azimuth=0.0, elevation=-8.0, lookat=(0.30, 0.0, 0.30)),
    "近看末端": dict(distance=0.9, azimuth=125.0, elevation=-15.0, lookat=(0.38, 0.0, 0.30)),
}


class Camera:
    """轨道相机（azimuth/elevation/distance/lookat），语义和 MuJoCo 一致。"""

    def __init__(self, distance=2.2, azimuth=135.0, elevation=-18.0,
                 lookat=(0.30, 0.0, 0.32)):
        self.mjv = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.mjv)
        self._home = dict(distance=float(distance), azimuth=float(azimuth),
                          elevation=float(elevation),
                          lookat=np.asarray(lookat, dtype=float))
        self.reset()

    def reset(self) -> None:
        """回到默认视角。"""
        h = self._home
        self.mjv.distance = h["distance"]
        self.mjv.azimuth = h["azimuth"]
        self.mjv.elevation = h["elevation"]
        self.mjv.lookat[:] = h["lookat"]

    def preset(self, name: str) -> None:
        """切到预设视角（名字见 ``VIEW_PRESETS``；不认识的名字当作复位）。"""
        p = VIEW_PRESETS.get(name)
        if p is None:
            self.reset()
            return
        self.mjv.distance = float(p["distance"])
        self.mjv.azimuth = float(p["azimuth"])
        self.mjv.elevation = float(p["elevation"])
        self.mjv.lookat[:] = np.asarray(p["lookat"], dtype=float)

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        az = np.radians(self.mjv.azimuth)
        el = np.radians(self.mjv.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        n = float(np.linalg.norm(right))
        right = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else right / n
        return fwd, right, np.cross(right, fwd)

    def orbit(self, dx: float, dy: float) -> None:
        """鼠标拖动转视角（dx/dy 是像素位移）。"""
        self.mjv.azimuth = float(self.mjv.azimuth - 0.4 * dx)
        self.mjv.elevation = float(np.clip(self.mjv.elevation + 0.3 * dy, -89.0, 89.0))

    def pan(self, dx: float, dy: float) -> None:
        _, right, up = self._basis()
        k = 0.0016 * self.mjv.distance
        self.mjv.lookat[:] = self.mjv.lookat - right * dx * k + up * dy * k

    def zoom(self, notches: float) -> None:
        self.mjv.distance = float(np.clip(self.mjv.distance * (0.88 ** notches), 0.35, 8.0))

    def look_at(self, pos) -> None:
        self.mjv.lookat[:] = np.asarray(pos, dtype=float).ravel()


# ---------------------------------------------------------------- 运动
class Motion:
    """一次运动：把 6 个驱动器的 ctrl 在**仿真时间**上从 q_start 平滑推到终点。

    * ``waypoints=None``：关节空间，smoothstep 直接在 q_start → q_end 之间插值；
    * ``waypoints=(M,6)``：先沿路点走（笛卡尔直线就是靠它），路点之间再线性细分，
      整体进度仍由同一条 smoothstep 时间曲线控制（首尾速度 0，不激起振动）。
    """

    def __init__(self, q_start, q_end, seconds: float, dt: float,
                 kind: str = "p2p", label: str = "", waypoints=None):
        self.q_start = np.asarray(q_start, dtype=float).ravel().copy()
        self.q_end = np.asarray(q_end, dtype=float).ravel().copy()
        self.waypoints = None if waypoints is None else np.asarray(waypoints, dtype=float)
        self.kind = kind
        self.label = label
        self.n = max(int(round(float(seconds) / dt)), 2)
        self.traj = ik.smoothstep(self.n)
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= self.n

    @property
    def progress(self) -> float:
        """0 → 1（界面进度条用）。"""
        return float(min(self.i, self.n)) / float(self.n)

    def value(self) -> np.ndarray:
        s = float(self.traj[min(self.i, self.n - 1)])
        if self.waypoints is None:
            return self.q_start + (self.q_end - self.q_start) * s
        m = self.waypoints.shape[0]
        if m == 1:
            return self.waypoints[0].copy()
        x = s * (m - 1)
        j = min(int(x), m - 2)
        f = x - j
        return self.waypoints[j] + (self.waypoints[j + 1] - self.waypoints[j]) * f


# ---------------------------------------------------------------- IK
def ik_seeds(seed) -> list[np.ndarray]:
    """给多初值 IK 准备初值列表（顺序 = 尝试顺序，命中即停）。

    为什么需要多初值：这台臂的腕是**偏置腕**（J4/J6 轴平行但不交），数值 IK 在奇异/局部极小
    时单初值可能解不出来、或解到"拧成麻花"的姿态。这里：

    1. 当前姿态（最自然，正常情况一次就中）；
    2. home 姿态（当前姿态卡住时的兜底）；
    3. 当前姿态 / home 再各加两处肩肘大幅扰动（跨到另一个解支上，等价于"肘上 / 肘下"换一支）。
    """
    seed = np.asarray(seed, dtype=float).ravel().copy()
    home = np.array(spec.HOME_QPOS, dtype=float)
    out = [seed, home]
    for base in (home, seed):
        for perturb in ((0.0, 0.6, -1.2, 0.0, 0.0, 0.0),
                        (0.0, -0.6, 1.2, 0.0, 0.0, 0.0)):
            out.append(base + np.asarray(perturb, dtype=float))
    return out


def solve_ik_pose(model, data, site: str, pos, axis=None, seed=None,
                  iters: int = 300) -> tuple[np.ndarray, float, float]:
    """单初值求解 + "解不够好就精修一轮"（等价于旧版 GUI 里那段逻辑）。"""
    q, ep, ea = ik.solve_ik_pose(model, data, site, pos, axis,
                                 q_seed=None if seed is None else np.asarray(seed, dtype=float),
                                 iters=iters)
    if ep > 3e-3:
        q2, ep2, ea2 = ik.refine_ik_pose(model, data, site, pos, axis, q_seed=q)
        if ep2 < ep:
            q, ep, ea = q2, ep2, ea2
    return q, float(ep), float(ea)


def solve_ik_best(model, data, pos, axis=None, seeds=None, site: str = "tool_site",
                  *, iters: int = 300) -> tuple[np.ndarray, float, float, int]:
    """多初值阻尼最小二乘 IK：谁解得好用谁。

    评分 = 位置误差 + 0.05 × 姿态误差（位置优先）；两者都够好就提前收工。
    返回 ``(q, 位置误差[m], 姿态误差[rad], 实际尝试的初值组数)``。
    """
    if seeds is None:
        seeds = ik_seeds(ik.arm_qpos(model, data))
    best: tuple[float, np.ndarray, float, float] | None = None
    tries = 0
    for seed in seeds:
        tries += 1
        q, ep, ea = solve_ik_pose(model, data, site, pos, axis, seed=seed, iters=iters)
        score = float(ep) + 0.05 * float(ea)
        if best is None or score < best[0]:
            best = (score, q, ep, ea)
        if ep < 1e-4 and (axis is None or ea < 1e-4):
            break
    assert best is not None
    return best[1], best[2], best[3], tries


@dataclass
class PlannedLine:
    """笛卡尔直线的规划结果（``ArmSim.plan_line`` 的返回值）。"""

    waypoints: np.ndarray            # (M,6) 关节路点（交给 Motion 回放）
    points: np.ndarray               # (M,3) 对应的工具尖期望位置（画面里画路径）
    q_end: np.ndarray                # 末点关节角
    target: np.ndarray               # 目标位置
    axis: np.ndarray | None          # 本次实际用的工具轴（None = 不约束姿态）
    ok: bool                         # 末点是否解出来
    pos_err_m: float                 # 末点 IK 残差
    axis_err_rad: float
    ms: float                        # 规划耗时
    failed_at: int = -1              # 第一个解不出来的路点序号（-1 = 全部解出来）
    reason: str = ""

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    @property
    def pos_err_mm(self) -> float:
        return float(self.pos_err_m) * 1000.0

    @property
    def axis_err_deg(self) -> float:
        return float(np.degrees(self.axis_err_rad))

    def straightness(self) -> float:
        """规划路点偏离"起点→终点"直线的最大距离[m]（=0 说明路点严格在直线上）。"""
        p0, p1 = self.points[0], self.points[-1]
        d = unit(p1 - p0)
        if float(np.linalg.norm(p1 - p0)) < 1e-12:
            return 0.0
        rel = self.points - p0
        off = rel - np.outer(rel @ d, d)
        return float(np.max(np.linalg.norm(off, axis=1)))


# =============================================================== 仿真 + 控制
class ArmSim:
    """模型 + 伺服 + 指令来源（纯逻辑，不含任何界面代码）。

    6 个驱动器的位置指令统一放在 ``self.cmd``（弧度），"谁在写它"决定当前模式：

    ==============  ==========================================================
    ``joint``       单关节（界面上每个关节的 −/+ 按钮）：直接改 ``cmd[i]``，点一下动一点
    ``p2p``         关节空间点到点：``Motion`` 每一步覆盖 ``cmd``（smoothstep 插值）
    ``cartesian``   笛卡尔直线：``Motion`` 带着一串 IK 路点覆盖 ``cmd``（工具尖走直线）
    ``hold``        急停：``cmd`` 定格在急停瞬间的**实测**姿态（原地撑住）
    ``follow``      真机跟随：每帧把 ``cmd`` 钉到真机来的关节角上（见 ``robot_link.py``）
    ==============  ==========================================================
    """

    def __init__(self, scene=spec.SCENE_XML, *, kp_scale: float = 1.0,
                 gravity: bool = True, move_seconds: float = 1.5, log_scene: bool = True):
        self.scene = Path(scene)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.move_seconds = float(move_seconds)
        self.kp0 = np.array([self.model.actuator_gainprm[i][0] for i in range(self.model.nu)])
        self.kv0 = np.array([-self.model.actuator_biasprm[i][2] for i in range(self.model.nu)])
        self.kp_scale = 1.0
        self.set_kp_scale(kp_scale)
        self.gravity_on = bool(gravity)
        self.set_gravity(gravity)
        self.lo = np.array([spec.LIMITS[j][0] for j in spec.JOINTS])
        self.hi = np.array([spec.LIMITS[j][1] for j in spec.JOINTS])
        self.mode = "joint"
        self.cmd = np.array(spec.HOME_QPOS, dtype=float)
        self.motion: Motion | None = None
        self.pending: dict | None = None
        self.last_report: dict | None = None
        self.last_ik: IKResult | None = None
        self.last_plan: PlannedLine | None = None
        self.ik_target: tuple[np.ndarray, np.ndarray | None, bool] | None = None
        self.path_points: np.ndarray | None = None
        self.messages: list[str] = []
        self.sim_time = 0.0
        self.reset_model(log_it=False)
        if log_scene:
            self.log(f"模型已加载：{self.scene.name}（{self.model.nu} 个位置伺服，"
                     f"dt = {self.dt * 1000:.0f} ms，home 姿态已就位）")

    # ---------------------------------------------------------------- 读
    def q_pos(self) -> np.ndarray:
        """6 个关节的**实测**角（弧度，顺序 joint1..joint6）。"""
        return ik.arm_qpos(self.model, self.data)

    def cmd_deg(self) -> np.ndarray:
        """6 个关节的**指令**角（度）。"""
        return np.degrees(self.cmd)

    def actual_deg(self) -> np.ndarray:
        """6 个关节的**实测**角（度）。"""
        return np.degrees(self.q_pos())

    def tip(self) -> np.ndarray:
        """工具尖（tool_site）的世界坐标。"""
        return ik.site_pos(self.model, self.data, "tool_site")

    def tip_axis(self) -> np.ndarray:
        """工具轴（= tool_site 的 z 轴）在世界系的方向。"""
        return ik.tool_axis(self.model, self.data, "tool_site")

    def tip_height(self) -> float:
        """工具尖离地高度[m]（地面在 spec.FLOOR_Z）。"""
        return float(self.tip()[2] - spec.FLOOR_Z)

    def track_err_deg(self) -> float:
        """最大关节跟踪误差[°]（指令 − 实测）。"""
        return float(np.degrees(np.abs(self.cmd - self.q_pos())).max())

    def moving(self) -> bool:
        return self.motion is not None

    def progress(self) -> float:
        """当前运动的进度 0 → 1（没有运动时为 0）。"""
        return 0.0 if self.motion is None else float(self.motion.progress)

    def log(self, msg: str) -> None:
        self.messages.append(msg)

    def drain(self) -> list[str]:
        out, self.messages = self.messages, []
        return out

    # ---------------------------------------------------------------- 写
    def set_kp_scale(self, scale: float) -> None:
        """整体缩放 kp/kv（体会"伺服软硬"对跟踪和抖动的影响）。"""
        self.kp_scale = float(max(scale, 0.01))
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = self.kp0[i] * self.kp_scale
            self.model.actuator_biasprm[i, 1] = -self.kp0[i] * self.kp_scale
            self.model.actuator_biasprm[i, 2] = -self.kv0[i] * self.kp_scale

    def set_gravity(self, on: bool) -> None:
        self.gravity_on = bool(on)
        self.model.opt.gravity[:] = (0.0, 0.0, -9.81) if self.gravity_on else (0.0, 0.0, 0.0)

    def set_joint(self, i: int, q_rad: float, *, log_it: bool = False) -> float:
        """单关节：把第 i 个关节的**目标角**直接设过去（界面上的 −/+ 按钮走这里）。

        手动微调意味着"人要接管"，所以顺手取消正在跑的自动运动（否则两边抢 ``cmd``）。
        """
        if self.motion is not None:
            self.motion = None
            self.pending = None
            self.log("手动微调关节：自动运动已取消")
        self.mode = "joint"
        val = float(np.clip(float(q_rad), self.lo[i], self.hi[i]))
        self.cmd = self.cmd.copy()
        self.cmd[i] = val
        if log_it:
            self.log(f"单关节：J{i + 1} 目标 → {np.degrees(val):+.1f}°")
        return val

    def nudge_joint(self, i: int, delta_deg: float) -> float:
        """在第 i 个关节当前**目标角**上加减 ``delta_deg`` 度（−/+ 按钮的语义）。"""
        return self.set_joint(i, float(self.cmd[i]) + np.radians(float(delta_deg)))

    def set_cmd(self, q) -> None:
        """直接设定 6 个目标（不插值，主要给复位/测试用）。"""
        self.cmd = np.clip(np.asarray(q, dtype=float).ravel(), self.lo, self.hi)
        self.mode = "joint"

    def follow(self, q_target) -> np.ndarray:
        """**真机跟随**：把 6 个目标角直接钉到给定角度（弧度，不插值，逐帧刷新）。

        和 :meth:`set_cmd` 的区别只有两点，但都是必须的：

        1. 顺手取消正在跑的自动运动（``p2p`` / ``cartesian`` 也在写 ``cmd``，
           两个来源抢目标会互相打架）；
        2. 模式标成 ``follow``（状态栏、日志能一眼看出"现在是人/真机在带"）。

        限位仍然生效：真机偶尔给出超限角（比如 ±400°）时按模型关节范围截断。
        """
        if self.motion is not None:
            self.motion = None
            self.pending = None
        self.mode = "follow"
        self.cmd = np.clip(np.asarray(q_target, dtype=float).ravel(), self.lo, self.hi)
        return self.cmd

    def hold(self) -> None:
        """急停：取消运动，把目标钉在**当前实测**姿态上（伺服原地撑住）。"""
        self.motion = None
        self.pending = None
        self.mode = "hold"
        self.cmd = self.q_pos().copy()
        self.log("急停：就地保持当前姿态")

    def reset_model(self, log_it: bool = True) -> None:
        """复位到 home（模型 + 指令一起回去）。"""
        if not ik.reset_home(self.model, self.data):
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        self.motion = None
        self.pending = None
        self.last_report = None
        self.mode = "joint"
        self.cmd = np.array(spec.HOME_QPOS, dtype=float)
        self.sim_time = 0.0
        if log_it:
            self.log("模型已复位到 home 姿态")

    def home(self, seconds: float | None = None) -> None:
        """平滑回 home（走关节空间 P2P，不是瞬移）。"""
        self.start_p2p(spec.HOME_QPOS, seconds, kind="p2p", src="home", label="回 home")

    def zero(self, seconds: float | None = None) -> None:
        """平滑回到"六轴全 0"（也是关节空间 P2P）。"""
        self.start_p2p(np.zeros(6), seconds, kind="p2p", src="zero", label="全部归零")

    def sync_cmd_to_actual(self) -> None:
        """把指令拉平到实测姿态（"跟住当前"：清掉跟踪误差，手臂不会突然抽一下）。"""
        self.cmd = self.q_pos().copy()
        self.mode = "joint"
        self.log(f"指令已同步到实测姿态：{np.round(self.cmd_deg(), 1).tolist()}°")

    # ---------------------------------------------------------------- IK
    def _ik_raw(self, pos, axis, seed, multi: bool = False):
        """解一次 IK：``multi=True`` 走多初值（慢但稳），否则单初值（快，沿直线递推用）。

        ⚠️ 数值 IK 会改 ``data.qpos``，调用方负责事后恢复。
        """
        if multi:
            return solve_ik_best(self.model, self.data, pos, axis, seeds=ik_seeds(seed))
        q, ep, ea = solve_ik_pose(self.model, self.data, "tool_site", pos, axis, seed=seed)
        return q, ep, ea, 1

    def solve_ik(self, pos, axis=None, seed=None) -> IKResult:
        """求逆解（多初值阻尼最小二乘）。

        算完会把 qpos 恢复原样 —— 这里只是"算"：**不会**把机械臂瞬移过去。
        """
        t0 = time.perf_counter()
        pos = np.asarray(pos, dtype=float).ravel()
        axis = None if axis is None else unit(axis)
        seed = np.asarray(self.q_pos() if seed is None else seed, dtype=float).ravel()
        qpos0, qvel0 = self.data.qpos.copy(), self.data.qvel.copy()
        try:
            q, ep, ea, tries = solve_ik_best(self.model, self.data, pos, axis,
                                            seeds=ik_seeds(seed))
        finally:
            self.data.qpos[:] = qpos0
            self.data.qvel[:] = qvel0
            mujoco.mj_forward(self.model, self.data)
        ms = (time.perf_counter() - t0) * 1000.0
        ok_pos = ep <= IK_TOL_POS_M
        ok_axis = axis is None or ea <= IK_TOL_AXIS_RAD
        reason = ""
        if not ok_pos:
            reason = f"位置差 {ep * 1000:.1f} mm（超出工作空间或顶到关节限位）"
        elif not ok_axis:
            reason = f"姿态差 {np.degrees(ea):.1f}°（这个点摆不出该工具朝向）"
        res = IKResult(q=np.asarray(q, dtype=float), pos_err_m=ep, axis_err_rad=ea, ms=ms,
                       ok=ok_pos and ok_axis, reason=reason, target=pos, axis=axis,
                       attempts=tries)
        self.last_ik = res
        self.ik_target = (pos, axis, res.ok)
        self.log(res.summary())
        self.log(f"    解 q = {np.round(np.degrees(res.q), 1).tolist()}°")
        return res

    # ---------------------------------------------------------------- 笛卡尔直线
    def plan_line(self, target, axis=None, seconds=None, *, step: float = LINE_STEP_M,
                  hold_axis: bool = True) -> PlannedLine:
        """笛卡尔直线规划：从当前工具尖沿**直线**走到 ``target``，每个路点解一次 IK。

        ``hold_axis=True``（默认）工具轴按最短弧从当前方向转到 ``axis``（``axis=None`` 时
        就保持当前方向 = "平移不转身"）；``hold_axis=False`` 姿态完全不约束（只保证位置）。
        """
        t0 = time.perf_counter()
        target = np.asarray(target, dtype=float).ravel()
        qpos0, qvel0 = self.data.qpos.copy(), self.data.qvel.copy()
        try:
            p0 = self.tip()                       # 必须在任何 IK 之前读（IK 会改 qpos）
            a0 = self.tip_axis()
            axis_goal = None if not hold_axis else unit(a0 if axis is None else axis)
            dist = float(np.linalg.norm(target - p0))
            m = int(np.clip(np.ceil(dist / max(float(step), 1e-3)) + 1,
                            LINE_MIN_POINTS, LINE_MAX_POINTS))
            qs: list[np.ndarray] = []
            pts: list[np.ndarray] = []
            seed = self.q_pos()
            ep_end, ea_end, failed_at = 0.0, 0.0, -1
            for i in range(m):
                last = i == m - 1
                t = i / float(m - 1)
                p = p0 + (target - p0) * t
                a = None if axis_goal is None else interp_axis(a0, axis_goal, t)
                q, ep, ea, _ = self._ik_raw(p, a, seed, multi=last)
                if float(ep) > LINE_MAX_ERR_M and failed_at < 0:
                    failed_at = i
                if last:
                    ep_end, ea_end = float(ep), float(ea)
                q = np.asarray(q, dtype=float)
                if float(ep) > LINE_MAX_ERR_M and qs:
                    q = qs[-1].copy()             # 这一点过不去：先停住，别让轨迹跳变
                qs.append(q)
                pts.append(p)
                seed = q
        finally:
            self.data.qpos[:] = qpos0
            self.data.qvel[:] = qvel0
            mujoco.mj_forward(self.model, self.data)
        waypoints, points = np.vstack(qs), np.vstack(pts)
        ok_pos = ep_end <= IK_TOL_POS_M
        ok_axis = axis_goal is None or ea_end <= IK_TOL_AXIS_RAD
        reason = ""
        if not ok_pos:
            reason = f"目标点位置差 {ep_end * 1000:.1f} mm"
        elif not ok_axis:
            reason = f"工具轴差 {np.degrees(ea_end):.1f}°"
        return PlannedLine(waypoints=waypoints, points=points, q_end=waypoints[-1].copy(),
                           target=target, axis=axis_goal, ok=ok_pos and ok_axis,
                           pos_err_m=ep_end, axis_err_rad=ea_end,
                           ms=(time.perf_counter() - t0) * 1000.0,
                           failed_at=failed_at, reason=reason)

    # ---------------------------------------------------------------- 运动指令
    def start_p2p(self, q_target, seconds=None, *, kind: str = "p2p", src: str = "cmd",
                  label: str = "", pos=None, axis=None) -> np.ndarray:
        """关节空间点到点：从当前**指令**角 smoothstep 平滑开到 ``q_target``。

        ⚠️ 关节空间插值下工具尖走的是弧线（旧版界面就是这个，看着不直观）；
        要做"沿直线过去"请用 :meth:`start_cartesian`。
        """
        qt = np.clip(np.asarray(q_target, dtype=float).ravel(), self.lo, self.hi)
        sec = self.move_seconds if seconds is None else float(seconds)
        self.motion = Motion(self.cmd, qt, sec, self.dt, kind=kind, label=label)
        self.mode = kind
        self.pending = dict(kind=kind, q_target=qt, deadline=self.sim_time + sec + 0.8,
                            pos=None if pos is None else np.asarray(pos, dtype=float),
                            axis=None if axis is None else np.asarray(axis, dtype=float))
        self.log(f"关节空间 P2P：{src} → {np.round(np.degrees(qt), 1).tolist()}°，{sec:.2f} s")
        return qt

    def start_cartesian(self, target, axis=None, seconds=None, *, step: float = LINE_STEP_M,
                        hold_axis: bool = True, label: str = "笛卡尔直线") -> PlannedLine:
        """笛卡尔直线点到点：先规划（沿直线解一串 IK 路点），再把路点交给 ``Motion`` 回放。"""
        plan = self.plan_line(target, axis, seconds, step=step, hold_axis=hold_axis)
        sec = self.move_seconds if seconds is None else float(seconds)
        self.last_plan = plan
        self.ik_target = (plan.target.copy(), None if plan.axis is None else plan.axis.copy(),
                          plan.ok)
        self.path_points = plan.points.copy()
        seg_mm = (float(np.linalg.norm(plan.points[-1] - plan.points[0]))
                  / max(plan.n - 1, 1)) * 1000.0
        self.log(f"直线规划：{np.round(plan.points[0], 3).tolist()} → "
                 f"{np.round(plan.target, 3).tolist()}，{plan.n} 个路点"
                 f"（间隔 {seg_mm:.0f} mm），用时 {plan.ms:.0f} ms，"
                 f"直线度 {plan.straightness() * 1000:.3f} mm")
        if plan.failed_at >= 0:
            self.log(f"    注意：第 {plan.failed_at + 1}/{plan.n} 个路点解不出来"
                     f"（沿线有碰限位/不可达的地方，轨迹会先停在那里）")
        if not plan.ok:
            self.log(f"直线运动取消：{plan.reason}"
                     f"（画面里目标点会标成红色；换个点或放宽工具轴试试）")
            return plan
        self.motion = Motion(self.cmd, plan.q_end, sec, self.dt, kind="cartesian",
                             label=label, waypoints=plan.waypoints)
        self.mode = "cartesian"
        self.pending = dict(kind="cartesian", q_target=plan.q_end.copy(),
                            deadline=self.sim_time + sec + 0.8, pos=plan.target.copy(),
                            axis=None if plan.axis is None else plan.axis.copy())
        self.log(f"直线运动开始：{sec:.2f} s（工具尖沿直线走，工具轴"
                 f"{'姿态不约束' if not hold_axis else ('保持当前朝向' if axis is None else '同步转向')}）")
        return plan

    def goto_pose(self, pos, axis=None, seconds=None, *, hold_axis: bool = True,
                  execute: bool = True, step: float = LINE_STEP_M):
        """「求解 IK（可选：并运动）」：先 IK 检查能不能到，再走笛卡尔直线过去。

        返回 ``(IKResult, PlannedLine | None)``；``execute=False`` 时只算不解、只把目标点标出来。
        """
        pos = np.asarray(pos, dtype=float).ravel()
        probe_axis = None if not hold_axis else unit(self.tip_axis() if axis is None else axis)
        res = self.solve_ik(pos, probe_axis)
        self.ik_target = (pos, probe_axis, res.ok)
        if not res.ok:
            self.path_points = None
            return res, None
        self.path_points = np.vstack([self.tip(), pos])
        if not execute:
            return res, None
        return res, self.start_cartesian(pos, axis, seconds, step=step, hold_axis=hold_axis)

    # ---------------------------------------------------------------- 每步
    def apply_ctrl(self) -> None:
        """把"这一刻该给的目标"写进 ``data.ctrl``（必须在 ``mj_step`` 之前调用）。"""
        if self.motion is not None:
            self.cmd = self.motion.value()
            self.motion.i += 1
            if self.motion.done:
                self.cmd = self.motion.q_end.copy()
                self.motion = None
        for j, v in zip(spec.JOINTS, self.cmd):
            self.data.ctrl[ik.dof_adr(self.model, j)] = float(v)

    def step(self) -> None:
        """推进一个仿真步（含运动插值与到点误差回报）。"""
        self.apply_ctrl()
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt
        self._check_pending()

    def _check_pending(self) -> None:
        """运动结束并静置一会儿后，量一次**实际**到位精度写进日志（不是看指令）。"""
        if self.pending is None or self.sim_time < self.pending["deadline"]:
            return
        p, self.pending = self.pending, None
        tip = self.tip()
        jerr = float(np.degrees(np.abs(self.q_pos() - p["q_target"])).max())
        rep: dict = dict(kind=p["kind"], joint_err_deg=jerr, tip=tip.copy(),
                         target=None if p.get("pos") is None else np.asarray(p["pos"]))
        if p.get("pos") is None:
            self.log(f"到位（{p['kind']}）：最大关节跟踪误差 {jerr:.3f}°，"
                     f"工具尖 {np.round(tip, 3).tolist()}")
        else:
            tgt = np.asarray(p["pos"], dtype=float)
            perr = float(np.linalg.norm(tip - tgt))
            aerr = None
            axis = p.get("axis")
            if axis is not None and np.any(axis):
                a = unit(axis)
                aerr = float(np.degrees(np.arccos(
                    np.clip(float(self.tip_axis() @ a), -1.0, 1.0))))
            rep.update(pos_err_m=perr, axis_err_deg=aerr)
            txt = (f"到位（{p['kind']}）：工具尖 {np.round(tip, 3).tolist()} / 目标 "
                   f"{np.round(tgt, 3).tolist()}，位置误差 {perr * 1000:.2f} mm")
            if aerr is not None:
                txt += f"，姿态误差 {aerr:.2f}°"
            self.log(txt + f"，最大关节跟踪误差 {jerr:.3f}°")
        self.last_report = rep

    def status(self) -> dict:
        """一屏状态（界面每次刷新都读它，别在界面里重复算）。"""
        return dict(mode=self.mode, moving=self.moving(), progress=self.progress(),
                    cmd_deg=self.cmd_deg(), q_deg=self.actual_deg(), tip=self.tip(),
                    axis=self.tip_axis(), height=self.tip_height(),
                    contacts=int(self.data.ncon), track_deg=self.track_err_deg(),
                    sim_time=self.sim_time, kp_scale=self.kp_scale,
                    gravity=self.gravity_on, last=self.last_report,
                    ik=self.last_ik, plan=self.last_plan)







