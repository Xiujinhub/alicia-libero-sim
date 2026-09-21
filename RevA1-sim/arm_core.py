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
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl()
import mujoco  # noqa: E402  (必须在设置 MUJOCO_GL 之后)

import revA1_config as cfg  # noqa: E402  (关节限位的用户配置，只依赖标准库)
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
LINE_MAX_BRANCH_JUMP_DEG = 45.0   # 单步"换解支"超过它就当直线走不过去（换支会让轨迹甩出去）

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


# ---------------------------------------------------------------- 工具 TCP 向量（画面箭头）
def ee_frame(model, data, site: str = "ee_site") -> tuple[np.ndarray, np.ndarray]:
    """末端法兰系在世界里的位姿 ``(原点, 旋转矩阵)``；``R`` 的三列 = 法兰的 x/y/z 轴。"""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise KeyError(f"场景里没有 site {site!r}（工具向量是相对它算的）")
    return (np.array(data.site_xpos[sid], dtype=float),
            np.array(data.site_xmat[sid], dtype=float).reshape(3, 3))


def tool_points(model, data, names=None, *, site: str = "ee_site") -> dict[str, np.ndarray]:
    """各工具 TCP 的**世界坐标** = 法兰原点 + ``R_法兰 @ v``（``v`` 见 ``spec.TOOLS``）。"""
    p, R = ee_frame(model, data, site)
    keys = spec.tool_names() if names is None else [n for n in names if n in spec.TOOLS]
    return {n: p + R @ spec.tool_vector(n) for n in keys}


def draw_tool_arrows(scene, origin, rot, names=None, *,
                     width: float = spec.TOOL_ARROW_R_M, head_scale: float = 2.2,
                     tip_dot: bool = False) -> int:
    """把每个工具画成一根**箭头**，返回画了几根。

    **三根箭头互相平行，而且都平行于末端姿态的工具轴**（= 法兰 ``+z``）—— 三个工具是并排装在
    同一片法兰上的，所以：

    * 起点 = 该工具在**法兰平面上的安装点** ``origin + rot @ (vx, vy, 0)``（侧偏就体现在这儿）；
    * 方向 = ``rot @ (0, 0, 1)`` = **末端工具轴**（世界系）；长度 = ``vz`` = 该工具的**轴向长度**；
    * 于是箭头尖 = ``起点 + 方向 · vz`` = ``origin + rot @ v`` = **该工具的 TCP**（与以前一致，
      所以"箭头尖落在 TCP 上"那个检查仍然成立）。

    （早先的版本是"从法兰中心直接画到 TCP"，也就是把 ``v`` 的侧偏也算进方向 —— 那样三根箭头的
    方向两两差 13.6°~14.9°，画面里是**扇形**；实际三个工具是平行的，这里按物理装法画。）

    * ``scene``：``mujoco.Renderer.update_scene()`` 之后的 ``MjvScene``（与 :func:`add_marker`
      一样——只在画面上，不参与物理/碰撞，也不改模型）；
    * 箭头用 MuJoCo 的**渲染专用**几何 ``mjGEOM_ARROW``（``mjv_connector`` 摆位：
      ``pos`` = 起点、本地 +z 指向终点）。
      **箭头长度 = 该工具的轴向长度、箭头尖就落在 TCP 上**（注意 MuJoCo 只画 ``size[2]`` 的一半，
      所以这里的 ``size[2]`` 会是 ``2·vz``，见下面注释），默认不再另画端点小球
      （``tip_dot=True`` 才会加）；
    * 每帧重算，所以箭头**跟着机械臂走**；几何数超上限时少画不报错；
    * 归到 ``mjCAT_DECOR``：箭头不投阴影，画面上不会在机械臂上多出跟着动的色斑。
    """
    if scene is None:
        return 0
    keys = spec.tool_names() if names is None else [n for n in names if n in spec.TOOLS]
    origin = np.asarray(origin, dtype=float).ravel()
    rot = np.asarray(rot, dtype=float).reshape(3, 3)
    axis = rot @ np.array([0.0, 0.0, 1.0])          # 末端工具轴（世界系）= 三根箭头的共同方向
    n = 0
    for name in keys:
        v = spec.tool_vector(name)
        rgba = np.asarray(spec.TOOLS[name]["rgba"], dtype=np.float32)
        vz = float(v[2])
        if vz > 1e-6:                                # 常规：装点 → 沿工具轴 → TCP
            start = origin + rot @ np.array([v[0], v[1], 0.0])   # 法兰平面上的安装点
            length = vz
        else:                                        # 退化保护（工具朝 −z 装时按老画法走）
            start, length = origin, float(np.linalg.norm(v))
        tip = start + axis * length                  # = origin + rot @ v（该工具的 TCP）
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_ARROW), np.zeros(3), np.zeros(3),
                            np.eye(3).reshape(-1), rgba)
        # ⚠️ 坑（实测，别删注释）：MuJoCo 3.13 画 mjGEOM_ARROW 时**只画 size[2] 的一半**
        # —— 沿本地 +z 从 pos 起、长度 size[2]/2。直接 mjv_connector(起点, 终点) 得到
        # size[2]=长度，画出来只有一半（箭头尖离 TCP 差 144~209 px）。
        # 所以终点放 2 倍处：画出来正好是 起点 → 起点+长度·ẑ = **安装点 → TCP**。
        # 空场景对照实测（箭长 300 mm，两端用小球标尺量像素）：size[2] = 150/300/600 mm
        # → 画出 75/150/300 mm，且起点始终在 pos 上。这个约定由自检里
        # "箭头尖到 TCP 的像素距离" 那一项守着（MuJoCo 若改行为会立刻被抓出来）。
        mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_ARROW), float(width),
                             start, start + 2.0 * (tip - start))
        # mjv_connector 把头部半径也设成杆粗，看着不像箭头 → 单独放大头
        g.size[1] = float(width) * float(head_scale)
        # 归到"装饰"类：这样箭头**不投阴影**（实测带上阴影时，画面上会在小臂/腕部多出
        # 一大片跟着动的色斑；改成 DECOR 后差异像素的纵向范围正好等于箭头长度）。
        g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1
        n += 1
        if tip_dot and scene.ngeom < scene.maxgeom:
            d = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(d, int(mujoco.mjtGeom.mjGEOM_SPHERE),
                                np.full(3, float(width) * 1.4), tip,
                                np.eye(3).reshape(-1), rgba)
            d.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            scene.ngeom += 1
    return n


# ---------------------------------------------------------------- 工具尖轨迹（任务信号驱动）
class Trail:
    """一根工具尖轨迹（世界系折线）：按最小间距采样、点数封顶、可清空。

    只存点；画的时候交给 :func:`draw_trail`（一串短胶囊、``mjCAT_DECOR``，不投阴影、
    不参与物理）。注意它跟路径规划里的 ``Motion`` / ``PlannedLine`` 是两回事 ——
    那些是"准备要走的关节路点"，这个是"已经走过了的末端痕迹"（真机开始作业时由 6501 的
    ``motion: start`` 触发记录，见 ``robot_link.TaskListener``）。

    * ``min_step_m``：两个点比它还近就不记（工具尖停着不动时不会每秒灌几百个点）；
    * ``max_points``：满了自动丢最老的点（滚动窗口，长时间作业也不会把内存/画面撑爆）；
    * ``length_m``：**累计路程**（被丢掉的点也算过，用来显示"走了多远"）。
    """

    def __init__(self, name: str, *, rgba=(1.0, 1.0, 1.0, 1.0),
                 min_step_m: float = spec.TRAIL_MIN_STEP_M,
                 max_points: int = spec.TRAIL_MAX_POINTS):
        self.name = str(name)
        self.rgba = tuple(float(v) for v in rgba)
        self.min_step_m = max(float(min_step_m), 0.0)
        self.max_points = max(int(max_points), 2)
        self._pts: deque = deque(maxlen=self.max_points)
        self.length_m = 0.0
        self.dropped = 0                     # 被滚动窗口丢掉的点数（显示"轨迹被截过"用）

    def __len__(self) -> int:
        return len(self._pts)

    @property
    def points(self) -> np.ndarray:
        """所有采样点（``(N,3)``；没点时是 ``(0,3)``）。"""
        return np.asarray(self._pts, dtype=float).reshape(-1, 3)

    def add(self, p) -> bool:
        """记一个点；离上一个点不够远（或完全没动）就不记，返回是否真的记了。"""
        q = np.asarray(p, dtype=float).ravel()
        if self._pts:
            d = float(np.linalg.norm(q - np.asarray(self._pts[-1], dtype=float)))
            if d < self.min_step_m:
                return False
            self.length_m += d
            if len(self._pts) == self.max_points:
                self.dropped += 1
        self._pts.append((float(q[0]), float(q[1]), float(q[2])))
        return True

    def clear(self) -> int:
        n = len(self._pts)
        self._pts.clear()
        self.length_m = 0.0
        self.dropped = 0
        return n

    def text(self) -> str:
        return (f"{self.name}  {len(self._pts)} 点 · 路程 {self.length_m:.3f} m"
                + (f"（已滚动丢掉 {self.dropped} 点）" if self.dropped else ""))


def draw_trail(scene, trail: Trail, *, width: float = spec.TRAIL_R_M) -> int:
    """把一根轨迹画出来：相邻两点之间一段**胶囊**，返回画了几段。

    只在画面上（``mjCAT_DECOR``：不投阴影、不参与物理、不进 ``mjModel``），每帧重画 →
    轨迹随机械臂一点点长出来。
    """
    if scene is None or len(trail) < 2:
        return 0
    pts = trail.points
    rgba = np.asarray(trail.rgba, dtype=np.float32)
    n = 0
    for i in range(len(pts) - 1):
        if scene.ngeom >= scene.maxgeom:
            break
        a, b = pts[i], pts[i + 1]
        if float(np.linalg.norm(b - a)) < 1e-9:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), np.zeros(3), np.zeros(3),
                            np.eye(3).reshape(-1), rgba)
        mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), float(width), a, b)
        g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1
        n += 1
    return n


# ---------------------------------------------------------------- 关节限位（用户配置）
def apply_joint_limits(model, limits=None) -> dict[str, tuple[float, float]]:
    """把每个关节的限位写进**模型**，返回生效值（**度**）。

    ``limits``：``{关节名: (最小°, 最大°)}``；没给的关节用 URDF 默认（``spec.LIMITS``）。

    ⚠️ **三处必须一起改，少一处就白改**：

    * ``model.jnt_range`` —— MuJoCo 的关节限位（``sim_ik`` 的迭代边界也读它）；
    * ``model.actuator_ctrlrange`` —— 不改的话 MuJoCo 会把 ``data.ctrl`` 再按旧范围静默夹一次
      （位置伺服就是这样：写进去的目标角会被夹回旧范围，表现成"卡在限位上"）；
    * 调用方的 ``lo/hi``（``ArmSim`` 的由 :meth:`ArmSim.set_joint_limits` 负责）——
      ``follow()`` / ``set_joint()`` / ``start_p2p()`` 用的是它。
    """
    eff = cfg.effective(limits)
    for jn, (lo_deg, hi_deg) in eff.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jn}")
        if jid >= 0:
            model.jnt_range[jid] = (np.radians(lo_deg), np.radians(hi_deg))
        if aid >= 0:
            model.actuator_ctrlrange[aid] = (np.radians(lo_deg), np.radians(hi_deg))
    return eff


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
    3. :data:`spec.NEUTRAL_QPOS`（工具轴**竖直朝下**的中性姿态——home 现在也是前倾的，
       "让工具轴朝下"这类常用目标从它出发一次就中）；
    4. 以上再各加两处肩肘大幅扰动（跨到另一个解支上，等价于"肘上 / 肘下"换一支）。
    """
    seed = np.asarray(seed, dtype=float).ravel().copy()
    home = np.array(spec.HOME_QPOS, dtype=float)
    neutral = np.array(spec.NEUTRAL_QPOS, dtype=float)
    out = [seed, home, neutral]
    for base in (home, seed, neutral):
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
                  *, iters: int = 300, tol_p: float = 1e-3, tol_a: float = 0.02,
                  ) -> tuple[np.ndarray, float, float, int]:
    """多初值阻尼最小二乘 IK。

    选择规则（很重要，直接决定机械臂"怎么动"）：

    1. 精度够好的解（位置 ≤ ``tol_p``、姿态 ≤ ``tol_a``）里，**挑离初值最近的那个**
       —— 关节动得最少、最像真机；
    2. 没有够好的，才退而挑打分最好的（``位置误差 + 0.05 × 姿态误差``）。

    为什么不能"只挑精度最高的"：同一目标常有多个解支，精度差 0.02 mm 时挑到另一支上，
    关节要翻 100°+，关节空间一动就把工具甩出去（实测笛卡尔直线能甩偏 0.64 m）。

    返回 ``(q, 位置误差[m], 姿态误差[rad], 实际尝试的初值组数)``。
    """
    if seeds is None:
        seeds = ik_seeds(ik.arm_qpos(model, data))
    seeds = [np.asarray(s, dtype=float).ravel() for s in seeds]
    q_ref = seeds[0]
    # 求解过程会反复改 qpos，算完恢复现场（调用方可能紧接着用当前状态跑运动/画图）
    qpos0, qvel0 = data.qpos.copy(), data.qvel.copy()
    best: tuple[float, np.ndarray, float, float] | None = None
    near: tuple[float, np.ndarray, float, float] | None = None      # 精度够好里离初值最近的
    tries = 0
    try:
        for seed in seeds:
            tries += 1
            q, ep, ea = solve_ik_pose(model, data, site, pos, axis, seed=seed, iters=iters)
            q = np.asarray(q, dtype=float)
            score = float(ep) + 0.05 * float(ea)
            if best is None or score < best[0]:
                best = (score, q, ep, ea)
            if ep <= tol_p and (axis is None or ea <= tol_a):
                dist = float(np.abs(q - q_ref).max())
                if near is None or dist < near[0]:
                    near = (dist, q, ep, ea)
            if ep < 1e-4 and (axis is None or ea < 1e-4) and seed is seeds[0]:
                break                       # 第一个初值就解好了 = 离初值最近，直接收工
    finally:
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mujoco.mj_forward(model, data)
    assert best is not None
    pick = near if near is not None else best
    return pick[1], pick[2], pick[3], tries


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
                 gravity: bool = True, move_seconds: float = 1.5, log_scene: bool = True,
                 joint_limits="auto", limits_path=None):
        """``joint_limits``：``"auto"`` = 读 ``config/joint_limits.json``（界面就是这条路）；
        ``None`` = 只用 URDF 默认；也可以直接给 ``{关节名: (最小°, 最大°)}``（给脚本/自检用）。
        ``limits_path``：指定配置文件路径（默认 :func:`revA1_config.limits_path`）。"""
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
        # 关节限位：URDF 默认 → 可被 config/joint_limits.json 覆盖（见 revA1_config / 界面「关节限位」卡片）
        self.limits_path = Path(limits_path) if limits_path else cfg.limits_path()
        self.limits_error = ""
        self.limits: dict[str, tuple[float, float]] = cfg.urdf_limits()
        self.limits_source = "URDF"
        saved = None
        if joint_limits == "auto":
            try:
                saved = cfg.load_limits(self.limits_path)
            except cfg.ConfigError as exc:
                self.limits_error = str(exc)
        elif isinstance(joint_limits, dict):
            saved = joint_limits
        self.set_joint_limits(saved, source=("config" if saved else "URDF"), log_it=False)
        self.floor_z = self._read_floor_z()
        self.mode = "joint"
        self.cmd = np.array(spec.HOME_QPOS, dtype=float)
        self.motion: Motion | None = None
        self.pending: dict | None = None
        self.last_report: dict | None = None
        self.last_ik: IKResult | None = None
        self.last_plan: PlannedLine | None = None
        self.ik_target: tuple[np.ndarray, np.ndarray | None, bool] | None = None
        self.path_points: np.ndarray | None = None
        self.show_tools: set[str] = set(spec.tool_names())      # 画哪几个工具的 TCP 向量箭头
        self.tool_width_m = float(spec.TOOL_ARROW_R_M)          # 箭头杆半径[m]
        self.trails: dict[str, Trail] = {}                      # 工具尖轨迹（任务信号驱动，见 record_trails）
        self.hidden_geoms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.messages: list[str] = []
        self.sim_time = 0.0
        self.reset_model(log_it=False)
        if log_scene:
            self.log(f"模型已加载：{self.scene.name}（{self.model.nu} 个位置伺服，"
                     f"dt = {self.dt * 1000:.0f} ms，home 姿态已就位）")
            if self.limits_error:
                self.log(f"关节限位：{self.limits_path} 读不了（{self.limits_error}）→ 用 URDF 默认")
            elif self.limits_source == "config":
                self.log(f"关节限位：{self.limits_path.name} 已加载 —— {cfg.describe(self.limits)}"
                         "（界面「关节限位」卡片可改）")

    # ---------------------------------------------------------------- 读
    def _read_floor_z(self) -> float:
        """地面高度：读场景里名字叫 ``floor`` 的 geom（点云场景会把它挪到小车脚下）。"""
        try:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        except Exception:  # noqa: BLE001
            gid = -1
        return float(self.model.geom_pos[gid][2]) if gid >= 0 else float(spec.FLOOR_Z)

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
        """工具尖离地高度[m]（地面 = 场景里那个叫 ``floor`` 的 geom；读不到才用 ``spec.FLOOR_Z``）。

        换成"带点云 / 带小车"的场景时，地面会被挪到小车脚下（例如 z = -1.1），
        这样界面上显示的"离地"才还是真的离地。
        """
        return float(self.tip()[2] - self.floor_z)

    # ---------------------------------------------------------------- 工具（TCP 向量）
    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """末端法兰（``ee_site``）在世界里的位姿 ``(原点, 旋转矩阵)``。"""
        return ee_frame(self.model, self.data)

    def tool_points(self, names=None) -> dict[str, np.ndarray]:
        """各工具 TCP 的**世界坐标**（默认三个都给；顺序 = ``spec.TOOLS``）。"""
        return tool_points(self.model, self.data, names)

    def tool_lines(self) -> list[str]:
        """每个工具一行文字：TCP 世界坐标 / 离法兰多远 / 离地多高（界面直接拿去显示）。"""
        out = []
        for name, q in self.tool_points().items():
            v = spec.tool_vector(name)
            out.append(f"{name}  TCP ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}) m"
                       f" · 离法兰 {float(np.linalg.norm(v)) * 1000:5.1f} mm"
                       f" · 离地 {q[2] - self.floor_z:.3f} m")
        return out

    def add_tool_arrows(self, scene, names=None, *, width: float | None = None,
                        tip_dot: bool = False) -> int:
        """按 :attr:`show_tools` 在当前末端位姿下画工具箭头（每帧调 → 跟着臂动）。

        三根箭头**互相平行、都平行于末端工具轴**（法兰 +z）：从法兰平面上的**安装点**
        （``(vx, vy, 0)``）沿工具轴画到各自 **TCP**（长度 = 该工具的轴向长度 ``vz``）。
        默认不加端点小球。返回画了几根。``names`` 给了就用它（忽略 :attr:`show_tools`），
        方便单独出图。
        """
        p, R = self.ee_pose()
        use = sorted(self.show_tools, key=spec.tool_names().index) if names is None else names
        return draw_tool_arrows(scene, p, R, use,
                                width=self.tool_width_m if width is None else float(width),
                                tip_dot=tip_dot)

    # ---------------------------------------------------------------- 工具尖轨迹
    def trail(self, name: str, *, create: bool = True) -> Trail | None:
        """取某个工具的轨迹（``create=True`` 时没有就建，颜色 = 该工具箭头颜色）。"""
        if name not in spec.TOOLS:
            raise KeyError(f"没有这个工具：{name!r}（只认 {spec.tool_names()}）")
        t = self.trails.get(name)
        if t is None and create:
            t = Trail(name, rgba=spec.TOOLS[name]["rgba"])
            self.trails[name] = t
        return t

    def record_trails(self, names=None) -> int:
        """把当前各工具的 **TCP**（= 画面上那根箭头的尖）记进轨迹，返回真新增的点数。

        界面在"任务中"（收到 6501 的 ``motion: start``）时每帧调它；收到 ``stop`` 就不再调，
        轨迹自然停住。``names=None`` = 三个工具都记；一般传"界面上勾选的那几个"。
        """
        n = 0
        for name, p in self.tool_points(names).items():
            t = self.trail(name)
            if t is not None and t.add(p):
                n += 1
        return n

    def draw_trails(self, scene, names=None) -> int:
        """把（非空的）轨迹画到画面上，返回画了几段；``names=None`` = 全部。"""
        use = list(self.trails) if names is None else [n for n in names if n in self.trails]
        return sum(draw_trail(scene, self.trails[n]) for n in use)

    def clear_trails(self, names=None) -> int:
        """清空轨迹（``names=None`` = 全清），返回清了几根。"""
        use = list(self.trails) if names is None else [n for n in names if n in self.trails]
        for n in use:
            self.trails[n].clear()
        return len(use)

    def trail_texts(self) -> list[str]:
        """界面显示用：每根轨迹一行（点数 / 路程）；没记过的工具不出现。"""
        return [self.trails[n].text() for n in spec.tool_names() if n in self.trails]

    def geom_names(self) -> list[str]:
        """场景里所有 geom 的名字（没名字的给 ``geom<id>``）。"""
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"
                for i in range(self.model.ngeom)]

    def hide_geom(self, name: str, hide: bool = True) -> bool:
        """把一个 geom **藏起来 / 放回去**（纯显示用，不动物理）。

        做法：挪到 y 轴外 10 km 再把透明度清零 —— 比只改 alpha 稳（有的渲染后端不吃 alpha=0）。
        原位置记在 ``self.hidden_geoms`` 里，能原样恢复。返回 ``False`` 表示场景里没有这个名字。

        典型用途：场景里那些"辅助标记"（地面/车顶上的名义作业点绿圆盘 ``target_pad``、
        小车 ``pc_cart``…）不想看时一键隐藏，不用改 XML。
        """
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            return False
        if hide:
            if name not in self.hidden_geoms:
                self.hidden_geoms[name] = (self.model.geom_pos[gid].copy(),
                                           self.model.geom_rgba[gid].copy())
            self.model.geom_pos[gid] = (0.0, 1.0e4, 0.0)
            self.model.geom_rgba[gid][3] = 0.0
        elif name in self.hidden_geoms:
            pos, rgba = self.hidden_geoms.pop(name)
            self.model.geom_pos[gid] = pos
            self.model.geom_rgba[gid] = rgba
        return True

    def geom_visible(self, name: str) -> bool:
        """这个 geom 现在是不是"看得见"（被 :meth:`hide_geom` 藏起来过的就是 False）。"""
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            return False
        return name not in self.hidden_geoms and abs(float(self.model.geom_pos[gid][1])) < 1.0e3

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

    # ---------------------------------------------------------------- 关节限位
    def set_joint_limits(self, limits=None, *, source: str = "应用",
                         log_it: bool = True) -> dict[str, tuple[float, float]]:
        """换一套关节限位（``{关节名: (最小°, 最大°)}``；``None`` = 全回 URDF 默认）。

        同时改三处（少一处就白改，见 :func:`apply_joint_limits`）：``self.lo/hi``、
        ``model.jnt_range``、``model.actuator_ctrlrange``；当前目标角 ``cmd`` 会夹回新范围。
        返回生效值（度）。值不合法抛 :class:`revA1_config.ConfigError`（原状不动）。
        """
        eff = apply_joint_limits(self.model, limits)       # 可能抛 ConfigError（调用方接住）
        self.lo = np.array([np.radians(eff[j][0]) for j in spec.JOINTS])
        self.hi = np.array([np.radians(eff[j][1]) for j in spec.JOINTS])
        cmd = getattr(self, "cmd", None)                    # __init__ 里调得比 cmd 早，这里要容错
        if cmd is not None:
            self.cmd = np.clip(np.asarray(cmd, dtype=float), self.lo, self.hi)
        self.limits = dict(eff)
        self.limits_source = str(source)
        if log_it:
            self.log(f"关节限位（{source}）：{cfg.describe(eff)}")
        return dict(eff)

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """当前生效的关节限位（度）。"""
        return dict(self.limits)

    def save_joint_limits(self, limits=None, *, log_it: bool = True) -> Path:
        """把限位存进 ``config/joint_limits.json``（只写与 URDF 不同的项）并立即生效。"""
        eff = self.set_joint_limits(limits, source="保存为默认", log_it=False)
        path = cfg.save_limits(eff, self.limits_path)
        self.limits_source = "config"
        if log_it:
            self.log(f"关节限位：已保存到 {path} —— {cfg.describe(eff)}（下次启动自动加载）")
        return path

    def reset_joint_limits(self, *, log_it: bool = True) -> tuple[dict, bool]:
        """回 URDF 默认（并删掉配置文件）。返回 ``(生效值, 是否删掉了文件)``。"""
        eff = self.set_joint_limits(None, source="URDF", log_it=False)
        removed = False
        if self.limits_path.exists():
            removed = cfg.clear_limits(self.limits_path)
        if log_it:
            self.log("关节限位：已恢复 URDF 默认（joint3 ±164°，其余 ±180°）"
                     + ("，配置文件已删除" if removed else ""))
        return eff, removed

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
            truncated = False
            for i in range(m):
                last = i == m - 1
                t = i / float(m - 1)
                p = p0 + (target - p0) * t
                a = None if axis_goal is None else interp_axis(a0, axis_goal, t)
                q, ep, ea, _ = self._ik_raw(p, a, seed, multi=last)
                q = np.asarray(q, dtype=float)
                # 关键：**每个路点都要跟上一个路点"同一支"**。单初值 IK 也可能游走到别的解支
                # （实测末点会从 J2≈-173° 跳到 -61°、J3 从 +129° 翻到 -129°），
                # 那样关节空间回放会把工具尖甩出门（实测 0.64 m）。
                jump = float(np.abs(np.degrees(q - seed)).max())
                if float(ep) > LINE_MAX_ERR_M or jump > LINE_MAX_BRANCH_JUMP_DEG:
                    q2, ep2, ea2, _ = self._ik_raw(p, a, seed, multi=True)
                    jump2 = float(np.abs(np.degrees(np.asarray(q2, dtype=float) - seed)).max())
                    if float(ep2) <= LINE_MAX_ERR_M and jump2 <= LINE_MAX_BRANCH_JUMP_DEG:
                        q, ep, ea, jump = np.asarray(q2, dtype=float), ep2, ea2, jump2
                    else:
                        truncated = True
                        failed_at = i
                        q = qs[-1].copy() if qs else np.asarray(seed, dtype=float).copy()
                        qs.append(q)
                        pts.append(p)
                        break
                if last:
                    ep_end, ea_end = float(ep), float(ea)
                if float(ep) > LINE_MAX_ERR_M:            # 兜底：过不去就停住，别跳变
                    failed_at = i if failed_at < 0 else failed_at
                    q = qs[-1].copy() if qs else q
                qs.append(q)
                pts.append(p)
                seed = q
        finally:
            self.data.qpos[:] = qpos0
            self.data.qvel[:] = qvel0
            mujoco.mj_forward(self.model, self.data)
        waypoints, points = np.vstack(qs), np.vstack(pts)
        ok_pos = ep_end <= IK_TOL_POS_M and not truncated
        ok_axis = (axis_goal is None or ea_end <= IK_TOL_AXIS_RAD) and not truncated
        reason = ""
        if truncated:
            reason = (f"直线走到第 {failed_at + 1}/{m} 个路点就走不过去了"
                      f"（这一支到不了：顶到限位或要换姿态）")
        elif not ok_pos:
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
                     f"（画面里目标点会标成红色。本支走不过去时：先用手/关节按钮把姿态转过去，"
                     f"或把目标改到这条直线走得到的位置）")
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







