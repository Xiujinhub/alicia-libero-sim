#!/usr/bin/env python3
# ============================================================================
# 旧版 tkinter 界面（保留可跑，逻辑与 revA1_gui.py 同源）
# ----------------------------------------------------------------------------
# 新版界面见 revA1_gui.py（PySide6）：关节换成 −/+ 按钮、点到点改成笛卡尔直线、
# IK 改多初值并明确报"能不能解 / 差多少"，配色也重做了。
# 这份 tkinter 版只作为对照/回退保留，命令行参数和以前完全一样：
#     python interactive_control_tk.py --selftest
#     python interactive_control_tk.py --ui-test
# ============================================================================
"""交互控制台：一个 tkinter 窗口，把 MuJoCo 的实时画面和控制面板放进同一个界面。

三种控制方式（窗口左上角单选）
------------------------------
1. **单关节 Joint**   每个关节一根滑条，拖动即发位置指令（可 ±0.5° 微调）；用于示教/挑姿态。
2. **点到点 P2P**     给定目标关节角，用 smoothstep 在设定时间内平滑开过去（首尾速度 0）。
3. **IK 笛卡尔**      给定 tool_site 的世界坐标 + 工具轴方向，先数值求逆解再走 P2P；
                      目标点会在画面里画一个绿色球 + 工具轴箭头。

其它：回 home / 急停（就地保持）/ 暂停物理 / 重力开关 / kp 缩放 / 实时状态栏 / 日志。

    python interactive_control.py                  # 开窗口
    python interactive_control.py --lang zh        # 有中文字体时窗口标签用中文
    python interactive_control.py --selftest       # 无窗口：把控制逻辑全跑一遍并断言
    python interactive_control.py --ui-test        # 真建窗口 -> 脚本化点一遍 -> 存图 -> 退出
    python interactive_control.py --exit-after 8   # 开窗口跑 8 秒自动退出（自动化用）

鼠标（画面区域）
----------------
    左键拖动 = 转视角    右键拖动 = 平移    滚轮 = 推拉    双击 = 视角复位

键盘
----
    空格 = 暂停/继续    h = 回 home    r = 复位模型    g = 重力开关
    1..6 = 选中关节，随后 [ / ] 微调 ±0.5°      ESC / 关窗口 = 退出

为什么是 tkinter
----------------
本机 conda 环境只有 tkinter + Pillow（没有 PyQt / imgui），所以画面用
``mujoco.Renderer`` 离屏渲染 -> PIL -> ``Canvas``。软件 OpenGL 下 640x480 约 30 fps，
嫌慢就用 ``--width/--height`` 调小窗口（或 ``--fps`` 降帧率）。

⚠️ 关于中文：本机 ``fc-list :lang=zh`` 为空（没装 CJK 字体），窗口标签会自动退回英文，
避免显示成方框；如果你的机器有中文字体就会自动用中文，也可以 ``--lang zh`` 强制。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl()
import mujoco  # noqa: E402  (必须在设置 MUJOCO_GL 之后)

import revA1_spec as spec  # noqa: E402
import sim_ik as ik  # noqa: E402

# tkinter / Pillow 都是本机已有的依赖；缺失时至少让 --selftest 还能跑（不给窗口）
try:
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk
except ImportError as _e:  # pragma: no cover - 只在本机缺依赖时触发
    tk = ttk = Image = ImageTk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR: str | None = str(_e)
else:
    _TK_IMPORT_ERROR = None

_BASE = ttk.Frame if ttk is not None else object

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent

# IK 快捷预设（和 viewer.py 保持一致）：工具尖目标（世界系）+ 工具轴方向
PRESETS = [
    ("home", (0.40, 0.00, 0.179), (0, 0, -1)),
    ("forward", (0.55, 0.00, 0.420), (0, 0, -1)),
    ("side", (0.30, 0.30, 0.450), (0, 0, -1)),
    ("low", (0.45, -0.15, 0.150), (0, 0, -1)),
]


def frame_from_z(z) -> np.ndarray:
    """构造一个正交基：第 3 列（z 轴）= 给定方向。用于画工具轴指示箭头。"""
    z = np.asarray(z, dtype=float).ravel()
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x = x / max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def fit_image(img, size, bg=(16, 20, 24)):
    """把 img 等比缩放到 size 里并居中（多出来的地方填底色），避免画面被拉变形。"""
    vw, vh = int(size[0]), int(size[1])
    if vw < 8 or vh < 8:
        return img
    iw, ih = img.size
    k = min(vw / float(iw), vh / float(ih))
    new = (max(int(round(iw * k)), 1), max(int(round(ih * k)), 1))
    if new != img.size:
        img = img.resize(new, Image.BILINEAR)
    out = Image.new(img.mode, (vw, vh), bg)
    out.paste(img, ((vw - new[0]) // 2, (vh - new[1]) // 2))
    return out


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


# =============================================================== 界面文字（i18n）
# 没有中文字体时窗口标签退回英文（否则全是方框）。--lang zh 可强制。
_STR: dict[str, dict[str, str]] = {
    "en": {
        "win_title": "TB6-R5-RevA1 interactive console",
        "canvas_no_gl": "Offscreen renderer unavailable (no GL).\nControls still work.",
        "mode_grp": "Control mode",
        "mode_joint": "Joint (sliders)",
        "mode_p2p": "Point-to-point",
        "mode_ik": "IK (cartesian)",
        "hint_joint": "All six sliders are live servo targets.\nDrag one -> that joint moves.",
        "hint_p2p": "Pick target angles, then Start.\nSmoothstep move over the given time.",
        "hint_ik": "Type tool-tip target (world) + tool axis,\nthen Solve & Move. Green marker = target.",
        "joint_grp": "Joints [deg]",
        "joint_sync": "Sliders <- current",
        "joint_zero": "All zero",
        "joint_home": "Home",
        "joint_minus": "−",
        "joint_plus": "+",
        "p2p_grp": "Point-to-point (joint space)",
        "p2p_src": "target from",
        "p2p_src_slider": "sliders",
        "p2p_src_actual": "measured qpos",
        "p2p_src_preset": "IK preset",
        "p2p_start": "Start P2P",
        "p2p_stop": "Stop (hold here)",
        "p2p_target": "target [deg] {t}",
        "ik_grp": "IK (cartesian)   tool_site target",
        "ik_axis": "tool axis",
        "ik_axis_free": "free",
        "ik_read": "Read current tip",
        "ik_solve": "Solve only",
        "ik_solve_move": "Solve & Move",
        "ik_result": "last solve: pos err {ep:.2f} mm, axis err {ea:.2f} deg, {ms:.0f} ms",
        "srv_grp": "Servo / run",
        "srv_dur": "move time [s]",
        "srv_kp": "kp scale",
        "srv_pause": "pause physics",
        "srv_gravity": "gravity",
        "srv_track": "follow target",
        "srv_estop": "E-stop (hold)",
        "srv_reset": "Reset model",
        "log_grp": "Log",
        "log_clear": "Clear",
        "st_mode": "mode {m}",
        "st_moving": "MOVING",
        "st_idle": "idle",
        "st_paused": "PAUSED",
        "st_tool": "tip {p}  (h {h:.3f} m)",
        "st_axis": "axis {a}",
        "st_track": "tracking err {e:.3f} deg",
        "st_contact": "contacts {n}",
        "st_fps": "{fps:.1f} fps (render {ms:.0f} ms)",
        "log_mode": "mode -> {m}",
        "log_joint": "J{i} -> {d:+.2f} deg",
        "log_cancel": "motion cancelled by joint slider",
        "log_p2p_start": "P2P start: {src} -> {t} deg, {s:.2f} s",
        "log_p2p_done": "P2P done: max joint tracking error {e:.3f} deg, tip {p}",
        "log_ik_solved": "IK solved: q = {q} deg, pos err {ep:.2f} mm, axis err {ea:.2f} deg",
        "log_ik_fail": "IK failed: pos err {ep:.1f} mm -> not executing",
        "log_ik_done": "IK done: tip {p} / target {t}, pos err {ep:.2f} mm, axis err {ea:.2f} deg",
        "log_estop": "E-stop: holding current pose",
        "log_reset": "model reset to home",
        "log_kp": "kp scale = {s:.2f}",
        "log_gravity": "gravity {v}",
        "log_pause": "physics {v}",
        "log_zero": "all joint targets zeroed",
        "log_read": "target box <- current tip {p}",
        "log_preset": "preset {n}: target {p}",
        "log_no_gl": "offscreen renderer unavailable: {e}",
        "log_frame": "frame saved: {p}",
        "log_bad_num": "not a number: {v!r} (check the red box)",
        "on": "on",
        "off": "off",
    },
    "zh": {
        "win_title": "TB6-R5-RevA1 交互控制台",
        "canvas_no_gl": "无法创建离屏渲染器（没有 GL）。\n控制面板依然可用。",
        "mode_grp": "控制模式",
        "mode_joint": "单关节（滑条）",
        "mode_p2p": "点到点（P2P）",
        "mode_ik": "IK 笛卡尔",
        "hint_joint": "六根滑条就是六个关节的位置指令，\n拖哪根哪个关节就动。",
        "hint_p2p": "设好目标关节角再点开始，\nsmoothstep 在设定时间内平滑开过去。",
        "hint_ik": "填工具尖目标（世界系）+ 工具轴方向，\n点“求解并运动”；绿色标记是目标点。",
        "joint_grp": "关节角 [度]",
        "joint_sync": "滑条←当前",
        "joint_zero": "全部归零",
        "joint_home": "回 home",
        "joint_minus": "−",
        "joint_plus": "+",
        "p2p_grp": "点到点（关节空间）",
        "p2p_src": "目标来自",
        "p2p_src_slider": "滑条",
        "p2p_src_actual": "实测姿态",
        "p2p_src_preset": "IK 预设",
        "p2p_start": "开始运动",
        "p2p_stop": "停止（就地保持）",
        "p2p_target": "目标 [度] {t}",
        "ik_grp": "IK 笛卡尔   tool_site 目标",
        "ik_axis": "工具轴",
        "ik_axis_free": "不约束",
        "ik_read": "读当前工具尖",
        "ik_solve": "只求解",
        "ik_solve_move": "求解并运动",
        "ik_result": "上次求解：位置误差 {ep:.2f} mm，姿态误差 {ea:.2f}°，{ms:.0f} ms",
        "srv_grp": "伺服 / 运行",
        "srv_dur": "运动时长 [s]",
        "srv_kp": "kp 缩放",
        "srv_pause": "暂停物理",
        "srv_gravity": "重力",
        "srv_track": "跟随目标",
        "srv_estop": "急停（就地保持）",
        "srv_reset": "复位模型",
        "log_grp": "日志",
        "log_clear": "清空",
        "st_mode": "模式 {m}",
        "st_moving": "运动中",
        "st_idle": "空闲",
        "st_paused": "已暂停",
        "st_tool": "工具尖 {p}（离地 {h:.3f} m）",
        "st_axis": "工具轴 {a}",
        "st_track": "跟踪误差 {e:.3f}°",
        "st_contact": "接触 {n} 对",
        "st_fps": "{fps:.1f} fps（渲染 {ms:.0f} ms）",
        "log_mode": "切换模式 -> {m}",
        "log_joint": "J{i} -> {d:+.2f}°",
        "log_cancel": "被关节滑条打断，运动已取消",
        "log_p2p_start": "P2P 开始：{src} -> {t}°，{s:.2f} s",
        "log_p2p_done": "P2P 完成：最大关节跟踪误差 {e:.3f}°，工具尖 {p}",
        "log_ik_solved": "IK 求解成功：q = {q}°，位置误差 {ep:.2f} mm，姿态误差 {ea:.2f}°",
        "log_ik_fail": "IK 失败：位置误差 {ep:.1f} mm -> 不执行",
        "log_ik_done": "IK 完成：工具尖 {p} / 目标 {t}，位置误差 {ep:.2f} mm，姿态误差 {ea:.2f}°",
        "log_estop": "急停：就地保持当前姿态",
        "log_reset": "模型已复位到 home",
        "log_kp": "kp 缩放 = {s:.2f}",
        "log_gravity": "重力 {v}",
        "log_pause": "物理 {v}",
        "log_zero": "六个关节目标已全部归零",
        "log_read": "目标框 <- 当前工具尖 {p}",
        "log_preset": "预设 {n}：目标 {p}",
        "log_no_gl": "无法创建离屏渲染器：{e}",
        "log_frame": "已存图：{p}",
        "log_bad_num": "不是数字：{v!r}（看标红的输入框）",
        "on": "开",
        "off": "关",
    },
}

_LANG = "en"

# 常见中文字体族名（fc-list 查不到时的第二道保险）
_CJK_FAMS = ("noto sans cjk", "noto serif cjk", "source han", "wenquanyi", "wqy",
             "microsoft yahei", "simhei", "simsun", "pingfang", "droid sans fallback",
             "sarasa", "arphic", "uming", "ukai", "fandol", "heiti", "songti")


def L(key: str, **kw) -> str:
    """取界面文字（当前语言缺失时退回英文，再退回到 key 本身）。"""
    tab = _STR.get(_LANG) or _STR["en"]
    s = tab.get(key) or _STR["en"].get(key) or key
    try:
        return s.format(**kw) if kw else s
    except (KeyError, IndexError, ValueError):
        return s


def detect_lang() -> str:
    """能否显示中文：先问 fontconfig，再看 tkinter 的字体族列表。"""
    exe = shutil.which("fc-list")
    if exe:
        try:
            out = subprocess.run([exe, ":lang=zh"], capture_output=True, text=True,
                                 timeout=5).stdout
            if out.strip():
                return "zh"
        except Exception:  # noqa: BLE001
            pass
    try:
        import tkinter.font as tkfont  # noqa: PLC0415

        fams = " ".join(tkfont.families()).lower()
        if any(k in fams for k in _CJK_FAMS):
            return "zh"
    except Exception:  # noqa: BLE001
        pass
    return "en"


# =============================================================== 相机 / 轨迹
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
        h = self._home
        self.mjv.distance = h["distance"]
        self.mjv.azimuth = h["azimuth"]
        self.mjv.elevation = h["elevation"]
        self.mjv.lookat[:] = h["lookat"]

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        az = np.radians(self.mjv.azimuth)
        el = np.radians(self.mjv.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        n = float(np.linalg.norm(right))
        right = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else right / n
        return fwd, right, np.cross(right, fwd)

    def orbit(self, dx: float, dy: float) -> None:
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


class Motion:
    """把 6 个驱动器的 ctrl 在仿真时间上从 q_start 平滑插值到 q_end。"""

    def __init__(self, q_start, q_end, seconds: float, dt: float,
                 kind: str, label: str = ""):
        self.q_start = np.asarray(q_start, dtype=float).copy()
        self.q_end = np.asarray(q_end, dtype=float).copy()
        self.kind = kind
        self.label = label
        self.n = max(int(round(float(seconds) / dt)), 2)
        self.traj = ik.smoothstep(self.n)
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= self.n

    def value(self) -> np.ndarray:
        s = self.traj[min(self.i, self.n - 1)]
        return self.q_start + (self.q_end - self.q_start) * s


# =============================================================== 仿真 + 控制核心
class ArmSim:
    """模型 + 伺服 + 三种控制方式（纯逻辑，不依赖 GUI，可被 --selftest 单独驱动）。

    位置指令统一放在 ``self.cmd``（6 个驱动器的目标角），三种模式的区别只在于谁写它：

    * ``joint`` 模式：滑条回调写 ``cmd``（拖动即动）；
    * ``p2p`` / ``ik`` 模式：``Motion`` 每个仿真步覆盖 ``cmd``；
    * ``hold`` 模式：``cmd`` 固定为急停那一瞬的实测姿态（原地撑住）。
    """

    def __init__(self, scene=spec.SCENE_XML, *, kp_scale: float = 1.0,
                 gravity: bool = True, move_seconds: float = 1.5):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.move_seconds = float(move_seconds)
        self.kp0 = np.array([self.model.actuator_gainprm[i][0] for i in range(self.model.nu)])
        self.kv0 = np.array([-self.model.actuator_biasprm[i][2] for i in range(self.model.nu)])
        self.kp_scale = 1.0
        self.set_kp_scale(kp_scale)
        self.set_gravity(gravity)
        self.lo = np.array([spec.LIMITS[j][0] for j in spec.JOINTS])
        self.hi = np.array([spec.LIMITS[j][1] for j in spec.JOINTS])
        self.mode = "joint"
        self.cmd = np.array(spec.HOME_QPOS, dtype=float)
        self.motion: Motion | None = None
        self.pending: dict | None = None
        self.last_report: dict | None = None
        self.last_ik: dict | None = None
        self.ik_target: tuple[np.ndarray, np.ndarray | None] | None = None
        self.messages: list[str] = []
        self.sim_time = 0.0
        self.reset_model(log_it=False)

    # ---------------------------------------------------------------- 读
    def q_pos(self) -> np.ndarray:
        return ik.arm_qpos(self.model, self.data)

    def tip(self) -> np.ndarray:
        return ik.site_pos(self.model, self.data, "tool_site")

    def tip_axis(self) -> np.ndarray:
        return ik.tool_axis(self.model, self.data, "tool_site")

    def track_err_deg(self) -> float:
        return float(np.degrees(np.abs(self.cmd - self.q_pos())).max())

    def moving(self) -> bool:
        return self.motion is not None

    def log(self, msg: str) -> None:
        self.messages.append(msg)

    def drain(self) -> list[str]:
        out, self.messages = self.messages, []
        return out

    # ---------------------------------------------------------------- 写
    def set_kp_scale(self, scale: float) -> None:
        self.kp_scale = float(max(scale, 0.01))
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = self.kp0[i] * self.kp_scale
            self.model.actuator_biasprm[i, 1] = -self.kp0[i] * self.kp_scale
            self.model.actuator_biasprm[i, 2] = -self.kv0[i] * self.kp_scale

    def set_gravity(self, on: bool) -> None:
        self.model.opt.gravity[:] = (0.0, 0.0, -9.81) if on else (0.0, 0.0, 0.0)

    def set_joint(self, i: int, q_rad: float) -> None:
        """单关节：直接改第 i 个驱动器目标（滑条回调走这里）。"""
        if self.motion is not None:
            self.motion = None
            self.pending = None
            self.log(L("log_cancel"))
        self.mode = "joint"
        self.cmd = self.cmd.copy()
        self.cmd[i] = float(np.clip(float(q_rad), self.lo[i], self.hi[i]))

    def set_cmd(self, q) -> None:
        """直接设定 6 个目标（不插值的「瞬移式」设定，主要给测试/复位用）。"""
        self.cmd = np.clip(np.asarray(q, dtype=float).ravel(), self.lo, self.hi)
        self.mode = "joint"

    def hold(self) -> None:
        """急停：取消运动，把目标钉在当前实测姿态上（伺服原地撑住）。"""
        self.motion = None
        self.pending = None
        self.mode = "hold"
        self.cmd = self.q_pos().copy()
        self.log(L("log_estop"))

    def reset_model(self, log_it: bool = True) -> None:
        if not ik.reset_home(self.model, self.data):
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        self.motion = None
        self.pending = None
        self.mode = "joint"
        self.cmd = np.array(spec.HOME_QPOS, dtype=float)
        self.sim_time = 0.0
        if log_it:
            self.log(L("log_reset"))

    def home(self, seconds: float | None = None) -> None:
        """平滑回 home（走 P2P，不是瞬移）。"""
        self.start_p2p(spec.HOME_QPOS, seconds, kind="p2p", src="home", label="home")

    # ---------------------------------------------------------------- 点到点
    def start_p2p(self, q_target, seconds: float | None = None, *,
                  kind: str = "p2p", src: str = "cmd", label: str = "",
                  pos=None, axis=None) -> np.ndarray:
        """从当前指令位置平滑开到 q_target（smoothstep 插值；到点后自动报告误差）。

        ``pos`` / ``axis`` 只用于"到点后回报笛卡尔误差"，不影响运动本身。
        """
        qt = np.clip(np.asarray(q_target, dtype=float).ravel(), self.lo, self.hi)
        sec = self.move_seconds if seconds is None else float(seconds)
        self.motion = Motion(self.cmd, qt, sec, self.dt, kind, label)
        self.mode = kind
        self.pending = dict(kind=kind, q_target=qt, deadline=self.sim_time + sec + 0.7,
                            pos=None if pos is None else np.asarray(pos, dtype=float),
                            axis=None if axis is None else np.asarray(axis, dtype=float))
        self.log(L("log_p2p_start", src=src, t=np.round(np.degrees(qt), 1).tolist(),
                   s=sec))
        return qt

    # ---------------------------------------------------------------- IK
    def solve_ik(self, pos, axis=None, seed=None) -> tuple[np.ndarray, float, float]:
        """求逆解。内部会临时改 qpos，算完原样恢复（不会把机械臂瞬移过去）。"""
        pos = np.asarray(pos, dtype=float).ravel()
        axis = None if axis is None else np.asarray(axis, dtype=float).ravel()
        if axis is not None and not np.any(axis):
            axis = None
        seed = np.asarray(self.q_pos() if seed is None else seed, dtype=float).ravel()
        qpos0, qvel0 = self.data.qpos.copy(), self.data.qvel.copy()
        try:
            q, ep, ea = ik.solve_ik_pose(self.model, self.data, "tool_site",
                                         pos, axis, q_seed=seed.copy())
            if ep > 3e-3 and axis is not None:
                q2, ep2, ea2 = ik.refine_ik_pose(self.model, self.data, "tool_site",
                                                 pos, axis, q_seed=q)
                if ep2 < ep:
                    q, ep, ea = q2, ep2, ea2
        finally:
            self.data.qpos[:] = qpos0
            self.data.qvel[:] = qvel0
            mujoco.mj_forward(self.model, self.data)
        return np.clip(q, self.lo, self.hi), float(ep), float(ea)

    def start_ik(self, pos, axis=None, seconds: float | None = None, *,
                 execute: bool = True):
        """解 IK（可选直接执行）：返回 ``(q, 位置误差[m], 工具轴夹角[rad])``。"""
        t0 = time.perf_counter()
        q, ep, ea = self.solve_ik(pos, axis)
        ms = (time.perf_counter() - t0) * 1000.0
        self.last_ik = dict(q=q.copy(), pos_err_m=ep, axis_err_rad=ea, ms=ms)
        self.ik_target = (np.asarray(pos, dtype=float).ravel(),
                          None if axis is None else np.asarray(axis, dtype=float).ravel())
        self.log(L("log_ik_solved", q=np.round(np.degrees(q), 1).tolist(),
                   ep=ep * 1000.0, ea=np.degrees(ea)))
        if ep > 5e-3:
            self.log(L("log_ik_fail", ep=ep * 1000.0))
            return q, ep, ea
        if execute:
            self.start_p2p(q, seconds, kind="ik", src="IK",
                           label="IK -> " + str(np.round(np.asarray(pos, float), 3).tolist()),
                           pos=pos, axis=axis)
        return q, ep, ea

    # ---------------------------------------------------------------- 每步
    def apply_ctrl(self) -> None:
        """把"这一刻该给的目标"写进 data.ctrl（mj_step 之前调用）。"""
        if self.motion is not None:
            self.cmd = self.motion.value()
            self.motion.i += 1
            if self.motion.done:
                self.cmd = self.motion.q_end.copy()
                self.motion = None
        for j, v in zip(spec.JOINTS, self.cmd):
            self.data.ctrl[ik.dof_adr(self.model, j)] = float(v)

    def step(self) -> None:
        """推进一个仿真步（含运动插值与到点误差汇报）。"""
        self.apply_ctrl()
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt
        self._check_pending()

    def _check_pending(self) -> None:
        """运动结束并静置一会儿后，量一次实际到位精度写进日志。"""
        if self.pending is None or self.sim_time < self.pending["deadline"]:
            return
        p, self.pending = self.pending, None
        tip = self.tip()
        jerr = float(np.degrees(np.abs(self.q_pos() - p["q_target"])).max())
        rep: dict = dict(kind=p["kind"], joint_err_deg=jerr, tip=tip,
                         target=None if p.get("pos") is None else np.asarray(p["pos"]))
        if p.get("pos") is None:
            self.log(L("log_p2p_done", e=jerr, p=np.round(tip, 3).tolist()))
        else:
            tgt = np.asarray(p["pos"], dtype=float)
            perr = float(np.linalg.norm(tip - tgt))
            aerr = None
            axis = p.get("axis")
            if axis is not None and np.any(axis):
                a = np.asarray(axis, dtype=float)
                a = a / max(float(np.linalg.norm(a)), 1e-12)
                aerr = float(np.degrees(np.arccos(np.clip(float(self.tip_axis() @ a),
                                                          -1.0, 1.0))))
            rep.update(pos_err_m=perr, axis_err_deg=aerr)
            self.log(L("log_ik_done", p=np.round(tip, 3).tolist(),
                       t=np.round(tgt, 3).tolist(), ep=perr * 1000.0,
                       ea=(0.0 if aerr is None else aerr)))
        self.last_report = rep

    def status(self) -> dict:
        tip = self.tip()
        return dict(mode=self.mode, moving=self.moving(),
                    cmd_deg=np.degrees(self.cmd), q_deg=np.degrees(self.q_pos()),
                    tip=tip, axis=self.tip_axis(),
                    height=float(tip[2] - spec.FLOOR_Z),
                    contacts=int(self.data.ncon), track_deg=self.track_err_deg(),
                    sim_time=self.sim_time, last=self.last_report,
                    kp_scale=self.kp_scale, target=self.ik_target)


# =============================================================== 右侧控制面板
class ControlPanel(_BASE):  # type: ignore[misc]
    """控制面板：控件 -> 回调 -> ArmSim。所有和模型有关的操作都走 ArmSim 的方法。"""

    def __init__(self, master, sim: "ArmSim", app: "App"):
        super().__init__(master, padding=6)
        self.sim = sim
        self.app = app
        self._sync = False            # True = 正在程序化改滑条，忽略回调
        self.joint_scale: list[tk.Scale] = []
        self.joint_val: list[ttk.Label] = []
        self.joint_tag: list[ttk.Label] = []
        self.target = np.degrees(sim.cmd).copy()      # P2P / IK 用滑条当"目标编辑器"
        self.p2p_q: np.ndarray | None = None           # 预设解出来的目标关节角
        self.selected = 0
        self._last_log: dict[str, float] = {}
        self._build_mode()
        self._build_joints()
        self._build_p2p()
        self._build_ik()
        self._build_run()
        self._build_log()
        self.set_mode("joint", log_it=False)

    # ------------------------------------------------------------ 模式
    def _build_mode(self) -> None:
        grp = ttk.LabelFrame(self, text=L("mode_grp"), padding=4)
        grp.pack(fill="x", pady=(0, 4))
        self.mode_var = tk.StringVar(value="joint")
        for i, key in enumerate(("joint", "p2p", "ik")):
            ttk.Radiobutton(grp, text=L("mode_" + key), value=key, variable=self.mode_var,
                            command=lambda k=key: self.set_mode(k)).grid(
                row=i, column=0, sticky="w")
        self.hint = ttk.Label(grp, text="", wraplength=232, justify="left",
                              foreground="#555")
        self.hint.grid(row=0, column=1, rowspan=3, sticky="nw", padx=(8, 0))

    def set_mode(self, key: str, log_it: bool = True) -> None:
        """切模式：joint=滑条直控；p2p/ik=滑条当目标编辑器。"""
        self.mode_var.set(key)
        self.hint.configure(text=L("hint_" + key))
        if key == "joint":
            self.sync_scales(self.sim.cmd)                 # 滑条恢复成"实时指令"
        else:
            self.target = np.degrees(self.sim.cmd).copy()   # 编辑器起点 = 当前姿态
        self._update_target_label()
        if log_it:
            self.sim.log(L("log_mode", m=L("mode_" + key)))

    # ------------------------------------------------------------ 关节
    def _build_joints(self) -> None:
        grp = ttk.LabelFrame(self, text=L("joint_grp"), padding=4)
        grp.pack(fill="x", pady=4)
        for i, name in enumerate(spec.JOINTS):
            lo, hi = (float(np.floor(np.degrees(spec.LIMITS[name][0]))),
                      float(np.ceil(np.degrees(spec.LIMITS[name][1]))))
            tag = ttk.Label(grp, text=f"J{i + 1}", width=3)
            tag.grid(row=i, column=0, sticky="w")
            sc = tk.Scale(grp, orient="horizontal", from_=lo, to=hi, resolution=0.5,
                          length=146, showvalue=0, sliderlength=16, width=11,
                          command=lambda v, i=i: self.on_joint(i, v))
            sc.grid(row=i, column=1, sticky="we")
            lab = ttk.Label(grp, text="+0.0", width=7, anchor="e")
            lab.grid(row=i, column=2, padx=(2, 3))
            ttk.Button(grp, text=L("joint_minus"), width=2,
                       command=lambda i=i: self.nudge(i, -0.5)).grid(row=i, column=3)
            ttk.Button(grp, text=L("joint_plus"), width=2,
                       command=lambda i=i: self.nudge(i, +0.5)).grid(row=i, column=4)
            self.joint_scale.append(sc)
            self.joint_val.append(lab)
            self.joint_tag.append(tag)
        bar = ttk.Frame(grp)
        bar.grid(row=len(spec.JOINTS), column=0, columnspan=5, sticky="we", pady=(4, 0))
        ttk.Button(bar, text=L("joint_sync"), width=11,
                   command=self.on_sync).pack(side="left")
        ttk.Button(bar, text=L("joint_zero"), width=9,
                   command=self.on_zero).pack(side="left", padx=3)
        ttk.Button(bar, text=L("joint_home"), width=8,
                   command=self.on_home).pack(side="left")

    def _throttled(self, key: str, seconds: float = 0.7) -> bool:
        """限流：同一个 key 在 seconds 秒内只放行一次（拖滑条别把日志刷爆）。"""
        now = time.perf_counter()
        if now - self._last_log.get(key, 0.0) < seconds:
            return False
        self._last_log[key] = now
        return True

    def on_joint(self, i: int, value: str) -> None:
        """滑条回调：joint 模式=直接驱动；p2p/ik 模式=只改目标值。"""
        v = float(value)
        self.joint_val[i].configure(text=f"{v:+.1f}")
        if self._sync:
            return
        if self.mode_var.get() == "joint":
            self.sim.set_joint(i, np.radians(v))
            self.target = np.degrees(self.sim.cmd).copy()
            if self._throttled(f"j{i}"):
                self.sim.log(L("log_joint", i=i + 1, d=v))
        else:
            self.target[i] = v
        self._update_target_label()

    def nudge(self, i: int, delta_deg: float) -> None:
        """±0.5° 微调：单独调某个关节时比拖滑条准。"""
        v = float(self.joint_scale[i].get()) + delta_deg
        self.joint_scale[i].set(v)          # set() 会触发 on_joint 回调
        self.select_joint(i)

    def select_joint(self, i: int) -> None:
        self.selected = int(i) % len(spec.JOINTS)
        for k, tag in enumerate(self.joint_tag):
            tag.configure(style="Sel.TLabel" if k == self.selected else "TLabel")

    def sync_scales(self, q_rad, sync_target: bool = True) -> None:
        """把 6 根滑条设成给定关节角（弧度）；不会触发驱动。"""
        self._sync = True
        try:
            for i, sc in enumerate(self.joint_scale):
                d = float(np.degrees(np.asarray(q_rad, dtype=float).ravel()[i]))
                sc.set(d)
                self.joint_val[i].configure(text=f"{d:+.1f}")
        finally:
            self._sync = False
        if sync_target:
            self.target = np.degrees(np.asarray(q_rad, dtype=float).ravel()).copy()

    def on_sync(self) -> None:
        self.sync_scales(self.sim.q_pos())

    def on_zero(self) -> None:
        self.sim.start_p2p(np.zeros(6), self.move_time(), kind="p2p", src="zero",
                           label="zero")
        self.sync_scales(np.zeros(6))
        self.sim.log(L("log_zero"))

    def on_home(self) -> None:
        self.sim.home(self.move_time())
        self.sync_scales(spec.HOME_QPOS)

    # ------------------------------------------------------------ 点到点
    def _build_p2p(self) -> None:
        grp = ttk.LabelFrame(self, text=L("p2p_grp"), padding=4)
        grp.pack(fill="x", pady=4)
        ttk.Label(grp, text=L("p2p_src")).grid(row=0, column=0, sticky="w")
        self.p2p_src = tk.StringVar(value="slider")
        for c, (key, txt) in enumerate((("slider", L("p2p_src_slider")),
                                        ("actual", L("p2p_src_actual")))):
            ttk.Radiobutton(grp, text=txt, value=key, variable=self.p2p_src).grid(
                row=0, column=1 + c, sticky="w")
        ttk.Radiobutton(grp, text=L("p2p_src_preset"), value="preset",
                        variable=self.p2p_src, command=self._rebuild_preset).grid(
            row=0, column=3, sticky="w")
        self.preset_var = tk.StringVar(value=PRESETS[1][0])
        self.preset_box = ttk.Combobox(grp, textvariable=self.preset_var, width=8,
                                       state="readonly",
                                       values=[p[0] for p in PRESETS])
        self.preset_box.grid(row=0, column=4, padx=(3, 0))
        self.preset_box.bind("<<ComboboxSelected>>", lambda e: self.on_preset())

        row = ttk.Frame(grp)
        row.grid(row=1, column=0, columnspan=5, sticky="we", pady=(4, 2))
        ttk.Button(row, text=L("p2p_start"), width=12,
                   command=self.on_p2p_start).pack(side="left")
        ttk.Button(row, text=L("p2p_stop"), width=15,
                   command=self.on_estop).pack(side="left", padx=3)
        self.p2p_label = ttk.Label(grp, text="", wraplength=300, justify="left",
                                   foreground="#555")
        self.p2p_label.grid(row=2, column=0, columnspan=5, sticky="w")

    def _update_target_label(self) -> None:
        t = np.round(np.asarray(self.target, dtype=float), 1).tolist()
        self.p2p_label.configure(text=L("p2p_target", t=t))

    def _rebuild_preset(self) -> None:
        self.on_preset()

    def on_preset(self) -> None:
        """选预设：先解 IK（不动），把解填进滑条/目标，供 P2P 或 IK 使用。"""
        name = self.preset_var.get()
        pos, axis = next(((p[1], p[2]) for p in PRESETS if p[0] == name), PRESETS[0][1:])
        q, ep, ea = self.sim.solve_ik(pos, axis)
        self.p2p_q = q
        self.sim.log(L("log_preset", n=name, p=list(np.round(np.asarray(pos), 3))))
        self.sync_scales(q)
        self._set_entries(self.ik_pos, pos)
        self._set_entries(self.ik_axis, axis)
        self.sim.last_ik = dict(q=q.copy(), pos_err_m=ep, axis_err_rad=ea, ms=0.0)
        self._show_ik_result()

    def move_time(self) -> float:
        try:
            return max(float(self.dur_var.get()), 0.2)
        except (tk.TclError, ValueError):
            return self.sim.move_seconds

    def on_p2p_start(self) -> None:
        src = self.p2p_src.get()
        if src == "actual":
            q = self.sim.q_pos()
        elif src == "preset" and self.p2p_q is not None:
            q = self.p2p_q
        else:
            q = np.radians(np.asarray(self.target, dtype=float))
        self.sim.start_p2p(q, self.move_time(), kind="p2p", src=src, label="P2P")
        self.sync_scales(q)

    # ------------------------------------------------------------ IK
    def _build_ik(self) -> None:
        grp = ttk.LabelFrame(self, text=L("ik_grp"), padding=4)
        grp.pack(fill="x", pady=4)
        self.ik_pos: list[ttk.Entry] = []
        self.ik_axis: list[ttk.Entry] = []
        for i, (nm, val) in enumerate(zip("xyz", spec.TARGET_POSE)):
            ttk.Label(grp, text=nm, width=2).grid(row=i, column=0, sticky="w")
            e = ttk.Entry(grp, width=8, justify="right")
            e.insert(0, f"{val:.3f}")
            e.grid(row=i, column=1, sticky="w", padx=1)
            ttk.Button(grp, text="−", width=2, command=lambda i=i: self.step_pos(i, -0.05)
                       ).grid(row=i, column=2)
            ttk.Button(grp, text="+", width=2, command=lambda i=i: self.step_pos(i, +0.05)
                       ).grid(row=i, column=3)
            self.ik_pos.append(e)

        ttk.Label(grp, text=L("ik_axis")).grid(row=3, column=0, sticky="w")
        axis_row = ttk.Frame(grp)
        axis_row.grid(row=3, column=1, columnspan=2, sticky="w", pady=(3, 0))
        for val in (0.0, 0.0, -1.0):
            e = ttk.Entry(axis_row, width=6, justify="right")
            e.insert(0, f"{val:.3f}")
            e.pack(side="left", padx=1)
            self.ik_axis.append(e)
        self.axis_free = tk.BooleanVar(value=False)
        ttk.Checkbutton(grp, text=L("ik_axis_free"), variable=self.axis_free
                        ).grid(row=3, column=3, columnspan=2, sticky="w", pady=(3, 0))

        row = ttk.Frame(grp)
        row.grid(row=4, column=0, columnspan=5, sticky="we", pady=(5, 2))
        ttk.Button(row, text=L("ik_read"), width=13,
                   command=self.on_ik_read).pack(side="left")
        ttk.Button(row, text=L("ik_solve"), width=9,
                   command=lambda: self.on_ik(False)).pack(side="left", padx=3)
        ttk.Button(row, text=L("ik_solve_move"), width=12,
                   command=lambda: self.on_ik(True)).pack(side="left")

        row2 = ttk.Frame(grp)
        row2.grid(row=5, column=0, columnspan=5, sticky="we")
        for name, _pos, _ax in PRESETS:
            ttk.Button(row2, text=name, width=8,
                       command=lambda n=name: self.quick_preset(n)).pack(side="left", padx=1)

        self.ik_label = ttk.Label(grp, text="", wraplength=300, justify="left",
                                  foreground="#555")
        self.ik_label.grid(row=6, column=0, columnspan=5, sticky="w", pady=(3, 0))

    def _set_entries(self, entries: list, vals) -> None:
        vals = np.asarray(vals, dtype=float).ravel()
        for e, v in zip(entries, vals):
            e.delete(0, "end")
            e.insert(0, f"{float(v):.3f}")
            e.configure(foreground="")

    def _read_entries(self, entries: list):
        """读三个输入框；有非法数字就标红 + 记一条日志，返回 None。"""
        out = []
        for e in entries:
            try:
                out.append(float(e.get()))
                e.configure(foreground="")
            except ValueError:
                e.configure(foreground="red")
                self.sim.log(L("log_bad_num", v=e.get()))
                return None
        return out

    def step_pos(self, i: int, delta: float) -> None:
        """±0.05 m 步进（并夹到工作空间内，别把目标设到地板下/身后）。"""
        try:
            v = float(self.ik_pos[i].get()) + delta
        except ValueError:
            v = float(spec.TARGET_POSE[i])
        lo = (0.05, -0.70, spec.FLOOR_Z + 0.05)[i]
        hi = (0.95, 0.70, 1.20)[i]
        self._set_entries([self.ik_pos[i]], [float(np.clip(v, lo, hi))])

    def _ik_inputs(self):
        pos = self._read_entries(self.ik_pos)
        if pos is None:
            return None, None
        axis = None
        if not self.axis_free.get():
            raw = self._read_entries(self.ik_axis)
            if raw is None:
                return None, None
            if any(raw):
                axis = raw
        return np.asarray(pos, dtype=float), (None if axis is None else np.asarray(axis, dtype=float))

    def on_ik_read(self) -> None:
        p = self.sim.tip()
        self._set_entries(self.ik_pos, p)
        self.sim.log(L("log_read", p=np.round(p, 3).tolist()))

    def quick_preset(self, name: str) -> None:
        self.preset_var.set(name)
        self.on_preset()

    def on_ik(self, execute: bool) -> None:
        pos, axis = self._ik_inputs()
        if pos is None:
            return
        q, ep, ea = self.sim.start_ik(pos, axis, self.move_time(), execute=execute)
        self._show_ik_result()
        self.p2p_q = q
        if execute and ep <= 5e-3:
            self.sync_scales(q)

    def _show_ik_result(self) -> None:
        st = self.sim.last_ik
        if not st:
            return
        self.ik_label.configure(text=L("ik_result", ep=st["pos_err_m"] * 1000.0,
                                       ea=np.degrees(st["axis_err_rad"]), ms=st["ms"]))

    # ------------------------------------------------------------ 伺服 / 运行
    def _build_run(self) -> None:
        grp = ttk.LabelFrame(self, text=L("srv_grp"), padding=4)
        grp.pack(fill="x", pady=4)
        ttk.Label(grp, text=L("srv_dur")).grid(row=0, column=0, sticky="w")
        self.dur_var = tk.StringVar(value=f"{self.sim.move_seconds:.1f}")
        ttk.Entry(grp, textvariable=self.dur_var, width=6, justify="right").grid(
            row=0, column=1, sticky="w", padx=3)

        ttk.Label(grp, text=L("srv_kp")).grid(row=1, column=0, sticky="w")
        self.kp_scale = tk.Scale(grp, orient="horizontal", from_=0.1, to=2.0,
                                 resolution=0.05, length=120, showvalue=0, width=11,
                                 command=self.on_kp)
        self.kp_scale.set(1.0)
        self.kp_scale.grid(row=1, column=1, sticky="we")
        self.kp_label = ttk.Label(grp, text="1.00", width=5, anchor="e")
        self.kp_label.grid(row=1, column=2, sticky="w")

        box = ttk.Frame(grp)
        box.grid(row=2, column=0, columnspan=3, sticky="we", pady=(3, 2))
        self.pause_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text=L("srv_pause"), variable=self.pause_var,
                        command=self.on_pause).pack(side="left")
        self.grav_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text=L("srv_gravity"), variable=self.grav_var,
                        command=self.on_gravity).pack(side="left", padx=6)
        self.track_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text=L("srv_track"), variable=self.track_var).pack(side="left")

        bar = ttk.Frame(grp)
        bar.grid(row=3, column=0, columnspan=3, sticky="we")
        ttk.Button(bar, text=L("srv_estop"), width=17,
                   command=self.on_estop).pack(side="left")
        ttk.Button(bar, text=L("srv_reset"), width=12,
                   command=self.on_reset).pack(side="left", padx=3)

    def on_kp(self, value: str) -> None:
        v = float(value)
        self.sim.set_kp_scale(v)
        self.kp_label.configure(text=f"{v:.2f}")
        if self._throttled("kp"):
            self.sim.log(L("log_kp", s=v))

    def on_pause(self) -> None:
        self.app.paused = bool(self.pause_var.get())
        self.sim.log(L("log_pause", v=L("on") if self.app.paused else L("off")))

    def on_gravity(self) -> None:
        on = bool(self.grav_var.get())
        self.sim.set_gravity(on)
        self.sim.log(L("log_gravity", v=L("on") if on else L("off")))

    def on_estop(self) -> None:
        self.sim.hold()
        self.sync_scales(self.sim.cmd)

    def on_reset(self) -> None:
        self.sim.reset_model()
        self.app.cam.reset()
        self.sync_scales(self.sim.cmd)
        self.p2p_q = None
        self._set_entries(self.ik_pos, spec.TARGET_POSE)
        self._set_entries(self.ik_axis, (0.0, 0.0, -1.0))
        self.sim.ik_target = None

    # ------------------------------------------------------------ 日志
    def _build_log(self) -> None:
        grp = ttk.LabelFrame(self, text=L("log_grp"), padding=4)
        grp.pack(fill="both", expand=True, pady=4)
        wrap = ttk.Frame(grp)
        wrap.pack(fill="both", expand=True)
        self.log_text = tk.Text(wrap, height=9, width=46, wrap="word",
                                state="disabled", background="#fbfbfb",
                                relief="flat", font=("TkFixedFont", 9))
        self.log_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.log_text.yview)
        sb.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=sb.set)
        ttk.Button(grp, text=L("log_clear"), width=8,
                   command=self.on_clear_log).pack(anchor="e", pady=(3, 0))

    def append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        print(f"[gui] {line}")

    def on_clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------ 定时刷新
    def refresh_from_sim(self) -> None:
        """由主循环按 ~10 Hz 调用：把滑条/标签跟仿真状态对齐。"""
        st = self.sim.status()
        mode = self.mode_var.get()
        if mode == "joint" or (st["moving"] and self.track_var.get()):
            cur = np.degrees(np.asarray(self.sim.cmd, dtype=float).ravel())
            self._sync = True
            try:
                for i, sc in enumerate(self.joint_scale):
                    if abs(float(sc.get()) - cur[i]) > 0.05:
                        sc.set(cur[i])
                    self.joint_val[i].configure(text=f"{cur[i]:+.1f}")
            finally:
                self._sync = False
            if mode == "joint":
                self.target = cur.copy()
                self._update_target_label()


# =============================================================== 窗口 + 主循环
class App:
    """窗口：左边 MuJoCo 画面（离屏渲染贴到 Canvas），右边控制面板，底下一行状态栏。"""

    def __init__(self, root, sim: "ArmSim", *, width: int = 640, height: int = 480,
                 fps: float = 25.0, render: bool = True, exit_after: float = 0.0):
        self.root = root
        self.sim = sim
        self.fps = float(max(fps, 1.0))
        self.cam = Camera()
        self.paused = False
        self.closed = False
        self.renderer = None
        self.frame = None                      # 最近一帧 RGB（自检/存图用）
        self._photo = None
        self._img_id = None
        self._rend_size = (int(width), int(height))
        self._view_size: tuple[int, int] | None = None
        self._drag: tuple[int, float, float] | None = None
        self._last_key: dict[str, float] = {}
        self.r_frames = 0
        self.r_fps = 0.0
        self.r_ms = 0.0
        self.acc = 0.0
        self.t_last = time.perf_counter()
        self.t_fps0 = self.t_last
        self.t_render = 0.0
        self.t_ui = 0.0

        root.title(L("win_title"))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        style = ttk.Style()
        try:
            style.configure("Sel.TLabel", foreground="#0a7", font=("TkDefaultFont", 9, "bold"))
        except tk.TclError:
            pass

        self.canvas = tk.Canvas(root, width=self._rend_size[0], height=self._rend_size[1],
                                background="#101418", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.panel = ControlPanel(root, sim, self)
        self.panel.grid(row=0, column=1, sticky="ns")
        self.status = ttk.Label(root, anchor="w", padding=(6, 3))
        self.status.grid(row=1, column=0, columnspan=2, sticky="we")

        self._bind_events()
        if render:
            # 渲染分辨率固定成 --width/--height（只建这一次渲染器）；
            # 窗口更大/更小时，画面按等比缩放居中显示，不重建 GL 上下文。
            self._init_renderer(width, height)
        else:
            self._canvas_text(L("canvas_no_gl"))
        self._update_status()
        self._loop()
        if exit_after > 0:
            root.after(int(exit_after * 1000), self.close)

    # ------------------------------------------------------------ 渲染
    def _init_renderer(self, w: int | None = None, h: int | None = None) -> bool:
        """建离屏渲染器（只建一次！）。

        ⚠️ 实测：`mujoco.Renderer` 反复创建而不 `close()` 会把 GL 上下文耗光，
        最终 **段错误**（本机软件 GL 下 12 次左右就崩）。所以这里：
        先 close 旧的；窗口缩放只重新缩放画面，绝不重建渲染器。
        """
        w = int(w or self._rend_size[0])
        h = int(h or self._rend_size[1])
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self.renderer = None
        try:
            self.renderer = mujoco.Renderer(self.sim.model, h, w)
        except Exception as e:  # noqa: BLE001
            self.renderer = None
            self.sim.log(L("log_no_gl", e=e))
            self._canvas_text(L("canvas_no_gl"))
            return False
        self._rend_size = (w, h)
        self._view_size = (w, h)
        self._img_id = None
        self._photo = None
        self.canvas.delete("all")
        return True

    def _canvas_text(self, msg: str) -> None:
        self.canvas.delete("all")
        self._img_id = None
        w, h = self._rend_size
        self.canvas.create_text(w // 2, h // 2, text=msg, fill="#93a6b8",
                                justify="center", font=("TkDefaultFont", 11))

    def _add_markers(self) -> None:
        """把 IK 目标点画出来：绿球 = 目标位置，黄色胶囊 = 期望工具轴。"""
        if self.renderer is None or self.sim.ik_target is None:
            return
        pos, axis = self.sim.ik_target
        pos = np.asarray(pos, dtype=float).ravel()
        scn = self.renderer.scene
        add_marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.022, 0.0, 0.0), pos,
                   np.eye(3).reshape(-1), (0.15, 0.95, 0.35, 0.75))
        if axis is not None and np.any(axis):
            a = np.asarray(axis, dtype=float).ravel()
            a = a / max(float(np.linalg.norm(a)), 1e-12)
            add_marker(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, (0.007, 0.11, 0.0),
                       pos + a * 0.11, frame_from_z(a).reshape(-1),
                       (0.98, 0.78, 0.15, 0.95))

    def _render(self) -> None:
        if self.renderer is None:
            return
        t0 = time.perf_counter()
        self.renderer.update_scene(self.sim.data, camera=self.cam.mjv)
        self._add_markers()
        arr = self.renderer.render()
        self.frame = arr
        img = Image.fromarray(arr)
        if self._view_size and self._view_size != img.size:
            img = fit_image(img, self._view_size)     # 只缩放显示，不重建 GL
        self._photo = ImageTk.PhotoImage(img)
        if self._img_id is None:
            self.canvas.delete("all")
            self._img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfigure(self._img_id, image=self._photo)
        self.r_ms = 0.9 * self.r_ms + 0.1 * ((time.perf_counter() - t0) * 1000.0)

    # ------------------------------------------------------------ 事件
    def _bind_events(self) -> None:
        c = self.canvas
        c.bind("<ButtonPress-1>", lambda e: self._drag_start(e, 1))
        c.bind("<B1-Motion>", self._drag_move)
        c.bind("<ButtonPress-3>", lambda e: self._drag_start(e, 3))
        c.bind("<B3-Motion>", self._drag_move)
        c.bind("<Double-Button-1>", lambda e: self.cam.reset())
        c.bind("<Button-4>", lambda e: self.cam.zoom(1.0))
        c.bind("<Button-5>", lambda e: self.cam.zoom(-1.0))
        c.bind("<MouseWheel>", lambda e: self.cam.zoom(1.0 if e.delta > 0 else -1.0))
        c.bind("<Configure>", self._on_resize)
        self.root.bind("<Key>", self.on_key)

    def _drag_start(self, event, button: int) -> None:
        self._drag = (button, float(event.x), float(event.y))

    def _drag_move(self, event) -> None:
        if self._drag is None:
            return
        b, x0, y0 = self._drag
        dx, dy = float(event.x) - x0, float(event.y) - y0
        self._drag = (b, float(event.x), float(event.y))
        if b == 1:
            self.cam.orbit(dx, dy)
        else:
            self.cam.pan(dx, dy)

    def _on_resize(self, event) -> None:
        """窗口尺寸变了：只记住新的显示尺寸，下一帧按它缩放（绝不重建 GL 上下文）。"""
        if self.renderer is None:
            return
        w, h = int(event.width), int(event.height)
        if w < 80 or h < 60:
            return
        self._view_size = (w, h)

    def on_key(self, event) -> None:
        """键盘快捷键。

        ⚠️ 两个坑（都踩过）：
        1. ``event.char`` 对 Shift/F1/输入法等功能键是空串，而 Python 里
           ``"" in "hH"`` 竟然是 **True** —— 所以判断单字符必须用元组/集合，
           并且空 char 直接忽略（否则随便按个没字符的键就会"回 home"）；
        2. 按住键的自动重复会疯狂触发动作，所以一次性动作用 0.35 s 去抖。
        """
        ch = event.char or ""
        if event.keysym == "Escape":
            self.close()
            return
        if ch == "":
            return
        if ch in {"h", "H", "r", "R", "g", "G", "p", "P", "i", "I", " "}:
            now = time.perf_counter()
            if now - self._last_key.get(ch, 0.0) < 0.35:
                return
            self._last_key[ch] = now
        if ch == " ":
            self.panel.pause_var.set(not self.panel.pause_var.get())
            self.panel.on_pause()
        elif ch in ("h", "H"):
            self.panel.on_home()
        elif ch in ("r", "R"):
            self.panel.on_reset()
        elif ch in ("g", "G"):
            self.panel.grav_var.set(not self.panel.grav_var.get())
            self.panel.on_gravity()
        elif ch in ("1", "2", "3", "4", "5", "6"):
            self.panel.select_joint(int(ch) - 1)
        elif ch == "[":
            self.panel.nudge(self.panel.selected, -0.5)
        elif ch == "]":
            self.panel.nudge(self.panel.selected, +0.5)
        elif ch in ("p", "P"):
            self.panel.on_p2p_start()
        elif ch in ("i", "I"):
            self.panel.on_ik(True)

    # ------------------------------------------------------------ 主循环
    def _loop(self) -> None:
        if self.closed:
            return
        t0 = time.perf_counter()
        dt_wall = min(t0 - self.t_last, 0.25)
        self.t_last = t0
        if not self.paused:
            self.acc += dt_wall
            n = 0
            while self.acc >= self.sim.dt and n < 80:
                self.sim.step()
                self.acc -= self.sim.dt
                n += 1
            if n >= 80:
                self.acc = 0.0                    # 追不上实时就丢掉欠账，避免雪崩
        for line in self.sim.drain():
            self.panel.append_log(line)
        if self.renderer is not None and (t0 - self.t_render) >= 1.0 / self.fps:
            self._render()
            self.t_render = time.perf_counter()
            self.r_frames += 1
        if t0 - self.t_ui >= 0.25:
            self.t_ui = t0
            self.panel.refresh_from_sim()
            self._update_status()
            self.r_fps = self.r_frames / max(t0 - self.t_fps0, 1e-6)
            self.r_frames, self.t_fps0 = 0, t0
        self.root.after(max(int(1000.0 / self.fps * 0.4), 8), self._loop)

    def _update_status(self) -> None:
        st = self.sim.status()
        state = L("st_moving") if st["moving"] else L("st_idle")
        if self.paused:
            state = L("st_paused")
        parts = [L("st_mode", m=L("mode_" + self.panel.mode_var.get())), state,
                 L("st_tool", p=np.round(st["tip"], 3).tolist(), h=st["height"]),
                 L("st_axis", a=np.round(st["axis"], 2).tolist()),
                 L("st_track", e=st["track_deg"]),
                 L("st_contact", n=st["contacts"]),
                 L("st_fps", fps=self.r_fps, ms=self.r_ms)]
        self.status.configure(text="   |   ".join(parts))

    # ------------------------------------------------------------ 工具方法
    def pump(self, seconds: float, dt: float = 0.005) -> None:
        """无 mainloop 地推进 Tk 事件循环若干秒（--ui-test / 自检用）。"""
        t_end = time.perf_counter() + float(seconds)
        while time.perf_counter() < t_end and not self.closed:
            try:
                self.root.update()
            except tk.TclError:
                break
            time.sleep(dt)

    def save_frame(self, path) -> bool:
        """把最近渲染的一帧存成 PNG（没有渲染器时返回 False）。"""
        if self.frame is None:
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.frame).save(path)
        self.sim.log(L("log_frame", p=str(path)))
        return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.renderer is not None:
            try:
                self.renderer.close()      # 必须显式释放 GL 上下文
            except Exception:  # noqa: BLE001
                pass
            self.renderer = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# =============================================================== 入口 / 自检
def _fmt_mm(x: float) -> str:
    return f"{float(x) * 1000.0:.2f} mm"


def _fmt_deg(x: float) -> str:
    return f"{float(x):.3f}°"


def run_selftest(args) -> int:
    """无窗口自检：三种控制模式 + 渲染通路 + 相机/标记几何。"""
    print("=" * 74)
    print("interactive_control.py 自检（不需要显示器；只验证控制与渲染逻辑）")
    print("=" * 74)
    fails: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    sim = ArmSim(args.scene, move_seconds=args.move_time)

    def run_until_idle(limit_s: float = 10.0) -> int:
        n, lim = 0, int(limit_s / sim.dt)
        while (sim.moving() or sim.pending is not None) and n < lim:
            sim.step()
            n += 1
        return n

    # 1) home 姿态
    err0 = float(np.linalg.norm(sim.tip() - np.asarray(spec.TARGET_POSE)))
    check("home 姿态：工具尖 = spec.TARGET_POSE", err0 < 1e-3,
          f"{_fmt_mm(err0)}  接触 {sim.data.ncon} 对")
    check("home 姿态：无自碰撞", int(sim.data.ncon) == 0)

    # 2) 单关节
    sim.set_joint(3, np.radians(25.0))
    for _ in range(int(0.6 / sim.dt)):
        sim.step()
    j4 = float(np.degrees(sim.q_pos()[3]))
    check("单关节：J4 拖到 +25°", abs(j4 - 25.0) < 0.8, f"实测 {j4:+.2f}°")
    other = float(np.degrees(np.abs(sim.q_pos() - np.array(spec.HOME_QPOS)))[[0, 1, 2]].max())
    check("单关节：其余关节基本不动", other < 0.8, f"最大变化 {other:.3f}°")
    sim.reset_model(log_it=False)

    # 3) 点到点（目标 = 预设 forward 的 IK 解）
    pos, axis = PRESETS[1][1], PRESETS[1][2]
    q_ik, ep_ik, ea_ik = sim.solve_ik(pos, axis)
    check("IK 求解（preset forward）", ep_ik < 5e-4 and np.degrees(ea_ik) < 0.1,
          f"{_fmt_mm(ep_ik)} / {_fmt_deg(np.degrees(ea_ik))}")
    sim.start_p2p(q_ik, args.move_time, kind="p2p", src="slider", pos=pos, axis=axis)
    n = run_until_idle()
    rep = sim.last_report or {}
    check("点到点：关节到位", rep.get("joint_err_deg", 9e9) < 0.8,
          f"最大关节误差 {rep.get('joint_err_deg', float('nan')):.3f}°（{n} 步）")
    check("点到点：工具尖到位", rep.get("pos_err_m", 9e9) < 10e-3,
          f"{_fmt_mm(rep.get('pos_err_m', float('nan')))}")

    # 4) IK 执行
    tgt = np.array([0.52, 0.18, 0.34])
    ax = np.array([0.0, 0.0, -1.0])
    q, ep, ea = sim.start_ik(tgt, ax, args.move_time)
    check("IK：求解精度", ep < 5e-4 and np.degrees(ea) < 0.1,
          f"{_fmt_mm(ep)} / {_fmt_deg(np.degrees(ea))}")
    run_until_idle()
    rep = sim.last_report or {}
    check("IK：运动后实际到位", rep.get("pos_err_m", 9e9) < 15e-3,
          f"工具尖 {np.round(rep.get('tip', np.zeros(3)), 3).tolist()} "
          f"误差 {_fmt_mm(rep.get('pos_err_m', float('nan')))}")
    check("IK：工具轴保持朝下", (rep.get("axis_err_deg") or 9e9) < 1.5,
          f"{_fmt_deg(rep.get('axis_err_deg', float('nan')))}")

    # 5) 急停：运动中按下 -> 先靠伺服减速停住，然后原地撑住（不再走）
    sim.start_p2p(np.radians([0.0, -60.0, -60.0, -10.0, 60.0, -30.0]), 1.5)
    for _ in range(int(0.3 / sim.dt)):
        sim.step()
    q_cmd_before = sim.cmd.copy()
    sim.hold()
    q_cmd_hold = sim.cmd.copy()          # 急停后指令应定格在这
    p_hold = sim.tip().copy()
    for _ in range(int(0.4 / sim.dt)):
        sim.step()
    travel = float(np.linalg.norm(sim.tip() - p_hold))
    p_settle = sim.tip().copy()
    for _ in range(int(0.5 / sim.dt)):
        sim.step()
    drift = float(np.linalg.norm(sim.tip() - p_settle))
    stopped = float(np.abs(sim.cmd - q_cmd_hold).max()) < 1e-12
    check("急停：目标指令定格不再前进", stopped,
          f"急停瞬间目标 {np.round(np.degrees(q_cmd_before), 2).tolist()} -> "
          f"{np.round(np.degrees(q_cmd_hold), 2).tolist()}")
    check("急停：停下后 0.5 s 不漂移", drift < 1.5e-3,
          f"漂移 {_fmt_mm(drift)}（停下前的减速行程 {_fmt_mm(travel)}，"
          f"残余速度 {float(np.linalg.norm(sim.data.qvel)):.4f}）")

    # 6) 渲染通路 + 目标标记
    try:
        r = mujoco.Renderer(sim.model, 240, 320)
        r.update_scene(sim.data)
        img0 = r.render().copy()
        n0 = int(r.scene.ngeom)
        r.update_scene(sim.data)
        add_marker(r.scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.022, 0.0, 0.0), tgt,
                   np.eye(3).reshape(-1), (0.15, 0.95, 0.35, 0.75))
        add_marker(r.scene, mujoco.mjtGeom.mjGEOM_CAPSULE, (0.007, 0.11, 0.0),
                   tgt - np.array([0.0, 0.0, 0.11]), frame_from_z((0, 0, 1)).reshape(-1),
                   (0.98, 0.78, 0.15, 0.95))
        img = r.render()
        check("离屏渲染出图", img.shape == (240, 320, 3) and float(img.mean()) > 3.0,
              f"shape={img.shape} 平均亮度={float(img.mean()):.1f}")
        check("画面标记（目标球 + 轴胶囊）", int(r.scene.ngeom) == n0 + 2,
              f"ngeom {n0} -> {int(r.scene.ngeom)}")
        dmask = np.abs(img.astype(int) - img0.astype(int)).sum(axis=2) > 12
        px = int(dmask.sum())
        rgb = img[dmask].mean(axis=0) if px else np.zeros(3)
        check("标记在画面里真的看得见", px > 60 and float(rgb[1]) > float(rgb[2]) + 20.0,
              f"变化像素 {px}，平均 RGB {np.round(rgb, 1).tolist()}（G 应最高）")
        out = HERE / "runs" / "gui_selftest.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out)
        print(f"       存图：{out}")
    except Exception as e:  # noqa: BLE001
        check("离屏渲染出图", False, f"异常 {e!r}")

    # 7) 相机与基向量
    cam = Camera()
    a0 = float(cam.mjv.azimuth)
    cam.orbit(10.0, 5.0)
    check("相机：拖动改变方位/俯仰", abs(float(cam.mjv.azimuth) - a0) > 1.0)
    lk = np.array(cam.mjv.lookat).copy()
    cam.pan(20.0, -10.0)
    check("相机：平移改变 lookat",
          float(np.linalg.norm(np.array(cam.mjv.lookat) - lk)) > 1e-4)
    d0 = float(cam.mjv.distance)
    cam.zoom(3.0)
    cam.zoom(-3.0)
    check("相机：推拉可逆", abs(float(cam.mjv.distance) - d0) < 1e-6)
    for z in ((0, 0, -1), (1, 0, 0), (0.3, -0.7, 0.5)):
        m = frame_from_z(z)
        zz = np.asarray(z, dtype=float) / np.linalg.norm(z)
        check(f"工具轴基向量 z={z}", bool(np.allclose(m.T @ m, np.eye(3), atol=1e-9)
                                          and np.allclose(m[:, 2], zz)))

    print("-" * 74)
    if fails:
        print(f"结论：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结论：全部通过 [OK]")
    return 0


def run_ui_test(args) -> int:
    """真建窗口，脚本化地点一遍所有控件，最后存一帧图（需要 DISPLAY）。"""
    if tk is None:
        print(f"需要 tkinter（{_TK_IMPORT_ERROR}）")
        return 1
    print("=" * 74)
    print("交互窗口 UI 自检：建窗口 + 脚本化操作三种模式 + 存图")
    print("=" * 74)
    fails: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    sim = ArmSim(args.scene, move_seconds=args.move_time)
    root = tk.Tk()
    app = App(root, sim, width=args.width, height=args.height, fps=15.0,
              render=not args.no_render)
    panel = app.panel

    def pump_until(cond, timeout: float) -> bool:
        """跑到条件成立或超时（比"睡固定秒数"稳，机器快慢都不怕）。"""
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            if cond():
                return True
            app.pump(0.05)
        return bool(cond())

    app.pump(0.5)
    bright = float(app.frame.mean()) if app.frame is not None else 0.0
    if args.no_render:
        check("窗口（--no-render：只验证控件）",
              root.winfo_exists() == 1 and app.renderer is None)
    else:
        check("窗口 & 画面", root.winfo_exists() == 1 and bright > 3.0,
              f"帧 shape={None if app.frame is None else app.frame.shape} 平均亮度={bright:.1f}")

    for k in ("joint", "p2p", "ik", "joint"):
        panel.set_mode(k, log_it=False)
        app.pump(0.15)
    check("三种模式都能切换", panel.mode_var.get() == "joint")

    panel.joint_scale[3].set(20.0)          # 拖滑条 = 单关节控制
    settled = pump_until(lambda: abs(float(np.degrees(sim.q_pos()[3])) - 20.0) < 1.0, 4.0)
    j4 = float(np.degrees(sim.q_pos()[3]))
    check("单关节滑条", settled, f"J4 实测 {j4:+.2f}°")

    panel.set_mode("p2p", log_it=False)
    panel.p2p_src.set("slider")
    panel.sync_scales(np.radians([10.0, -70.0, -110.0, -20.0, 80.0, -100.0]))
    panel.on_p2p_start()
    pump_until(lambda: not sim.moving() and sim.pending is None, 12.0)
    rep = sim.last_report or {}
    check("点到点按钮", rep.get("joint_err_deg", 9e9) < 1.0,
          f"最大关节误差 {rep.get('joint_err_deg', float('nan')):.3f}°")

    panel.set_mode("ik", log_it=False)
    tgt = np.array([0.50, 0.20, 0.30])
    panel._set_entries(panel.ik_pos, tgt)
    panel._set_entries(panel.ik_axis, (0.0, 0.0, -1.0))
    panel.on_ik(True)
    pump_until(lambda: not sim.moving() and sim.pending is None, 12.0)
    perr = float(np.linalg.norm(sim.tip() - tgt))
    check("IK 按钮（求解并运动）", perr < 15e-3,
          f"工具尖 {np.round(sim.tip(), 3).tolist()} 误差 {_fmt_mm(perr)}")

    panel.sync_scales(np.radians([25.0, -60.0, -100.0, -20.0, 70.0, -90.0]))
    panel.on_p2p_start()
    app.pump(0.6)
    panel.on_estop()
    app.pump(0.3)
    check("急停按钮", not sim.moving())

    panel.on_reset()
    app.pump(0.6)
    p0 = float(np.linalg.norm(sim.tip() - np.asarray(spec.TARGET_POSE)))
    check("复位按钮", p0 < 5e-3, f"回 home 误差 {_fmt_mm(p0)}")

    panel.on_ik_read()
    panel.step_pos(0, +0.05)
    app.pump(0.2)
    log_len = len(panel.log_text.get("1.0", "end").strip())
    check("日志面板有内容", log_len > 10, f"{log_len} 字符")
    panel.on_clear_log()
    check("清空日志按钮", panel.log_text.get("1.0", "end").strip() == "")
    panel.on_zero()
    app.pump(0.3)
    check("其它按钮不报错", root.winfo_exists() == 1)

    if args.no_render:
        print("  [SKIP] 渲染 / 存图 / 缩放相关检查（--no-render）")
    else:
        ok_img = app.save_frame(HERE / "runs" / "gui_ui_test.png")
        check("窗口画面存图", ok_img, str(HERE / "runs" / "gui_ui_test.png"))

        # 缩放窗口：只缩放显示，不重建 GL 上下文（重建会漏 GL 上下文 -> 段错误）
        rend_before = app.renderer
        ev = tk.Event()
        ev.width, ev.height = 500, 400
        app._on_resize(ev)
        app.pump(0.5)
        b2 = float(app.frame.mean()) if app.frame is not None else 0.0
        check("缩放窗口：不重建渲染器、画面仍正常",
              app.renderer is rend_before and b2 > 3.0, f"平均亮度 {b2:.1f}")
        app._view_size = (320, 240)             # 直接验证"缩放只作用于显示"
        app._render()
        ph = app._photo
        check("缩放只改显示、不改渲染分辨率",
              ph is not None and (ph.width(), ph.height()) == (320, 240)
              and app.frame.shape[:2] == (app._rend_size[1], app._rend_size[0]),
              f"显示 {None if ph is None else (ph.width(), ph.height())} "
              f"渲染 {app._rend_size[0]}x{app._rend_size[1]}")
        app._view_size = app._rend_size
        app._render()

    # 键盘：空 char 的功能键绝不能触发动作（"" in "hH" 是 True 的经典坑）
    q_before = sim.cmd.copy()
    for ks in ("Shift_L", "F1", "Caps_Lock", "Insert", "Super_L"):
        ev2 = tk.Event()
        ev2.keysym, ev2.char = ks, ""
        app.on_key(ev2)
    check("空 char 的功能键不触发动作", bool(np.allclose(sim.cmd, q_before)))
    ev3 = tk.Event()
    ev3.keysym, ev3.char = "j", "j"
    app.on_key(ev3)                          # 未绑定的键也不该炸
    ev4 = tk.Event()
    ev4.keysym, ev4.char = "h", "h"
    app.on_key(ev4)
    app.on_key(ev4)                          # 连按第二次要被去抖挡掉（只记一次 home）
    app.pump(0.3)
    homes = sum("P2P start: home" in ln for ln in
                panel.log_text.get("1.0", "end").splitlines())
    check("h 键回 home（且自动重复被去抖）", homes == 1, f"日志里 home 动作 {homes} 次")

    # 画面等比缩放：不能变形
    img = Image.new("RGB", (640, 480), (200, 30, 30))
    out = fit_image(img, (907, 640))
    check("等比缩放 + 居中（不变形）",
          out.size == (907, 640) and np.asarray(out)[5, 5].tolist() != [200, 30, 30],
          f"输出 {out.size}，上下留边（底色）")

    check("状态栏有内容", len(str(app.status.cget("text"))) > 20,
          str(app.status.cget("text"))[:70] + " ...")
    if args.no_render:
        print("  [SKIP] 渲染帧率检查（--no-render）")
    else:
        check("渲染 fps > 0", app.r_fps > 0.0,
              f"{app.r_fps:.1f} fps, 渲染 {app.r_ms:.0f} ms/帧")

    app.close()
    print("-" * 74)
    if fails:
        print(f"结论：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结论：全部通过 [OK]")
    return 0


def run_gui(args) -> int:
    if tk is None:
        print(f"需要 tkinter + Pillow（{_TK_IMPORT_ERROR}）。无窗口时可用 --selftest。")
        return 1
    sim = ArmSim(args.scene, kp_scale=args.kp_scale, gravity=not args.no_gravity,
                 move_seconds=args.move_time)
    root = tk.Tk()
    app = App(root, sim, width=args.width, height=args.height, fps=args.fps,
              render=not args.no_render, exit_after=args.exit_after)
    print(f"关节顺序 : {spec.JOINTS}")
    print(f"home qpos: {np.round(spec.HOME_QPOS, 4)}")
    print(f"语言     : {_LANG}（无中文字体时自动用英文；--lang zh/en 可强制）")
    print("鼠标     : 左键拖动=转视角  右键拖动=平移  滚轮=推拉  双击=复位视角")
    print("键盘     : 空格=暂停 h=home r=复位 g=重力 1..6=选关节 [ ]=±0.5° p=P2P i=IK ESC=退出")
    print("控制模式 : ① 单关节滑条（拖动即动） ② 点到点（滑条当目标编辑器）"
          " ③ IK 笛卡尔（解完再走 P2P）")
    app.panel.append_log("ready. mode=joint")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TB6-R5-RevA1 交互控制台（tkinter + MuJoCo）")
    ap.add_argument("--scene", default=str(spec.SCENE_XML))
    ap.add_argument("--lang", choices=("auto", "en", "zh"), default="auto",
                    help="界面语言（auto = 看系统有没有中文字体）")
    ap.add_argument("--width", type=int, default=640, help="画面渲染宽度")
    ap.add_argument("--height", type=int, default=480, help="画面渲染高度")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="渲染帧率上限（软件 GL 慢就调小；物理始终按实时步进）")
    ap.add_argument("--kp-scale", type=float, default=1.0, help="初始 kp/kv 缩放")
    ap.add_argument("--no-gravity", action="store_true", help="关掉重力启动")
    ap.add_argument("--no-render", action="store_true", help="不建离屏渲染器（省 CPU）")
    ap.add_argument("--move-time", type=float, default=1.5, help="默认运动时长 [s]")
    ap.add_argument("--exit-after", type=float, default=0.0,
                    help="开窗口后自动退出 [s]（自动化/录屏用）")
    ap.add_argument("--selftest", action="store_true", help="无窗口自检（控制+渲染逻辑）")
    ap.add_argument("--ui-test", action="store_true", help="真建窗口并脚本化点一遍控件")
    args = ap.parse_args(argv)

    global _LANG
    _LANG = detect_lang() if args.lang == "auto" else args.lang
    if args.selftest:
        return run_selftest(args)
    if args.ui_test:
        return run_ui_test(args)
    return run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
