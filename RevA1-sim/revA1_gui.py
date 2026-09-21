#!/usr/bin/env python3
"""TB6-R5-RevA1 交互控制台（PySide6 界面）。

界面布局
--------
左边是 MuJoCo 实时画面（离屏渲染成 QImage 贴上去，不另开窗口），右边是可滚动的控制面板，
底下一行状态栏；面板分 7 张卡片：

1. **关节微调**：每个关节一行 ``− 目标角 + 实测角``——**点一下动一点**（按住还能连点）。
   旧版是 6 根滑条，拖起来又不准又不直观，这里直接换成 ± 按钮 + 步长选择。
2. **真机跟随**：接真机的姿态——真机在广播/开着状态接口，这里一按就把**真机姿态**搬到
   模型上实时跟着动（HTTP 状态接口 / UDP 广播两路，见 ``robot_link.py``）；
   卡片上同时显示延迟、包率、真机与仿真的**TCP 位置/工具轴误差**。
3. **点云（相机 → 机械臂）**：深度相机拍的点云，按**手眼标定 + 拍摄姿态**换算到机械臂基座系，
   生成"带点云 + 小车"的世界场景再整体加载（见 ``point_cloud.py``）——机械臂与点云同框、
   相对位置真实（深度 z 越小越靠近相机），地面按小车高度下移。
4. **末端目标 / IK**：填工具尖目标（世界系 x/y/z）+ 工具轴方向，``求解 IK`` 只算不动、
   ``求解并沿直线运动`` 算完就走。结果里写清楚位置误差 / 姿态误差 / 耗时 / 解出来的 6 个关节角，
   目标点在画面里画成**绿球 + 黄轴**（不可达时变红）。
5. **点位（示教 / 点到点）**：``记录当前位姿`` 存点，``走到选中点`` 沿直线过去（双击列表也行）。
   这是最直观的"点到点"：先手动摆到位置存下来，以后一键复现。
6. **运行 / 伺服**：运动时长、进度条、kp 缩放（体会伺服软硬）、暂停、重力、**急停**、复位、存图。
7. **日志**：每一步动作 + **实测**到位精度（不是只看指令）。

点到点为什么改成"直线"
----------------------
旧版是关节空间插值，工具尖走弧线，看着不直观；现在用 ``arm_core.ArmSim.plan_line``：
沿"当前工具尖 → 目标点"的直线每 1 cm 解一次 IK，再按 smoothstep 回放时间——
工具尖就是沿直线过去的（与真机 MoveL 语义一致），画面里还会画出这条**青色路径**。

IK 为什么以前"用不动"
--------------------
旧版是单初值阻尼最小二乘：偏置腕（J4/J6 轴平行但不交）在奇异/局部极小时会解不出来、或解到
"拧成麻花"的姿态，然后界面只是悄悄记一条日志，看起来就是"按了没反应"。现在：

* 换成**多初值**（当前姿态 / home / 两个肩肘大扰动，命中即停）——正常一次就中，卡住会自动换解支；
* 结果里明确给出"可解 / 不可解 + 原因（位置差多少 mm、姿态差多少度）"，界面上直接显示；
* 求解过程不动 qpos（算完恢复），只有真正"运动"时才下发指令。

运行
----
    python revA1_gui.py                     # 开窗口
    python revA1_gui.py --selftest          # 不开窗口：把控制逻辑全跑一遍并断言
    python revA1_gui.py --ui-test           # 真建窗口 → 脚本化点一遍控件 → 存图 → 退出
    python revA1_gui.py --exit-after 10     # 开窗口跑 10 秒自动退出（自动化/截图用）

鼠标（画面区域）：左键拖动=转视角，中键拖动=平移，滚轮=推拉，双击=视角复位。
键盘：空格=暂停，H=回 home，R=复位，G=重力，1..6=选关节，``-``/``=``=微调一个步长，
Ctrl+S=存图，ESC=退出。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl()
import mujoco  # noqa: E402  (必须在设置 MUJOCO_GL 之后)

from PySide6.QtCore import Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QImage,  # noqa: E402
                           QPainter, QPixmap)
from PySide6.QtTest import QTest  # noqa: E402  (UI 自检里模拟真实按键)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,  # noqa: E402
                               QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSplitter,
                               QVBoxLayout, QWidget)

import arm_core as core  # noqa: E402
import point_cloud as pc  # noqa: E402
import revA1_spec as spec  # noqa: E402
import robot_link as link  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
# 点云下拉框里代表「config.json 的多组点云（多位姿 + 多点云）」的哨兵值（不是真实文件路径）
MULTI_CLOUD = "__config_multi_cloud__"

# IK 快捷预设（与 viewer.py / demo_trajectory.py 同一批点）：工具尖目标（世界系）+ 工具轴
POSE_PRESETS = [
    ("home", (0.40, 0.00, 0.179), (0, 0, -1)),
    ("前伸", (0.55, 0.00, 0.420), (0, 0, -1)),
    ("侧向", (0.30, 0.30, 0.450), (0, 0, -1)),
    ("低位", (0.45, -0.15, 0.150), (0, 0, -1)),
]
AXIS_CHOICES = [
    ("保持当前朝向", None),
    ("竖直朝下 (0,0,-1)", (0.0, 0.0, -1.0)),
    ("水平朝前 (1,0,0)", (1.0, 0.0, 0.0)),
    ("水平朝左 (0,1,0)", (0.0, 1.0, 0.0)),
    ("自定义", "custom"),
]
JOINT_STEPS = ["0.5°", "1°", "5°", "15°"]
IK_STEPS = [("5 mm", 0.005), ("1 cm", 0.01), ("5 cm", 0.05)]
TICK_MS = 8                      # 主循环定时器周期
UI_REFRESH_S = 0.2               # 面板 / 状态栏刷新间隔

# 深色主题（QSS 集中在这一处，换配色只动这里）
QSS = """
QWidget { background: #12161c; color: #e7edf4; font-size: 13px; }
QFrame#Card { background: #1a2129; border: 1px solid #262f3a; border-radius: 10px; }
QLabel#CardTitle { color: #6fd3bd; font-size: 14px; font-weight: 600; }
QLabel#Hint { color: #8296a8; font-size: 12px; }
QLabel#Head { color: #7d8b9a; font-size: 12px; }
QLabel#Mono { color: #dbe6f1; font-family: "Consolas", "Cascadia Mono", monospace; }
QLabel#Value { color: #eaf3fb; font-family: "Consolas", "Cascadia Mono", monospace;
               font-size: 15px; font-weight: 600; }
QLabel#Tag { color: #cfe0ee; font-family: "Consolas", "Cascadia Mono", monospace;
             font-weight: 700; }
QLabel#Tag[sel="true"] { color: #37c9a8; }
QLabel#Result { color: #b9d3e6; font-size: 12px; }
QLabel#Result[fail="true"] { color: #ff9aa6; }
QPushButton { background: #232c37; border: 1px solid #313d4b; border-radius: 6px;
              padding: 5px 10px; }
QPushButton:hover { background: #2b3644; }
QPushButton:pressed { background: #1c242e; }
QPushButton:disabled { color: #5b6a79; background: #1b222b; border-color: #262f3a; }
QPushButton#Primary { background: #1f7f6e; border-color: #2ba48c; color: #ffffff;
                      font-weight: 600; }
QPushButton#Primary:hover { background: #24907c; }
QPushButton#Danger { background: #7d2f3a; border-color: #a23c4a; color: #ffe9ec;
                     font-weight: 600; }
QPushButton#Danger:hover { background: #8e3743; }
QPushButton#Step { min-width: 30px; max-width: 30px; padding: 2px 0; font-size: 15px;
                   font-weight: 700; }
QPushButton#Tiny { min-width: 24px; max-width: 24px; padding: 1px 0; }
QPushButton#Preset { padding: 4px 8px; font-size: 12px; }
QComboBox, QLineEdit { background: #101720; border: 1px solid #2c3743;
                       border-radius: 5px; padding: 3px 6px; }
QComboBox:focus, QLineEdit:focus { border-color: #2ba48c; }
QLabel#NumBox { background: #101720; border: 1px solid #2c3743; border-radius: 5px;
                padding: 3px 6px; color: #eaf3fb;
                font-family: "Consolas", "Cascadia Mono", monospace; }
QPushButton#NumBtn { min-width: 22px; max-width: 22px; padding: 1px 0; font-size: 13px;
                     font-weight: 700; border-radius: 5px; }
QComboBox::drop-down { border: 0; width: 16px; }
QComboBox QAbstractItemView { background: #161d26; border: 1px solid #2c3743;
                              selection-background-color: #1f7f6e; }
QCheckBox { spacing: 6px; }
QSlider::groove:horizontal { height: 4px; background: #2a333f; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #2ba48c; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; background: #d6e4f0;
                             border-radius: 6px; }
QPlainTextEdit#Log { background: #0e1319; border: 1px solid #232c36; border-radius: 8px;
                     color: #a8c1d6; font-family: "Consolas", "Cascadia Mono", monospace;
                     font-size: 12px; }
QListWidget { background: #101720; border: 1px solid #2c3743; border-radius: 8px; }
QListWidget::item { padding: 3px 4px; }
QListWidget::item:selected { background: #1f7f6e; color: #ffffff; }
QProgressBar { background: #101720; border: 1px solid #2c3743; border-radius: 5px;
               height: 8px; text-align: center; color: transparent; }
QProgressBar::chunk { background: #2ba48c; border-radius: 4px; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #33404e; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QStatusBar { background: #0e1319; color: #8ba0b4; }
QStatusBar::item { border: 0; }
QToolTip { background: #1a2129; color: #e7edf4; border: 1px solid #2c3743; }
"""


# =============================================================== 界面小工具
def pick_font() -> str:
    """挑一个能显示中文的界面字体（本机实测有 Microsoft YaHei UI；Linux 常见 Noto/思源）。"""
    cands = ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
             "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei", "SimSun", "DejaVu Sans")
    have = set(QFontDatabase.families())
    for name in cands:
        if name in have:
            return name
    return "Sans Serif"


def vec_str(v, nd: int = 3) -> str:
    """把向量印成 ``(0.400, 0.000, 0.179)`` 这样的一行（日志/列表都用它）。"""
    return "(" + ", ".join(f"{float(x):.{nd}f}" for x in np.asarray(v).ravel()) + ")"


def restyle(widget: QWidget) -> None:
    """改过 ``setProperty`` 之后要重新算一遍样式，QSS 才会生效（Qt 的老规矩）。"""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class RepeatButton(QPushButton):
    """按住会连发的按钮（关节 ± 微调用：短按 = 一个步长，长按 = 连续动）。

    实现：按下立刻发一次；按住 0.4 s 后每 0.06 s 再发一次；松开就停。
    """

    def __init__(self, text: str, callback, *, delay_ms: int = 400, period_ms: int = 60):
        super().__init__(text)
        self._cb = callback
        self._timer = QTimer(self)
        self._timer.setInterval(int(period_ms))
        self._timer.timeout.connect(self._fire)
        self._delay = QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.setInterval(int(delay_ms))
        self._delay.timeout.connect(self._timer.start)
        self.pressed.connect(self._on_press)
        self.released.connect(self._stop)

    def _on_press(self) -> None:
        self._fire()
        self._delay.start()

    def _stop(self) -> None:
        self._delay.stop()
        self._timer.stop()

    def _fire(self) -> None:
        self._cb()


class Card(QFrame):
    """一张卡片：标题 + 可选提示 + 内容区（面板上的每一块都是它）。"""

    def __init__(self, title: str, hint: str = ""):
        super().__init__()
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(12, 10, 12, 12)
        self.body.setSpacing(8)
        head = QLabel(title)
        head.setObjectName("CardTitle")
        self.body.addWidget(head)
        if hint:
            lab = QLabel(hint)
            lab.setObjectName("Hint")
            lab.setWordWrap(True)
            self.body.addWidget(lab)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget

    def add_row(self, *widgets) -> QHBoxLayout:
        """把若干控件横着排一行；传 ``None`` 表示那里放一个弹性空格。"""
        row = QHBoxLayout()
        row.setSpacing(6)
        for w in widgets:
            if w is None:
                row.addStretch(1)
            elif isinstance(w, QWidget):
                row.addWidget(w)
            else:                       # 已经是 layout
                row.addLayout(w)
        self.body.addLayout(row)
        return row


class NumBox(QWidget):
    """数字输入框：``[-] 数值 [+]``（替代 QSpinBox / QDoubleSpinBox）。

    * 上/下箭头改成 ``+`` / ``−`` 按钮（RepeatButton：按住连发）；
    * 数值是只读 QLabel，天然不响应鼠标滚轮 → 不会再被滚轮误改；
    * 保留了 spinbox 常用的 setter 链，替换时其它调用不用改。
    """

    valueChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._lo = -1e9
        self._hi = 1e9
        self._step = 1.0
        self._decimals = 0
        self._suffix = ""
        self._value = 0.0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._minus = RepeatButton("−", self._dec)
        self._plus = RepeatButton("+", self._inc)
        for b in (self._minus, self._plus):
            b.setObjectName("NumBtn")
            b.setFixedWidth(22)
            b.setFocusPolicy(Qt.NoFocus)
            b.setCursor(Qt.PointingHandCursor)
        self._label = QLabel()
        self._label.setObjectName("NumBox")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self._minus)
        lay.addWidget(self._label, 1)
        lay.addWidget(self._plus)
        self._refresh()

    # ---------------------------------------------------------- 兼容 spinbox 的 setter 链
    def setRange(self, lo: float, hi: float) -> "NumBox":
        self._lo, self._hi = float(lo), float(hi)
        self._clamp()
        return self

    def setSingleStep(self, step: float) -> "NumBox":
        self._step = float(step)
        return self

    def setDecimals(self, d: int) -> "NumBox":
        self._decimals = int(d)
        self._refresh()
        return self

    def setSuffix(self, s: str) -> "NumBox":
        self._suffix = str(s)
        self._refresh()
        return self

    def setValue(self, v: float) -> "NumBox":
        self._value = float(v)
        if self._decimals == 0:
            self._value = float(round(self._value))
        self._clamp()
        self._refresh()
        return self

    def value(self) -> float:
        return self._value

    # ---------------------------------------------------------- 内部
    def _inc(self) -> None:
        self._step_by(+1.0)

    def _dec(self) -> None:
        self._step_by(-1.0)

    def _step_by(self, sign: float) -> None:
        self._value = float(round(self._value + sign * self._step, self._decimals))
        self._clamp()
        self._refresh()
        self.valueChanged.emit(self._value)

    def _clamp(self) -> None:
        self._value = float(np.clip(self._value, self._lo, self._hi))

    def _refresh(self) -> None:
        if self._decimals == 0:
            txt = f"{int(self._value)}{self._suffix}"
        else:
            txt = f"{self._value:.{self._decimals}f}{self._suffix}"
        self._label.setText(txt)


# =============================================================== 画面区
class ArmView(QWidget):
    """MuJoCo 画面区：离屏渲染 → QImage → 等比缩放居中显示；鼠标直接操作相机。

    画面里会额外画三样东西（都是 ``mjv`` 几何，不参与物理）：

    * **当前工具尖**：青色小球（site 在 group 3，某些设置下不显示，干脆自己画一个）；
    * **直线路径**：起点到目标的**青色胶囊串**——一眼看出"要沿哪条直线过去"；
    * **目标位姿**：绿球（可达）/ 红球（不可达）+ 黄色胶囊（期望工具轴方向）。
    """

    def __init__(self, sim: core.ArmSim, *, width: int = 960, height: int = 600,
                 render: bool = True):
        super().__init__()
        self.sim = sim
        self.cam = core.Camera()
        self.render_enabled = bool(render)        # 换场景时按它决定要不要重建渲染器
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.renderer = None
        self.frame_rgb: np.ndarray | None = None      # 最近一帧（存图 / 自检用）
        self.fps = 0.0
        self.render_ms = 0.0
        self._rend = (int(width), int(height))
        self._image: QImage | None = None
        self._pix: QPixmap | None = None
        self._pix_sig: tuple | None = None
        self._drag: tuple[int, float, float] | None = None
        self.message = ""                              # 没有 GL 时显示的替代文字
        if render:
            self.init_renderer(width, height)

    # ------------------------------------------------------------ 渲染器
    def init_renderer(self, w: int | None = None, h: int | None = None) -> bool:
        """建离屏渲染器（**只建这一次**）。

        ⚠️ 踩过的坑：``mujoco.Renderer`` 反复创建而不 ``close()`` 会把 GL 上下文耗光，
        最终段错误（本机软件 GL 下 12 次左右就崩）。所以窗口缩放**绝不重建渲染器**：
        渲染分辨率固定成 ``--width/--height``，窗口变大变小只是把画面等比缩放居中。
        """
        w = int(w or self._rend[0])
        h = int(h or self._rend[1])
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
            self.message = f"离屏渲染器不可用：{e}\n控制面板依然可用"
            return False
        self._rend = (w, h)
        self.frame_rgb = None
        self._image = None
        self._pix = None
        self._pix_sig = None
        return True

    def close(self) -> None:
        """释放 GL 上下文（不释放会拖垮进程）。"""
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self.renderer = None

    def _add_markers(self) -> None:
        """往场景里塞"当前工具尖 / 直线路径 / 目标位姿"三类标记。"""
        if self.renderer is None:
            return
        scn = self.renderer.scene
        core.add_marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.009, 0.0, 0.0),
                        self.sim.tip(), np.eye(3).reshape(-1), (0.25, 0.95, 1.0, 0.9))
        pts = self.sim.path_points
        if pts is not None and len(pts) >= 2:
            pts = np.asarray(pts, dtype=float)
            step = max(int(len(pts) // 40), 1)
            for i in range(0, len(pts) - 1, step):
                a, b = pts[i], pts[i + 1]
                d = b - a
                seg = float(np.linalg.norm(d))
                if seg < 1e-6:
                    continue
                core.add_marker(scn, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                (0.0016, seg / 2 + 0.0016, 0.0), (a + b) / 2.0,
                                core.frame_from_z(d).reshape(-1), (0.25, 0.85, 1.0, 0.5))
        tgt = self.sim.ik_target
        if tgt is not None:
            pos, axis, ok = tgt
            pos = np.asarray(pos, dtype=float).ravel()
            col = (0.15, 0.95, 0.35, 0.85) if ok else (0.95, 0.25, 0.30, 0.9)
            core.add_marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.022, 0.0, 0.0), pos,
                            np.eye(3).reshape(-1), col)
            if axis is not None and np.any(axis):
                a = core.unit(axis)
                core.add_marker(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, (0.006, 0.11, 0.0),
                                pos + a * 0.11, core.frame_from_z(a).reshape(-1),
                                (0.98, 0.78, 0.15, 0.95))

    def render_frame(self) -> bool:
        """渲染一帧并刷新显示（返回是否成功）。"""
        if self.renderer is None:
            return False
        t0 = time.perf_counter()
        try:
            self.renderer.update_scene(self.sim.data, camera=self.cam.mjv)
            self._add_markers()
            self.sim.add_tool_arrows(self.renderer.scene)   # 三个工具 TCP 向量箭头（平行、随臂动）
            self.sim.draw_trails(self.renderer.scene)       # 工具尖轨迹（任务信号驱动）
            arr = self.renderer.render()
        except Exception as e:  # noqa: BLE001
            self.message = f"渲染失败：{e}"
            self.renderer = None
            self.update()
            return False
        self.frame_rgb = arr
        w, h = self._rend
        self._image = QImage(arr.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._pix = None
        self._pix_sig = None
        self.render_ms = 0.9 * self.render_ms + 0.1 * ((time.perf_counter() - t0) * 1000.0)
        self.update()
        return True

    # ------------------------------------------------------------ 画
    def paintEvent(self, ev) -> None:  # noqa: N802  (Qt 的命名习惯)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0f14"))
        if self._image is not None and not self._image.isNull():
            sig = (self.width(), self.height(), id(self._image))
            if self._pix is None or self._pix_sig != sig:
                self._pix = QPixmap.fromImage(self._image.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self._pix_sig = sig
            pm = self._pix
            p.drawPixmap((self.width() - pm.width()) // 2,
                         (self.height() - pm.height()) // 2, pm)
        else:
            p.setPen(QColor("#7d8b9a"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       self.message or "正在初始化离屏渲染…")
        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF() - 1.0, 8.0))
        p.setFont(font)
        p.setPen(QColor("#8ea3b6"))
        p.drawText(12, self.height() - 12, "左键旋转 · 中键平移 · 滚轮缩放 · 双击复位视角")
        if self.renderer is not None:
            p.drawText(self.width() - 100, 22, f"{self.fps:5.1f} fps")

    # ------------------------------------------------------------ 鼠标
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        btn = ev.button()
        if btn in (Qt.LeftButton, Qt.MiddleButton):
            self._drag = (1 if btn == Qt.LeftButton else 2,
                          float(ev.position().x()), float(ev.position().y()))

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._drag is None:
            return
        b, x0, y0 = self._drag
        dx = float(ev.position().x()) - x0
        dy = float(ev.position().y()) - y0
        self._drag = (b, float(ev.position().x()), float(ev.position().y()))
        if b == 1:
            self.cam.orbit(dx, dy)
        else:
            self.cam.pan(dx, dy)
        self.render_frame()                       # 拖动时立刻出图，手感才跟得上

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        self._drag = None

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802
        self.cam.reset()
        self.render_frame()

    def wheelEvent(self, ev) -> None:  # noqa: N802
        notches = ev.angleDelta().y() / 120.0
        if notches:
            self.cam.zoom(notches)
            self.render_frame()

    def set_view(self, name: str) -> None:
        """切预设视角（斜视 / 俯视 / 侧视 / 近看末端，见 ``arm_core.VIEW_PRESETS``）。"""
        self.cam.preset(name)
        self.render_frame()


# =============================================================== 控制面板
# =============================================================== 真机跟随的体检文本
def follow_status_text(sim: core.ArmSim, lk, out=None, err: str = "") -> str:
    """「真机跟随」卡片下半部分那几行（状态 / 延迟 / 包率 / 真机 ↔ 仿真误差）。

    做成模块级函数是为了**自检能直接调**（不用建窗口），界面里的
    ``ControlPanel._follow_text`` 只是转发到这里。
    """
    st_sim = sim.status()
    if lk is None:
        note = f"（上次启动失败：{err}）" if err else ""
        return (f"状态：未启动{note}\n"
                f"当前模式：{MODE_NAMES.get(st_sim['mode'], st_sim['mode'])} · "
                f"仿真工具尖 {vec_str(st_sim['tip'])}\n"
                f"默认：UDP :{link.DEFAULT_UDP_PORT} 广播 + HTTP {link.DEFAULT_HTTP_URL}\n"
                f"提示：点「启动跟随」开始，键盘 F 也能切；停止时会就地保持姿态。")
    n_pkt, n_err = lk.packets()
    head = "跟随中" if lk.running else "已停止"
    if out is not None and out.stale:
        head += " · 掉线！"
    line1 = f"状态：{head} · {lk.describe_sources()}"
    age = lk.age()
    line2 = (f"数据：{lk.rate_text()} · 延迟 "
             f"{'—' if age == float('inf') else f'{age * 1000:.0f} ms'} · "
             f"包 {n_pkt} / 错 {n_err}")
    if out is not None and out.note:
        line2 += f" · {out.note}"
    st = out.state if out is not None else lk.latest()
    if st is None:
        return "\n".join([line1, line2, "还没收到数据……检查真机是否在广播 / 地址端口对不对"])
    lines = [line1, line2,
             f"真机：q {np.round(st.deg, 2).tolist()}（原单位 {st.unit}）",
             f"仿真：q {np.round(st_sim['q_deg'], 2).tolist()} · "
             f"关节跟踪 {st_sim['track_deg']:.3f}°"]
    tcp = st.tcp()
    if tcp is not None:
        d_mm = float(np.linalg.norm(sim.tip() - tcp)) * 1000.0
        axis = st.axis()
        if axis is None:
            lines.append(f"误差：TCP {d_mm:.2f} mm（真机 pose vs 模型 tool_site）")
        else:
            ang = math.degrees(math.acos(float(np.clip(sim.tip_axis() @ axis, -1.0, 1.0))))
            lines.append(f"误差：TCP {d_mm:.2f} mm · 工具轴 {ang:.2f}°"
                         f"（真机 pose vs 模型 tool_site）")
        lines.append(f"真机 TCP {np.round(tcp, 4).tolist()} · "
                     f"仿真工具尖 {vec_str(sim.tip(), 4)}")
    return "\n".join(lines)


class ControlPanel(QWidget):
    """右侧控制面板：5 张卡片；所有控件回调都只做两件事——调 ``ArmSim`` 的方法、刷新显示。"""

    def __init__(self, sim: core.ArmSim, win: "RevA1Window"):
        super().__init__()
        self.sim = sim
        self.win = win
        self.selected = 0                     # 键盘选中的关节（1..6）
        self.joint_rows: list[dict] = []      # 关节表格的控件引用（刷新用）
        self.points: list[dict] = []          # 示教点位（工具尖 + 工具轴 + 关节角）
        self._last_log: dict[str, float] = {}
        self.link: link.JointLink | None = None    # 真机 ↔ 仿真的姿态链路（没启动时 None）
        self.follow_out: link.FollowOut | None = None
        self.follow_error = ""                # 启动失败的原因（留在卡片上）
        self.cloud_scene: Path | None = None  # 当前场景里是否接了点云（None = 干净场景）
        self.cloud_report: list[str] = []     # 点云的体检报告（卡片上显示）
        self.cloud_points = None              # 基座系点云（存图时算视角用）
        self.cloud_error = ""
        self.task_listener: link.TaskListener | None = None   # 任务信号（6501）监听，没启动时 None
        self.task_active = False              # 最近一包是 start 且还没收到 over → 正在记轨迹

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(10)
        for builder in (self._build_joints, self._build_tools, self._build_follow,
                        self._build_task, self._build_cloud,
                        self._build_cartesian, self._build_points, self._build_run,
                        self._build_log):
            lay.addWidget(builder())
        lay.addStretch(1)
        self.select_joint(0)
        self.refresh()

    # ------------------------------------------------------------ 小工具
    def _throttled(self, key: str, seconds: float = 0.6) -> bool:
        """限流：同一个 key 在 seconds 秒内只放行一次（连点按钮别把日志刷爆）。"""
        now = time.perf_counter()
        if now - self._last_log.get(key, 0.0) < seconds:
            return False
        self._last_log[key] = now
        return True

    def move_time(self) -> float:
        """「运动时长」输入框的值（秒）。"""
        return max(float(self.dur_spin.value()), 0.2)

    # ------------------------------------------------------------ 1. 关节微调
    def _build_joints(self) -> Card:
        card = Card("关节微调", "− / + 就是「目标角 −步长 / +步长」，点一下动一点（按住 0.4 s 后"
                               "自动连点）。左边是目标角（伺服指令），右边括号里是实测角。")
        card.add_row(QLabel("步长"), self._make_step_box(), None,
                     self._btn("同步实测", self.on_sync),
                     self._btn("全部归零", self.on_zero),
                     self._btn("回 home", self.on_home))
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        for c, txt in enumerate(("关节", "", "目标角", "", "实测角")):
            head = QLabel(txt)
            head.setObjectName("Head")
            head.setAlignment(Qt.AlignCenter)
            grid.addWidget(head, 0, c)
        for i, _name in enumerate(spec.JOINTS):
            tag = QLabel(f"J{i + 1}")
            tag.setObjectName("Tag")
            tag.setFixedWidth(24)
            minus = RepeatButton("−", lambda i=i: self.nudge(i, -1.0))
            minus.setObjectName("Step")
            minus.setToolTip(f"J{i + 1}：目标角 −一个步长（可按住）")
            plus = RepeatButton("+", lambda i=i: self.nudge(i, +1.0))
            plus.setObjectName("Step")
            plus.setToolTip(f"J{i + 1}：目标角 +一个步长（可按住）")
            val = QLabel("+0.0°")
            val.setObjectName("Value")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setMinimumWidth(72)
            act = QLabel("(+0.0°)")
            act.setObjectName("Mono")
            act.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            act.setMinimumWidth(70)
            grid.addWidget(tag, i + 1, 0)
            grid.addWidget(minus, i + 1, 1)
            grid.addWidget(val, i + 1, 2)
            grid.addWidget(plus, i + 1, 3)
            grid.addWidget(act, i + 1, 4)
            self.joint_rows.append(dict(tag=tag, minus=minus, plus=plus, val=val, act=act))
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(4, 1)
        card.body.addLayout(grid)
        return card

    # ------------------------------------------------------------ 1.5 工具 TCP 向量（画面箭头）
    def _build_tools(self) -> Card:
        card = Card("工具 TCP 向量（跟随机械臂）",
                    "同一片法兰装了三个工具（真机 TCP 标定给的是**末端系**向量）。"
                    "画面上每个工具一根箭头，三根互相平行、都平行于末端工具轴，箭头尖落在各自 TCP 上，"
                    "随机械臂实时动。")
        self.tool_boxes: dict[str, QCheckBox] = {}
        row = QHBoxLayout()
        row.setSpacing(8)
        for name in spec.tool_names():
            r, g, b, _ = spec.TOOLS[name]["rgba"]
            box = QCheckBox(name)
            box.setChecked(name in self.sim.show_tools)
            box.setStyleSheet(f"color: rgb({int(r * 255)},{int(g * 255)},{int(b * 255)});"
                              " font-weight: bold;")
            box.stateChanged.connect(lambda _s, n=name: self._on_tool_toggle(n))
            self.tool_boxes[name] = box
            row.addWidget(box)
        row.addStretch(1)
        card.add_row(row)

        self.tool_width = NumBox()
        self.tool_width.setRange(1.0, 12.0)
        self.tool_width.setSingleStep(0.5)
        self.tool_width.setDecimals(1)
        self.tool_width.setValue(float(spec.TOOL_ARROW_R_M) * 1000.0)
        self.tool_width.setSuffix(" mm")
        self.tool_width.setMaximumWidth(110)
        self.tool_width.setToolTip("箭头杆半径（头部 = 2.2 倍）")
        self.tool_width.valueChanged.connect(self._on_tool_width)
        card.add_row(QLabel("箭头粗细"), self.tool_width, None)

        self.tool_label = QLabel()
        self.tool_label.setObjectName("Mono")
        self.tool_label.setWordWrap(True)
        self.tool_label.setMinimumHeight(54)
        self.tool_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        card.add(self.tool_label)
        return card

    def _on_tool_toggle(self, name: str) -> None:
        if self.tool_boxes[name].isChecked():
            self.sim.show_tools.add(name)
        else:
            self.sim.show_tools.discard(name)
        self.refresh()

    def _on_tool_width(self, value: float) -> None:
        self.sim.tool_width_m = float(value) / 1000.0

    def _make_step_box(self) -> QComboBox:
        self.step_box = QComboBox()
        self.step_box.addItems(JOINT_STEPS)
        self.step_box.setCurrentIndex(1)          # 默认 1°
        self.step_box.setToolTip("−/+ 按钮每次改变多少度")
        return self.step_box

    @staticmethod
    def _btn(text: str, slot, *, name: str = "") -> QPushButton:
        b = QPushButton(text)
        if name:
            b.setObjectName(name)
        b.clicked.connect(slot)
        return b

    def step_deg(self) -> float:
        """当前关节微调步长[度]。"""
        return (0.5, 1.0, 5.0, 15.0)[self.step_box.currentIndex()]

    def nudge(self, i: int, sign: float) -> None:
        """−/+ 按钮：改第 i 个关节的目标角（并按需记一条日志）。"""
        self.select_joint(i)
        if self.link is not None and self.link.running and self._throttled("follow-nudge", 3.0):
            self.sim.log("真机跟随中：手动微调下一秒就会被真机姿态覆盖（要手动先点「停止跟随」）")
        self.sim.nudge_joint(i, sign * self.step_deg())
        if self._throttled(f"j{i}"):
            self.sim.log(f"单关节：J{i + 1} 目标 → {self.sim.cmd_deg()[i]:+.1f}°")
        self.refresh()

    def select_joint(self, i: int) -> None:
        """选中关节（键盘 1..6 / 点 −+ 都会选），用来高亮 + 给 ``-``/``=`` 用。"""
        self.selected = int(i) % len(spec.JOINTS)
        for k, row in enumerate(self.joint_rows):
            row["tag"].setProperty("sel", k == self.selected)
            row["tag"].style().unpolish(row["tag"])
            row["tag"].style().polish(row["tag"])

    def on_sync(self) -> None:
        """指令 ← 实测（清掉跟踪误差，手臂不会突然抽一下）。"""
        self.sim.sync_cmd_to_actual()
        self.refresh()

    def on_zero(self) -> None:
        self.sim.zero(self.move_time())

    def on_home(self) -> None:
        self.sim.home(self.move_time())

    # ------------------------------------------------------------ 2. 真机跟随（真机 → 仿真）
    def _build_follow(self) -> Card:
        card = Card("真机跟随（UDP 广播 / HTTP 状态）",
                    "真机把姿态发出来（本机实测：往 **UDP 6001** 广播，约 96 Hz；也开着 HTTP 状态接口），"
                    "点「启动跟随」就把它的 6 个关节角实时搬到模型上。下面几行是现场体检："
                    "延迟、包率，以及**真机 TCP ↔ 仿真工具尖**的误差。")
        self.f_source = QComboBox()
        self.f_source.addItems(link.SOURCE_LABELS)
        self.f_source.setCurrentIndex(link.SOURCES.index("auto"))
        self.f_source.setToolTip("自动 = HTTP 与 UDP 同时开，哪路新用哪路（推荐）")
        self.f_source.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.f_source.setMaximumWidth(180)
        self.f_fmt = QComboBox()
        self.f_fmt.addItems(link.FORMAT_LABELS)
        self.f_fmt.setToolTip("UDP 报文的格式；真机是 JSON，认不出来就换这里试")
        self.f_fmt.setMaximumWidth(96)
        card.add_row(QLabel("数据源"), self.f_source, QLabel("报文"), self.f_fmt)

        self.f_url = QLineEdit(link.DEFAULT_HTTP_URL)
        self.f_url.setToolTip("真机 HTTP 状态接口（本机实测 http://192.168.66.169:8080/api/state）")
        self.f_url.setMinimumWidth(150)
        card.add_row(QLabel("HTTP"), self.f_url)

        self.f_port = NumBox()
        self.f_port.setRange(1, 65535)
        self.f_port.setValue(int(link.DEFAULT_UDP_PORT))
        self.f_port.setToolTip("真机 UDP 广播端口（本机实测往 6001 发）")
        self.f_unit = QComboBox()
        self.f_unit.addItems(link.UNIT_LABELS)
        self.f_unit.setToolTip("报文里的角度单位；自动 = |角| > 7 当度，否则当弧度")
        card.add_row(QLabel("UDP 端口"), self.f_port, QLabel("单位"), self.f_unit)

        self.f_mode = QComboBox()
        self.f_mode.addItems(link.MODE_LABELS)
        self.f_mode.setToolTip("绝对 = 完全镜像真机姿态；相对 = 只镜像增量（两边本来就不同姿时用）")
        card.add_row(QLabel("映射"), self.f_mode)

        self.f_smooth = NumBox()
        self.f_smooth.setRange(0.0, 0.5)
        self.f_smooth.setSingleStep(0.02)
        self.f_smooth.setDecimals(2)
        self.f_smooth.setValue(0.08)
        self.f_smooth.setSuffix(" s")
        self.f_smooth.setToolTip("0 = 最跟手；越大越平滑但滞后（真机信号抖的时候调大）")
        self.f_speed = NumBox()
        self.f_speed.setRange(0.0, 720.0)
        self.f_speed.setSingleStep(30.0)
        self.f_speed.setDecimals(0)
        self.f_speed.setValue(180.0)
        self.f_speed.setSuffix(" °/s")
        self.f_speed.setToolTip("每秒最多跟多少度（防真机跳变/毛刺把仿真甩出去）；0 = 不限")
        self.f_timeout = NumBox()
        self.f_timeout.setRange(0.2, 10.0)
        self.f_timeout.setSingleStep(0.1)
        self.f_timeout.setDecimals(2)
        self.f_timeout.setValue(1.0)
        self.f_timeout.setSuffix(" s")
        self.f_timeout.setToolTip("多久没有新包就算掉线（看门狗：掉线就保持不动）")
        card.add_row(QLabel("平滑"), self.f_smooth, QLabel("限速"), self.f_speed)
        card.add_row(QLabel("看门狗"), self.f_timeout, None)
        for w in (self.f_port, self.f_unit, self.f_smooth, self.f_speed, self.f_timeout):
            w.setMaximumWidth(96)

        card.add_row(self._btn("启动跟随", self.on_follow_start, name="Primary"),
                     self._btn("停止跟随", self.on_follow_stop, name="Danger"),
                     None,
                     self._btn("读一次", self.on_follow_probe, name="Preset"))

        self.follow_label = QLabel()
        self.follow_label.setObjectName("Mono")
        self.follow_label.setWordWrap(True)
        self.follow_label.setMinimumHeight(132)
        self.follow_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        card.add(self.follow_label)
        return card

    # ------------------------------------------------------------ 2.5 任务信号 6501 → 工具尖轨迹
    def _build_task(self) -> Card:
        card = Card("任务信号 6501 → 工具尖轨迹",
                    "真机开始/结束作业时往 **UDP 6501** 广播 ``{\"motion\": \"start\"}`` / "
                    "``{\"motion\": \"over\"}``。收到 start 就把勾上的工具尖连成轨迹（随机械臂长出来），"
                    "收到 over 就立刻清空。")
        self.t_task_port = NumBox()
        self.t_task_port.setRange(1, 65535)
        self.t_task_port.setValue(int(link.DEFAULT_TASK_PORT))
        self.t_task_port.setMaximumWidth(96)
        self.t_task_port.setToolTip("真机任务信号 UDP 端口")
        card.add_row(QLabel("端口"), self.t_task_port,
                     self._btn("开始监听", self.on_task_start, name="Primary"),
                     self._btn("停止监听", self.on_task_stop, name="Danger"))

        self.trail_boxes: dict[str, QCheckBox] = {}
        row = QHBoxLayout()
        row.setSpacing(8)
        for name in spec.tool_names():
            r, g, b, _ = spec.TOOLS[name]["rgba"]
            box = QCheckBox(name)
            box.setStyleSheet(f"color: rgb({int(r * 255)},{int(g * 255)},{int(b * 255)});"
                              " font-weight: bold;")
            box.setToolTip("勾上后，收到 start 就记这个工具尖的轨迹")
            self.trail_boxes[name] = box
            row.addWidget(box)
        row.addStretch(1)
        card.add_row(row)

        card.add_row(self._btn("清空轨迹", self.on_task_clear, name="Preset"), None)

        self.task_label = QLabel()
        self.task_label.setObjectName("Mono")
        self.task_label.setWordWrap(True)
        self.task_label.setMinimumHeight(80)
        self.task_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        card.add(self.task_label)
        return card

    # ---- 任务信号 → 轨迹：按钮 / 每帧 / 显示
    def on_task_start(self) -> None:
        if self.task_listener is not None and self.task_listener.running:
            self.sim.log("任务信号监听已经在跑了（要改端口先点「停止监听」）")
            return
        try:
            tk = link.TaskListener(int(self.t_task_port.value()))
            tk.start()
        except link.RobotLinkError as exc:
            self.sim.log(f"任务信号监听启动失败：{exc}")
            return
        self.task_listener = tk
        self.sim.log(f"任务信号监听已启动：{tk.describe()}"
                     f"（收到 start 开始记勾选工具的轨迹，over 清空）")
        self.refresh()

    def on_task_stop(self) -> None:
        if self.task_listener is None:
            self.sim.log("任务信号监听还没启动")
            return
        self.task_listener.stop()
        self.task_listener = None
        self.task_active = False
        self.sim.log("任务信号监听已停止")
        self.refresh()

    def on_task_clear(self) -> None:
        n = self.sim.clear_trails()
        self.sim.log(f"已清空 {n} 根轨迹" if n else "没有轨迹可清")
        self.refresh()

    def task_tick(self) -> None:
        """主循环每帧调用：收到 start 就记勾选工具的轨迹，收到 over 就清空。"""
        tk = self.task_listener
        if tk is None:
            return
        for sig in tk.drain_signals():
            if sig.motion == link.MOTION_START:
                self.task_active = True
                names = [n for n, b in self.trail_boxes.items() if b.isChecked()]
                self.sim.log(f"任务开始（{sig.src or '?'}）：开始记轨迹"
                             + (f" {names}" if names else "（没勾工具，只监听）"))
            elif sig.motion == link.MOTION_OVER:
                self.task_active = False
                self.sim.clear_trails()
                self.sim.log("任务结束：轨迹已清空")
        if self.task_active:
            names = [n for n, b in self.trail_boxes.items() if b.isChecked()]
            if names:
                self.sim.record_trails(names)

    def _task_text(self) -> str:
        tk = self.task_listener
        lines = []
        if tk is None:
            lines.append("状态：未监听（点「开始监听」接 6501 任务信号）")
        else:
            st = tk.stats()
            lines.append(f"状态：{'运行中' if tk.running else '已停止'} · 端口 :{st['port']} · "
                         f"包 {st['packets']} / 错 {st['errors']}")
            last = st["motion"]
            lines.append("最近：" + (link.MOTION_LABELS.get(last, last) if last else "还没收到信号")
                         + f" · 来自 {st['last_addr'] or '—'}")
        if self.task_active:
            lines.append("任务中：正在记勾选工具的轨迹")
        trails = self.sim.trail_texts()
        lines.append("轨迹：" + ("；".join(trails) if trails else "空（收到 start 且勾了工具才会长）"))
        return "\n".join(lines)

    # ---- 真机跟随：按钮 / 每帧 / 显示
    def follow_source(self) -> str:
        """当前选的数据源（``auto`` / ``http`` / ``udp``）。"""
        return link.SOURCES[self.f_source.currentIndex()]

    def follow_config(self) -> dict:
        """把卡片上的控件读成 :class:`robot_link.JointLink` 的构造参数。"""
        return dict(source=self.follow_source(),
                    http_url=self.f_url.text().strip(),
                    port=int(self.f_port.value()),
                    fmt=link.FORMATS[self.f_fmt.currentIndex()],
                    unit=link.UNITS[self.f_unit.currentIndex()],
                    mode=link.MODES[self.f_mode.currentIndex()],
                    smooth=float(self.f_smooth.value()),
                    max_speed_deg=float(self.f_speed.value()),
                    timeout=float(self.f_timeout.value()))

    def on_follow_start(self) -> None:
        """「启动跟随」：起接收线程，之后主循环每帧把真机姿态下发到仿真。"""
        if self.link is not None and self.link.running:
            self.sim.log("真机跟随已经在跑了（要改参数先点「停止跟随」）")
            return
        cfg = self.follow_config()
        self.follow_error = ""
        if cfg["source"] in ("http", "auto") and not cfg["http_url"]:
            self.follow_error = "HTTP 地址是空的（或者把数据源改成「UDP 广播」）"
            self.link = None
        else:
            try:
                new_link = link.JointLink(**cfg)
                new_link.start()
                self.link = new_link
            except (link.RobotLinkError, OSError) as exc:
                self.link = None
                self.follow_error = f"{type(exc).__name__}: {exc}"
        if self.link is None:
            self.sim.log(f"真机跟随启动失败：{self.follow_error}")
        else:
            speed = "不限" if cfg["max_speed_deg"] <= 0 else f"{cfg['max_speed_deg']:.0f}°/s"
            self.sim.log(f"真机跟随已启动：{self.link.describe_sources()}"
                         f"（{link.MODE_LABELS[self.f_mode.currentIndex()]}，"
                         f"平滑 {cfg['smooth']:.2f} s，限速 {speed}，"
                         f"看门狗 {cfg['timeout']:.2f} s）")
        self.refresh()

    def on_follow_stop(self) -> None:
        """「停止跟随」：收线程关掉，并**就地保持**当前姿态（不会掉下来）。"""
        if self.link is None:
            self.sim.log("真机跟随还没启动（没什么可停的）")
            self.refresh()
            return
        name = self.link.describe_sources()
        self.link.stop()
        self.link = None
        self.follow_out = None
        self.sim.hold()
        self.sim.log(f"真机跟随已停止（{name}）：就地保持当前姿态"
                     f"（要回 home 按 H，或点「回 home」）")
        self.refresh()

    def on_follow_probe(self) -> None:
        """「读一次」：不持续跟，先读一帧看看通不通（HTTP 专有；UDP 只能一直听）。"""
        if self.follow_source() == "udp":
            self.sim.log("UDP 只能持续听：点「启动跟随」后看卡片上的包率 / 延迟")
            return
        try:
            st = link.HttpStateSource(self.f_url.text().strip(), timeout=1.5).poll_once()
        except Exception as exc:  # noqa: BLE001
            self.follow_error = f"{type(exc).__name__}: {exc}"
            self.sim.log(f"「读一次」失败：{self.follow_error}")
            self.refresh()
            return
        self.follow_error = ""
        extra = "" if st.tcp() is None else f"，TCP {np.round(st.tcp(), 4).tolist()}"
        self.sim.log(f"真机在线：q(deg) {np.round(st.deg, 2).tolist()}{extra}")
        self.refresh()

    def toggle_follow(self) -> None:
        """键盘 ``F``：在「启动跟随 / 停止跟随」之间切。"""
        if self.link is not None and self.link.running:
            self.on_follow_stop()
        else:
            self.on_follow_start()

    def follow_tick(self) -> None:
        """主循环**每帧**调用（在推进物理之前）：把最近一帧真机姿态下发到仿真。"""
        lk = self.link
        if lk is None:
            return
        for line in lk.drain_notes():          # 链路自己的事件也写进日志
            self.sim.log(line)
        out = lk.update(self.sim.cmd)          # 映射 / 平滑 / 限速 / 看门狗
        self.follow_out = out
        if out.ok:
            self.sim.follow(out.q_rad)

    def shutdown(self) -> None:
        """退出前收尾：把接收线程停掉（别留 socket / 线程）。"""
        if self.link is not None:
            self.link.stop()
            self.link = None
            self.follow_out = None
        if self.task_listener is not None:
            self.task_listener.stop()
            self.task_listener = None

    def _follow_text(self) -> str:
        """卡片下半部分那几行状态：转发给模块级 :func:`follow_status_text`（自检也能用）。"""
        return follow_status_text(self.sim, self.link, self.follow_out, self.follow_error)

    # ------------------------------------------------------------ 3. 点云（相机 → 机械臂）
    def _build_cloud(self) -> Card:
        card = Card("点云（相机 → 机械臂）",
                    "深度相机拍的点云（本机数据是 **o2e**：已经换算到**末端系**），"
                    "按**拍摄姿态**搬到机械臂基座系，再生成一个「机械臂 + 点云 + 小车」的世界场景整体加载。"
                    "相机 z 越小 = 离相机越近；地面按小车高度下移，所以「离地高度」还是真的。")
        self.c_cloud = QComboBox()
        for p in pc.list_clouds():
            self.c_cloud.addItem(p.name, str(p))
        if pc.load_cloud_groups():           # config.json 里配了多组点云 → 放第一项并默认选中
            self.c_cloud.insertItem(0, self.multi_cloud_label(), MULTI_CLOUD)
            self.c_cloud.setCurrentIndex(0)
        if self.c_cloud.count() == 0:
            self.c_cloud.addItem("（point_cloud/ 里还没有文件）", "")
        self.c_cloud.setToolTip("point_cloud/ 下的点云（*.json：3D 点 或 16 位深度图；也认 *.npy）\n"
                                "第一项「配置的多组点云」= config.json 的 point_cloud_groups："
                                "每组用自己的拍摄姿态换算，和网页端一样一起渲染")
        self.c_cloud.setMaximumWidth(200)
        card.add_row(QLabel("点云"), self.c_cloud, self._btn("刷新", self.on_cloud_refresh, name="Preset"))

        self.c_pose = QLineEdit(pc.pose_text(pc.DEFAULT_POSE))
        self.c_pose.setToolTip("拍摄姿态：x y z(mm) + rx ry rz(deg)（= 真机 pose 的 6 个数）")
        card.add_row(QLabel("拍摄姿态"), self.c_pose)
        card.add_row(self._btn("取真机姿态", self.on_cloud_pose_from_robot, name="Preset"), None)

        self.c_calib = QLineEdit(str(pc.DEFAULT_EXTRINSIC))
        self.c_calib.setToolTip("手眼标定（OpenCV XML 里的 R / t：相机 → 末端，单位 mm）")
        card.add_row(QLabel("手眼标定"), self.c_calib)

        self.c_frame = QComboBox()
        self.c_frame.addItems(["自动判定", "相机光学系", "末端系（o2e）"])
        self.c_frame.setToolTip("点云自带哪个坐标系。自动判定 = 看有多少点反投影落回文件自带的 ROI\n"
                                "（本机的 o2e 数据会判成末端系：光学→末端的转换已经做过，不再重复乘手眼）")
        card.add_row(QLabel("坐标系"), self.c_frame, None, QLabel("（o2e = 末端系）"))

        self.c_ref = QComboBox()
        self.c_ref.addItems(["TCP（工具尖）", "法兰（ee_site）"])
        self.c_ref.setToolTip("标定矩阵是相对哪个点算的（真机 pose 报的是工具尖）")
        self.c_cart = NumBox()
        self.c_cart.setRange(0.0, 2.5)
        self.c_cart.setSingleStep(0.05)
        self.c_cart.setDecimals(2)
        self.c_cart.setValue(float(pc.DEFAULT_CART_HEIGHT_M))
        self.c_cart.setSuffix(" m")
        self.c_cart.setToolTip("机械臂装在小车上，基座离地多高（地面就铺在这个高度）")
        self.c_show_cart = QCheckBox("画小车")
        self.c_show_cart.setChecked(True)
        self.c_cart_x = NumBox()
        self.c_cart_y = NumBox()
        for w, v in ((self.c_cart_x, float(pc.DEFAULT_CART_CENTER_MM[0])),
                     (self.c_cart_y, float(pc.DEFAULT_CART_CENTER_MM[1]))):
            w.setRange(-500.0, 500.0)
            w.setSingleStep(10.0)
            w.setDecimals(0)
            w.setSuffix(" mm")
            w.setMaximumWidth(96)
            w.setValue(v)
        self.c_cart_x.setToolTip("小车中心在**基座系**里的位置（x）。默认 (+150, +150) 表示机械臂"
                                 "贴着小车的**左前缘**装；想让基座回到车正中就填 0 0")
        self.c_cart_y.setToolTip(self.c_cart_x.toolTip())
        self.c_target_pad = QCheckBox("作业点标记")
        self.c_target_pad.setChecked(bool(pc.DEFAULT_SHOW_TARGET_PAD))
        self.c_target_pad.setToolTip("场景里那个绿色圆盘（名义作业点标记，直径 10 cm）。\n"
                                     "默认不显示：它在点云场景里像机械臂旁多出来的一块。\n"
                                     "勾上就显示（对当前场景立刻生效，切场景也记着）")
        self.c_target_pad.stateChanged.connect(lambda _s: self._apply_pad_visibility())
        card.add_row(QLabel("参考点"), self.c_ref, QLabel("小车高"), self.c_cart)
        card.add_row(QLabel("小车中心"), self.c_cart_x, self.c_cart_y, None)
        card.add_row(self.c_show_cart, self.c_target_pad, None)

        self.c_points = NumBox()
        self.c_points.setRange(0, 200000)
        self.c_points.setSingleStep(5000)
        self.c_points.setValue(int(pc.DEFAULT_MAX_POINTS))
        self.c_points.setToolTip("点数上限（0 = 不抽稀）；整云 8 万点也能跑，就是网格文件大一些")
        self.c_size = NumBox()
        self.c_size.setRange(1.0, 20.0)
        self.c_size.setSingleStep(0.5)
        self.c_size.setDecimals(1)
        self.c_size.setValue(float(pc.DEFAULT_POINT_R_MM))
        self.c_size.setSuffix(" mm")
        self.c_size.setToolTip("每个点画多大（点越稀疏就开大一点，看起来才是连续的面）")
        self.c_bands = NumBox()
        self.c_bands.setRange(0, 8)
        self.c_bands.setValue(int(pc.DEFAULT_BANDS))
        self.c_bands.setToolTip("按深度分几带颜色（近红 → 远紫）；1 = 单色，0 = 也当单色")
        card.add_row(QLabel("点数上限"), self.c_points, QLabel("点大小"), self.c_size)
        card.add_row(QLabel("深度分色"), self.c_bands, None, QLabel("（近红 → 远紫）"))

        self.c_dx = NumBox()
        self.c_dy = NumBox()
        self.c_dz = NumBox()
        for w in (self.c_dx, self.c_dy, self.c_dz):
            w.setRange(-500.0, 500.0)
            w.setSingleStep(10.0)
            w.setDecimals(0)
            w.setSuffix(" mm")
            w.setMaximumWidth(96)
            w.setToolTip("手动微调：把整片点云沿基座 xyz 平移（标定有残差时用）")
        self.c_yaw = NumBox()
        self.c_yaw.setRange(-180.0, 180.0)
        self.c_yaw.setSingleStep(1.0)
        self.c_yaw.setDecimals(1)
        self.c_yaw.setSuffix(" °")
        self.c_yaw.setMaximumWidth(96)
        self.c_yaw.setToolTip("手动微调：绕基座 z 轴转一点（对不准时用）")
        card.add_row(QLabel("微调 Δx"), self.c_dx, QLabel("Δy"), self.c_dy)
        card.add_row(QLabel("Δz"), self.c_dz, QLabel("绕 z"), self.c_yaw)

        card.add_row(self._btn("加载点云", self.on_cloud_load, name="Primary"),
                     self._btn("清除点云", self.on_cloud_clear, name="Danger"),
                     None,
                     self._btn("存当前画面", self.on_cloud_snapshot, name="Preset"))

        self.cloud_label = QLabel()
        self.cloud_label.setObjectName("Mono")
        self.cloud_label.setWordWrap(True)
        self.cloud_label.setMinimumHeight(150)
        self.cloud_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        card.add(self.cloud_label)
        self.c_cloud.currentIndexChanged.connect(self._sync_cloud_mode)   # 多组 ↔ 单份 切换
        self._sync_cloud_mode()                   # 默认可能是多组模式 → 先同步一次拍摄姿态框
        return card

    # ---- 点云：按钮 / 状态
    def cloud_file(self) -> str:
        """下拉框里选的点云文件（完整路径；选在「多组点云」那一项时是 :data:`MULTI_CLOUD`）。"""
        return str(self.c_cloud.currentData() or "")

    def cloud_is_multi(self) -> bool:
        """下拉框是不是选在「配置的多组点云」那一项（= config 的 ``point_cloud_groups``）。"""
        return self.cloud_file() == MULTI_CLOUD

    def multi_cloud_label(self) -> str:
        """「配置的多组点云」那一项显示的名字（N 组 / M 份）。"""
        groups = pc.load_cloud_groups()
        return (f"（配置的多组点云：{len(groups)} 组 / "
                f"{sum(len(g['files']) for g in groups)} 份）")

    def _sync_cloud_mode(self, *_args) -> None:
        """切到多组模式时把「拍摄姿态」框置灰 —— 每组的位姿写在 config 里，改这个框不管用。"""
        multi = self.cloud_is_multi()
        self.c_pose.setEnabled(not multi)
        self.c_pose.setToolTip("多组点云模式：每组的拍摄姿态写在 config.json 里（本框不参与）" if multi
                               else "拍摄姿态：x y z(mm) + rx ry rz(deg)（= 真机 pose 的 6 个数）")

    def set_cloud_file(self, path) -> None:
        """把下拉框切到某个文件（不在列表里就临时加进去）。"""
        want = Path(path)
        for i in range(self.c_cloud.count()):
            if Path(str(self.c_cloud.itemData(i) or "")) == want:
                self.c_cloud.setCurrentIndex(i)
                return
        self.c_cloud.addItem(want.name, str(want))
        self.c_cloud.setCurrentIndex(self.c_cloud.count() - 1)

    def on_cloud_refresh(self) -> None:
        """重新扫一遍 ``point_cloud/``。"""
        cur = self.cloud_file()
        self.c_cloud.clear()
        for p in pc.list_clouds():
            self.c_cloud.addItem(p.name, str(p))
        if pc.load_cloud_groups():                       # 多组那一项也补回来
            self.c_cloud.insertItem(0, self.multi_cloud_label(), MULTI_CLOUD)
        if self.c_cloud.count() == 0:
            self.c_cloud.addItem("（point_cloud/ 里还没有文件）", "")
        if cur:
            self.set_cloud_file(cur)
        self.sim.log(f"点云列表已刷新：{len(pc.list_clouds())} 个文件")
        self.refresh()

    def on_cloud_pose_from_robot(self) -> None:
        """「取真机姿态」：读一次真机 pose（mm + 度）填进姿态框。"""
        pose = None
        try:
            lk = self.link
            if lk is not None and lk.latest() is not None and lk.latest().pose is not None:
                pose = lk.latest().pose                      # 正在跟随：直接用最近一帧
            else:
                st = link.HttpStateSource(self.f_url.text().strip(), timeout=1.5).poll_once()
                pose = st.pose
        except Exception as exc:  # noqa: BLE001
            self.sim.log(f"取真机姿态失败：{type(exc).__name__}: {exc}")
            return
        if pose is None:
            self.sim.log("真机这一帧里没有 pose（只有关节角），请手动填拍摄姿态")
            return
        vals = [float(v) for v in np.r_[np.asarray(pose[:3], float) * 1000.0,
                                        np.degrees(np.asarray(pose[3:6], float))]]
        self.c_pose.setText(pc.pose_text(vals))
        self.sim.log("拍摄姿态已填成真机当前 pose：["
                     + ", ".join(f"{v:.2f}" for v in vals) + "]（mm, deg）")
        self.refresh()

    def cloud_kwargs(self) -> dict:
        """把卡片上的控件读成 :func:`point_cloud.build_cloud_scene` 的参数。"""
        return dict(pose=self.c_pose.text().strip(),
                    extrinsic=self.c_calib.text().strip(),
                    frame=("auto", "cam", "end")[self.c_frame.currentIndex()],
                    ref="flange" if self.c_ref.currentIndex() == 1 else "tcp",
                    cart_height=float(self.c_cart.value()),
                    show_cart=bool(self.c_show_cart.isChecked()),
                    cart_center_mm=(float(self.c_cart_x.value()), float(self.c_cart_y.value())),
                    # 绿圆盘总是写进 XML，显不显示交给「作业点标记」那个勾（运行时开关，
                    # 换场景也管用；见 _apply_pad_visibility）
                    show_target_pad=True,
                    max_points=int(self.c_points.value()),
                    point_mm=float(self.c_size.value()),
                    bands=int(self.c_bands.value()),
                    offset_mm=(float(self.c_dx.value()), float(self.c_dy.value()),
                               float(self.c_dz.value())),
                    yaw_deg=float(self.c_yaw.value()))

    def cloud_multi_kwargs(self) -> dict:
        """多组点云用：卡片控件里 :func:`point_cloud.build_multi_cloud_scene` 认识的那些。

        逐组的**拍摄姿态**写在 config.json 里（每组一个），所以「拍摄姿态」框不参与
        （见 :meth:`_sync_cloud_mode`）。
        """
        kw = self.cloud_kwargs()
        kw.pop("pose", None)
        return kw

    def on_cloud_load(self) -> None:
        """「加载点云」：换算 → 生成场景 → 重建模型（机械臂 + 点云 + 小车同框）。

        下拉框选具体文件 = 只看那一份；选「配置的多组点云」= 按 config.json 的
        ``point_cloud_groups``，**每组用自己的拍摄姿态**换算后一起渲染（和网页端一致）。
        """
        multi = self.cloud_is_multi()
        groups = pc.load_cloud_groups() if multi else []
        if multi and not groups:
            self.sim.log("config.json 里没有可用的多组点云（point_cloud_groups）")
            return
        if groups:
            n_files = sum(len(g["files"]) for g in groups)
            self.sim.log(f"正在把配置的多组点云接进场景：{len(groups)} 组 / {n_files} 份点云"
                         "（写网格 + 重建场景）……")
        else:
            path = self.cloud_file()
            if not path:
                self.sim.log("没有选点云文件（point_cloud/ 里放 *.json 或 *.npy，再点「刷新」）")
                return
            self.sim.log(f"正在把点云接进场景：{Path(path).name}（写网格 + 重建场景，约 1 秒）……")
        QApplication.processEvents()                          # 让上面这行日志先显示出来
        t0 = time.perf_counter()
        try:
            if groups:
                res = pc.build_multi_cloud_scene(groups, **self.cloud_multi_kwargs())
                pts = np.vstack([np.asarray(p, dtype=float).reshape(-1, 3)
                                 for p in res["points_base"]])
            else:
                res = pc.build_cloud_scene(cloud_path=path, **self.cloud_kwargs())
                pts = np.asarray(res["points_base"], dtype=float)
        except Exception as exc:  # noqa: BLE001
            self.cloud_error = f"{type(exc).__name__}: {exc}"
            self.cloud_report = []
            self.sim.log(f"点云加载失败：{self.cloud_error}")
            self.refresh()
            return
        self.cloud_error = ""
        self.cloud_report = list(res["report"])
        self.cloud_points = pts
        self.cloud_scene = Path(res["scene"].scene)
        if not self.win.load_scene(self.cloud_scene, log=False):
            self.cloud_error = "场景重建失败（看日志）"
            self.cloud_scene = None
        else:
            self.sim.log(f"点云已接进场景：{res['scene'].summary()}"
                         f"（总耗时 {(time.perf_counter() - t0) * 1000:.0f} ms）")
            for line in self.cloud_report:
                self.sim.log("    " + line)
            self.sim.log("    显示设置: 作业点标记（绿圆盘）"
                         f"{'显示' if self.c_target_pad.isChecked() else '隐藏'}"
                         f" · 小车中心 ({self.c_cart_x.value():.0f}, {self.c_cart_y.value():.0f}) mm"
                         "（点云卡片上都能改）")
            self.sim.log(f"当前场景：{Path(self.cloud_scene).name}"
                         f" · 地面 z = {self.sim.floor_z:.3f} m（「离地高度」按它算）")
        self.refresh()

    def on_cloud_clear(self) -> None:
        """「清除点云」：回到干净场景（不带点云 / 小车）。"""
        if self.cloud_scene is None:
            self.sim.log("现在就是干净场景（没加载过点云）")
            return
        ok = self.win.load_scene(spec.SCENE_XML, log=False)
        self.cloud_scene = None
        self.cloud_points = None
        self.cloud_report = []
        self.cloud_error = "" if ok else "回不到基场景（看日志）"
        self.sim.log(f"已清除点云：回到 {Path(spec.SCENE_XML).name}"
                     f" · 地面 z = {self.sim.floor_z:.3f} m")
        self.refresh()

    def on_cloud_snapshot(self) -> None:
        """「存当前画面」：把"机械臂 + 点云"同框这一帧存成 PNG。"""
        if self.cloud_scene is None:
            self.sim.log("先「加载点云」，再存画面")
            return
        out = RUNS / "pc_view.png"
        if self.win.save_frame(out):
            self.sim.log(f"已存图：{out}（机械臂 + 点云同框）")
        else:
            self.sim.log("存图失败：还没有渲染过画面（--no-render？）")

    def apply_pad_visibility(self) -> bool:
        """「作业点标记」那个勾 → 当前场景里 ``target_pad``（绿圆盘）的显隐。

        对**任何**场景都管用（基场景那个圆盘在地面上，点云场景那个在车顶平面），
        换场景后会再调一次，把勾选状态接上。场景里没这个 geom 就返回 ``False``。
        """
        return self._apply_pad_visibility(log_it=False)

    def _apply_pad_visibility(self, *, log_it: bool = True) -> bool:
        want = bool(self.c_target_pad.isChecked())
        ok = self.sim.hide_geom("target_pad", hide=not want)
        if log_it:
            self.sim.log(f"作业点标记（绿圆盘）：{'显示' if want else '隐藏'}"
                         + ("" if ok else "（当前场景里没有这个 geom）"))
        self.refresh()
        return ok

    def _cloud_text(self) -> str:
        """点云卡片下半部分的状态（加载过没有 + 体检报告）。"""
        if self.cloud_scene is None:
            note = f"（上次失败：{self.cloud_error}）" if self.cloud_error else ""
            files = pc.list_clouds()
            return (f"状态：未加载点云{note}\n"
                    f"可选点云：{len(files)} 个（{files[0].name if files else '—'} 等）\n"
                    f"点「加载点云」→ 世界换成「机械臂 + 点云 + 小车」；「清除点云」回到原场景。\n"
                    f"位置由**手眼标定 + 拍摄姿态**决定；「离地高度」按小车高度铺的地面算"
                    f"（默认 {pc.DEFAULT_CART_HEIGHT_M:.2f} m）。")
        return "\n".join([f"状态：已接进场景 {Path(self.cloud_scene).name}"] + self.cloud_report[:9])

    # ------------------------------------------------------------ 4. 末端目标 / IK
    def _build_cartesian(self) -> Card:
        card = Card("末端目标 / IK（笛卡尔）",
                    "目标位置是世界系（米），默认就是 home 时工具尖的位置。「求解 IK」只算不动，"
                    "并把目标点标在画面里；「求解并沿直线运动」先解、再让工具尖沿直线走过去。")
        self.pos_spins: list[NumBox] = []
        for r, (nm, val) in enumerate(zip("XYZ", spec.TARGET_POSE)):
            tag = QLabel(nm)
            tag.setObjectName("Tag")
            tag.setFixedWidth(14)
            spin = self._pos_spin(float(val), f"{nm} 目标坐标 [m]（世界系）")
            minus = self._btn("−", lambda _, r=r: self.step_pos(r, -1.0), name="Tiny")
            plus = self._btn("+", lambda _, r=r: self.step_pos(r, +1.0), name="Tiny")
            minus.setToolTip(f"{nm} 目标 −一个步长")
            plus.setToolTip(f"{nm} 目标 +一个步长")
            card.add_row(tag, spin, minus, plus, None)
            self.pos_spins.append(spin)
        self.ik_step_box = QComboBox()
        self.ik_step_box.addItems([name for name, _ in IK_STEPS])
        self.ik_step_box.setCurrentIndex(1)
        self.ik_step_box.setToolTip("位置 ± 按钮每次走多少")
        card.add_row(QLabel("位置步进"), self.ik_step_box, None)

        self.axis_box = QComboBox()
        self.axis_box.addItems([name for name, _ in AXIS_CHOICES])
        self.axis_box.currentIndexChanged.connect(self.on_axis_choice)
        self.axis_spins: list[NumBox] = []
        for nm, val in zip("XYZ", (0.0, 0.0, -1.0)):
            spin = self._pos_spin(val, f"工具轴方向分量 {nm}（选「自定义」时生效）")
            spin.setDecimals(2)
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.1)
            self.axis_spins.append(spin)
        card.add_row(QLabel("工具轴"), self.axis_box, None)
        card.add_row(QLabel("自定义"), *self.axis_spins, None)
        self.free_axis = QCheckBox("不约束工具姿态（只保证位置，腕部自己找姿势）")
        self.free_axis.toggled.connect(self.on_axis_choice)
        card.add(self.free_axis)

        card.add_row(self._btn("求解 IK", self.on_solve),
                     self._btn("求解并沿直线运动", self.on_move_to_target, name="Primary"),
                     None,
                     self._btn("目标←当前工具尖", self.on_read_tip, name="Preset"))
        self.ik_label = QLabel("还没求解过：先填目标点，再点「求解 IK」看误差。")
        self.ik_label.setObjectName("Result")
        self.ik_label.setWordWrap(True)
        self.ik_label.setProperty("fail", False)
        card.add(self.ik_label)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        head = QLabel("预设")
        head.setObjectName("Head")
        preset_row.addWidget(head)
        for name, pos, axis in POSE_PRESETS:
            b = self._btn(name, lambda _, p=pos, a=axis: self.apply_preset(p, a), name="Preset")
            b.setToolTip(f"填到 {vec_str(pos)}，工具轴 {vec_str(axis, 2)}")
            preset_row.addWidget(b)
        preset_row.addStretch(1)
        card.add_row(preset_row)
        self.on_axis_choice()
        return card

    def _pos_spin(self, value: float, tip: str) -> NumBox:
        spin = NumBox()
        spin.setRange(-1.5, 1.5)
        spin.setDecimals(3)
        spin.setSingleStep(0.01)
        spin.setValue(float(value))
        spin.setMinimumWidth(82)
        spin.setToolTip(tip)
        return spin

    def ik_step_m(self) -> float:
        """位置 ± 按钮的步长[m]。"""
        return IK_STEPS[self.ik_step_box.currentIndex()][1]

    def step_pos(self, i: int, sign: float) -> None:
        """第 i 个坐标 ± 一个步长（顺手夹到工作空间附近，别把机械臂指到地下/身后）。"""
        spin = self.pos_spins[i]
        lo = (-0.60, -0.70, spec.FLOOR_Z + 0.02)[i]
        hi = (1.00, 0.70, 1.20)[i]
        spin.setValue(float(np.clip(spin.value() + sign * self.ik_step_m(), lo, hi)))

    def target_pos(self) -> np.ndarray:
        """界面上的目标位置。"""
        return np.array([s.value() for s in self.pos_spins], dtype=float)

    def target_axis(self) -> tuple[np.ndarray | None, bool]:
        """界面上的工具轴设置 → ``(方向 or None, 是否约束姿态)``，直接喂 ``ArmSim.goto_pose``。"""
        if self.free_axis.isChecked():
            return None, False
        choice = AXIS_CHOICES[self.axis_box.currentIndex()][1]
        if choice is None:                        # 保持当前朝向
            return None, True
        if choice == "custom":
            return np.array([s.value() for s in self.axis_spins], dtype=float), True
        return np.asarray(choice, dtype=float), True

    def on_axis_choice(self) -> None:
        """只有选「自定义」且没勾"不约束"时，才让三个方向分量可编辑。"""
        custom = AXIS_CHOICES[self.axis_box.currentIndex()][1] == "custom"
        for spin in self.axis_spins:
            spin.setEnabled(custom and not self.free_axis.isChecked())

    def apply_preset(self, pos, axis) -> None:
        """预设点：填进输入框并**只求解**（先看能不能到、误差多少，再决定动不动）。"""
        for spin, v in zip(self.pos_spins, pos):
            spin.setValue(float(v))
        for i, (_name, val) in enumerate(AXIS_CHOICES):
            if val in (None, "custom"):
                continue
            if np.allclose(np.asarray(val, dtype=float), np.asarray(axis, dtype=float)):
                self.axis_box.setCurrentIndex(i)
                break
        self.sim.log(f"预设目标：{vec_str(pos)}，工具轴 {vec_str(axis, 2)}")
        self.on_solve()

    def on_read_tip(self) -> None:
        """把当前工具尖读进目标框（想"就地微调 1 cm"时很方便）。"""
        tip = self.sim.tip()
        for spin, v in zip(self.pos_spins, tip):
            spin.setValue(float(v))
        self.sim.log(f"目标 ← 当前工具尖 {vec_str(tip)}")

    def on_solve(self) -> None:
        """只求解（不动）：能到就绿点，到不了就红点 + 写清差多少。"""
        pos = self.target_pos()
        axis, hold = self.target_axis()
        res, _plan = self.sim.goto_pose(pos, axis, self.move_time(),
                                        hold_axis=hold, execute=False)
        self._show_ik(res)

    def on_move_to_target(self) -> None:
        """求解 + 沿直线运动过去。"""
        pos = self.target_pos()
        axis, hold = self.target_axis()
        res, plan = self.sim.goto_pose(pos, axis, self.move_time(),
                                       hold_axis=hold, execute=True)
        self._show_ik(res, plan)

    def _show_ik(self, res: core.IKResult, plan: core.PlannedLine | None = None) -> None:
        """把 IK / 规划结果写进卡片（界面上"看得见的结果"就靠它）。"""
        lines = [res.summary(), f"解 q = {np.round(np.degrees(res.q), 1).tolist()}°"]
        if plan is not None:
            seg = (float(np.linalg.norm(plan.points[-1] - plan.points[0]))
                   / max(plan.n - 1, 1)) * 1000.0
            lines.append(f"直线：{plan.n} 个路点（间隔 {seg:.0f} mm），规划 {plan.ms:.0f} ms，"
                         f"直线度 {plan.straightness() * 1000:.3f} mm")
        self.ik_label.setText("\n".join(lines))
        self.ik_label.setProperty("fail", not res.ok)
        restyle(self.ik_label)

    # ------------------------------------------------------------ 5. 点位（示教 / 点到点）
    def _build_points(self) -> Card:
        card = Card("点位（示教 / 点到点）",
                    "「记录当前位姿」把当前工具尖 + 工具轴存成一条点位；「走到选中点」让工具尖沿"
                    "**直线**走过去（双击列表项是同一个动作）。想先手动摆姿势就用手/关节按钮，"
                    "摆好记一条即可。")
        self.point_list = QListWidget()
        self.point_list.setMinimumHeight(96)
        self.point_list.itemDoubleClicked.connect(lambda _item: self.goto_selected_point())
        card.add(self.point_list)
        card.add_row(self._btn("记录当前位姿", self.record_point),
                     self._btn("走到选中点", self.goto_selected_point, name="Primary"),
                     None,
                     self._btn("删除选中", self.delete_point, name="Preset"),
                     self._btn("清空", self.clear_points, name="Preset"))
        return card

    def record_point(self) -> None:
        """把当前工具尖 + 工具轴 + 关节角存成一条点位（含防重复）。"""
        tip = self.sim.tip().copy()
        axis = self.sim.tip_axis().copy()
        if self.points and float(np.linalg.norm(self.points[-1]["tip"] - tip)) < 1e-3:
            self.sim.log("这个位姿和上一条点位几乎一样（<1 mm），就不用再记了")
            return
        self.points.append(dict(tip=tip, axis=axis, q=self.sim.q_pos().copy()))
        idx = len(self.points)
        item = QListWidgetItem(f"#{idx:02d}  tip {vec_str(tip)}  轴 {vec_str(axis, 2)}")
        item.setToolTip(f"关节角 {np.round(np.degrees(self.points[-1]['q']), 1).tolist()}°")
        self.point_list.addItem(item)
        self.point_list.setCurrentRow(self.point_list.count() - 1)
        self.sim.log(f"记录点位 #{idx}：{vec_str(tip)}，工具轴 {vec_str(axis, 2)}")

    def goto_selected_point(self) -> None:
        """让工具尖沿直线走到选中点位（姿态保持当前朝向，最不容易中途拐出去）。"""
        idx = self.point_list.currentRow()
        if not (0 <= idx < len(self.points)):
            self.sim.log("还没选点位：先在列表里点一条（或先「记录当前位姿」）")
            return
        p = self.points[idx]
        self.sim.log(f"走到点位 #{idx + 1}：目标 {vec_str(p['tip'])}")
        self.sim.goto_pose(p["tip"], None, self.move_time(), hold_axis=True, execute=True)

    def delete_point(self) -> None:
        idx = self.point_list.currentRow()
        if 0 <= idx < len(self.points):
            self.points.pop(idx)
            self.point_list.takeItem(idx)
            self.sim.log(f"删除点位 #{idx + 1}")

    def clear_points(self) -> None:
        self.points.clear()
        self.point_list.clear()
        self.sim.log("点位已清空")

    # ------------------------------------------------------------ 6. 运行 / 伺服
    def _build_run(self) -> Card:
        card = Card("运行 / 伺服",
                    "运动时长作用在「点到点 / 直线运动」上（smoothstep，首尾速度 0）；"
                    "急停 = 就地保持当前姿态（指令钉在实测角上，不会掉下来）。")
        self.dur_spin = NumBox()
        self.dur_spin.setRange(0.2, 10.0)
        self.dur_spin.setSingleStep(0.1)
        self.dur_spin.setDecimals(2)
        self.dur_spin.setValue(float(self.sim.move_seconds))
        self.dur_spin.setSuffix(" s")
        self.dur_spin.setToolTip("走得越慢越稳；太快跟踪误差会变大")
        card.add_row(QLabel("运动时长"), self.dur_spin, None, QLabel("kp 缩放"))
        self.kp_slider = QSlider(Qt.Horizontal)
        self.kp_slider.setRange(10, 200)
        self.kp_slider.setValue(100)
        self.kp_slider.setToolTip("整体缩放 kp/kv：小了伺服软（扛不住重力、稳态偏差大），"
                                  "大了硬（可能抖）")
        self.kp_slider.valueChanged.connect(self.on_kp)
        self.kp_label = QLabel("1.00×")
        self.kp_label.setObjectName("Mono")
        self.kp_label.setMinimumWidth(52)
        card.add_row(self.kp_slider, self.kp_label, None)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        card.add(self.bar)

        self.pause_box = QCheckBox("暂停物理")
        self.pause_box.toggled.connect(self.on_pause)
        self.grav_box = QCheckBox("重力")
        self.grav_box.setChecked(self.sim.gravity_on)
        self.grav_box.toggled.connect(self.on_gravity)
        self.track_box = QCheckBox("状态栏跟随")
        self.track_box.setChecked(True)
        card.add_row(self.pause_box, self.grav_box, self.track_box, None)

        card.add_row(self._btn("急停（就地保持）", self.on_estop, name="Danger"),
                     self._btn("复位模型", self.on_reset),
                     None,
                     self._btn("存图", self.on_snapshot, name="Preset"))
        view_row = QHBoxLayout()
        view_row.setSpacing(6)
        head = QLabel("视角")
        head.setObjectName("Head")
        view_row.addWidget(head)
        for name in core.VIEW_PRESETS:
            view_row.addWidget(self._btn(name, lambda _, n=name: self.win.view.set_view(n),
                                         name="Preset"))
        view_row.addStretch(1)
        card.add_row(view_row)
        return card

    def on_kp(self, value: int) -> None:
        scale = float(value) / 100.0
        self.sim.set_kp_scale(scale)
        self.kp_label.setText(f"{scale:.2f}×")
        if self._throttled("kp", 0.4):
            self.sim.log(f"kp 缩放 = {scale:.2f}×")

    def on_pause(self) -> None:
        self.win.paused = bool(self.pause_box.isChecked())
        self.sim.log(f"物理{'已暂停' if self.win.paused else '继续'}")

    def on_gravity(self) -> None:
        on = bool(self.grav_box.isChecked())
        self.sim.set_gravity(on)
        self.sim.log(f"重力{'开' if on else '关（看纯运动学效果）'}")

    def on_estop(self) -> None:
        """急停：就地保持（指令钉在急停瞬间的实测姿态上）。"""
        self.sim.hold()

    def on_reset(self) -> None:
        """复位模型 + 视角，并清掉画面上的目标点/路径。"""
        self.sim.reset_model()
        self.win.view.cam.reset()
        self.sim.ik_target = None
        self.sim.path_points = None
        self.ik_label.setText("还没求解过：先填目标点，再点「求解 IK」看误差。")
        self.ik_label.setProperty("fail", False)
        restyle(self.ik_label)

    def on_snapshot(self) -> None:
        """把当前画面存成 PNG（runs/gui_snapshot.png）。"""
        out = RUNS / "gui_snapshot.png"
        if self.win.save_frame(out):
            self.sim.log(f"已存图：{out}")
        else:
            self.sim.log("存图失败：还没有渲染过画面")

    # ------------------------------------------------------------ 7. 日志
    def _build_log(self) -> Card:
        card = Card("日志", "动作和**实测**到位精度都记在这里（同一份也打印到终端）。")
        self.log_text = QPlainTextEdit()
        self.log_text.setObjectName("Log")
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(800)
        self.log_text.setMinimumHeight(150)
        card.add(self.log_text)
        card.add_row(self._btn("清空日志", self.on_clear_log, name="Preset"), None)
        return card

    def append_log(self, line: str) -> None:
        self.log_text.appendPlainText(line)
        print(f"[gui] {line}")

    def on_clear_log(self) -> None:
        self.log_text.clear()

    # ------------------------------------------------------------ 刷新（主循环调用）
    def refresh(self) -> None:
        """按 ~5 Hz 由主循环调用：目标角 / 实测角 / 进度条 / 真机跟随体检 跟状态对齐。"""
        st = self.sim.status()
        for i, row in enumerate(self.joint_rows):
            row["val"].setText(f"{st['cmd_deg'][i]:+.1f}°")
            row["act"].setText(f"({st['q_deg'][i]:+.1f}°)")
        self.bar.setValue(int(round(st["progress"] * 1000)))
        self.follow_label.setText(self._follow_text())
        self.cloud_label.setText(self._cloud_text())
        self.tool_label.setText("\n".join(self.sim.tool_lines()))
        self.task_label.setText(self._task_text())

    def nudge_selected(self, sign: float) -> None:
        """键盘 ``-`` / ``=``：给选中关节 ± 一个步长。"""
        self.nudge(self.selected, sign)


# =============================================================== 主窗口
MODE_NAMES = {"joint": "单关节", "p2p": "关节点到点", "cartesian": "笛卡尔直线",
              "hold": "急停保持", "follow": "真机跟随"}


class RevA1Window(QMainWindow):
    """主窗口：左边 MuJoCo 画面、右边控制面板、底下一行状态栏；定时器推进物理 + 渲染。"""

    def __init__(self, sim: core.ArmSim, *, width: int = 960, height: int = 600,
                 fps: float = 30.0, render: bool = True, exit_after: float = 0.0,
                 win_w: int = 1420, win_h: int = 900):
        super().__init__()
        self.sim = sim
        self.fps = float(max(fps, 1.0))
        self.paused = False
        self.closed = False
        self.r_frames = 0
        self.r_fps = 0.0
        self.acc = 0.0
        self.t_last = time.perf_counter()
        self.t_fps0 = self.t_last
        self.t_render = 0.0
        self.t_ui = 0.0
        self._manual = False                      # True = pump() 正在手动推进

        self.setWindowTitle("TB6-R5-RevA1 交互控制台 · PySide6 + MuJoCo")
        self.resize(int(win_w), int(win_h))
        self.view = ArmView(sim, width=width, height=height, render=render)
        self.panel = ControlPanel(sim, self)
        self.panel.apply_pad_visibility()          # 「作业点标记」默认不显示（绿圆盘）

        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(430)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.view)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 0)
        split.setSizes([980, 460])
        self.setCentralWidget(split)

        self._build_statusbar()
        self.panel.append_log("就绪：用关节 ± 按钮摆姿态，或填目标点按「求解 IK」。")
        self._update_status()
        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        if self.view.renderer is not None:
            self.view.render_frame()
        if exit_after > 0:
            QTimer.singleShot(int(float(exit_after) * 1000), self.close)

    # ------------------------------------------------------------ 状态栏
    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        self.st_mode = QLabel()
        self.st_tip = QLabel()
        self.st_axis = QLabel()
        self.st_track = QLabel()
        self.st_follow = QLabel()
        self.st_perf = QLabel()
        self.st_hint = QLabel("左键旋转 · 中键平移 · 滚轮缩放 · 空格暂停 · H 回 home · "
                              "F 真机跟随 · 1..6 选关节 · −/= 微调")
        self.st_hint.setObjectName("Hint")
        bar.addWidget(self.st_hint, 1)
        for w in (self.st_mode, self.st_tip, self.st_axis, self.st_track, self.st_follow,
                  self.st_perf):
            w.setObjectName("Sub")
            bar.addPermanentWidget(w)

    def _update_status(self) -> None:
        st = self.sim.status()
        state = "运动中" if st["moving"] else ("已暂停" if self.paused else "空闲")
        if st["moving"]:
            state += f" {st['progress'] * 100:.0f}%"
        self.st_mode.setText(f"模式 {MODE_NAMES.get(st['mode'], st['mode'])} · {state}")
        self.st_tip.setText(f"工具尖 {vec_str(st['tip'])} · 离地 {st['height']:.3f} m")
        self.st_axis.setText(f"工具轴 {vec_str(st['axis'], 2)}")
        self.st_track.setText(f"跟踪误差 {st['track_deg']:.3f}° · 接触 {st['contacts']} 对")
        lk = self.panel.link
        if lk is None:
            self.st_follow.setText("真机跟随 关")
        else:
            age = lk.age()
            delay = "—" if age == float("inf") else f"{age * 1000:.0f} ms"
            n_pkt, n_err = lk.packets()
            self.st_follow.setText(f"真机 {lk.rate_text()} · 延迟 {delay} · "
                                   f"包 {n_pkt} / 错 {n_err}")
        self.st_perf.setText(f"{self.r_fps:.1f} fps（渲染 {self.view.render_ms:.0f} ms）")

    # ------------------------------------------------------------ 主循环
    def _advance(self) -> None:
        """推进一帧：真机跟随下发目标 → 物理（按真实时间补步）→ 日志 → 渲染 → 面板/状态栏。"""
        t0 = time.perf_counter()
        dt_wall = min(t0 - self.t_last, 0.25)      # 卡顿时最多补 0.25 s，别一次算几百步
        self.t_last = t0
        self.panel.follow_tick()                   # 真机跟随：先更新目标，再推进物理
        self.panel.task_tick()                     # 任务信号 → 工具尖轨迹（start 记 / over 清）
        if not self.paused:
            self.acc += dt_wall
            n = 0
            while self.acc >= self.sim.dt and n < 80:
                self.sim.step()
                self.acc -= self.sim.dt
                n += 1
            if n >= 80:
                self.acc = 0.0                     # 追不上实时就丢掉欠账，避免雪崩
        self._flush_logs()
        if self.view.renderer is not None and (t0 - self.t_render) >= 1.0 / self.fps:
            self.view.render_frame()
            self.t_render = time.perf_counter()
            self.r_frames += 1
        if t0 - self.t_ui >= UI_REFRESH_S:
            self.t_ui = t0
            self.panel.refresh()
            self._update_status()
            now = time.perf_counter()
            self.r_fps = self.r_frames / max(now - self.t_fps0, 1e-6)
            self.view.fps = self.r_fps
            self.r_frames, self.t_fps0 = 0, now

    def _tick(self) -> None:
        if self.closed or self._manual:
            return
        self._advance()

    def _flush_logs(self) -> None:
        for line in self.sim.drain():
            self.panel.append_log(line)

    def toggle_pause(self) -> None:
        self.panel.pause_box.setChecked(not self.panel.pause_box.isChecked())

    def toggle_gravity(self) -> None:
        self.panel.grav_box.setChecked(not self.panel.grav_box.isChecked())

    # ------------------------------------------------------------ 键盘
    def keyPressEvent(self, ev) -> None:  # noqa: N802  (Qt 的命名习惯)
        """键盘快捷键。

        ⚠️ 两个坑：
        1. 输入框里按键（数字、负号）**不能抢** —— 否则在位置上打 "-0.05" 会被当成快捷键；
        2. 一次性动作（回 home / 复位 / 重力）要挡掉自动重复，不然按住键会刷爆动作。
        """
        if isinstance(QApplication.focusWidget(), (QLineEdit, QPlainTextEdit)):
            super().keyPressEvent(ev)
            return
        key, txt = ev.key(), (ev.text() or "")
        if ev.isAutoRepeat() and (txt.lower() in ("h", "r", "g", "f") or key == Qt.Key_Escape):
            return
        if Qt.Key_1 <= key <= Qt.Key_6:
            self.panel.select_joint(key - Qt.Key_1)
            return
        if txt in ("-", "_"):
            self.panel.nudge_selected(-1.0)
            return
        if txt in ("=", "+"):
            self.panel.nudge_selected(+1.0)
            return
        if key == Qt.Key_Space:
            self.toggle_pause()
            return
        if key == Qt.Key_Escape:
            self.close()
            return
        if txt.lower() == "h":
            self.panel.on_home()
            return
        if txt.lower() == "r":
            self.panel.on_reset()
            return
        if txt.lower() == "g":
            self.toggle_gravity()
            return
        if txt.lower() == "f":
            self.panel.toggle_follow()
            return
        if key == Qt.Key_S and (ev.modifiers() & Qt.ControlModifier):
            self.panel.on_snapshot()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------ 工具方法
    def pump(self, seconds: float, dt: float = 0.005) -> None:
        """不开 ``mainloop`` 地推进"事件循环 + 物理"若干秒（``--ui-test`` / 自动截图用）。"""
        self._manual = True
        try:
            t_end = time.perf_counter() + float(seconds)
            while time.perf_counter() < t_end and not self.closed:
                QApplication.processEvents()
                self._advance()
                time.sleep(dt)
        finally:
            self._manual = False
            self.t_last = time.perf_counter()

    def save_frame(self, path) -> bool:
        """把最近渲染的一帧存成 PNG（没有渲染器时返回 False）。"""
        arr = self.view.frame_rgb
        if arr is None:
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w, h = self.view._rend
        QImage(arr.tobytes(), w, h, 3 * w, QImage.Format_RGB888).save(str(path))
        return True

    def close(self) -> None:  # noqa: A003 (Qt 的接口名)
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        self.panel.shutdown()      # 真机跟随的接收线程 / socket
        self.view.close()          # 必须显式释放 GL 上下文
        super().close()

    # ------------------------------------------------------------ 换场景
    def load_scene(self, scene_path, *, log: bool = True) -> bool:
        """换一个场景（例如把点云接进世界）：重建 ``ArmSim`` + 渲染器。

        伺服软硬 / 重力开关 / 运动时长都沿用当前的；真机跟随会先停掉
        （模型换了，接收线程不该再往旧模型上写）。
        """
        old = self.sim
        try:
            new = core.ArmSim(scene_path, kp_scale=old.kp_scale, gravity=old.gravity_on,
                              move_seconds=old.move_seconds, log_scene=False)
        except Exception as exc:  # noqa: BLE001
            self.panel.append_log(f"换场景失败（{Path(scene_path).name}）："
                                  f"{type(exc).__name__}: {exc}")
            return False
        self.panel.shutdown()
        self.sim = new
        self.view.sim = new
        self.panel.sim = new
        self.panel.link = None
        self.panel.follow_out = None
        self.acc = 0.0
        self.t_last = time.perf_counter()
        if self.view.render_enabled:
            self.view.close()               # 先放掉旧 GL 上下文，再按新模型重建
            self.view.init_renderer()
        if log:
            self.panel.append_log(f"已换场景：{Path(scene_path).name}"
                                  f"（{new.model.nq} 轴 · {new.model.ngeom} 几何 · "
                                  f"地面 z = {new.floor_z:.3f} m）")
        self.panel.apply_pad_visibility()      # 换场景后把「作业点标记」的显示状态接上
        self.panel.refresh()
        return True


# =============================================================== 入口 / 自检
def make_app() -> QApplication:
    """建 QApplication：字体 + 深色主题一次配好（界面只在这里定全局样式）。"""
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("TB6-R5-RevA1 交互控制台")
    font = QFont(pick_font())
    font.setPointSizeF(10.0)
    app.setFont(font)
    app.setStyleSheet(QSS)
    return app


def _fmt_mm(x) -> str:
    return f"{float(x) * 1000.0:.2f} mm"


def _fmt_deg(x) -> str:
    return f"{float(x):.3f}°"


def run_selftest(args) -> int:
    """无窗口自检：三种控制方式（单关节 / 关节 P2P / 笛卡尔直线 + IK）+ 渲染通路。"""
    print("=" * 78)
    print("revA1_gui.py 自检（不开窗口；只验证控制与渲染逻辑）")
    print("=" * 78)
    fails: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    sim = core.ArmSim(args.scene, move_seconds=args.move_time, log_scene=False)

    def run_until_idle(limit_s: float = 12.0) -> int:
        n, lim = 0, int(limit_s / sim.dt)
        while (sim.moving() or sim.pending is not None) and n < lim:
            sim.step()
            n += 1
        return n

    # 1) home 姿态
    err0 = float(np.linalg.norm(sim.tip() - np.asarray(spec.TARGET_POSE)))
    check("home 姿态：工具尖 = spec.TARGET_POSE", err0 < 2e-3,
          f"{_fmt_mm(err0)}  接触 {int(sim.data.ncon)} 对")
    check("home 姿态：无自碰撞", int(sim.data.ncon) == 0)

    # 2) 单关节 −/+ 按钮（就是 nudge_joint）
    q0 = sim.cmd_deg().copy()
    for _ in range(5):                            # 5 下 = +5°（默认步长 1°）
        sim.nudge_joint(3, +1.0)
    for _ in range(int(0.8 / sim.dt)):
        sim.step()
    delta = float(sim.actual_deg()[3] - q0[3])
    other = float(np.abs(sim.actual_deg() - q0)[[0, 1, 2, 4, 5]].max())
    check("单关节：5 下「+」把 J4 送到 +5°", abs(delta - 5.0) < 0.8, f"实测 {delta:+.2f}°")
    check("单关节：其余关节基本不动", other < 0.8, f"最大变化 {other:.3f}°")
    for _ in range(5):
        sim.nudge_joint(3, -1.0)
    check("单关节：5 下「−」能原样退回来",
          abs(float(sim.cmd_deg()[3]) - float(q0[3])) < 1e-9)
    sim.set_joint(3, 9.0)                          # 顶到限位：必须被夹住
    check("单关节：目标角被关节限位夹住",
          float(sim.cmd[3]) <= float(sim.hi[3]) + 1e-12,
          f"输入 9 rad → 目标 {np.degrees(sim.cmd[3]):.1f}°"
          f"（限位 {np.degrees(sim.hi[3]):.1f}°）")
    sim.reset_model(log_it=False)

    # 3) IK：四个预设点都要能解
    worst_p, worst_a = 0.0, 0.0
    for name, pos, axis in POSE_PRESETS:
        res = sim.solve_ik(pos, axis)
        worst_p = max(worst_p, res.pos_err_m)
        worst_a = max(worst_a, res.axis_err_rad)
        if not res.ok:
            check(f"IK 预设「{name}」", False, res.summary())
    check("IK 预设（home / 前伸 / 侧向 / 低位）全部可解",
          worst_p <= core.IK_TOL_POS_M and worst_a <= core.IK_TOL_AXIS_RAD,
          f"最差 {_fmt_mm(worst_p)} / {_fmt_deg(np.degrees(worst_a))}")

    # 4) 不可达目标：必须明确报"不可用"，而且不许动
    far = np.array([1.30, 0.0, 0.30])
    res_far = sim.solve_ik(far, (0, 0, -1))
    check("不可达目标：明确判为不可用 + 给出原因",
          (not res_far.ok) and bool(res_far.reason), res_far.summary())
    sim.goto_pose(far, (0, 0, -1), args.move_time, execute=True)
    check("不可达目标：不会偷偷开始运动", sim.motion is None)

    # 5) 笛卡尔直线：规划直线度 / 执行到位精度
    #    目标挑"从当前姿态这一支沿直线走得到"的点（新 home 是前倾姿态，
    #    把工具轴摆成竖直朝下往往要换解支 → 那种直线会被明确拒绝，见 5b）
    tgt = np.array([0.25, 0.10, 0.30])
    plan = sim.plan_line(tgt, (0, 0, -1))
    check("直线规划：路点严格落在直线上（直线度 = 0）",
          plan.straightness() < 1e-9 and plan.n >= core.LINE_MIN_POINTS and plan.ok,
          f"{plan.n} 个路点，直线度 {plan.straightness() * 1000:.6f} mm，"
          f"规划 {plan.ms:.0f} ms")
    check("直线规划：终点 IK 残差够小", plan.pos_err_mm < 0.5 and plan.ok,
          f"{plan.pos_err_mm:.2f} mm / {plan.axis_err_deg:.2f}°")

    sim.start_cartesian(tgt, (0, 0, -1), args.move_time)
    p0, p1 = plan.points[0], plan.points[-1]
    d = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-12)
    dev, n = 0.0, 0
    while sim.moving() and n < int(12.0 / sim.dt):
        sim.step()
        n += 1
        rel = sim.tip() - p0
        dev = max(dev, float(np.linalg.norm(rel - (rel @ d) * d)))
    run_until_idle()
    rep = sim.last_report or {}
    check("直线执行：实际轨迹基本贴着直线（含重力下垂 / 伺服跟踪）",
          dev < 0.012, f"最大偏离 {dev * 1000:.2f} mm（{n} 步）")
    check("直线执行：到位精度", rep.get("pos_err_m", 9e9) < 0.015,
          f"{_fmt_mm(rep.get('pos_err_m', float('nan')))}，"
          f"末端 {vec_str(rep.get('tip', np.zeros(3)))}")
    check("直线执行：工具轴保持竖直朝下", (rep.get("axis_err_deg") or 9e9) < 2.0,
          _fmt_deg(rep.get("axis_err_deg", float("nan"))))
    check("直线执行：关节跟踪误差", rep.get("joint_err_deg", 9e9) < 1.0,
          _fmt_deg(rep.get("joint_err_deg", float("nan"))))

    # 5b) 这一支沿直线走不到的目标：必须**明确拒绝**，而不是换解支把工具甩出去
    bad = np.array([0.52, 0.18, 0.34])
    plan_bad = sim.plan_line(bad, (0, 0, -1))
    check("直线规划：本支到不了的目标 → 判为不 OK 且给出原因（不许偷偷换支）",
          (not plan_bad.ok) and bool(plan_bad.reason) and plan_bad.failed_at >= 0,
          f"{plan_bad.n} 点，{plan_bad.reason}")
    sim.start_cartesian(bad, (0, 0, -1), args.move_time)
    check("直线规划：不 OK 的直线不会开始运动", sim.motion is None)

    # 6) 关节空间 P2P（归零）+ 急停
    sim.zero(1.2)
    check("关节空间 P2P：归零动作已发出（终点 = 0）",
          sim.motion is not None and bool(np.allclose(sim.motion.q_end, 0.0, atol=1e-12)))
    run_until_idle()
    check("关节空间 P2P：到位误差（含重力下垂的稳态偏差）",
          (sim.last_report or {}).get("joint_err_deg", 9e9) < 1.5,
          _fmt_deg((sim.last_report or {}).get("joint_err_deg", float("nan"))))
    sim.start_p2p(np.radians([0.0, -60.0, -60.0, -10.0, 60.0, -30.0]), 1.5)
    for _ in range(int(0.3 / sim.dt)):
        sim.step()
    sim.hold()
    cmd_hold = sim.cmd.copy()
    for _ in range(int(0.4 / sim.dt)):
        sim.step()
    p_settle = sim.tip().copy()
    for _ in range(int(0.5 / sim.dt)):
        sim.step()
    drift = float(np.linalg.norm(sim.tip() - p_settle))
    check("急停：目标指令定格不再前进", bool(np.array_equal(sim.cmd, cmd_hold)))
    check("急停：停下后 0.5 s 不漂移", drift < 2e-3, f"漂移 {_fmt_mm(drift)}")
    sim.reset_model(log_it=False)

    # 7) 离线渲染 + 画面标记
    try:
        r = mujoco.Renderer(sim.model, 320, 480)
        r.update_scene(sim.data)
        n0 = int(r.scene.ngeom)
        r.update_scene(sim.data)
        markers = core.add_marker(r.scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.022, 0, 0), tgt,
                                  np.eye(3).reshape(-1), (0.15, 0.95, 0.35, 0.85))
        markers = core.add_marker(r.scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                  (0.0016, 0.02, 0.0), (0.45, 0.05, 0.25),
                                  np.eye(3).reshape(-1), (0.25, 0.85, 1.0, 0.5)) and markers
        img = r.render()
        check("离屏渲染出图", img.shape == (320, 480, 3) and float(img.mean()) > 3.0,
              f"shape={img.shape} 平均亮度={float(img.mean()):.1f}")
        check("画面标记：目标球 + 路径胶囊能塞进场景",
              bool(markers) and int(r.scene.ngeom) == n0 + 2,
              f"ngeom {n0} -> {int(r.scene.ngeom)}")
        runs = RUNS
        runs.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio                    # 自检不依赖 Qt / Pillow
        imageio.imwrite(str(runs / "gui_selftest.png"), img)
        print(f"       存图：{runs / 'gui_selftest.png'}")
        r.close()                                  # ⚠️ 不 close 会耗光 GL 上下文
    except Exception as e:  # noqa: BLE001
        check("离屏渲染出图", False, f"异常 {e!r}")

    # 8) 相机 / 姿态插值
    cam = core.Camera()
    a0 = float(cam.mjv.azimuth)
    cam.orbit(10.0, 5.0)
    check("相机：拖动改变方位 / 俯仰", abs(float(cam.mjv.azimuth) - a0) > 1.0)
    lk = np.array(cam.mjv.lookat).copy()
    cam.pan(20.0, -10.0)
    check("相机：平移改变 lookat", float(np.linalg.norm(np.array(cam.mjv.lookat) - lk)) > 1e-4)
    d0 = float(cam.mjv.distance)
    cam.zoom(3.0)
    cam.zoom(-3.0)
    check("相机：推拉可逆", abs(float(cam.mjv.distance) - d0) < 1e-9)
    cam.preset("俯视")
    check("相机：预设视角生效", abs(float(cam.mjv.elevation) + 68.0) < 1e-6,
          f"俯视 elevation = {float(cam.mjv.elevation):.1f}°")
    for a, b, t in (((0, 0, -1), (1, 0, 0), 0.5), ((0, 0, -1), (0, 0, 1), 0.25)):
        mid = core.interp_axis(a, b, t)
        check(f"工具轴插值 {a} → {b} @ t={t} 仍是单位向量",
              abs(float(np.linalg.norm(mid)) - 1.0) < 1e-9)
    check("工具轴插值：t=0 / t=1 两端精确",
          bool(np.allclose(core.interp_axis((0, 0, -1), (1, 0, 0), 0.0), (0, 0, -1)))
          and bool(np.allclose(core.interp_axis((0, 0, -1), (1, 0, 0), 1.0), (1, 0, 0))))

    # 9) 真机跟随全链路：假真机（HTTP，带 MuJoCo 正运动学算的 pose）→ JointLink → ArmSim
    q_real = np.array([0.35, -1.30, -2.10, -0.60, 1.45, -2.10])
    q_wave = np.radians([3.0, -3.0, 2.0, -2.0, 2.0, -2.0])

    def q_fn(t):
        """真机在慢慢动（±3°、0.3 Hz）：要跟得住**运动**，不只是停在某一点。"""
        return q_real + q_wave * math.sin(2.0 * math.pi * 0.3 * t)

    fk_model = mujoco.MjModel.from_xml_path(str(spec.SCENE_XML))
    fk_data = mujoco.MjData(fk_model)
    fk_site = mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")

    def pose_fn(qq):
        """用模型正运动学造一帧真机 pose（位置 + 外旋 XYZ 欧拉角），误差链也能一起测。"""
        fk_data.qpos[:6] = qq
        mujoco.mj_forward(fk_model, fk_data)
        R = fk_data.site_xmat[fk_site].reshape(3, 3)
        ry = math.asin(-float(np.clip(R[2, 0], -1.0, 1.0)))
        rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
        rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
        return list(fk_data.site_xpos[fk_site]) + [rx, ry, rz]

    server = link.FakeStateServer(q_fn=q_fn, pose_fn=pose_fn).start()
    follow_sim = core.ArmSim(args.scene, move_seconds=args.move_time, log_scene=False)
    lk = link.JointLink(source="http", http_url=server.url, smooth=0.0, timeout=2.0)
    lk.start()
    t_end = time.perf_counter() + 2.5
    while time.perf_counter() < t_end:
        out = lk.update(follow_sim.cmd)
        if out.ok:
            follow_sim.follow(out.q_rad)
        follow_sim.step()
    server.stop()
    now = time.time()
    err_deg = float(np.degrees(np.abs(follow_sim.q_pos() - q_fn(now))).max())
    check("真机跟随：仿真关节实时跟上真机（运动中 < 3°）", err_deg < 3.0,
          f"最大差 {err_deg:.3f}°（真机此刻 {np.round(np.degrees(q_fn(now)), 1).tolist()}°）")
    check("真机跟随：仿真进入 follow 模式", follow_sim.mode == "follow", follow_sim.mode)
    tcp_err = float(np.linalg.norm(follow_sim.tip() - np.asarray(pose_fn(q_fn(now))[:3]))) * 1000.0
    check("真机跟随：工具尖 ↔ 真机 TCP < 8 mm", tcp_err < 8.0, f"{tcp_err:.2f} mm")
    check("真机跟随：数据率 > 10 Hz", lk.rate_hz() > 10.0, lk.rate_text())
    txt = follow_status_text(follow_sim, lk, out, "")
    check("真机跟随：卡片体检文本成形（延迟 / 包率 / 误差）",
          ("Hz" in txt) and ("延迟" in txt) and ("TCP" in txt) and ("工具轴" in txt),
          txt.splitlines()[1] if len(txt.splitlines()) > 1 else txt)
    lk.stop()
    check("真机跟随：停止后接收线程退出", not lk.running and not lk.sources)

    # 10) 点云 → 场景（点数压小一点，别拖慢自检）
    try:
        res_pc = pc.build_cloud_scene(max_points=2000, bands=3, point_mm=5.0,
                                      scene_out=pc.ASSETS / "revA1_pc_scene_selftest.xml")
        sim_pc = core.ArmSim(res_pc["scene"].scene, log_scene=False)
        names = [mujoco.mj_id2name(sim_pc.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                 for i in range(sim_pc.model.ngeom)]
        n_pc = sum(1 for n in names if n and n.startswith("pc_pts"))
        check("点云：ArmSim 能加载「机械臂 + 点云 + 小车」场景",
              sim_pc.model.nq == 6 and n_pc >= 3 and "pc_cart" in names,
              f"ngeom={sim_pc.model.ngeom}（点云几何 {n_pc} 个）")
        check("点云：地面按小车高度下移（ArmSim 读到 floor_z）",
              abs(sim_pc.floor_z + 1.1) < 1e-6, f"floor_z = {sim_pc.floor_z:.4f} m")
        check("点云：离地高度 = 工具尖 z − 场景地面 z",
              abs(sim_pc.tip_height() - (sim_pc.tip()[2] - sim_pc.floor_z)) < 1e-9,
              f"{sim_pc.tip_height():.3f} m")
    except pc.PointCloudError as exc:
        check("点云：全链路（点云 → 场景 → ArmSim）", False, str(exc))

    print("-" * 78)
    if fails:
        print(f"结论：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结论：全部通过 [OK]")
    return 0


def run_ui_test(args) -> int:
    """真建窗口 → 脚本化点一遍控件（按钮 + 键盘）→ 存图 → 退出（``--ui-test``）。"""
    print("=" * 78)
    print("revA1_gui.py UI 自检：建窗口 + 脚本化点一遍控件 + 存图")
    print("=" * 78)
    make_app()
    fails: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    RUNS.mkdir(parents=True, exist_ok=True)
    sim = core.ArmSim(args.scene, move_seconds=args.move_time, log_scene=False)
    win = RevA1Window(sim, width=args.width, height=args.height, fps=args.fps,
                      render=not args.no_render)
    win.show()
    win.pump(0.3)
    panel = win.panel

    check("窗口标题包含机型", "RevA1" in win.windowTitle(), win.windowTitle())
    check("关节控件 6 行", len(panel.joint_rows) == 6)
    check("状态栏有内容", len(win.st_mode.text()) > 6 and len(win.st_tip.text()) > 10,
          win.st_mode.text() + " ｜ " + win.st_tip.text())

    # 关节 −/+ 按钮：真的点按钮（走信号链路，不是直接调函数）
    q0 = sim.cmd_deg().copy()
    panel.step_box.setCurrentIndex(0)                     # 步长 0.5°
    panel.joint_rows[2]["plus"].click()
    check("点 J3「+」：目标角 +0.5°",
          abs(float(sim.cmd_deg()[2] - q0[2]) - 0.5) < 1e-6,
          f"{q0[2]:+.2f}° → {sim.cmd_deg()[2]:+.2f}°")
    panel.joint_rows[2]["minus"].click()
    check("点 J3「−」：目标角回到原值", abs(float(sim.cmd_deg()[2] - q0[2])) < 1e-9)
    win.pump(0.3)
    check("面板刷新：目标角 / 实测角标签有值",
          "°" in panel.joint_rows[2]["val"].text()
          and "°" in panel.joint_rows[2]["act"].text(),
          panel.joint_rows[2]["val"].text() + " " + panel.joint_rows[2]["act"].text())
    check("点 −/+ 会顺手选中该关节（高亮）", panel.selected == 2)

    # 键盘
    QTest.keyClick(win, Qt.Key_5)
    check("键盘 5：选中 J5", panel.selected == 4)
    q5 = float(sim.cmd_deg()[4])
    QTest.keyClick(win, Qt.Key_Equal)
    check("键盘 =：选中关节 +0.5°", abs(float(sim.cmd_deg()[4]) - q5 - 0.5) < 1e-6)
    q_before = sim.cmd.copy()
    for key in (Qt.Key_Shift, Qt.Key_F1, Qt.Key_Insert, Qt.Key_CapsLock):
        QTest.keyClick(win, key)                          # 空 char 的功能键不能触发动作
    check("功能键（Shift / F1 / Insert）不触发动作", bool(np.array_equal(sim.cmd, q_before)))
    QTest.keyClick(win, Qt.Key_2)
    QTest.keyClick(win, Qt.Key_Minus)
    check("键盘 −：选中关节 −0.5°",
          panel.selected == 1 and abs(float(sim.cmd_deg()[1] - q0[1]) + 0.5) < 1e-6,
          f"J2 {q0[1]:+.2f}° → {sim.cmd_deg()[1]:+.2f}°")

    # IK 求解（目标挑"从当前姿态这一支沿直线走得到"的点；
    # 新 home 是前倾姿态，把工具轴摆竖直朝下往往要换解支，那种直线会被拒绝，见下面那条）
    for spin, v in zip(panel.pos_spins, (0.25, 0.10, 0.30)):
        spin.setValue(v)
    panel.on_solve()
    check("「求解 IK」结果显示可解", "可解" in panel.ik_label.text(),
          panel.ik_label.text().splitlines()[0])
    check("IK 结果带误差 / 关节角 / 直线信息",
          ("mm" in panel.ik_label.text()) and ("q =" in panel.ik_label.text()))

    # 求解并沿直线运动
    panel.on_move_to_target()
    win.pump(args.move_time + 2.5)
    err = float(np.linalg.norm(sim.tip() - np.array([0.25, 0.10, 0.30])))
    check("「求解并沿直线运动」后工具尖到位", err < 0.015, f"误差 {err * 1000:.2f} mm")
    check("日志里记了实测到位精度", "到位（cartesian）" in panel.log_text.toPlainText())

    # 本支沿直线走不到的目标：界面要明确拒绝 + 说明原因，而且**不许动**
    tip0 = sim.tip().copy()
    for spin, v in zip(panel.pos_spins, (0.52, 0.18, 0.34)):
        spin.setValue(v)
    panel.on_move_to_target()
    win.pump(1.0)
    moved = float(np.linalg.norm(sim.tip() - tip0))
    check("「求解并沿直线运动」：本支到不了的目标 → 拒绝并说明原因、工具尖不动",
          ("直线运动取消" in panel.log_text.toPlainText()) and moved < 0.02,
          f"工具尖只挪了 {moved * 1000:.1f} mm")

    # 不可达目标：界面必须说清楚
    panel.pos_spins[0].setValue(1.30)
    panel.on_solve()
    check("不可达点：结果显示不可用 + 原因", "不可用" in panel.ik_label.text(),
          panel.ik_label.text().splitlines()[0])
    panel.pos_spins[0].setValue(0.25)

    # 点位（示教 / 点到点）
    panel.record_point()
    check("记录点位：列表多一条", panel.point_list.count() == 1,
          panel.point_list.item(0).text() if panel.point_list.count() else "")
    panel.on_home()
    win.pump(args.move_time + 1.5)
    panel.goto_selected_point()
    win.pump(args.move_time + 2.5)
    p_err = float(np.linalg.norm(sim.tip() - panel.points[0]["tip"]))
    check("走到点位：工具尖回到记录点", p_err < 0.015, f"误差 {p_err * 1000:.2f} mm")
    panel.delete_point()
    check("删除点位：列表清空", panel.point_list.count() == 0)

    # 运行 / 伺服
    panel.pause_box.setChecked(True)
    check("暂停物理：窗口 paused = True", win.paused)
    panel.pause_box.setChecked(False)
    panel.kp_slider.setValue(150)
    check("kp 缩放滑块生效", abs(sim.kp_scale - 1.5) < 1e-9, f"kp_scale = {sim.kp_scale:.2f}")
    panel.kp_slider.setValue(100)
    panel.on_estop()
    check("急停：模式变成 hold（就地保持）", sim.mode == "hold" and sim.motion is None)
    panel.on_reset()
    win.pump(0.3)
    err_home = float(np.linalg.norm(sim.tip() - np.asarray(spec.TARGET_POSE)))
    check("复位模型：回到 home（静置 0.3 s 的重力下垂 ≈ 4 mm）", err_home < 6e-3,
          f"{err_home * 1000:.2f} mm")

    # 渲染 / 缩放 / 存图
    if not args.no_render:
        rend_before = win.view.renderer
        win.resize(1120, 720)
        win.pump(0.5)
        check("缩放窗口：不重建渲染器（GL 上下文只有一个）",
              win.view.renderer is rend_before and win.view.renderer is not None)
        check("渲染 fps > 0", win.r_fps > 0.0,
              f"{win.r_fps:.1f} fps（渲染 {win.view.render_ms:.0f} ms/帧）")
        check("渲染帧尺寸 = --width x --height",
              win.view.frame_rgb is not None
              and win.view.frame_rgb.shape[:2] == (args.height, args.width),
              f"{None if win.view.frame_rgb is None else win.view.frame_rgb.shape}")
    shot = RUNS / "gui_ui_test.png"
    win.grab().save(str(shot))
    check("窗口截图存图", shot.exists() and shot.stat().st_size > 5000,
          f"{shot}（{shot.stat().st_size} B）")
    if args.no_render:
        print("  [SKIP] 保存渲染帧到 PNG（--no-render，本来就没有渲染器）")
    else:
        check("保存渲染帧到 PNG", win.save_frame(RUNS / "gui_ui_test_view.png"))
    check("日志面板有内容", len(panel.log_text.toPlainText()) > 60,
          f"{len(panel.log_text.toPlainText())} 字符")

    # 真机跟随卡片：控件齐全 + 启停不崩（用 UDP 收一个没人发的端口，不碰真机）
    check("真机跟随卡片：控件齐全（数据源 / 地址 / 端口）",
          panel.f_source.count() == len(link.SOURCE_LABELS)
          and panel.f_port.value() == int(link.DEFAULT_UDP_PORT)
          and panel.f_url.text() == link.DEFAULT_HTTP_URL)
    panel.f_source.setCurrentIndex(link.SOURCES.index("udp"))
    panel.f_port.setValue(link.free_udp_port())
    panel.on_follow_start()
    win.pump(0.4)
    check("真机跟随：启动后接收线程在跑",
          panel.link is not None and panel.link.running,
          panel.link.describe_sources() if panel.link is not None else "")
    txt = panel.follow_label.text()
    check("真机跟随：没数据时卡片给提示（不崩）",
          ("还没收到数据" in txt) or ("看门狗" in txt), txt.splitlines()[0] if txt else "")
    panel.on_follow_stop()
    win.pump(0.2)
    check("真机跟随：停止后链路释放、姿态就地保持",
          panel.link is None and win.sim.mode == "hold", win.sim.mode)

    # 点云卡片：默认值 + 加载/清除各一遍
    check("点云卡片：默认值（自动判坐标系 / 小车 1.10 m / 6 万点 / TCP / 6 带 / 有点云文件）",
          abs(panel.c_cart.value() - 1.10) < 1e-9 and panel.c_points.value() == 60000
          and panel.c_frame.currentIndex() == 0 and panel.c_ref.currentIndex() == 0
          and panel.c_bands.value() == 6 and panel.c_cloud.count() >= 1)
    check("点云卡片：小车中心默认 (+150, +150) mm（基座靠车头左前缘）",
          abs(panel.c_cart_x.value() - 150.0) < 1e-9 and abs(panel.c_cart_y.value() - 150.0) < 1e-9,
          f"({panel.c_cart_x.value():.0f}, {panel.c_cart_y.value():.0f}) mm")

    # 「作业点标记」= 那个绿色圆盘：默认不显示，勾上就回来（任何场景都管用）
    gid_pad = mujoco.mj_name2id(win.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "target_pad")
    check("作业点标记：默认不勾 → 绿圆盘被藏起来",
          (not panel.c_target_pad.isChecked()) and not win.sim.geom_visible("target_pad"),
          f"geom_visible = {win.sim.geom_visible('target_pad')}")
    z_pad0 = float(win.sim.hidden_geoms["target_pad"][0][2]) \
        if "target_pad" in win.sim.hidden_geoms else float("nan")     # 藏之前的原位（地面上）
    panel.c_target_pad.setChecked(True)
    win.pump(0.2)
    z_pad1 = float(win.sim.model.geom_pos[gid_pad][2]) if gid_pad >= 0 else float("nan")
    check("作业点标记：勾上 → 绿圆盘回到原位（地面上）",
          win.sim.geom_visible("target_pad") and abs(z_pad1 - z_pad0) < 1e-9,
          f"z {z_pad0:.4f} → {z_pad1:.4f} m")
    panel.c_target_pad.setChecked(False)
    win.pump(0.2)

    panel.c_points.setValue(3000)
    panel.c_bands.setValue(3)
    panel.on_cloud_load()
    win.pump(0.5)
    names = [mujoco.mj_id2name(win.sim.model, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(win.sim.model.ngeom)]
    n_pc = sum(1 for n in names if n and n.startswith("pc_pts"))
    check("点云：加载后场景里真的有点云几何",
          panel.cloud_scene is not None and n_pc >= 3,
          f"点云几何 {n_pc} 个 · {Path(panel.cloud_scene).name if panel.cloud_scene else ''}")
    check("点云：小车按「小车中心」摆放、且绿圆盘依然不显示",
          abs(float(win.sim.model.geom_pos[mujoco.mj_name2id(win.sim.model,
                                                              mujoco.mjtObj.mjOBJ_GEOM,
                                                              "pc_cart")][0]) - 0.15) < 1e-6
          and not win.sim.geom_visible("target_pad"),
          f"pc_cart x = {float(win.sim.model.geom_pos[mujoco.mj_name2id(win.sim.model, mujoco.mjtObj.mjOBJ_GEOM, 'pc_cart')][0]):.3f} m"
          f"（= 基座在小车上靠 −x 150 mm）")
    check("点云：地面挪到小车脚下", abs(win.sim.floor_z + 1.1) < 1e-6,
          f"floor_z = {win.sim.floor_z:.3f} m")
    txt = panel.cloud_label.text()
    check("点云：卡片上有体检报告（相机位置 / 基座系 / 离地）",
          ("相机位置" in txt) and ("基座系" in txt), txt.splitlines()[0] if txt else "")
    panel.on_cloud_clear()
    win.pump(0.3)
    check("点云：清除后回到干净场景",
          panel.cloud_scene is None and abs(win.sim.floor_z + 0.171217) < 2e-3,
          f"floor_z = {win.sim.floor_z:.4f} m")

    win.close()
    print("-" * 78)
    if fails:
        print(f"结论：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结论：全部通过 [OK]")
    return 0


def run_gui(args) -> int:
    """开真正的窗口（``mainloop``）。"""
    make_app()
    sim = core.ArmSim(args.scene, kp_scale=args.kp_scale, gravity=not args.no_gravity,
                      move_seconds=args.move_time)
    win = RevA1Window(sim, width=args.width, height=args.height, fps=args.fps,
                      render=not args.no_render, exit_after=args.exit_after)
    win.show()
    print(f"关节顺序 : {spec.JOINTS}")
    print(f"home qpos: {np.round(spec.HOME_QPOS, 4)}")
    print("鼠标     : 左键拖动=转视角  中键拖动=平移  滚轮=推拉  双击=复位视角")
    print("键盘     : 空格=暂停  H=回 home  R=复位  G=重力  F=真机跟随  1..6=选关节  "
          "−/=±一个步长  Ctrl+S=存图  ESC=退出")
    print("面板     : ① 关节 −/+ 微调  ② 真机跟随（UDP 广播 / HTTP 状态）  ③ 点云（相机 → 机械臂）"
          "  ④ 末端目标 / IK  ⑤ 点位（示教点到点）  ⑥ 运行 / 伺服  ⑦ 日志")
    if args.follow_source:
        win.panel.f_source.setCurrentIndex(link.SOURCES.index(args.follow_source))
    if args.follow_url:
        win.panel.f_url.setText(args.follow_url)
    if args.follow_port:
        win.panel.f_port.setValue(int(args.follow_port))
    if args.follow:
        win.panel.on_follow_start()
    cloud = getattr(args, "point_cloud", None)
    if cloud is None:                                  # 没给参数 → 用 config 里配好的点云
        groups = pc.load_cloud_groups()
        if groups:
            cloud = MULTI_CLOUD
            print(f"点云     : config.json 的 {len(groups)} 组 / "
                  f"{sum(len(g['files']) for g in groups)} 份点云（每组按自己的拍摄姿态一起渲染）")
        else:
            try:
                cloud = str(pc.default_cloud())
                print(f"点云     : {cloud}（来自 config.json，可用 --point-cloud 覆盖）")
            except pc.PointCloudError as exc:
                cloud = ""
                win.sim.log(f"配置里没有可用点云，先不加载：{exc}")
    if cloud:
        win.panel.set_cloud_file(cloud)
        win.panel.on_cloud_load()
    if getattr(args, "task_port", 0):
        win.panel.t_task_port.setValue(int(args.task_port))
    if getattr(args, "task_listen", False):
        win.panel.on_task_start()
    return QApplication.instance().exec() if not args.exit_after else _exec(win)


def _exec(win: RevA1Window) -> int:
    """带 ``--exit-after`` 的运行：跑完自动退出（自动化 / 截图用）。"""
    app = QApplication.instance()
    win._manual = True                     # 手动推进，定时器就别再插一脚
    while not win.closed:
        app.processEvents()
        win._advance()
        time.sleep(0.005)
    win.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="TB6-R5-RevA1 交互控制台（PySide6 + MuJoCo）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scene", default=str(spec.SCENE_XML), help="场景 XML")
    ap.add_argument("--width", type=int, default=960, help="离屏渲染宽度[px]")
    ap.add_argument("--height", type=int, default=600, help="离屏渲染高度[px]")
    ap.add_argument("--fps", type=float, default=30.0, help="画面刷新上限[fps]")
    ap.add_argument("--kp-scale", type=float, default=1.0, help="启动时的 kp/kv 缩放")
    ap.add_argument("--move-time", type=float, default=1.5, help="默认运动时长[s]")
    ap.add_argument("--no-gravity", action="store_true", help="启动时关掉重力")
    ap.add_argument("--no-render", action="store_true", help="不建离屏渲染器（没有 GL 时用）")
    ap.add_argument("--selftest", action="store_true", help="无窗口自检（控制逻辑 + 渲染通路）")
    ap.add_argument("--ui-test", action="store_true", help="建窗口 → 脚本化点一遍控件 → 存图 → 退出")
    ap.add_argument("--exit-after", type=float, default=0.0,
                    help="开窗口跑这么多秒后自动退出（自动化 / 截图用）")
    ap.add_argument("--follow", action="store_true",
                    help="启动时自动开「真机跟随」（用卡片里的默认值：UDP 6001 + HTTP 8080）")
    ap.add_argument("--follow-source", default="", choices=("",) + link.SOURCES,
                    help="覆盖跟随的数据源（auto/http/udp）")
    ap.add_argument("--follow-url", default="", help="覆盖真机 HTTP 状态接口")
    ap.add_argument("--follow-port", type=int, default=0, help="覆盖真机 UDP 广播端口")
    ap.add_argument("--point-cloud", default=None,
                    help="启动时接进场景的点云文件；**不给这个参数 = 用 config.json 里配的**"
                         "（point_cloud_groups / point_cloud_file），传空串 \"\" = 不加载点云（干净场景）")
    ap.add_argument("--task-listen", action="store_true",
                    help="启动时自动开「任务信号」监听（默认 UDP 6501）")
    ap.add_argument("--task-port", type=int, default=0, help="覆盖任务信号 UDP 端口")
    ap.add_argument("--snapshot-dir", default=str(here / "runs"), help="存图目录")
    args = ap.parse_args(argv)

    global RUNS
    RUNS = Path(args.snapshot_dir)

    if args.selftest:
        return run_selftest(args)
    if args.ui_test:
        return run_ui_test(args)
    return run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
















