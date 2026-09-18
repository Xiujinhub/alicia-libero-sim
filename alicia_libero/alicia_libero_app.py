#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alicia-D × LIBERO 桌面操作仿真台（PySide6 界面）。

功能
----
* 左侧下拉框选择任务（任务来自对 ``Datasets/libero_assets`` 的自动分析），
  切换任务即重新生成并加载 MuJoCo 场景
* 中间 3D 视图：用 ``mujoco.Renderer`` 离屏渲染成 QImage 显示（不另开窗口）
* 三种操作方式：
  1. **鼠标拖拽**（推荐）：按住鼠标左键在视图里拖动 = 拖动夹爪目标点，IK 实时求解
  2. **键盘虚拟示教臂**：``1..6`` 选关节、``,`` ``.`` 微调、``o``/``c`` 开关夹爪
  3. **关节滑块**：6 个滑块直接给关节目标角
* 实时判据：任务的完成条件（目标区域、水平偏差、落座高度）每帧计算并显示
 * 相机切换、目标区域显示开关、场景复位

动作规划状态
------------
整个动作规划（接近/抓取/搬运/释放高度反算、辅助按钮、▶ 执行动作计划）
已从本文件与工程中删除，等待重写。当前界面只提供**手动操作 + 实时判分**。

运行
----
    python alicia_libero_app.py            # 或双击 run_app.bat
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from alicia_ik import AliciaIK, finger_targets  # noqa: E402
from libero_catalog import catalog_object, load_catalog  # noqa: E402
from libero_scene import TABLE_TOP_Z  # noqa: E402
from libero_tasks import TASKS, build_task_scene, check_success, sample_spawn  # noqa: E402
import skills  # noqa: E402  （闭环技能库：接触/跟随判据驱动的自动执行）

VIEW_W, VIEW_H = 900, 640          # 离屏渲染分辨率
JOINT_STEP_MAX_DEG = 8.0           # 一帧内单个关节最多动多少度（IK 限速，防"换臂形甩臂"）
CAMERAS = ["cam_front", "cam_top", "cam_side", "cam_wrist"]
CAMERA_LABELS = ["斜前方", "正上方", "侧前方", "腕部相机"]


class SimSession:
    """一次任务场景的仿真会话：模型 / 数据 / IK / 渲染器 / 目标点。"""

    def __init__(self, task: dict, catalog: dict, render: bool = True,
                 spawn_seed: int | None = None) -> None:
        """``render=False`` 时不建 MuJoCo 离屏渲染器（无头批量试验用，快很多）。

        ``spawn_seed``：带 ``spawn_region`` 的任务（当前是 t1）随机安放**抓取物和目标物**
        用的种子；给同一个种子就复现同一串随机位置（基准脚本用它保证可复现），``None`` = 真随机。
        """
        self.task = task
        self.catalog = catalog
        xml, _ = build_task_scene(task, catalog=catalog)
        self.xml_path = xml
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.ik = AliciaIK(self.model)
        self.renderer = (mujoco.Renderer(self.model, height=VIEW_H, width=VIEW_W)
                         if render else None)
        # 物体碰撞盒在 group 3：界面里默认不渲染（不然会看到一堆灰色方块套在物体外面）
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[3] = 0
        self.arm_act = [self.model.actuator(f"pos{i + 1}").id for i in range(6)]
        self.grip_act = [self.model.actuator("pos_grip_l").id, self.model.actuator("pos_grip_r").id]
        self.marker_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ik_marker")
        self.joint_target = np.zeros(6)
        self.gripper = 1.0                     # 1 = 张开
        self.camera = "cam_front"
        self.grasp_yaw = self._grasp_yaw(task)
        # 随机安放（任务多样化）：随机源 + "AABB 中心 → body 原点"的 xy 偏移（见 _apply_spawn）
        self.spawn_rng = np.random.default_rng(spawn_seed)
        self.spawn: dict[str, tuple[float, float]] = {}
        self.spawn_offset: dict[str, np.ndarray] = {}
        for region_name in task.get("spawn_region", {}):
            entry = next((e for e in task["objects"]
                          if e.get("name", e["key"].split("/")[-1]) == region_name), None)
            if entry is None:
                continue
            adr = self.model.jnt_qposadr[self.model.joint(f"{region_name}_joint").id]
            self.spawn_offset[region_name] = (self.model.qpos0[adr:adr + 2]
                                              - np.asarray(entry["xy"], dtype=float))
        self.reset()

    def _grasp_yaw(self, task: dict) -> float:
        """用素材分析结果决定夹爪的闭合方向（偏航角）。

        ⚠ 踩坑：夹爪是平行开合，**必须沿物体的窄边闭合**。
        实测番茄酱 56×37mm，若沿 56mm 侧闭合，50mm 开口根本夹不上，
        搬运途中就掉了（最后掉到地上）。这里按 catalog 里 AABB 的较短边自动选方向：
        窄边在 X → 闭合轴沿 X（yaw=0）；窄边在 Y → 闭合轴沿 Y（yaw=90°）。
        """
        objects = {item["key"]: item for item in task["objects"]}
        item = objects.get(task["grasp_object"]) or next(
            (v for k, v in objects.items() if k.endswith("/" + task["grasp_object"])), None)
        obj_yaw = float(item.get("yaw", 0.0)) if item else 0.0
        obj = catalog_object(self.catalog, task["grasp_object"])
        narrow_is_x = obj["size"][0] <= obj["size"][1]
        return obj_yaw + (0.0 if narrow_is_x else 90.0)

    def reset(self, resample: bool = True) -> None:
        """回到任务初始状态（物体按 XML 初始位姿复位）。

        ``resample=True``（默认）：带 ``spawn_region`` 的抓取物**重新抽一个位置** ——
        所以界面上的"复位"= 换一局新摆位（任务多样化的入口）；``resample=False``：
        沿用本次会话已抽到的位置（可复现的实验用，例如基准脚本按种子生成的每一局）。
        """
        mujoco.mj_resetData(self.model, self.data)
        if resample:
            self.spawn = sample_spawn(self.task, self.spawn_rng, self.catalog)
        self._apply_spawn()
        self.joint_target = np.zeros(6)
        self.gripper = 1.0
        self.apply_control()
        mujoco.mj_forward(self.model, self.data)
        self.ee_target = self.tcp_position().copy()
        # 零位（6 关节目标全 0）下的末端位置：技能库"回程"的终点，见 skills.return_home
        self.home_tcp = self.tcp_position().copy()

    def _apply_spawn(self) -> None:
        """把随机抽到的 XY 写进物体自由关节。

        只动 x/y：z 与朝向沿用 XML（``SceneBuilder`` 已把底面精确摆在桌面上），
        所以瓶子还是立着的、偏航角也不变 —— 夹爪闭合方向不受随机化影响。
        """
        for name, (x, y) in self.spawn.items():
            offset = self.spawn_offset.get(name)
            try:
                adr = self.model.jnt_qposadr[self.model.joint(f"{name}_joint").id]
            except Exception:  # noqa: BLE001  物体不是自由关节就跳过
                continue
            self.data.qpos[adr] = x + (float(offset[0]) if offset is not None else 0.0)
            self.data.qpos[adr + 1] = y + (float(offset[1]) if offset is not None else 0.0)

    # ── 控制 ──
    def apply_control(self) -> None:
        for i, act in enumerate(self.arm_act):
            self.data.ctrl[act] = self.joint_target[i]
        left, right = finger_targets(self.model, self.gripper)
        self.data.ctrl[self.grip_act[0]] = left
        self.data.ctrl[self.grip_act[1]] = right

    def step(self, n_substeps: int = 8) -> None:
        self.apply_control()
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def tcp_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ik.site_id].copy()

    def set_ee_target(self, target, yaw: float | None = None) -> None:
        """给末端目标点做 IK，并把解写进关节目标。

        ``yaw``：夹爪闭合方向的偏航角（度）；None = 用本任务的 ``self.grasp_yaw``。
        技能库在抓取/推滑时会传入自己算的偏航角（例如"闭合轴沿推进方向"）。

        ⚠ 踩坑 1：IK 从"上一帧解"做种子时可能陷进局部极小（实测目标 z=0.892 只走到 0.948），
        所以位置误差偏大时改用多个种子重试，取误差最小的解。
        """
        target = np.asarray(target, dtype=float).copy()
        target[2] = float(np.clip(target[2], TABLE_TOP_Z + 0.015, TABLE_TOP_Z + 0.45))
        base_xy = self.model.body("base_link").pos[:2]
        radius = float(np.linalg.norm(target[:2] - base_xy))
        if radius > 0.62:                      # 超出舒适工作半径就拉回来
            direction = (target[:2] - base_xy) / radius
            target[:2] = base_xy + direction * 0.62
        self.ee_target = target

        orientation = self.ik.down_orientation(self.grasp_yaw if yaw is None else yaw)
        q_prev = self.joint_target.copy()
        # ⚠ 性能：IK 每轮迭代都调 mj_forward，而 mj_forward 会做**完整碰撞检测**
        # （2 种子 × 2 阶段 × 60 轮 ≈ 240 次/帧，实测 4~13 fps 的元凶）。
        # IK 只要运动学，所以整段求解期间临时关掉接触生成，算完立刻恢复。
        flags = self.model.opt.disableflags
        self.model.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_CONTACT
        try:
            best_q, best_score = None, np.inf
            for seed in (q_prev, np.zeros(6)):
                q, err = self.ik.solve(target, orientation, q_seed=seed, iters=60)
                # 代价 = 位置误差 + 关节空间偏离。
                # ⚠ 踩坑 2：零位附近是奇异点，**只按位置误差挑解会跳到"另一个臂形"**
                # （实测 J6 甩到 ±180°），整条臂跟着甩过去 —— "抬臂准备"时 TCP 反而先掉
                # 74mm、手指蹭到桌面再慢慢爬回来。加上关节偏离项后能稳定挑到连续解。
                score = float(err + 0.05 * float(np.linalg.norm(q - q_prev)))
                if score < best_score:
                    best_q, best_score = q, score
        finally:
            self.model.opt.disableflags = flags
        # 关节限速：一帧最多 8°，即使 IK 换臂形也不会让整条臂"甩"过去
        limit = np.radians(JOINT_STEP_MAX_DEG)
        self.joint_target = np.clip(best_q, q_prev - limit, q_prev + limit)
        if self.marker_body >= 0:
            self.data.mocap_pos[0] = target

    # ── 渲染与判分 ──

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera,
                                   scene_option=self.scene_option)
        return self.renderer.render()

    def success(self) -> tuple[bool, str]:
        return check_success(self.task, self.model, self.data, self.catalog)

    def object_report(self) -> str:
        name = self.task["grasp_object"]
        obj = catalog_object(self.catalog, name)
        text = (f"抓取物 {name}：{obj['size'][0]*1000:.0f}×{obj['size'][1]*1000:.0f}"
                f"×{obj['size'][2]*1000:.0f} mm，最窄处 {obj['body_width']*1000:.0f} mm，"
                f"50mm 夹爪{'可夹' if obj['graspable_strict'] else '偏宽（可改用侧面/边缘）'}")
        report = self.spawn_report()
        return text + ("；" + report if report else "")

    def spawn_report(self) -> str:
        """本局随机安放的位置（没有 ``spawn_region`` 的任务返回空串）。"""
        if not self.spawn:
            return ""
        parts = [f"{name} ({x * 1000:+.0f}, {y * 1000:+.0f}) mm"
                 for name, (x, y) in self.spawn.items()]
        return "本局随机安放 " + "、".join(parts)


class SimView(QLabel):
    """把 MuJoCo 渲染结果显示出来，并把鼠标拖动/滚轮转成末端目标点移动。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(VIEW_W // 2, VIEW_H // 2)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#1b1e24; border:1px solid #333a45;")
        self._last = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._last = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        self._last = None
        self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._last is None or self.pixmap() is None:
            return
        pos = event.position()
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        self.window().drag_end_effector(dx, dy, self.pixmap().width(), self.pixmap().height())

    def wheelEvent(self, event) -> None:
        window = self.window()
        steps = event.angleDelta().y() / 120.0
        if event.modifiers() & Qt.ShiftModifier:
            window.nudge_gripper(0.1 * steps)
        else:
            window.nudge_end_effector(steps)


class MainWindow(QMainWindow):
    """主窗口：左侧任务/操作面板，中间 3D 视图。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Alicia-D × LIBERO 桌面操作仿真台")
        self.catalog = load_catalog()
        self.session: SimSession | None = None
        self.auto: skills.SkillRunner | None = None
        self._auto_reported = False
        self.mode = "drag"
        self.selected_joint = 0
        self.success_count = 0
        self.was_success = False
        self._fps_t = time.perf_counter()
        self._fps = 0.0

        self._build_ui()
        self.callback_load_task(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    # ────────────────── 界面搭建 ──────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.view = SimView()
        root.addWidget(self.view, stretch=1)
        root.addWidget(self._build_side_panel(), stretch=0)
        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.addWidget(self._build_task_box())
        layout.addWidget(self._build_control_box())
        layout.addWidget(self._build_auto_box())
        layout.addWidget(self._build_view_box())
        layout.addWidget(self._build_help_box())
        layout.addStretch(1)
        return panel

    def _build_task_box(self) -> QGroupBox:
        box = QGroupBox("任务（来自 libero_assets 素材分析）")
        layout = QVBoxLayout(box)
        self.task_combo = QComboBox()
        for task in TASKS:
            self.task_combo.addItem(task["name"])
        self.task_combo.currentIndexChanged.connect(self.callback_load_task)
        layout.addWidget(self.task_combo)

        for attr, style in (
            ("task_text", "color:#d7e3f4;"),
            ("hint_text", "color:#8fa6c4; font-size:11px;"),
            ("object_text", "color:#7f8fa6; font-size:11px;"),
            ("judge_text", "color:#ffd479; font-size:11px;"),
            ("result_text", "font-size:13px; font-weight:bold;"),
        ):
            label = QLabel()
            label.setWordWrap(True)
            label.setStyleSheet(style)
            setattr(self, attr, label)
            layout.addWidget(label)
        return box

    def _build_control_box(self) -> QGroupBox:
        box = QGroupBox("操作方式")
        layout = QVBoxLayout(box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["鼠标拖拽（IK 跟随）", "键盘虚拟示教臂", "关节滑块"])
        self.mode_combo.currentIndexChanged.connect(self.callback_mode_changed)
        layout.addWidget(self.mode_combo)

        self.joint_sliders, self.joint_labels = [], []
        for i in range(6):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"J{i + 1}"))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-180, 180)
            slider.valueChanged.connect(self.callback_joint_slider)
            label = QLabel("0.0°")
            label.setFixedWidth(54)
            row.addWidget(slider, stretch=1)
            row.addWidget(label)
            layout.addLayout(row)
            self.joint_sliders.append(slider)
            self.joint_labels.append(label)

        grip_row = QHBoxLayout()
        grip_row.addWidget(QLabel("夹爪"))
        self.grip_slider = QSlider(Qt.Horizontal)
        self.grip_slider.setRange(0, 100)
        self.grip_slider.setValue(100)
        self.grip_slider.valueChanged.connect(self.callback_grip_slider)
        grip_row.addWidget(self.grip_slider, stretch=1)
        layout.addLayout(grip_row)

        return box

    def _build_auto_box(self) -> QGroupBox:
        """自动执行面板：一条按钮 + 状态行 + 每步明细。

        背后的 ``skills`` 是**闭环**的：每帧读实际接触/位置决定下一步，
        因此按钮只有"开始/暂停/单步"，没有"执行固定脚本"的说法。
        """
        box = QGroupBox("自动执行（闭环技能库 skills.py）")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.auto_button = QPushButton("▶ 自动完成本关")
        self.auto_button.clicked.connect(self.callback_auto_toggle)
        row.addWidget(self.auto_button)
        self.auto_step_button = QPushButton("单步")
        self.auto_step_button.clicked.connect(self.callback_auto_single)
        row.addWidget(self.auto_step_button)
        layout.addLayout(row)
        self.auto_status = QLabel("就绪")
        self.auto_status.setWordWrap(True)
        self.auto_status.setStyleSheet("color:#7fd1ff; font-size:11px;")
        layout.addWidget(self.auto_status)
        self.auto_result = QLabel("")
        self.auto_result.setWordWrap(True)
        self.auto_result.setStyleSheet("color:#cfd8e3; font-size:11px;")
        layout.addWidget(self.auto_result)
        return box

    def _build_view_box(self) -> QGroupBox:
        box = QGroupBox("视图 / 场景")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(CAMERA_LABELS)
        self.camera_combo.currentIndexChanged.connect(self.callback_camera)
        row.addWidget(QLabel("相机"))
        row.addWidget(self.camera_combo, stretch=1)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        reset_btn = QPushButton("重新开始（复位场景）")
        reset_btn.clicked.connect(self.callback_reset)
        buttons.addWidget(reset_btn)
        layout.addLayout(buttons)
        return box

    def _build_help_box(self) -> QGroupBox:
        box = QGroupBox("快捷键")
        layout = QVBoxLayout(box)
        help_text = QLabel(
            "鼠标左键拖动 = 移动夹爪目标点（IK 实时求解）\n"
            "滚轮 = 升降夹爪    Shift+滚轮 = 夹爪开合\n"
            "1..6 选关节    , / . 关节 ±2°    O / C 夹爪开合\n"
            "R 复位场景    T 切换相机\n"
            "自动执行 = 闭环技能库（接触/跟随判据），手动操作会自动暂停它"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color:#9fb3c8; font-size:11px;")
        layout.addWidget(help_text)
        return box

    # ────────────────── 回调 ──────────────────
    def callback_load_task(self, index: int) -> None:
        task = TASKS[index]
        self.statusBar().showMessage(f"正在加载任务 {task['id']} ……")
        try:
            if self.session is not None and self.session.renderer is not None:
                self.session.renderer.close()
            self.session = SimSession(task, self.catalog)
            self.auto = skills.SkillRunner(self.session)
            self._auto_reported = False
            self.auto_status.setText("就绪（点 ▶ 自动完成本关）")
            self.auto_result.setText("")
            self.auto_button.setText("▶ 自动完成本关")
        except Exception as exc:  # noqa: BLE001
            self.result_text.setText(f"❌ 场景加载失败：{exc}")
            return

        self.task_text.setText(task["task_text"])
        self.hint_text.setText("提示：" + task["hint"])
        self.object_text.setText(self.session.object_report())
        self.result_text.setText("状态：进行中")
        self.result_text.setStyleSheet("font-size:13px; font-weight:bold; color:#d7e3f4;")
        self.was_success = False
        self.mode_combo.setCurrentIndex(0)
        self.callback_mode_changed(0)
        self._sync_controls()
        self.statusBar().showMessage(
            f"已加载 {task['id']}（{task['kind']}，难度 {task['difficulty']}）"
            + ("；" + self.session.spawn_report() if self.session.spawn_report() else ""))

    def _sync_controls(self) -> None:
        """把界面控件同步到 session 当前状态。

        ⚠ 坑：换任务/复位后 session 的夹爪会回到 1.0、关节回 0，但滑块还停在旧值；
        不主动同步就会出现"滑块显示 0 而仿真以为全张开"的错位（实测按 C 无反应）。
        """
        if self.session is None:
            return
        for slider, value in zip(self.joint_sliders, self.session.joint_target):
            slider.blockSignals(True)
            slider.setValue(int(round(math.degrees(value))))
            slider.blockSignals(False)
        self.grip_slider.blockSignals(True)
        self.grip_slider.setValue(int(round(self.session.gripper * 100)))
        self.grip_slider.blockSignals(False)

    def callback_mode_changed(self, index: int) -> None:
        self.mode = ["drag", "keyboard", "slider"][index]
        for slider in self.joint_sliders:
            slider.setEnabled(self.mode == "slider")

    def callback_joint_slider(self, _value: int) -> None:
        if self.session is None or self.mode != "slider":
            return
        self.session.joint_target = np.radians(
            np.array([s.value() for s in self.joint_sliders], dtype=float))

    def callback_grip_slider(self, value: int) -> None:
        if self.session is not None:
            self.session.gripper = value / 100.0

    def callback_camera(self, index: int) -> None:
        if self.session is not None:
            self.session.camera = CAMERAS[index]

    def callback_reset(self) -> None:
        if self.session is not None:
            if self.auto is not None:
                self.auto.stop()
                self.auto_status.setText("已停止（场景复位）")
                self.auto_button.setText("▶ 自动完成本关")
            self.session.reset()
            self._sync_controls()
            self.object_text.setText(self.session.object_report())
            self.was_success = False
            self.result_text.setStyleSheet("font-size:13px; font-weight:bold; color:#d7e3f4;")
            self.result_text.setText("状态：进行中")
            spawn = self.session.spawn_report()
            self.statusBar().showMessage("场景已复位" + ("（" + spawn + "）" if spawn else ""))

    # ────────────────── 自动执行（闭环技能库） ──────────────────
    def _pause_auto(self) -> None:
        """手动接管（拖拽/微调/滑块/夹爪）时自动暂停，避免和技能抢控制权。"""
        if self.auto is not None and self.auto.active and not self.auto.paused:
            self.auto.pause()
            self.auto_button.setText("▶ 继续自动")
            self.statusBar().showMessage("检测到手动操作：自动执行已暂停")

    def callback_auto_toggle(self) -> None:
        if self.auto is None:
            return
        if not self.auto.active:
            self.auto.start()
            self._auto_reported = False
            self.auto_button.setText("⏸ 暂停自动")
            self.auto_status.setText("开始")
            self.auto_result.setText("")
            self.statusBar().showMessage("自动执行开始（每帧读接触/位置决定下一步）")
        elif self.auto.paused:
            self.auto.resume()
            self.auto_button.setText("⏸ 暂停自动")
        else:
            self.auto.pause()
            self.auto_button.setText("▶ 继续自动")

    def callback_auto_single(self) -> None:
        """单步：暂停状态下推进一帧，便于逐步观察每个判据。"""
        if self.auto is None:
            return
        if not self.auto.active:
            self.auto.start()
        self.auto.pause()
        self.auto.step()
        self.auto_button.setText("▶ 继续自动")
        self.auto_status.setText("单步：" + self.auto.status)

    # ────────────────── 鼠标拖拽 → IK ──────────────────
    def _screen_axes(self):
        """返回相机在世界系下的右/上方向，以及渲染图上"每像素多少米"。"""
        sess = self.session
        cam_id = mujoco.mj_name2id(sess.model, mujoco.mjtObj.mjOBJ_CAMERA, sess.camera)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, sess.model.cam_quat[cam_id])
        rot = rot.reshape(3, 3)
        right, up = rot[:, 0].copy(), rot[:, 1].copy()
        distance = float(np.linalg.norm(sess.model.cam_pos[cam_id] - sess.ee_target))
        fovy = math.radians(float(sess.model.vis.global_.fovy))
        return right, up, 2.0 * distance * math.tan(fovy / 2.0) / VIEW_H

    def drag_end_effector(self, dx_px: float, dy_px: float, disp_w: int, disp_h: int) -> None:
        if self.session is None or self.mode != "drag" or disp_w <= 0:
            return
        self._pause_auto()
        right, up, meters_per_px = self._screen_axes()
        meters_per_px *= VIEW_W / float(disp_w)        # 显示缩放折算回渲染像素
        delta = right * (dx_px * meters_per_px) + up * (-dy_px * meters_per_px)
        self.session.set_ee_target(self.session.ee_target + delta)

    def nudge_end_effector(self, steps: float) -> None:
        if self.session is None or self.mode != "drag":
            return
        self._pause_auto()
        target = self.session.ee_target.copy()
        target[2] += 0.012 * steps
        self.session.set_ee_target(target)

    def nudge_gripper(self, delta: float) -> None:
        """键盘/滚轮调夹爪：同样直接改 session 再同步控件，避免只动滑块漏改状态。"""
        if self.session is None:
            return
        self._pause_auto()
        self.session.gripper = float(np.clip(self.session.gripper + delta, 0.0, 1.0))
        self._sync_controls()

    def _nudge_joint(self, delta_deg: float) -> None:
        sess = self.session
        self._pause_auto()
        q_deg = np.degrees(sess.joint_target.copy())
        q_deg[self.selected_joint] += delta_deg
        sess.joint_target = np.radians(q_deg)
        for i, slider in enumerate(self.joint_sliders):
            slider.blockSignals(True)
            slider.setValue(int(q_deg[i]))
            slider.blockSignals(False)
        self.statusBar().showMessage(
            f"Joint{self.selected_joint + 1} → {q_deg[self.selected_joint]:+.1f}°")

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        if self.session is None:
            return
        text = event.text().lower()
        key = event.key()
        # ⚠ 坑：某些情况下（例如 QTest 模拟、部分输入法）event.text() 是空串，
        # 而 Python 里 "" in "123456" 恒为 True，会误入数字分支并 int("") 崩掉。
        if not text and 0x20 <= key < 0x7F:
            text = chr(key).lower()
        if text and text in "123456":
            self.selected_joint = int(text) - 1
            self.statusBar().showMessage(f"已选中 Joint{self.selected_joint + 1}")
        elif text == "," or key == Qt.Key_Left:
            self._nudge_joint(-2.0)
        elif text == "." or key == Qt.Key_Right:
            self._nudge_joint(+2.0)
        elif text == "o":
            self.nudge_gripper(+0.2)
        elif text == "c":
            self.nudge_gripper(-0.2)
        elif text == "r":
            self.callback_reset()
        elif text == "t":
            self.camera_combo.setCurrentIndex((self.camera_combo.currentIndex() + 1) % len(CAMERAS))
        else:
            super().keyPressEvent(event)

    # ────────────────── 主循环 ──────────────────
    def tick(self) -> None:
        sess = self.session
        if sess is None:
            return
        if self.auto is not None:                     # 闭环技能：先按当前状态定目标，再跑物理
            self.auto.step()
            self.auto_status.setText("自动：" + self.auto.status)
            if self.auto.finished and not self._auto_reported:
                self._auto_reported = True
                self.auto_result.setText(self.auto.summary())
                self.auto_button.setText("▶ 自动完成本关")
                self.statusBar().showMessage(
                    "自动执行结束：" + ("成功 ✅" if self.auto.ok else "未完成 ❌"))
        sess.step(8)                                  # dt=2ms × 8 ≈ 1/60 s
        frame = sess.render()
        image = QImage(frame.data, VIEW_W, VIEW_H, 3 * VIEW_W, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(pixmap)

        if self.mode != "slider":                     # 关节滑块回显当前关节角
            for i, (slider, label) in enumerate(zip(self.joint_sliders, self.joint_labels)):
                deg = float(np.degrees(sess.joint_target[i]))
                label.setText(f"{deg:+.1f}°")
                slider.blockSignals(True)
                slider.setValue(int(np.clip(deg, -180, 180)))
                slider.blockSignals(False)

        self.grip_slider.blockSignals(True)
        self.grip_slider.setValue(int(round(sess.gripper * 100)))
        self.grip_slider.blockSignals(False)

        ok, message = sess.success()
        self.judge_text.setText(message)
        if ok and not self.was_success:
            self.success_count += 1
            self.result_text.setStyleSheet(
                "font-size:13px; font-weight:bold; color:#3ddc84;")
            self.result_text.setText(f"✅ 任务完成！本次会话已完成 {self.success_count} 个任务")
        elif not ok and self.was_success:
            self.result_text.setStyleSheet(
                "font-size:13px; font-weight:bold; color:#ffd479;")
            self.result_text.setText("状态：进行中（物体被挪动了）")
        self.was_success = ok

        now = time.perf_counter()
        if now - self._fps_t >= 0.5:
            self._fps = 1.0 / max(now - self._fps_t, 1e-6)
            self._fps_t = now
        self.statusBar().showMessage(
            f"渲染 {self._fps:.0f} FPS | 任务 {sess.task['id']} | 已完成 {self.success_count}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

