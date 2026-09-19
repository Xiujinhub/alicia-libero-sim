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
* **连续任务（自动连跑）**：单任务连跑 N 局 / 多任务轮转（勾选若干关，一轮跑完再下一轮），
  每局自动重制场景（带随机安放的任务**重新抽摆位**），跑完给出逐局明细与成功率

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
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from alicia_ik import AliciaIK, finger_targets  # noqa: E402
from libero_catalog import (  # noqa: E402
    catalog_object,
    load_catalog,
    quat_from_axis_angle,
    quat_to_mat,
)
from libero_scene import TABLE_TOP_Z  # noqa: E402
from libero_tasks import (  # noqa: E402
    TASKS,
    as_spawn_pose,
    build_task_scene,
    check_success,
    placed_pose,
    sample_spawn,
    SpawnPose,
)
import skills  # noqa: E402  （闭环技能库：接触/跟随判据驱动的自动执行）

VIEW_W, VIEW_H = 900, 640          # 离屏渲染分辨率（单格；2×2 时整图是 2 倍加分隔线）
GRID_GAP = 2                       # 多视角拼接时格子之间的分隔线宽度（像素）
JOINT_STEP_MAX_DEG = 8.0           # 一帧内单个关节最多动多少度（IK 限速，防"换臂形甩臂"）
CAMERAS = ["cam_front", "cam_top", "cam_side", "cam_wrist"]
CAMERA_LABELS = ["斜前方", "正上方", "侧前方", "腕部相机"]
PANEL_W = 400                      # 左侧面板宽度
CONT_DWELL_FRAMES = 15             # 连续任务：一局跑完后停留多少帧再重制场景
                                   # （≈0.25s，让画面停在"结束状态"上，肉眼看得见成绩）
CONT_MAX_RUNS = 999                # 连续任务：次数/轮数上限（防手滑把界面卡死）


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
        self.spawn: dict[str, SpawnPose] = {}
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

    def _grasp_yaw(self, task: dict, quat=None) -> float:
        """用素材分析结果决定夹爪的闭合方向（偏航角，**按物体实际姿态算**）。

        ⚠ 踩坑：夹爪是平行开合，**必须沿物体的窄边闭合**。
        实测番茄酱 56×37mm，若沿 56mm 侧闭合，50mm 开口根本夹不上，
        搬运途中就掉了（最后掉到地上）。这里按 catalog 里 AABB 的较短边自动选方向：
        窄边在 X → 闭合轴沿 X（yaw=0）；窄边在 Y → 闭合轴沿 Y（yaw=90°）。

        姿态随机之后（§8.9：瓶子可能平放、还斜 45°）不能再按"世界 X/Y 谁短"猜 ——
        那是**包围盒**的短边，斜躺时骨架会退化成一个近似正方形（实测 45° 时 AABB 129×129mm），
        猜错就会横着扎进瓶身。改成把物体的**局部轴**逐条转到世界系，只在**大致水平**的轴里
        挑**最短**的一条闭合（平行夹爪只能水平开合）：

        * 立着的瓶子/黄油：最薄的局部 Y 本来就在水平面上 → 选它（与旧算法逐字一致）；
        * 平放的瓶子：三条轴都水平，最薄的局部 Y 仍是最短的 → 结果也不变；
        * 平放的**黄油**（t2，§8.9）：最薄的局部 Y 被转成**竖直**了 → 退而选次短的局部 Z
          （39.5mm，正好是它横躺时那对侧面），旧算法只能按 AABB 猜、斜 45° 时
          AABB 退化成 81.8×81.8mm 近乎正方形（50% 概率夹错方向）。

        返回的仍然是"要沿哪条水平轴闭合"的偏航角（度）。
        """
        name = task["grasp_object"]
        obj = catalog_object(self.catalog, name)
        objects = {item["key"]: item for item in task["objects"]}
        item = objects.get(name) or next(
            (v for k, v in objects.items() if k.endswith("/" + name)), None)
        obj_yaw = float(item.get("yaw", 0.0)) if item else 0.0
        local = np.asarray(obj["size"], dtype=float)
        rot = quat_to_mat(np.asarray(quat, dtype=float) if quat is not None
                          else quat_from_axis_angle("z", obj_yaw))
        axes = rot.T                                # ⚠ axes[i] = 第 i 列 = 局部轴 i 在世界系的方向
        horiz = [i for i in range(3) if abs(float(axes[i][2])) < 0.9]
        if not horiz:                                     # 三条轴都竖直：不可能，兜底用旧规则
            narrow_is_x = local[0] <= local[1]
            return obj_yaw + (0.0 if narrow_is_x else 90.0)
        thin = np.asarray(axes[min(horiz, key=lambda i: local[i])], dtype=float)
        yaw = math.degrees(math.atan2(thin[1], thin[0]))
        # 躺姿再折算到"负角"那半圈（等价方向，见 skills.reachable_yaw 的实测说明）
        return yaw if skills.is_upright(self, name) else skills.reachable_yaw(yaw)

    def reset(self, resample: bool = True) -> None:
        """回到任务初始状态（物体按 XML 初始位姿复位）。

        ``resample=True``（默认）：带 ``spawn_region`` 的抓取物**重新抽一个位置/姿态** ——
        所以界面上的"复位"= 换一局新摆位（任务多样化的入口）；``resample=False``：
        沿用本次会话已抽到的摆位（可复现的实验用，例如基准脚本按种子生成的每一局）。
        """
        mujoco.mj_resetData(self.model, self.data)
        if resample:
            self.spawn = sample_spawn(self.task, self.spawn_rng, self.catalog)
        self._apply_spawn()
        # ⚠ 必须先 mj_forward 把 xpos/xquat 刷新出来，下面 _grasp_yaw 里的 is_upright 才读得到
        # 本局真实姿态（mj_resetData 之后这些派生量还是旧的/零，会让"立着"被误判成"平放"，
        # 闭合轴被折算成 −90°）。
        mujoco.mj_forward(self.model, self.data)
        # 夹爪闭合方向跟着**本局姿态**走（平放/斜躺与立着完全不同），必须在摆位之后重算
        self.grasp_yaw = self._grasp_yaw(self.task, self._spawned_quat(self.task["grasp_object"]))
        self.joint_target = np.zeros(6)
        self.gripper = 1.0
        self.apply_control()
        mujoco.mj_forward(self.model, self.data)
        self.ee_target = self.tcp_position().copy()
        # 零位（6 关节目标全 0）下的末端位置：技能库"回程"的终点，见 skills.return_home
        self.home_tcp = self.tcp_position().copy()

    def _spawned_quat(self, name: str):
        """本局该物体的姿态四元数（没随机姿态就返回 None = 沿用 XML）。"""
        entry = self.spawn.get(name)
        return as_spawn_pose(entry).quat if entry is not None else None

    def _apply_spawn(self) -> None:
        """把随机抽到的位姿写进物体自由关节。

        没有随机姿态时只动 x/y：z 与朝向沿用 XML（``SceneBuilder`` 已把底面精确摆在桌面上）。
        有随机姿态时（§8.9 的"平放"）用 ``placed_pose`` 重算 body 位置：平放是**侧面**贴桌，
        沿用立姿的 z 会让瓶子悬空或穿桌；同时把抽到的四元数写进自由关节。
        """
        for name, raw in self.spawn.items():
            pose = as_spawn_pose(raw)
            x, y = pose.xy
            try:
                adr = self.model.jnt_qposadr[self.model.joint(f"{name}_joint").id]
            except Exception:  # noqa: BLE001  物体不是自由关节就跳过
                continue
            if pose.quat is None:
                offset = self.spawn_offset.get(name)
                self.data.qpos[adr] = x + (float(offset[0]) if offset is not None else 0.0)
                self.data.qpos[adr + 1] = y + (float(offset[1]) if offset is not None else 0.0)
                continue
            entry = next((e for e in self.task["objects"]
                          if e.get("name", e["key"].split("/")[-1]) == name), None)
            if entry is None:
                continue
            pos, quat = placed_pose(entry["key"], (x, y), pose.quat, self.catalog)
            # 物体要**坐在容器内底面**上时（t9：番茄酱立在木托盘里，z_offset = 内底高度 7.7mm），
            # 抬高量必须跟"无姿态"分支保持一致 —— 那条分支沿用 XML 的 z（已含 z_offset），
            # 这里不补的话瓶子会陷进托盘地板 7.7mm（实测会卡住/歪倒）。其余任务的 z_offset = 0，
            # 所以这一行对它们逐字不变。
            pos[2] += float(entry.get("z_offset", 0.0))
            self.data.qpos[adr:adr + 3] = pos
            self.data.qpos[adr + 3:adr + 7] = quat

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

    def render(self, camera: str | None = None) -> np.ndarray:
        """渲染一帧（``camera=None`` = 本会话当前相机）。

        多视角模式（界面的 2×2 网格）就是**对每个机位各调一次**这个函数再把图拼起来，
        所以这里把相机名做成参数；单视角 / 各测试脚本的老用法（不传参）逐字不变。
        """
        self.renderer.update_scene(self.data, camera=camera or self.camera,
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
        """本局随机安放的位姿（没有 ``spawn_region`` 的任务返回空串）。"""
        if not self.spawn:
            return ""
        parts = []
        for name, raw in self.spawn.items():
            pose = as_spawn_pose(raw)
            text = f"{name} ({pose.xy[0] * 1000:+.0f}, {pose.xy[1] * 1000:+.0f}) mm"
            if pose.quat is not None:
                text += f"·{pose.label}"
            parts.append(text)
        return "本局随机安放 " + "、".join(parts)


class SimView(QLabel):
    """把 MuJoCo 渲染结果显示出来，并把鼠标拖动/滚轮转成末端目标点移动。

    多视角模式下画面是 2×2（或放大的单格）拼图：按下时记下**那一格的机位**
    （``drag_camera``），拖动就按该机位的屏幕方向换算；双击 / 右键 = 放大该格。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(VIEW_W // 2, VIEW_H // 2)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#1b1e24; border:1px solid #333a45;")
        self._last = None
        self.drag_camera: str | None = None     # 当前拖动对应的机位（None = 用 session.camera）
        self.hover_camera: str | None = None    # 鼠标底下是哪一格（状态栏提示用）

    def mousePressEvent(self, event) -> None:
        camera = self.window().view_camera_at(event.position())
        if event.button() == Qt.RightButton:            # 右键 = 放大/还原该格
            self.window().toggle_view_maximize(camera)
            return
        if event.button() == Qt.LeftButton:
            self.drag_camera = camera
            self._last = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:      # 双击 = 放大/还原该格
        if event.button() == Qt.LeftButton:
            self.window().toggle_view_maximize(self.window().view_camera_at(event.position()))
            self._last = None
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:
        self._last = None
        self.drag_camera = None
        self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event) -> None:
        self.hover_camera = self.window().view_camera_at(event.position())
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
        # ── 多视角显示（2×2 网格）状态 ──
        self.view_maximized: str | None = None     # 放大的机位名；None = 按勾选拼网格
        self.view_cells: list[tuple[str, tuple[int, int, int, int]]] = []
        self.view_image_size = (VIEW_W, VIEW_H)    # 当前拼图（未缩放）的尺寸
        # ── 连续任务（自动连跑）状态 ──
        # cont_plan：调度表，元素 = (第几轮, TASKS 下标)；单任务 = 同一关排 N 次，
        # 多任务 = **按轮展开**（一轮里把勾选的关按列表顺序跑一遍，再进下一轮）
        self.cont_plan: list[tuple[int, int]] = []
        self.cont_active = False           # 连跑进行中（暂停时仍为 True）
        self.cont_done = 0                 # 已经跑完几局
        self.cont_ok = 0                   # 其中成功几局
        self.cont_counts: dict[str, list[int]] = {}   # 任务 id → [成功, 局数]
        self.cont_dwell = 0                # 本局结束后还要停留几帧
        self.cont_frames = 0               # 本局已经跑了多少帧
        self.cont_wall = 0.0               # 上一局耗时（秒）
        self.cont_started = 0.0            # 本局开始时刻（算每局耗时）
        self._cont_internal = False        # 连跑自己在切关/复位时别把自己停掉
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
        """左侧面板。⚠ 控件变多之后 800px 高的窗口装不下，套一层滚动区（不滚动就看不到
        「视图/场景」「快捷键」两组，实测窗口缩到 800px 时最下面一组被压扁）。"""
        panel = QWidget()
        panel.setFixedWidth(PANEL_W)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.addWidget(self._build_task_box())
        layout.addWidget(self._build_control_box())
        layout.addWidget(self._build_auto_box())
        layout.addWidget(self._build_cont_box())
        layout.addWidget(self._build_view_box())
        layout.addWidget(self._build_help_box())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedWidth(PANEL_W + 16)
        scroll.setWidget(panel)
        return scroll

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

    def _build_cont_box(self) -> QGroupBox:
        """连续任务面板：单任务连跑 N 局 / 多任务轮转 R 轮。

        两条要求都能满足：
        * **单任务**：次数 = 局数，一局跑完自动"重新开始"，走 ``callback_reset()`` →
          ``sample_spawn`` 重新抽一次摆位，所以带随机安放的任务**每局都不一样**；
        * **多任务**：在下面列表里勾选若干关，次数 = **轮数**；调度表按轮展开
          （一轮把勾选的关依次跑完，再进下一轮）。
        """
        box = QGroupBox("连续任务（自动连跑）")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self.cont_mode_combo = QComboBox()
        self.cont_mode_combo.addItems(["单任务连跑", "多任务轮转"])
        self.cont_mode_combo.currentIndexChanged.connect(self.callback_cont_mode)
        row.addWidget(QLabel("模式"))
        row.addWidget(self.cont_mode_combo, stretch=1)
        self.cont_count_label = QLabel("次数（局）")
        self.cont_count_spin = QSpinBox()
        self.cont_count_spin.setRange(1, CONT_MAX_RUNS)
        self.cont_count_spin.setValue(5)
        row.addWidget(self.cont_count_label)
        row.addWidget(self.cont_count_spin)
        layout.addLayout(row)

        self.cont_task_list = QListWidget()
        self.cont_task_list.setMaximumHeight(84)      # 约 3 行高：够点，又不太占侧栏
        self.cont_task_list.setStyleSheet("font-size:11px;")
        for task in TASKS:
            item = QListWidgetItem(task["name"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.cont_task_list.addItem(item)
        self.cont_task_list.setEnabled(False)          # 单任务模式用不到这张表
        layout.addWidget(self.cont_task_list)

        row2 = QHBoxLayout()
        self.cont_all_button = QPushButton("全选")
        self.cont_all_button.clicked.connect(self.callback_cont_check_all)
        self.cont_none_button = QPushButton("全不选")
        self.cont_none_button.clicked.connect(self.callback_cont_clear_all)
        self.cont_button = QPushButton("▶ 开始连续任务")
        self.cont_button.clicked.connect(self.callback_cont_toggle)
        row2.addWidget(self.cont_all_button)
        row2.addWidget(self.cont_none_button)
        row2.addWidget(self.cont_button, stretch=1)
        layout.addLayout(row2)

        # ⚠ 连跑**不打印逐局运行日志**（用户要求）：所以这里既没有"计划：…"这种说明行、
        # 也没有"✅ 第 k/N 局 … 帧 / 秒"的历史明细，只有**一行**实时状态（跑完时换成汇总）。
        # 逐局的账仍然在记（cont_ok / cont_done / cont_counts），只是不往界面上刷。
        self.cont_status = QLabel("就绪")
        self.cont_status.setWordWrap(True)
        self.cont_status.setStyleSheet("color:#7fd1ff; font-size:11px;")
        layout.addWidget(self.cont_status)
        return box

    def _build_view_box(self) -> QGroupBox:
        """视图/场景：**多视角**（最多 2×2 四个机位，各占 1/4）+ 逐格勾选 + 逐格放大。

        * 勾选框：默认四个全开；只勾一个时它自动占满整块画面；
        * 放大：下拉选一格（或**双击 / 右键**画面里的格子）→ 该格占满、其它三格隐藏，
          再操作一次还原成网格；
        * 鼠标拖动按**光标所在那一格**的机位换算（每格自己的方位角），互不干扰。
        """
        box = QGroupBox("视图 / 场景（多视角）")
        layout = QVBoxLayout(box)

        layout.addWidget(QLabel("显示的视角（默认全开，各自可关）"))
        check_row = QHBoxLayout()
        self.view_checks: dict[str, QCheckBox] = {}
        for camera, label in zip(CAMERAS, CAMERA_LABELS):
            box_ = QCheckBox(label.replace("相机", ""))       # 斜前方 / 正上方 / 侧前方 / 腕部
            box_.setChecked(True)
            box_.setToolTip(f"显示 {label}（{camera}）")
            box_.toggled.connect(self.callback_view_checked)
            self.view_checks[camera] = box_
            check_row.addWidget(box_)
        layout.addLayout(check_row)

        row = QHBoxLayout()
        row.addWidget(QLabel("放大"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItem("无（2×2 网格）")
        self.zoom_combo.addItems(CAMERA_LABELS)
        self.zoom_combo.currentIndexChanged.connect(self.callback_view_zoom)
        self.zoom_combo.setToolTip("选一格放大：它占满画面、其它三格隐藏；"
                                   "也可以直接双击 / 右键画面里的格子")
        row.addWidget(self.zoom_combo, stretch=1)
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
            "R 复位场景    T 依次放大四格（再按回到 2×2 网格）\n"
            "视图=2×2 多视角：拖动按**光标所在那一格**换算；双击/右键某格=放大占满\n"
            "（左上角那个「放大」下拉也能选；勾选框控制显示哪几路，默认全开）\n"
            "自动执行 = 闭环技能库（接触/跟随判据），手动操作会自动暂停它\n"
            "连续任务 = 单任务连跑 N 局 / 多任务轮转 R 轮；每局都重制场景（随机化的关\n"
            "          每局换新摆位），手动拖拽会暂停，R 复位或切任务则结束连跑"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color:#9fb3c8; font-size:11px;")
        layout.addWidget(help_text)
        return box

    # ────────────────── 回调 ──────────────────
    def callback_load_task(self, index: int) -> None:
        task = TASKS[index]
        # 手动切关时把连跑停掉（连跑自己切关会置 _cont_internal，不会被误停）
        if self.cont_active and not self._cont_internal:
            self.stop_continuous("手动切换任务")
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

    # ────────────────── 多视角显示（2×2 网格 + 逐格放大） ──────────────────
    def visible_cameras(self) -> list[str]:
        """当前要渲染哪几路：放大时只有它；否则是勾选的那几路（按固定顺序）。"""
        if self.view_maximized in CAMERAS:
            return [self.view_maximized]
        return [cam for cam in CAMERAS if self.view_checks[cam].isChecked()]

    def compose_view(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """把各路画面拼成 2×2（每格 1/4，中间 2px 分隔线；不足 4 格留黑）。

        同时把每格在**图像坐标系**下的矩形记进 ``self.view_cells``，
        供鼠标换算（拖动按格子机位、双击放大哪一格）使用。只有一路时直接返回它。
        """
        cams = [c for c in self.visible_cameras() if c in frames]
        if not cams:                                   # 理论上不会（勾选回调会兜底）
            self.view_cells = []
            return np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        if len(cams) == 1:
            self.view_cells = [(cams[0], (0, 0, VIEW_W, VIEW_H))]
            return frames[cams[0]]
        canvas = np.zeros((2 * VIEW_H + GRID_GAP, 2 * VIEW_W + GRID_GAP, 3), dtype=np.uint8)
        cells: list[tuple[str, tuple[int, int, int, int]]] = []
        for i, cam in enumerate(cams[:4]):
            row, col = divmod(i, 2)
            x = col * (VIEW_W + GRID_GAP)
            y = row * (VIEW_H + GRID_GAP)
            canvas[y:y + VIEW_H, x:x + VIEW_W] = frames[cam]
            cells.append((cam, (x, y, VIEW_W, VIEW_H)))
        self.view_cells = cells
        return canvas

    def callback_view_checked(self, *_args) -> None:
        """勾选/取消某一路视角。至少要留一路；放大中的那路被取消时退回网格。"""
        checked = [cam for cam in CAMERAS if self.view_checks[cam].isChecked()]
        if not checked:                                # 全关掉就没画面了 → 兜底留第一路
            self.view_checks[CAMERAS[0]].setChecked(True)
            self.statusBar().showMessage("至少保留一路视角，已自动勾回「斜前方」")
            self.refresh_view()
            return
        if self.view_maximized is not None and self.view_maximized not in checked:
            self.set_view_maximize(None)
        self.statusBar().showMessage(
            f"视角：显示 {len(checked)} 路 —— "
            + "、".join(CAMERA_LABELS[CAMERAS.index(c)] for c in checked))
        self.refresh_view()

    def callback_view_zoom(self, index: int) -> None:
        """下拉框选放大哪一格（0 = 不放大，回到 2×2 网格）。"""
        self.set_view_maximize(None if index == 0 else CAMERAS[index - 1])

    def set_view_maximize(self, camera: str | None) -> None:
        """放大某一格（占满画面、其它三格隐藏）或还原成网格；同步下拉框与提示。"""
        self.view_maximized = camera if camera in CAMERAS else None
        index = 0 if self.view_maximized is None else CAMERAS.index(self.view_maximized) + 1
        self.zoom_combo.blockSignals(True)
        self.zoom_combo.setCurrentIndex(index)
        self.zoom_combo.blockSignals(False)
        self.statusBar().showMessage(
            "视图：2×2 网格" if camera is None
            else f"视图：{CAMERA_LABELS[CAMERAS.index(camera)]} 放大占满（其它三格已隐藏）")
        self.refresh_view()

    def toggle_view_maximize(self, camera: str | None) -> None:
        """双击 / 右键某一格：放大它；再操作一次还原成网格。"""
        if camera not in CAMERAS:
            return
        self.set_view_maximize(None if self.view_maximized == camera else camera)

    def view_camera_at(self, pos) -> str | None:
        """视图控件里的一个点 → 落在哪一格（返回机位名）。"""
        if not self.view_cells:
            return None
        pixmap = self.view.pixmap()
        if pixmap is None or pixmap.isNull():
            return self.view_cells[0][0]
        width = max(self.view.width(), 1)
        height = max(self.view.height(), 1)
        # QLabel 把按 KeepAspectRatio 缩放后的图**居中**，先减掉居中留白再折算回图像坐标
        ox = (width - pixmap.width()) / 2.0
        oy = (height - pixmap.height()) / 2.0
        px, py = pos.x() - ox, pos.y() - oy
        if not (0 <= px < pixmap.width() and 0 <= py < pixmap.height()):
            return None
        image_w, image_h = self.view_image_size
        ix = px * image_w / pixmap.width()
        iy = py * image_h / pixmap.height()
        for camera, (x, y, w, h) in self.view_cells:
            if x <= ix < x + w and y <= iy < y + h:
                return camera
        return None

    def _render_view(self, sess) -> None:
        """按当前设置渲染 + 拼成 2×2 + 贴到视图控件（tick 与"设置变了立刻重画"共用）。"""
        frames = {camera: sess.render(camera) for camera in self.visible_cameras()}
        canvas = self.compose_view(frames)
        self.view_image_size = (canvas.shape[1], canvas.shape[0])
        image = QImage(canvas.data, canvas.shape[1], canvas.shape[0],
                       3 * canvas.shape[1], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.draw_view_labels(pixmap)                # 每格左上角标机位名（多格时才画）
        self.view.setPixmap(pixmap)

    def refresh_view(self) -> None:
        """勾选 / 放大之后**立刻**重画一次（不等下一帧，双击手感更跟手）。"""
        if self.session is not None and self.session.renderer is not None:
            self._render_view(self.session)

    def draw_view_labels(self, pixmap: QPixmap) -> None:
        """在放大的 QPixmap 上给每格左上角标机位名（只在多格时画）。"""
        if len(self.view_cells) < 2:
            return
        font = QFont()
        font.setPointSize(9)
        painter = QPainter(pixmap)
        try:
            painter.setFont(font)
            scale = pixmap.width() / max(self.view_image_size[0], 1)
            for camera, (x, y, _w, _h) in self.view_cells:
                text = CAMERA_LABELS[CAMERAS.index(camera)]
                rect = painter.fontMetrics().boundingRect(text).adjusted(-3, -2, 3, 2)
                rect.moveTopLeft(QPoint(int(x * scale) + 4, int(y * scale) + 4))
                painter.fillRect(rect, QColor(0, 0, 0, 130))
                painter.setPen(QColor(230, 238, 248))
                painter.drawText(rect, Qt.AlignCenter, text)
        finally:
            painter.end()

    def callback_reset(self) -> None:
        if self.session is not None:
            # 手动复位 = 宣布接管：把连跑停掉（连跑自己每局都会复位，走 _cont_internal 分支）
            if self.cont_active and not self._cont_internal:
                self.stop_continuous("手动复位")
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
            if self.cont_active:                       # 连跑也一起暂停（点按钮接着跑）
                self.cont_button.setText("▶ 继续连跑")
                self.cont_status.setText(
                    "已暂停（手动接管）—— 点「▶ 继续连跑」接着跑；"
                    "按 R 复位或切任务则结束连跑")
            self.statusBar().showMessage("检测到手动操作：自动执行已暂停")

    def callback_auto_toggle(self) -> None:
        if self.auto is None:
            return
        if self.cont_active and not (self.auto.active and self.auto.paused):
            self.stop_continuous("改为单局自动")        # 单局自动与连跑互斥
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
        if self.cont_active:                    # 连跑被单步暂停：按钮改成"继续连跑"
            self.cont_button.setText("▶ 继续连跑")

    # ────────────────── 连续任务（自动连跑） ──────────────────
    def callback_cont_mode(self, index: int) -> None:
        """切换单任务 / 多任务：只有多任务模式才用得上那张任务勾选表。"""
        self.cont_task_list.setEnabled(index == 1)
        self.cont_count_label.setText("轮数" if index == 1 else "次数（局）")

    def callback_cont_check_all(self, *_args) -> None:
        self._cont_set_all_checked(True)

    def callback_cont_clear_all(self, *_args) -> None:
        self._cont_set_all_checked(False)

    def _cont_set_all_checked(self, checked: bool) -> None:
        for i in range(self.cont_task_list.count()):
            self.cont_task_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _cont_checked_indices(self) -> list[int]:
        """多任务模式下勾选的关（按列表顺序 = 一轮里的执行顺序）。"""
        return [i for i in range(self.cont_task_list.count())
                if self.cont_task_list.item(i).checkState() == Qt.Checked]

    def _cont_build_plan(self) -> list[tuple[int, int]]:
        count = int(self.cont_count_spin.value())
        if self.cont_mode_combo.currentIndex() == 0:          # 单任务：同一关排 N 局
            return [(0, self.task_combo.currentIndex())] * count
        # 多任务：**按轮展开** —— 一轮里依次跑完勾选的关，再进下一轮
        return [(r, i) for r in range(count) for i in self._cont_checked_indices()]

    def start_continuous(self) -> bool:
        """按当前设置开始连跑；返回是否真的开始了。"""
        if self.session is None or self.auto is None:
            return False
        plan = self._cont_build_plan()
        if not plan:
            self.cont_status.setText("❌ 多任务模式至少要勾选一个任务")
            return False
        self.cont_plan = plan
        self.cont_active = True
        self.cont_done = self.cont_ok = 0
        self.cont_counts = {}
        self.cont_dwell = 0
        self.cont_button.setText("⏹ 停止连跑")
        self.statusBar().showMessage(f"连续任务开始：共 {len(plan)} 局")
        self._cont_begin_run()
        return True

    def _cont_begin_run(self) -> None:
        """开始调度表里的下一局：需要时切任务 → **复位重制场景**（重新随机摆位）→ 启动技能库。"""
        if not self.cont_active:
            return
        if self.cont_done >= len(self.cont_plan):
            self._cont_finish()
            return
        round_idx, task_idx = self.cont_plan[self.cont_done]
        # ⚠ 连跑自己切关/复位时不能把自己停掉（callback_load_task / callback_reset 里会检查它）
        self._cont_internal = True
        try:
            if task_idx != self.task_combo.currentIndex():
                self.task_combo.setCurrentIndex(task_idx)      # 重建场景 + 新的 SkillRunner
            self.callback_reset()                             # 每局都重制：resample=True 换新摆位
        finally:
            self._cont_internal = False
        self.cont_frames = 0
        self.cont_started = time.perf_counter()
        self._auto_reported = False
        self.auto.start()
        self.auto_button.setText("⏸ 暂停自动")
        self.auto_status.setText("连跑中")
        self.auto_result.setText("")
        self.cont_dwell = 0                                # 本局刚开跑，还没有"结束停留"
        self.cont_status.setText(self._cont_status_text(round_idx))
        self.statusBar().showMessage(
            f"连续任务 第 {self.cont_done + 1}/{len(self.cont_plan)} 局：{self.session.task['id']}")

    def _cont_status_text(self, round_idx: int | None = None) -> str:
        """一行实时状态（不打印逐局日志）。"""
        head = f"第 {self.cont_done + 1}/{len(self.cont_plan)} 局"
        if self.cont_mode_combo.currentIndex() == 1:
            head += f"（第 {(round_idx if round_idx is not None else 0) + 1} 轮）"
        return (f"{head} · {self.session.task['id']} · 累计成功 {self.cont_ok}/{self.cont_done}")

    def _cont_record_run(self) -> None:
        """一局跑完：**只记账**（成功/失败、帧数），不往界面写运行日志。"""
        task_id = self.session.task["id"]
        ok = bool(self.auto.ok)                       # auto.ok = 技能库"判分"那一步的结论
        entry = self.cont_counts.setdefault(task_id, [0, 0])
        entry[0] += int(ok)
        entry[1] += 1
        self.cont_ok += int(ok)
        self.cont_done += 1
        self.cont_wall = time.perf_counter() - self.cont_started      # 供汇总里的平均用时
        self.cont_dwell = CONT_DWELL_FRAMES           # 停一会儿，让画面留在结束状态上

    def _cont_advance(self) -> None:
        """结束停留走完：开下一局；整张调度表跑完就收尾。"""
        if not self.cont_active:
            return
        if self.cont_done >= len(self.cont_plan):
            self._cont_finish()
        else:
            self._cont_begin_run()

    def _cont_finish(self) -> None:
        total = len(self.cont_plan)
        self.cont_active = False
        self.cont_dwell = 0
        self.cont_button.setText("▶ 开始连续任务")
        rate = 100.0 * self.cont_ok / max(total, 1)
        per_task = "，".join(f"{tid.split('_')[0]} {o}/{n}"
                             for tid, (o, n) in self.cont_counts.items())
        self.cont_status.setText(
            f"✅ 跑完 {self.cont_ok}/{total} 成功（{rate:.0f}%）"
            + (f"·分任务 {per_task}" if per_task else ""))
        self.statusBar().showMessage(
            f"连续任务结束：{self.cont_ok}/{total} 成功（{rate:.0f}%）")

    def stop_continuous(self, reason: str = "手动停止") -> None:
        """停掉连跑（不清场）：手动接管/复位/切关，或按停止按钮时调用。"""
        if not self.cont_active:
            return
        self.cont_active = False
        self.cont_dwell = 0
        if self.auto is not None:
            self.auto.stop()
            self.auto_button.setText("▶ 自动完成本关")
        self.cont_button.setText("▶ 开始连续任务")
        self.cont_status.setText(
            f"已停止（{reason}）· 已完成 {self.cont_done}/{len(self.cont_plan)} 局，"
            f"成功 {self.cont_ok}")
        self.auto_status.setText(f"已停止（{reason}）")

    def callback_cont_toggle(self) -> None:
        """开始 / 继续 / 停止连跑。"""
        if self.cont_active and self.auto is not None and self.auto.paused:
            self.auto.resume()                       # 被手动操作暂停过 → 接着跑
            self.auto_button.setText("⏸ 暂停自动")
            self.cont_button.setText("⏹ 停止连跑")
            self.cont_status.setText(self._cont_status_text())
            return
        if self.cont_active:
            self.stop_continuous("点停止")
            return
        self.start_continuous()

    # ────────────────── 鼠标拖拽 → IK ──────────────────
    def _screen_axes(self, camera: str | None = None):
        """返回相机在世界系下的右/上方向，以及渲染图上"每像素多少米"（多视角时传格子机位）。"""
        sess = self.session
        camera = camera or sess.camera
        cam_id = mujoco.mj_name2id(sess.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
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
        # 按"拖动起点那一格"的机位换算（每格的屏幕方向不同）；单视角时就是 session.camera
        right, up, meters_per_px = self._screen_axes(self.view.drag_camera)
        meters_per_px *= VIEW_W / float(disp_w)         # 显示缩放折算回渲染像素
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
        elif text == "t":                              # 依次放大四格（再按一次回到网格）
            nxt = CAMERAS[0] if self.view_maximized is None else (
                None if self.view_maximized == CAMERAS[-1]
                else CAMERAS[CAMERAS.index(self.view_maximized) + 1])
            self.set_view_maximize(nxt)
        else:
            super().keyPressEvent(event)

    # ────────────────── 主循环 ──────────────────
    def tick(self) -> None:
        sess = self.session
        if sess is None:
            return
        if self.auto is not None:                     # 闭环技能：先按当前状态定目标，再跑物理
            if self.cont_active and not self.auto.finished:
                self.cont_frames += 1                 # 连跑：统计本局帧数（与 bench 口径一致）
            self.auto.step()
            self.auto_status.setText("自动：" + self.auto.status)
            if self.auto.finished and not self._auto_reported:
                self._auto_reported = True
                if not self.cont_active:              # 连跑不打印逐局运行日志，只更新状态
                    self.auto_result.setText(self.auto.summary())
                self.auto_button.setText("▶ 自动完成本关")
                self.statusBar().showMessage(
                    "自动执行结束：" + ("成功 ✅" if self.auto.ok else "未完成 ❌"))
                if self.cont_active:                  # 连跑：记下这一局的成绩
                    self._cont_record_run()
            # 连跑调度：停留 CONT_DWELL_FRAMES 帧 → 重制场景 → 下一局（或收尾）
            if self.cont_active and self.auto.finished and self._auto_reported:
                if self.cont_dwell > 0:
                    self.cont_dwell -= 1
                else:
                    self._cont_advance()
        # ⚠ 连跑会**在这一帧里切任务**（_cont_advance → callback_load_task 重建 session 并
        # close 掉旧 renderer），上面那个 sess 可能已经被关掉了 —— 必须重新取一次，
        # 否则本帧接着 render 旧 session 会抛 "render cannot be called after close"。
        sess = self.session
        if sess is None:
            return
        sess.step(8)                                  # dt=2ms × 8 ≈ 1/60 s
        self._render_view(sess)                       # 多视角：各路各渲一帧 → 2×2 拼图 → 贴到控件

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
            f"渲染 {self._fps:.0f} FPS | 任务 {sess.task['id']} | 已完成 {self.success_count}"
            + (f" | 连跑 第 {min(self.cont_done + 1, len(self.cont_plan))}/{len(self.cont_plan)} 局"
               f"（成功 {self.cont_ok}）" if self.cont_active else ""))


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

