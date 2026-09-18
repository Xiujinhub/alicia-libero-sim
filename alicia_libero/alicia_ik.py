#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alicia 50mm 手臂的运动学工具：正向/逆向（阻尼最小二乘）与夹爪控制。

为什么需要
----------
鼠标拖动机械臂时，用户给的是**末端位置**，而模型接受的是**6 个关节目标角**，
所以要在两步之间做 IK。这里用阻尼最小二乘（DLS）解 6×6 的位姿误差，
在**独立的 MjData** 上求解（不干扰正在跑的物理），解出关节角后再写进 ctrl。

坐标与朝向约定
------------
``tool0_site`` 是末端参考点；它的局部 -Z 指向夹爪外侧（指尖方向）。
"朝下抓取" = 让 site 的 -Z 轴对准世界 -Z，同时保持当前偏航角。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

ARM_JOINTS = [f"Joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ("left_finger", "right_finger")


def rotvec_from_matrix(mat: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 轴角向量（罗德里格斯向量），用于 IK 的姿态误差。"""
    cos_theta = float(np.clip((np.trace(mat) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if abs(math.pi - theta) < 1e-6:          # 接近 180°，用对角线求轴
        axis = np.sqrt(np.maximum((np.diag(mat) + 1.0) / 2.0, 0.0))
        if mat[2, 1] - mat[1, 2] < 0:
            axis[0] = -axis[0]
        if mat[0, 2] - mat[2, 0] < 0:
            axis[1] = -axis[1]
        if mat[1, 0] - mat[0, 1] < 0:
            axis[2] = -axis[2]
        return axis / (np.linalg.norm(axis) + 1e-12) * theta
    axis = np.array([mat[2, 1] - mat[1, 2], mat[0, 2] - mat[2, 0], mat[1, 0] - mat[0, 1]])
    return axis / (2.0 * math.sin(theta)) * theta


def down_orientation(yaw_deg: float = 0.0) -> np.ndarray:
    """（备用）直接指定 tool 帧 -Z 朝下的姿态；实际请用 AliciaIK.down_orientation。"""
    yaw = math.radians(yaw_deg)
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


class AliciaIK:
    """在独立 MjData 上做位置+姿态 IK，返回 6 个关节的目标角。

    ⚠ 踩坑记录：不能想当然认为 ``tool0_site`` 的某个坐标轴就是"接近轴"。
    实测零位姿态下 tool 帧 Y=(0,-1,0)（水平 = 手指夹紧方向）、
    Z=(-0.707,0,+0.707)（斜向上 = 接近方向），
    若强行要求 Z 朝下，手腕会被顶到 ±90° 限位，IK 全盘失败。
    因此这里从**模型的零位姿态**反推「接近轴/夹紧轴在 tool 帧下的方向」，
    再构造"接近轴竖直向下 + 夹紧轴水平"的目标姿态。
    """

    def __init__(self, model: mujoco.MjModel, site_name: str = "tool0_site",
                 link6_name: str = "link6") -> None:
        self.model = model
        self.data = mujoco.MjData(model)          # 专用，不影响物理状态
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise RuntimeError(f"模型里找不到 site '{site_name}'")
        self.joint_ids = [model.joint(name).id for name in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.lo = model.jnt_range[self.joint_ids, 0].copy()
        self.hi = model.jnt_range[self.joint_ids, 1].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self.approach_local, self.closing_local = self._derive_grasp_axes(link6_name)

    # ── 从零位姿态推导抓取轴 ──
    def _derive_grasp_axes(self, link6_name: str):
        q_zero = np.zeros(6)
        self.data.qpos[self.qpos_adr] = q_zero
        self.data.qpos[6:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        tool_rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        link6_rot = self.data.xmat[self.model.body(link6_name).id].reshape(3, 3).copy()
        # 夹爪沿 link6 的 -Z 伸出 → 接近轴；手指沿 link6 的 Y 分开 → 夹紧轴
        approach_local = tool_rot.T @ (-link6_rot[:, 2])
        closing_local = tool_rot.T @ link6_rot[:, 1]
        return (approach_local / np.linalg.norm(approach_local),
                closing_local / np.linalg.norm(closing_local))

    def down_orientation(self, yaw_deg: float = 0.0) -> np.ndarray:
        """构造"接近轴竖直向下、夹紧轴水平（按 yaw 指向）"的 tool 目标姿态。"""
        yaw = math.radians(yaw_deg)
        approach_world = np.array([0.0, 0.0, -1.0])
        closing_world = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        local = np.column_stack([self.closing_local,
                                 np.cross(self.approach_local, self.closing_local),
                                 self.approach_local])
        world = np.column_stack([closing_world,
                                 np.cross(approach_world, closing_world),
                                 approach_world])
        return world @ local.T

    def forward(self, q_arm) -> np.ndarray:
        """正运动学：给关节角，返回末端位置。"""
        self.data.qpos[self.qpos_adr] = q_arm
        self.data.qpos[6:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def tool_matrix(self, q_arm) -> np.ndarray:
        self.data.qpos[self.qpos_adr] = q_arm
        self.data.qpos[6:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def solve(self, target_pos, target_mat, q_seed=None, iters: int = 30,
              damping: float = 0.05, step: float = 0.6, pos_tol: float = 2e-4,
              rot_weight: float = 0.5, leak: float = 0.0,
              keep_joints=None) -> tuple[np.ndarray, float]:
        """阻尼最小二乘 IK，返回 (关节目标角, 位置误差)。分两阶段：

        1. **纯位置阶段**：先只消除位置误差。实测若一开始就带姿态约束，
           解会陷进"肘部朝向不对"的局部解，远处目标直接卡死（桌面网格可达率仅 55%）；
        2. **姿态微调阶段**：以阶段 1 的解为种子，在位置基本满足的前提下把姿态拉向目标。

        * ``rot_weight``：姿态误差权重。实测「严格竖直向下」会把 J5 顶到 ±90° 限位，
          所以姿态只当**软约束**（位置优先）
        * ``leak``：把关节往量程中点拉的泄漏项。注意别设大：每轮迭代都加一次，
          0.02 × 60 轮 = 1.2 的等效拉力会把位置精度拖到 10mm 以上（实测踩过），
          通常保持 0 即可
        """
        q, err = self._iterate(target_pos, target_mat, q_seed, iters,
                               damping, step, pos_tol, 0.0, leak, keep_joints)
        if rot_weight > 0.0:
            q, err = self._iterate(target_pos, target_mat, q, max(60, iters),
                                   damping, step * 0.5, pos_tol, rot_weight, leak,
                                   keep_joints, stop_on_pos=False)
        return q, err

    def _iterate(self, target_pos, target_mat, q_seed, iters: int, damping: float,
                 step: float, pos_tol: float, rot_weight: float, leak: float,
                 keep_joints=None, stop_on_pos: bool = True,
                 orientation_mode: str = "axis", spin_weight: float = 0.6) -> tuple[np.ndarray, float]:
        if q_seed is None:
            q_seed = self.data.qpos[self.qpos_adr].copy()
        q = np.clip(np.asarray(q_seed, dtype=float).copy(), self.lo + 1e-4, self.hi - 1e-4)
        self.data.qpos[self.qpos_adr] = q
        self.data.qpos[6:] = 0.0
        q_mid = 0.5 * (self.lo + self.hi)
        freeze = np.zeros(6, dtype=bool)
        if keep_joints is not None:
            freeze[np.asarray(keep_joints, dtype=int)] = True

        target_pos = np.asarray(target_pos, dtype=float)
        err_norm = 1e9
        for _ in range(iters):
            mujoco.mj_forward(self.model, self.data)
            err_pos = target_pos - self.data.site_xpos[self.site_id]
            err_norm = float(np.linalg.norm(err_pos))
            if stop_on_pos and err_norm < pos_tol:
                break
            err = err_pos
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            if rot_weight > 0.0:
                # 位置修正 + 只在位置雅可比零空间里改姿态：
                # 6 关节 - 3 位置约束 = 3 个冗余自由度，姿态能变而位置严格不掉
                jac_p = self._jacp[:, self.dof_adr]
                jac_r = self._jacr[:, self.dof_adr]
                cur_mat = self.data.site_xmat[self.site_id].reshape(3, 3)
                if orientation_mode == "axis":
                    # 接近轴对齐（2 自由度）+ 绕接近轴的自转对齐（1 自由度，主要靠 J6）。
                    # 只对齐接近轴不够：平行夹爪必须让"闭合轴"对上物体的窄边，
                    # 否则会出现"夹爪方向对了但夹不上"（实测番茄酱搬运中掉落）。
                    a_cur = cur_mat @ self.approach_local
                    a_tgt = target_mat @ self.approach_local
                    c_cur = cur_mat @ self.closing_local
                    c_tgt = target_mat @ self.closing_local
                    err_approach = np.cross(a_cur, a_tgt)
                    spin_error = float(np.dot(np.cross(c_cur, c_tgt), a_tgt))
                    err_rot = err_approach + (spin_weight * spin_error) * a_tgt
                else:
                    err_rot = rotvec_from_matrix(target_mat @ cur_mat.T)
                dq_p = jac_p.T @ np.linalg.solve(
                    jac_p @ jac_p.T + (damping ** 2) * np.eye(3), err_pos)
                dq_r = jac_r.T @ np.linalg.solve(
                    jac_r @ jac_r.T + (damping ** 2) * np.eye(3), err_rot * rot_weight)
                null_space = np.eye(6) - np.linalg.pinv(jac_p) @ jac_p
                dq = dq_p + null_space @ dq_r
            else:
                jac = self._jacp[:, self.dof_adr]        # 只用位置雅可比（3×6）
                dq = jac.T @ np.linalg.solve(
                    jac @ jac.T + (damping ** 2) * np.eye(3), err)
            if leak:
                dq -= leak * (q - q_mid)
            dq[freeze] = 0.0
            q = np.clip(q + step * dq, self.lo + 1e-4, self.hi - 1e-4)
            self.data.qpos[self.qpos_adr] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos[self.qpos_adr].copy(), err_norm


def finger_targets(model: mujoco.MjModel, opening: float) -> tuple[float, float]:
    """把"张开度"换算成左右手指的 ctrl（米）。

    ``opening``：0 = 完全闭合，1 = 完全张开。

    ⚠ 踩坑：模型里 ``left_finger`` 滑轨 range 是 ``0 0.025``，且 **0 = 两指分开 50mm（张开）**、
    ``0.025`` = 两指并拢（闭合）—— 与"张开度"正好相反。
    直接把 opening 当位移写进去会导致"想夹紧却全张开"（实测抓取全部失败）。
    """
    travel = float(model.jnt_range[model.joint("left_finger").id][1])
    displacement = (1.0 - max(0.0, min(1.0, opening))) * travel
    return displacement, -displacement
