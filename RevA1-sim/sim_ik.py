"""末端位姿数值 IK（阻尼最小二乘）+ "平滑开到目标位姿"的仿真小工具。

这套机械臂的腕部是偏置腕（J4 与 J6 轴线平行但不交于一点），没有简洁的解析 IK，
所以统一用数值法：直接用 MuJoCo 的 site 雅可比做阻尼最小二乘迭代。

典型用法::

    from sim_ik import solve_ik_pose, site_pos, tool_axis, drive_to
    q, ep, eo = solve_ik_pose(model, data, "tool_site", (0.45, 0.0, 0.25), axis=(0, 0, -1))
    drive_to(model, data, q, seconds=2.0)

命令行（只做 IK，不渲染）::

    python sim_ik.py --target 0.45 0 0.25 --axis 0 0 -1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl()
import mujoco  # noqa: E402  (必须在设置 MUJOCO_GL 之后)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


# ----------------------------------------------------------------------------- 基础
def jid(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise KeyError(f"模型里没有关节 {name!r}")
    return int(i)


def dof_adr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[jid(model, name)])


def qpos_adr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[jid(model, name)])


def arm_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """取 6 个臂关节的 qpos（顺序 joint1..joint6）。"""
    return np.array([data.qpos[qpos_adr(model, j)] for j in ARM_JOINTS])


def set_arm_qpos(model: mujoco.MjModel, data: mujoco.MjData, q) -> None:
    for j, v in zip(ARM_JOINTS, np.asarray(q).ravel()):
        data.qpos[qpos_adr(model, j)] = v


def site_id(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0:
        raise KeyError(f"模型里没有 site {name!r}")
    return int(i)


def site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return np.array(data.site_xpos[site_id(model, name)])


def site_mat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.site_xmat[site_id(model, name)].reshape(3, 3).copy()


def tool_axis(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """site 的 z 轴（对 tool_site 来说就是工具指向）。"""
    return site_mat(model, data, name)[:, 2]


# ----------------------------------------------------------------------------- IK
def solve_ik_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site: str = "tool_site",
    target_pos=(0.0, 0.0, 0.3),
    axis=None,
    q_seed=None,
    iters: int = 300,
    pos_tol: float = 1e-4,
    axis_tol: float = 1e-3,
    damping: float = 1e-2,
    step: float = 0.5,
    null_gain: float = 0.02,
) -> tuple[np.ndarray, float, float]:
    """解"末端到 target_pos、且 site 的 z 轴对齐 axis"的关节角。

    参数
    ----
    axis      : 期望的工具轴方向（世界系）；None = 只约束位置（6 轴臂还有 3 个冗余）。
    q_seed    : 迭代起点；同时用零空间投影把解往它拉，避免拧成奇怪姿态。
    damping   : 阻尼最小二乘的 λ，越大越保守（接近奇异时靠它救命）。
    step      : 每次迭代的步长比例。

    返回 ``(q, 位置误差[m], 工具轴夹角[rad])``；迭代结束时 ``data`` 停在解上，可直接读 site。
    """
    sid = site_id(model, site)
    if q_seed is None:
        q_seed = arm_qpos(model, data)
    q_seed = np.asarray(q_seed, dtype=float).copy()
    q = q_seed.copy()
    target_pos = np.asarray(target_pos, dtype=float)
    target_axis = None if axis is None else np.asarray(axis, dtype=float)
    if target_axis is not None:
        target_axis = target_axis / np.linalg.norm(target_axis)

    dofs = [dof_adr(model, j) for j in ARM_JOINTS]
    lo = np.array([model.jnt_range[jid(model, j)][0] for j in ARM_JOINTS])
    hi = np.array([model.jnt_range[jid(model, j)][1] for j in ARM_JOINTS])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    e_p, e_a = np.inf, np.inf

    for _ in range(iters):
        set_arm_qpos(model, data, q)
        mujoco.mj_forward(model, data)
        p = np.array(data.site_xpos[sid])
        z = data.site_xmat[sid].reshape(3, 3)[:, 2].copy()
        e_p = float(np.linalg.norm(target_pos - p))

        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, dofs], jacr[:, dofs]])            # 6 x 6
        err = np.concatenate([target_pos - p, np.zeros(3)])
        if target_axis is not None:
            # 让 z 轴转向 target_axis：小角度下最省力的角速度就是 z × target_axis
            err[3:] = np.cross(z, target_axis)
            e_a = float(np.arccos(np.clip(np.dot(z, target_axis), -1.0, 1.0)))
        else:
            e_a = 0.0
        if e_p < pos_tol and e_a < axis_tol:
            break

        # 阻尼最小二乘：dq = J^T (J J^T + λ² I)^-1 err
        JJt = J @ J.T + (damping ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        # 零空间：往 seed 方向靠，保持姿态自然（J4/J6 轴平行，不约束会乱拧）
        if null_gain > 0:
            Jpinv = J.T @ np.linalg.inv(JJt)
            dq = dq + (np.eye(6) - Jpinv @ J) @ (null_gain * (q_seed - q))
        q = np.clip(q + step * dq, lo, hi)

    set_arm_qpos(model, data, q)
    mujoco.mj_forward(model, data)
    p = np.array(data.site_xpos[sid])
    z = data.site_xmat[sid].reshape(3, 3)[:, 2]
    e_p = float(np.linalg.norm(target_pos - p))
    if target_axis is None:
        e_a = 0.0
    else:
        e_a = float(np.arccos(np.clip(np.dot(z / np.linalg.norm(z), target_axis), -1.0, 1.0)))
    return q, e_p, e_a


def refine_ik_pose(model, data, site, target_pos, axis=None, q_seed=None,
                   iters: int = 400) -> tuple[np.ndarray, float, float]:
    """求解质量不够时用更小的阻尼再迭代一轮（不做零空间偏置，纯精度）。"""
    return solve_ik_pose(model, data, site, target_pos, axis, q_seed,
                         iters=iters, damping=5e-3, step=0.35, null_gain=0.0)


# ----------------------------------------------------------------------------- 运动
def smoothstep(n: int, eps: float = 0.1) -> np.ndarray:
    """0->1 的 S 型插值（首尾速度都为 0，最不容易激起振动）。"""
    t = np.linspace(0.0, 1.0, max(n, 2))
    s = np.clip((t - eps) / (1.0 - 2 * eps), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def reset_home(model: mujoco.MjModel, data: mujoco.MjData, key: str = "home") -> bool:
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
    if kid < 0:
        return False
    mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    return True


def drive_to(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_target,
    *,
    seconds: float = 2.0,
    settle: float = 0.4,
    q_start=None,
    callback=None,
) -> None:
    """把 6 个位置驱动器的 ctrl 从当前值平滑开到 q_target，再静置 settle 秒。

    ``callback(k, data)`` 每仿真步调用一次，用于录视频 / 记录数据（k 是步序号）。
    """
    dt = model.opt.timestep
    if q_start is None:
        q_start = np.array([data.ctrl[dof_adr(model, j)] for j in ARM_JOINTS])
    q_start = np.asarray(q_start, dtype=float)
    q_target = np.asarray(q_target, dtype=float)
    n = max(int(seconds / dt), 1)
    traj = smoothstep(n)
    k = 0
    for i in range(n):
        val = q_start + (q_target - q_start) * traj[i]
        for j, v in zip(ARM_JOINTS, val):
            data.ctrl[dof_adr(model, j)] = v
        mujoco.mj_step(model, data)
        if callback:
            callback(k, data)
        k += 1
    for j, v in zip(ARM_JOINTS, q_target):
        data.ctrl[dof_adr(model, j)] = v
    for _ in range(max(int(settle / dt), 1)):
        mujoco.mj_step(model, data)
        if callback:
            callback(k, data)
        k += 1


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="末端位姿数值 IK 小工具")
    ap.add_argument("--scene", default=str(here / "assets" / "revA1_scene.xml"))
    ap.add_argument("--target", nargs=3, type=float, default=[0.45, 0.0, 0.25],
                    help="目标位置（世界系，米）")
    ap.add_argument("--axis", nargs=3, type=float, default=[0.0, 0.0, -1.0],
                    help="工具轴期望方向（世界系）；0 0 0 表示不约束")
    ap.add_argument("--site", default="tool_site")
    ap.add_argument("--seed", nargs=6, type=float, default=None,
                    help="迭代起点（弧度），默认用 home 姿态")
    args = ap.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    if not reset_home(model, data):
        mujoco.mj_forward(model, data)
    seed = np.array(args.seed) if args.seed else arm_qpos(model, data)
    axis = None if not any(args.axis) else args.axis

    q, e_p, e_a = solve_ik_pose(model, data, args.site, args.target, axis, q_seed=seed)
    print(f"目标位置 = {np.round(args.target, 4)}   工具轴 = {axis}")
    print(f"解       = {np.round(q, 4)}")
    print(f"位置误差 = {e_p * 1000:.2f} mm    姿态误差 = {np.degrees(e_a):.2f}°")
    print(f"到位后 {args.site} = {np.round(site_pos(model, data, args.site), 4)}"
          f"   工具轴 = {np.round(tool_axis(model, data, args.site), 4)}")
    return 0 if (e_p < 5e-3 and e_a < np.radians(2)) else 1


if __name__ == "__main__":
    sys.exit(main())

