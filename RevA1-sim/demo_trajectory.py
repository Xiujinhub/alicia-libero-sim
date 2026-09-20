#!/usr/bin/env python3
"""演示：让这台机械臂自动跑几段轨迹（关节空间 / 笛卡尔空间）。

    python demo_trajectory.py                           # 关节空间：home -> 前伸 -> 收拢 -> home
    python demo_trajectory.py --cartesian               # 笛卡尔：工具尖沿正方形+圆走（IK 驱动）
    python demo_trajectory.py --view                    # 边跑边开窗口看（需要显示环境）
    python demo_trajectory.py --video runs/demo.mp4     # 录视频（离屏渲染，无需显示器）
    python demo_trajectory.py --cartesian --video runs/cart.mp4 --camera overview_cam

两种演示分别验证什么
--------------------
· 关节空间：位置伺服带着 ~90kg 的重力负载能不能稳稳地"从 A 摆到 B"（看跟踪误差）。
· 笛卡尔空间：每走一小步都用数值 IK 解一次关节角，验证"工具尖指哪打哪"的精度，
  这也是以后接 lerobot 做训练 / 遥操作的基础。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl()
import mujoco  # noqa: E402

import revA1_spec as spec  # noqa: E402
import sim_ik as ik  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 关节空间路点：用"工具尖要到哪里（世界系）+ 工具轴朝下"来定义，运行时解 IK。
# 这样每个路点的含义都是可验证的（IK 误差 < 0.1mm，实测见 check_model.log）。
# 首尾两个点直接用 revA1_spec.TARGET_POSE（= home 姿态的工具尖位置），避免两处数值漂移。
_HOME_T = tuple(spec.TARGET_POSE)
WP_TARGETS = [
    ("home         ", _HOME_T, (0, 0, -1)),
    ("抬起 0.25 m  ", (0.40, 0.00, 0.430), (0, 0, -1)),
    ("侧移 -0.30 m ", (0.40, -0.30, 0.430), (0, 0, -1)),
    ("下降取件      ", (0.40, -0.30, 0.200), (0, 0, -1)),
    ("回到 home     ", _HOME_T, (0, 0, -1)),
]


def cartesian_path(center_xy=(spec.TARGET_POSE[0], spec.TARGET_POSE[1]),
                   height=0.35, side=0.16, radius=0.09):
    """在水平面上生成"正方形 + 圆"的笛卡尔路点（工具轴始终朝下）。"""
    cx, cy = center_xy
    z = spec.FLOOR_Z + height
    pts = []
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for (sx, sy), (ex, ey) in zip(corners, corners[1:] + corners[:1]):
        for t in np.linspace(0.0, 1.0, 9):
            pts.append((cx + side / 2 * (sx + (ex - sx) * t),
                        cy + side / 2 * (sy + (ey - sy) * t), z))
    for t in np.linspace(0.0, 2 * np.pi, 25):
        pts.append((cx + radius * np.cos(t), cy + radius * np.sin(t), z))
    return pts


class Recorder:
    """按固定帧率把渲染结果存下来，最后写成 mp4 / gif。"""

    def __init__(self, model, path: Path, fps: int, camera: str | None):
        import imageio.v2 as imageio
        self.imageio = imageio
        self.path = path
        self.fps = fps
        self.camera = camera
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.n = max(int(round(1.0 / fps / model.opt.timestep)), 1)
        self.frames: list = []

    def on_step(self, k: int, data: mujoco.MjData) -> None:
        if k % self.n:
            return
        # mujoco 3.8 里 camera=None 不能传给 update_scene，得分成两种情况
        if self.camera:
            self.renderer.update_scene(data, camera=self.camera)
        else:
            self.renderer.update_scene(data)
        self.frames.append(self.renderer.render())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.imageio.mimsave(self.path, self.frames, fps=self.fps, quality=8)
        print(f"视频已保存: {self.path}  ({len(self.frames)} 帧 @ {self.fps} fps)")


def run_waypoint_demo(model, data, args, cb) -> None:
    """5 个"工具尖目标点"依次到位，关节空间做平滑插值（最接近真机点到点运动）。"""
    print("\n=== 路点演示：工具尖到指定点（工具轴竖直朝下），关节空间平滑过渡 ===")
    print(f"{'阶段':<15}{'目标':>24}{'实际':>24}{'误差(mm)':>10}{'轴偏差(°)':>10}"
          f"{'接触':>6}{'末态误差(°)':>13}")
    q_seed = np.array(spec.HOME_QPOS)
    for label, target, axis in WP_TARGETS:
        q, _, _ = ik.solve_ik_pose(model, data, "tool_site", target, axis, q_seed=q_seed)
        ik.drive_to(model, data, q, seconds=args.seconds, settle=0.5, callback=cb)
        p = ik.site_pos(model, data, "tool_site")
        z = ik.tool_axis(model, data, "tool_site")
        e_p = float(np.linalg.norm(p - np.array(target)))
        e_a = float(np.degrees(np.arccos(np.clip(np.dot(z, np.array(axis)), -1, 1))))
        err = np.degrees(np.abs(ik.arm_qpos(model, data) - q)).max()
        print(f"{label:<15}{str(np.round(target, 3)):>24}{str(np.round(p, 3)):>24}"
              f"{e_p * 1000:>10.2f}{e_a:>10.2f}{data.ncon:>6}{err:>13.3f}")
        q_seed = q


def run_cartesian_demo(model, data, args, cb) -> None:
    print("\n=== 笛卡尔演示：工具尖朝下画正方形 + 圆（每步都解 IK）===")
    q_seed = np.array(spec.HOME_QPOS)
    pts = cartesian_path()
    if not ik.reset_home(model, data):
        mujoco.mj_forward(model, data)
    for j, v in zip(spec.JOINTS, q_seed):
        data.ctrl[ik.dof_adr(model, j)] = v
    errs = []
    print(f"{'路点':>6}{'目标':>26}{'实际':>26}{'误差(mm)':>10}{'接触':>6}")
    for i, p in enumerate(pts):
        q, e_p, _ = ik.solve_ik_pose(model, data, "tool_site", p, (0, 0, -1),
                                     q_seed=q_seed)
        if e_p > 3e-3:                       # 单次解不动就再精修一轮
            q, e_p, _ = ik.refine_ik_pose(model, data, "tool_site", p, (0, 0, -1),
                                          q_seed=q)
        ik.drive_to(model, data, q, seconds=args.segment, settle=0.05, callback=cb)
        got = ik.site_pos(model, data, "tool_site")
        err = float(np.linalg.norm(got - np.array(p)))
        errs.append(err)
        if i % 8 == 0 or i == len(pts) - 1:
            print(f"{i:>6}{str(np.round(p, 3)):>26}{str(np.round(got, 3)):>26}"
                  f"{err * 1000:>10.2f}{data.ncon:>6}")
        q_seed = q
    errs = np.array(errs)
    print(f"到位误差: 平均 {errs.mean() * 1000:.2f} mm  最大 {errs.max() * 1000:.2f} mm"
          f"  （{len(pts)} 个路点，含伺服跟踪误差）")


def _view_demo(model, data, args) -> None:
    """带窗口的演示：物理由本循环推进（launch_passive 不自己跑物理）。"""
    import time

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -18
        viewer.cam.lookat[:] = (0.1, 0.0, 0.35)

        def spin(steps: int) -> bool:
            for _ in range(steps):
                if not viewer.is_running():
                    return False
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(model.opt.timestep - 0.0005, 0.0))
            return True

        if args.cartesian:
            print("窗口模式: 笛卡尔演示（关窗口即退出）")
            q_seed = np.array(spec.HOME_QPOS)
            for p in cartesian_path():
                q, e_p, _ = ik.solve_ik_pose(model, data, "tool_site", p, (0, 0, -1),
                                             q_seed=q_seed)
                for j, v in zip(spec.JOINTS, q):
                    data.ctrl[ik.dof_adr(model, j)] = v
                if not spin(int(args.segment / model.opt.timestep)):
                    return
                q_seed = q
        else:
            print("窗口模式: 路点演示（关窗口即退出）")
            q_seed = np.array(spec.HOME_QPOS)
            for label, target, axis in WP_TARGETS:
                q, e_p, _ = ik.solve_ik_pose(model, data, "tool_site", target, axis,
                                             q_seed=q_seed)
                print(f"-> {label}  target={np.round(target, 3)}  IK误差={e_p * 1000:.2f}mm")
                for j, v in zip(spec.JOINTS, q):
                    data.ctrl[ik.dof_adr(model, j)] = v
                if not spin(int((args.seconds + 0.5) / model.opt.timestep)):
                    return
                q_seed = q


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TB6-R5-RevA1 MuJoCo 演示")
    ap.add_argument("--scene", default=str(spec.SCENE_XML))
    ap.add_argument("--cartesian", action="store_true", help="跑笛卡尔轨迹（默认关节空间）")
    ap.add_argument("--seconds", type=float, default=1.5, help="关节空间每段用时")
    ap.add_argument("--segment", type=float, default=0.35, help="笛卡尔每个路点用时")
    ap.add_argument("--view", action="store_true", help="边跑边开窗口")
    ap.add_argument("--video", default=None, help="录视频路径，例如 runs/demo.mp4")
    ap.add_argument("--camera", default=None, help="固定相机名（默认自由视角）")
    ap.add_argument("--fps", type=int, default=30, help="视频帧率")
    args = ap.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    ik.reset_home(model, data)

    if args.view:
        _view_demo(model, data, args)
        return 0

    rec = Recorder(model, Path(args.video), args.fps, args.camera) if args.video else None
    cb = rec.on_step if rec else None
    if args.cartesian:
        run_cartesian_demo(model, data, args, cb)
    else:
        run_waypoint_demo(model, data, args, cb)
    if rec:
        rec.save()
    print("\n完事。想看窗口版：python viewer.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

