#!/usr/bin/env python3
"""可视化 / 交互调试：开 MuJoCo 窗口，用滑条和键盘玩这个机械臂。

    python viewer.py                                        # 开窗口（WSL 需要 WSLg / X 转发）
    python viewer.py --headless --seconds 3 --out runs/h.png # 无窗口：离屏渲染出图

窗口里的操作
------------
   空格        暂停 / 继续物理
   g           重力开关（关掉重力看纯运动学效果）
   r           复位到 home
   1 / 2 / 3 / 4  四个预设姿态（用 IK 解到指定工具位置，启动时会打印实际目标）
   [ / ]       当前 kp 整体缩小 / 放大（看伺服软硬的区别）
   ESC 或关窗口 退出

窗口左侧 "Control" 面板里的滑条就是 6 个关节的 ctrl（弧度），拖它就能让手臂动起来。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl(verbose=True)
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

import revA1_spec as spec  # noqa: E402
import sim_ik as ik  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent

# 预设姿态：用"工具尖要到哪 + 工具轴朝下"定义，启动时解 IK 得到关节角（标签=实际目标）。
PRESET_TARGETS = [
    ("1", "home", (0.40, 0.00, 0.179), (0, 0, -1)),
    ("2", "前伸俯视", (0.55, 0.00, 0.420), (0, 0, -1)),
    ("3", "侧向左前", (0.30, 0.30, 0.450), (0, 0, -1)),
    ("4", "低位对位", (0.45, -0.15, 0.150), (0, 0, -1)),
]


def build_presets(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, tuple[str, np.ndarray]]:
    """把上面 4 个目标点解成关节角；解不出来的就用 home 顶上并打印提示。"""
    out: dict[str, tuple[str, np.ndarray]] = {}
    seed = np.array(spec.HOME_QPOS)
    for key, label, target, axis in PRESET_TARGETS:
        q, e_p, e_a = ik.solve_ik_pose(model, data, "tool_site", target, axis, q_seed=seed)
        if e_p > 3e-3:
            q, e_p, e_a = ik.refine_ik_pose(model, data, "tool_site", target, axis,
                                            q_seed=q)
        ok = e_p < 5e-3
        out[key] = (f"{label} -> {np.round(target, 2)}", q if ok else seed)
        print(f"  预设 {key}: {label:8s} 目标={np.round(target, 3)} "
              f"IK误差={e_p * 1000:.2f} mm {'OK' if ok else 'FAIL(用 home 代替)'}")
    return out


class State:
    """窗口回调与主循环之间共享的状态。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.paused = False
        self.gravity_on = True
        self.scale = 1.0
        self.kp0 = np.array([model.actuator_gainprm[i][0] for i in range(model.nu)])
        self.kv0 = np.array([-model.actuator_biasprm[i][2] for i in range(model.nu)])

    def set_gains(self, scale: float) -> None:
        """整体缩放 kp/kv（直观感受"伺服软硬"对跟踪和抖动的影响）。"""
        self.scale = scale
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = self.kp0[i] * scale
            self.model.actuator_biasprm[i, 1] = -self.kp0[i] * scale
            self.model.actuator_biasprm[i, 2] = -self.kv0[i] * scale

    def goto(self, q) -> None:
        """把 ctrl 设成目标（位置伺服自己会平滑过去）。"""
        for j, v in zip(spec.JOINTS, np.asarray(q).ravel()):
            self.data.ctrl[ik.dof_adr(self.model, j)] = v

    def reset(self) -> None:
        if not ik.reset_home(self.model, self.data):
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)


def run_headless(model, data, st, args) -> int:
    """无窗口模式：离屏渲染成 PNG 序列（不需要显示器）。"""
    import imageio.v2 as imageio

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    step_per_frame = max(int(round(1.0 / args.fps / model.opt.timestep)), 1)
    n_frames = max(int(args.seconds * args.fps), 1)
    renderer = mujoco.Renderer(model, height=480, width=640)
    for k in range(n_frames):
        for _ in range(step_per_frame):
            mujoco.mj_step(model, data)
        # 注意：mujoco 3.8 里 camera=None 不能传给 update_scene，必须显式分两种情况
        if args.camera:
            renderer.update_scene(data, camera=args.camera)
        else:
            renderer.update_scene(data)
        imageio.imwrite(out.with_name(f"{out.stem}_{k:04d}{out.suffix}"), renderer.render())
    print(f"已输出 {n_frames} 帧 -> {out.parent}（640x480，每帧 {step_per_frame} 个仿真步）")
    p = ik.site_pos(model, data, "tool_site")
    print(f"末态 tool_site = {np.round(p, 4)}  离地 {p[2] - spec.FLOOR_Z:.3f} m  "
          f"接触 {data.ncon} 对")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TB6-R5-RevA1 MuJoCo 可视化")
    ap.add_argument("--scene", default=str(spec.SCENE_XML))
    ap.add_argument("--camera", default=None, help="固定相机名（默认自由视角）")
    ap.add_argument("--kp-scale", type=float, default=1.0, help="kp/kv 整体缩放")
    ap.add_argument("--headless", action="store_true", help="不起窗口，离屏渲染")
    ap.add_argument("--seconds", type=float, default=3.0, help="headless 模式跑多久")
    ap.add_argument("--fps", type=int, default=30, help="headless 模式出图帧率")
    ap.add_argument("--out", default=str(HERE / "runs" / "viewer.png"),
                    help="headless 出图路径（自动加帧号）")
    args = ap.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    st = State(model, data)
    st.reset()
    if args.kp_scale != 1.0:
        st.set_gains(args.kp_scale)

    print(f"关节顺序 : {spec.JOINTS}")
    print(f"home qpos: {np.round(spec.HOME_QPOS, 4)}")
    print("预设姿态（启动时用 IK 解出来的）:")
    presets = build_presets(model, data)
    st.reset()
    print("按键     : 空格=暂停 g=重力 r=复位 1/2/3/4=预设 [ ]=kp 缩放 关窗口=退出")

    if args.headless:
        return run_headless(model, data, st, args)

    def on_key(keycode: int) -> None:
        ch = chr(keycode) if 0 < keycode < 256 else ""
        if ch == " ":
            st.paused = not st.paused
            print(f"-> {'暂停' if st.paused else '继续'}")
        elif ch in ("g", "G"):
            st.gravity_on = not st.gravity_on
            model.opt.gravity[:] = (0, 0, -9.81) if st.gravity_on else (0, 0, 0)
            print(f"-> 重力 {'开' if st.gravity_on else '关'}")
        elif ch in ("r", "R"):
            st.reset()
            print("-> 复位到 home")
        elif ch in presets:
            label, q = presets[ch]
            st.goto(q)
            print(f"-> 预设 {ch}: {label}  ctrl={np.round(q, 3)}")
        elif ch == "[":
            st.set_gains(max(st.scale * 0.5, 0.01))
            print(f"-> kp/kv 缩放 = {st.scale:.3f}")
        elif ch == "]":
            st.set_gains(min(st.scale * 2.0, 100.0))
            print(f"-> kp/kv 缩放 = {st.scale:.3f}")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.cam.lookat[:] = (0.1, 0.0, 0.35)

        last = time.time()
        while viewer.is_running():
            t0 = time.time()
            if not st.paused:
                mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)   # 实时跑，避免慢动作
            if dt > 0:
                time.sleep(dt)
            if time.time() - last > 5.0:
                last = time.time()
                p = ik.site_pos(model, data, "tool_site")
                print(f"  tool_site = {np.round(p, 3)}  离地 {p[2] - spec.FLOOR_Z:.3f} m"
                      f"  接触 {data.ncon} 对")
    return 0


if __name__ == "__main__":
    sys.exit(main())

