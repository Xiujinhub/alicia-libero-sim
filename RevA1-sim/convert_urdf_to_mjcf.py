#!/usr/bin/env python3
"""把 TB6-R5-RevA1 的 URDF 转成 MuJoCo 用的 MJCF + 场景文件。

    python convert_urdf_to_mjcf.py                # 用 revA1_spec.py 里的默认参数
    python convert_urdf_to_mjcf.py --help         # 看所有可调项
    python convert_urdf_to_mjcf.py --arm-only     # 只生成机器人本体

产物（默认都在 assets/ 下）::

    revA1_arm.xml      机器人本体（自包含：mesh 已拷到 meshes/，含驱动器/site/接触排除）
    revA1_scene.xml    场景（include 本体 + 地面 + 灯光 + 相机 + home keyframe）
    meshes/*.STL       从 RobotSDK/TB6-R5-RevA1/meshes 拷来的网格

细节（改了 URDF 也能直接复用）见 model_import.py 顶部的说明。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl()
import mujoco  # noqa: E402

import model_import as mi  # noqa: E402
import revA1_spec as spec  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def sanity_check(model: mujoco.MjModel, data: mujoco.MjData, qpos,
                 floor_z: float, site: str = "tool_site") -> tuple[str, float]:
    """home 姿态下的一次前向计算：位置 / 工具轴 / 自碰撞 / 重力力矩。

    返回 ``(可打印的一行摘要, tool_site 离地高度)``。
    """
    data.qpos[:] = 0.0
    for i, name in enumerate(spec.JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[jid])] = qpos[i]
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    pos = data.site_xpos[sid]
    axis = data.site_xmat[sid].reshape(3, 3)[:, 2]
    tau = data.qfrc_bias[:6]
    text = (f"{site}={np.round(pos, 3)} (离地 {pos[2] - floor_z:.3f} m) "
            f"工具轴={np.round(axis, 3)} 自碰撞={data.ncon} 对 "
            f"重力力矩={np.round(tau, 1)}")
    return text, float(pos[2] - floor_z)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="TB6-R5-RevA1 URDF -> MuJoCo MJCF + 场景",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--urdf", default=str(spec.URDF), help="输入 URDF")
    ap.add_argument("--package-root", default=str(spec.PACKAGE_ROOT),
                    help="用于解析 package:// 的目录（mesh 所在的包根）")
    ap.add_argument("--out-dir", default=str(spec.ASSETS), help="输出目录")
    ap.add_argument("--name", default=spec.NAME, help="生成文件名前缀")
    ap.add_argument("--home", nargs=6, type=float, default=list(spec.HOME_QPOS),
                    help="home/ready 姿态的 6 个关节角（弧度）")
    ap.add_argument("--floor-z", type=float, default=None,
                    help="地面高度；默认自动量 base_link mesh 的最低点")
    ap.add_argument("--arm-only", action="store_true", help="只生成机器人本体，不生成场景")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    print("=" * 78)
    print("TB6-R5-RevA1  ->  MuJoCo MJCF")
    print("=" * 78)

    res = mi.build_arm_mjcf(
        args.urdf, out_dir / f"{args.name}_arm.xml",
        package_root=args.package_root, model_name=args.name,
        gains=spec.GAINS, sites=spec.SITES, verbose=True)
    model = res["model"]

    floor_z = args.floor_z
    if floor_z is None:
        floor_z = mi.geom_world_min_z(model, "base_link_geom")
        print(f"地面高度: z = {floor_z:.4f}（自动量取 base_link mesh 最低点）")

    data = mujoco.MjData(model)
    summary, tip_z = sanity_check(model, data, args.home, floor_z)
    print(f"home 自检: {summary}")

    if not args.arm_only:
        scene = mi.build_scene(
            res["arm_xml"], out_dir / f"{args.name}_scene.xml", name=args.name,
            floor_z=floor_z,
            target_xy=(spec.TARGET_POSE[0], spec.TARGET_POSE[1]),
            home_qpos=args.home,
            home_tip_z=tip_z,
            stat_center_z=0.5)
        # 场景能不能编译，这里也顺手验一遍
        mujoco.MjModel.from_xml_path(str(scene))
        print(f"场景    : {scene}  (编译通过)")

    print("\n下一步：")
    print("    python check_model.py       # 结构/质量/工作空间/稳定性自检")
    print("    python viewer.py            # 可视化（可拖关节滑条）")
    print("    python demo_trajectory.py   # 自动跑一段轨迹")
    return 0


if __name__ == "__main__":
    sys.exit(main())
