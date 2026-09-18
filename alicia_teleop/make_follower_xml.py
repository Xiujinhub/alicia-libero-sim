#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Synria-Robot-Descriptions 里的 MJCF 转成「可遥操作」的版本。

背景
----
官方遥操作示例（Teleoperation-SDK / 02_demo_mujoco_follower.py）要求模型里有一套
位置伺服 actuator：``pos1..pos6`` / ``pos_grip_l`` / ``pos_grip_r``，
这样脚本只要往 ``data.ctrl`` 写「目标角度」即可。

但 Synria-Robot-Descriptions 里的模型并不满足：

* ``Alicia_D_v5_6_gripper_50mm.xml``  → 完全没有 <actuator>（nu = 0）
* ``Alicia_D_v5_6_gripper_100mm.xml`` → 只有 ``actuator1..actuator8``，
  其中 6 个臂关节是纯力矩 actuator（``dyntype=none biastype=none``，ctrl = 力矩）

本脚本生成 ``<原名>_teleop.xml``，做三件事：

1. 用 <position> 位置伺服 actuator 替换 <actuator> 段（kp / kv 对齐官方遥操作模型）
2. 补上 <option>（timestep / integrator）与 <default>（joint damping / armature）；
   原模型没有关节阻尼与 armature，高 kp 位置伺服在仿真里会抖振
3. 保留原 ``<compiler meshdir="...">``，因此生成文件默认与原 MJCF 同目录；
   用 ``--outdir`` 输出到别处时会自动把 meshdir 改写成绝对路径

用法
----
    python make_follower_xml.py                      # 默认转换 50mm + 100mm follower
    python make_follower_xml.py --src path/to/any.xml
    python make_follower_xml.py --src any.xml --outdir my_models
    python make_follower_xml.py --list               # 列出所有可转换的 MJCF
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_MJCF = HERE.parent / "Synria-Robot-Descriptions-main" / "synriard" / "mjcf"

# 官方遥操作模型的增益（Alicia_D_v5_6 / gripper_50mm / alicia_d_follower.xml）
ARM_GAINS = [(800.0, 40.0), (800.0, 40.0), (800.0, 40.0), (400.0, 20.0), (300.0, 15.0), (300.0, 15.0)]
FINGER_GAIN = (200.0, 10.0)

DEFAULT_JOINT_DAMPING = 1.0
DEFAULT_JOINT_ARMATURE = 0.02

ARM_JOINTS = [f"Joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["left_finger", "right_finger"]


# ─────────────────────────── helpers ───────────────────────────
def find_child(root: ET.Element, tag: str) -> ET.Element | None:
    """返回 root 下第一个名为 tag 的直接子元素。"""
    return root.find(tag)


def joint_names(root: ET.Element) -> list[str]:
    """按出现顺序收集所有 joint 名字。"""
    return [j.get("name") for j in root.iter("joint") if j.get("name")]


def mesh_dir_of(root: ET.Element) -> str:
    """读取 <compiler meshdir>。"""
    compiler = find_child(root, "compiler")
    if compiler is None:
        return ""
    return compiler.get("meshdir", "")


def insert_params_blocks(root: ET.Element) -> None:
    """补 <option> / <visual> / <default>，并按 MuJoCo 要求的顺序重排顶层元素。

    MuJoCo 对 compiler / option / size / visual / statistic / default 有顺序要求，
    必须排在 asset、worldbody 之前，因此统一插到 <compiler> 后面。
    """
    if find_child(root, "compiler") is None:
        root.insert(0, ET.Element("compiler", {"angle": "radian"}))

    if find_child(root, "option") is None:
        ET.SubElement(
            root,
            "option",
            {
                "timestep": "0.002",
                "gravity": "0 0 -9.81",
                "integrator": "implicit",
                "iterations": "100",
                "noslip_iterations": "20",
                "impratio": "10",
            },
        )

    if find_child(root, "visual") is None:
        vis = ET.SubElement(root, "visual")
        ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "720"})

    if find_child(root, "default") is None:
        default = ET.SubElement(root, "default")
        ET.SubElement(
            default,
            "joint",
            {"damping": str(DEFAULT_JOINT_DAMPING), "armature": str(DEFAULT_JOINT_ARMATURE)},
        )
    else:
        print(
            "[WARN] 源文件已有 <default>，未自动注入 joint damping/armature；"
            '若仿真抖动请手动加 <joint damping="1.0" armature="0.02"/>'
        )

    order = {"compiler": 0, "option": 1, "size": 2, "visual": 3, "statistic": 4,
             "default": 5, "asset": 6, "worldbody": 7}
    children = sorted(list(root), key=lambda e: order.get(e.tag, 99))
    for child in children:
        root.remove(child)
    for child in children:
        root.append(child)


def build_actuators(root: ET.Element) -> int:
    """用位置伺服替换 / 新建 <actuator> 段，返回 actuator 数量。"""
    names = joint_names(root)
    missing = [n for n in ARM_JOINTS if n not in names]
    if missing:
        raise SystemExit(f"[ERROR] 模型里找不到关节 {missing}，无法生成遥操作 actuator")

    old = find_child(root, "actuator")
    if old is not None:
        root.remove(old)

    actuator = ET.Element("actuator")
    for idx, (joint, (kp, kv)) in enumerate(zip(ARM_JOINTS, ARM_GAINS), start=1):
        ET.SubElement(
            actuator,
            "position",
            {"name": f"pos{idx}", "joint": joint, "kp": f"{kp:g}", "kv": f"{kv:g}"},
        )
    for joint, name in zip(FINGER_JOINTS, ("pos_grip_l", "pos_grip_r")):
        if joint in names:
            ET.SubElement(
                actuator,
                "position",
                {"name": name, "joint": joint,
                 "kp": f"{FINGER_GAIN[0]:g}", "kv": f"{FINGER_GAIN[1]:g}"},
            )
    root.append(actuator)
    return len(list(actuator))


def convert(src: Path, outdir: Path | None) -> Path:
    """转换单个 MJCF，返回生成文件路径。"""
    tree = ET.parse(src)
    root = tree.getroot()

    insert_params_blocks(root)
    n_act = build_actuators(root)
    root.set("model", f"{root.get('model', src.stem)}_teleop")

    if outdir is None:
        dst = src.with_name(f"{src.stem}_teleop.xml")
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        dst = outdir / f"{src.stem}_teleop.xml"
        meshdir = mesh_dir_of(root)
        if meshdir:
            abs_mesh = (src.parent / meshdir).resolve()
            find_child(root, "compiler").set("meshdir", abs_mesh.as_posix())
            print(f"[INFO] meshdir 改写为绝对路径: {abs_mesh}")

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"[OK] {src.name} → {dst.name}  ({n_act} actuators)")
    return dst


def self_check(dst: Path) -> None:
    """用 MuJoCo 真正加载一次，确认能编译且 actuator 类型符合预期。"""
    try:
        import mujoco
    except ImportError:
        print("[WARN] 未安装 mujoco，跳过自检")
        return

    model = mujoco.MjModel.from_xml_path(str(dst))
    names, kinds = [], []
    for aid in range(model.nu):
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid))
        bias = model.actuator_biastype[aid]
        kinds.append("pos" if bias == int(mujoco.mjtBias.mjBIAS_AFFINE) else "frc")
    print(f"     自检: nq={model.nq} nu={model.nu} {list(zip(names, kinds))}")


def list_candidates() -> list[Path]:
    """列出仓库里所有含 6 轴关节、可转换的 MJCF。"""
    if not REPO_MJCF.is_dir():
        return []
    found = []
    for path in sorted(REPO_MJCF.rglob("*.xml")):
        if path.stem.endswith("_teleop"):
            continue
        names = set(joint_names(ET.parse(path).getroot()))
        if set(ARM_JOINTS).issubset(names):
            found.append(path)
    return found


def default_sources() -> list[Path]:
    """默认转换对象：Alicia_D_v5_6 的 50mm / 100mm follower（对应官方示例机型）。"""
    return [
        REPO_MJCF / "Alicia_D_v5_6" / "Alicia_D_v5_6_gripper_50mm.xml",
        REPO_MJCF / "Alicia_D_v5_6" / "Alicia_D_v5_6_gripper_100mm.xml",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成可用于遥操作的 MJCF（position 位置伺服版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", type=str, default=None, help="源 MJCF 路径")
    parser.add_argument("--outdir", type=str, default=None,
                        help="输出目录；默认与源文件同目录（保持 meshdir 相对路径不变）")
    parser.add_argument("--list", action="store_true", help="列出所有可转换的 MJCF")
    parser.add_argument("--all", action="store_true", help="转换仓库里所有可转换的 MJCF")
    args = parser.parse_args(argv)

    if args.list:
        candidates = list_candidates()
        print(f"可转换的 MJCF（共 {len(candidates)} 个，位于 {REPO_MJCF}）:")
        for path in candidates:
            print(f"  {path.relative_to(REPO_MJCF)}")
        return 0

    outdir = Path(args.outdir).resolve() if args.outdir else None
    if args.src:
        sources = [Path(args.src).resolve()]
    elif args.all:
        sources = list_candidates()
    else:
        sources = default_sources()

    if not sources:
        print("[ERROR] 没找到源 MJCF，请检查 Synria-Robot-Descriptions-main 是否在本目录的上一级")
        return 1

    for src in sources:
        if not src.is_file():
            print(f"[ERROR] 源文件不存在: {src}")
            return 1

    last = None
    for src in sources:
        last = convert(src, outdir)
        self_check(last)

    print("\n完成。可用下面命令打开确认：")
    print(f'  "{HERE.parent / "bin" / "simulate.exe"}" "{last}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())


