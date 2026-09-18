#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""场景拼装：把 Alicia_D_v5_6_gripper_50mm 机械臂 + LIBERO 桌子 + 素材物体 合成为一个 MJCF。

设计要点
--------
* 以 ``Alicia_D_v5_6_gripper_50mm_teleop.xml``（position 位置伺服版）为骨架，
  用 ElementTree 把桌子/物体/相机插进它的 worldbody，保留原有的 actuator 与 contact 排除
* 手臂的 ``meshdir`` 改写成绝对路径，物体网格用「相对该 meshdir 的相对路径」引用，
  这样生成的场景 XML 放在任何目录都能加载
* 物体碰撞体直接用 libero_assets 里**预烘焙的凸盒分解**（已归一化四元数），
  视觉用同目录的 ``.obj``（MuJoCo 不读 OBJ 贴图，故用 catalog 里的纯色材质）
* 摆放：``xy`` 指物体 AABB 中心，``z`` 由「桌面高 - 物体底部偏移」精确算出，不会陷进桌子
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from libero_catalog import (
    catalog_object,
    boxes_aabb,
    load_catalog,
    normalize_quat,
    quat_from_axis_angle,
    rotate_boxes,
)

HERE = Path(__file__).resolve().parent
ARM_XML = (HERE.parent / "Synria-Robot-Descriptions-main" / "synriard" / "mjcf"
           / "Alicia_D_v5_6" / "Alicia_D_v5_6_gripper_50mm_teleop.xml")
SCENE_DIR = HERE / "scenes"

# LIBERO 标准桌面（scenes/libero_tabletop_base_style.xml：table 在 z=0.4、半尺寸 0.4 → 桌面 0.8 m）
TABLE_TOP_Z = 0.8
TABLE_HALF = 0.4
ARM_BASE_POS = [-0.27, 0.0, TABLE_TOP_Z]   # 底座放在桌面靠边，+X 朝桌子中心
ARM_BASE_YAW = 0.0

# 物体接触参数：LIBERO 原值 solref="0.001 1" 在 dt=2ms 下偏硬，放软一点避免抖动/穿透
OBJ_SOLREF = "0.004 1"
OBJ_SOLIMP = "0.95 0.95 0.001"


def indent(elem: ET.Element, level: int = 0) -> None:
    """给 ElementTree 加缩进，输出更易读。"""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


class SceneBuilder:
    """按布局描述生成可直接被 MuJoCo 加载的场景 XML。"""

    def __init__(self, arm_xml: Path = ARM_XML, catalog: dict | None = None) -> None:
        self.arm_xml = Path(arm_xml)
        self.catalog = catalog or load_catalog()
        raw_meshdir = ET.parse(self.arm_xml).getroot().find("compiler").get("meshdir", "")
        self.mesh_dir = (self.arm_xml.parent / raw_meshdir).resolve()

    # ── 物体摆放计算 ──
    def placed_boxes(self, key: str, xy, yaw_deg: float = 0.0):
        """返回物体在世界系下的碰撞盒，以及让 AABB 中心落在 xy 上的 body 位置。"""
        obj = catalog_object(self.catalog, key)
        boxes = [([float(v) for v in pos], [float(v) for v in quat], [float(v) for v in size])
                 for pos, quat, size in obj["boxes"]]
        if abs(yaw_deg) > 1e-6:
            boxes = rotate_boxes(boxes, quat_from_axis_angle("z", yaw_deg))
        lo, hi = boxes_aabb(boxes)
        pos = [float(xy[0]) - (lo[0] + hi[0]) / 2.0,
               float(xy[1]) - (lo[1] + hi[1]) / 2.0,
               TABLE_TOP_Z - float(lo[2])]
        return boxes, pos, lo, hi

    # ── 生成 XML ──
    def build(self, layout: dict, out_path: Path | None = None) -> Path:
        tree = ET.parse(self.arm_xml)
        root = tree.getroot()
        root.set("model", f"alicia_libero_{layout.get('name', 'scene')}")
        root.find("compiler").set("meshdir", self.mesh_dir.as_posix())

        asset = root.find("asset")
        worldbody = root.find("worldbody")

        # 1) 场地：地面 + 桌面
        ET.SubElement(asset, "material", {"name": "scene_floor_mat", "rgba": "0.55 0.57 0.60 1",
                                          "reflectance": "0.15"})
        ET.SubElement(worldbody, "geom", {
            "name": "scene_floor", "type": "plane", "size": "3 3 0.05", "pos": "0 0 0",
            "material": "scene_floor_mat", "contype": "1", "conaffinity": "1",
            "condim": "3", "friction": "0.9 0.05 0.001", "group": "0"})
        ET.SubElement(asset, "material", {"name": "scene_table_mat", "rgba": "0.62 0.48 0.32 1",
                                          "reflectance": "0.1"})
        table = ET.SubElement(worldbody, "body", {"name": "table", "pos": f"0 0 {TABLE_TOP_Z - 0.4}"})
        ET.SubElement(table, "geom", {
            "name": "table_top", "type": "box", "size": f"{TABLE_HALF} {TABLE_HALF} 0.4",
            "pos": "0 0 0", "material": "scene_table_mat", "contype": "1", "conaffinity": "1",
            "condim": "3", "friction": "1 0.02 0.001", "group": "0"})
        ET.SubElement(table, "site", {"name": "table_top_site", "pos": "0 0 0.4",
                                      "size": "0.001 0.001 0.001", "rgba": "0 0 0 0"})

        # 2) 机械臂：挪到桌面上
        base = worldbody.find("body[@name='base_link']")
        if base is None:
            raise RuntimeError("骨架 XML 里找不到 base_link，请确认用的是 Alicia follower 模型")
        base.set("pos", " ".join(f"{v:.4f}" for v in layout.get("arm_base", ARM_BASE_POS)))
        base.set("quat", " ".join(f"{v:.6f}" for v in
                                  quat_from_axis_angle("z", layout.get("arm_yaw", ARM_BASE_YAW))))

        # 3) 物体
        for item in layout.get("objects", []):
            self._add_object(asset, worldbody, item)

        # 4) 额外几何体（目标区域示意等）
        for geom in layout.get("extra_geoms", []):
            ET.SubElement(worldbody, "geom", {k: str(v) for k, v in geom.items()})

        # 4b) IK 目标点标记（mocap 体，界面上拖动夹爪时显示目标位置）
        marker = ET.SubElement(worldbody, "body", {"name": "ik_marker", "mocap": "true",
                                                   "pos": "0 0 0"})
        ET.SubElement(marker, "geom", {"name": "ik_marker_geom", "type": "sphere",
                                       "size": "0.012", "rgba": "1.0 0.25 0.25 0.55",
                                       "contype": "0", "conaffinity": "0", "group": "1",
                                       "condim": "1"})

        # 5) 相机
        for cam in layout.get("cameras", default_cameras()):
            ET.SubElement(worldbody, "camera", {k: str(v) for k, v in cam.items()})
        self._attach_wrist_camera(worldbody)

        # 6) 补一盏灯（原模型只有一盏斜光，桌面会偏暗）
        ET.SubElement(worldbody, "light", {
            "name": "scene_light", "pos": "0.4 -0.4 2.0", "dir": "-0.3 0.3 -1",
            "diffuse": "0.9 0.9 0.9", "specular": "0.2 0.2 0.2", "castshadow": "false"})

        indent(root)
        out_path = Path(out_path) if out_path else SCENE_DIR / f"{layout.get('name', 'scene')}.xml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return out_path

    def _add_object(self, asset: ET.Element, worldbody: ET.Element, item: dict) -> ET.Element:
        obj = catalog_object(self.catalog, item["key"])
        yaw = float(item.get("yaw", 0.0))
        boxes, pos, _, _ = self.placed_boxes(item["key"], item.get("xy", (0.2, 0.0)), yaw)
        pos[2] += float(item.get("z_offset", 0.0))   # 需要放在容器内部时可抬高一点

        body = ET.SubElement(worldbody, "body", {
            "name": item.get("name", obj["name"]),
            "pos": " ".join(f"{v:.5f}" for v in pos),
            "quat": " ".join(f"{v:.6f}" for v in quat_from_axis_angle("z", yaw))})
        ET.SubElement(body, "freejoint", {"name": f"{obj['name']}_joint"})

        if obj["mesh"]:
            mesh_name = f"{obj['name']}_vis"
            file_ref = os.path.relpath(Path(obj["mesh"]), self.mesh_dir).replace("\\", "/")
            ET.SubElement(asset, "mesh", {
                "name": mesh_name, "file": file_ref,
                "scale": " ".join(f"{v:g}" for v in obj["mesh_scale"])})
            ET.SubElement(asset, "material", {
                "name": f"{obj['name']}_mat", "rgba": obj["color"],
                "specular": "0.3", "shininess": "0.4"})
            ET.SubElement(body, "geom", {
                "name": f"{obj['name']}_visual", "type": "mesh", "mesh": mesh_name,
                # ⚠ 视觉网格必须套上同一套"摆正"旋转：libero 素材里碰撞盒是用
                #   ``upright_quat`` 转过再用的，但 .obj 自己还在原始坐标系里
                #   （很多是 Y 轴朝上）。以前漏了这个 quat，结果"瓶子躺在地上 +
                #   旁边立着一个灰色碰撞盒方块"。
                "quat": " ".join(f"{v:.6f}" for v in obj["upright_quat"]),
                "material": f"{obj['name']}_mat", "contype": "0", "conaffinity": "0",
                "group": "1", "mass": "0"})
        else:
            # 连 .obj 都没有的物体（实测只有 cookies：目录里只有 .mtl/.xml/.png）：
            # 用碰撞盒做一份"看得见"的替代外观，否则物体完全不可见
            ET.SubElement(asset, "material", {
                "name": f"{obj['name']}_mat", "rgba": obj["color"],
                "specular": "0.2", "shininess": "0.2"})
            for idx, (bpos, bquat, bsize) in enumerate(boxes):
                ET.SubElement(body, "geom", {
                    "name": f"{obj['name']}_proxy{idx}", "type": "box",
                    "pos": " ".join(f"{v:.5f}" for v in bpos),
                    "quat": " ".join(f"{v:.6f}" for v in normalize_quat(bquat)),
                    "size": " ".join(f"{v:.5f}" for v in bsize),
                    "material": f"{obj['name']}_mat",
                    "contype": "0", "conaffinity": "0", "mass": "0", "group": "2"})

        for idx, (bpos, bquat, bsize) in enumerate(boxes):
            ET.SubElement(body, "geom", {
                "name": f"{obj['name']}_col{idx}", "type": "box",
                "pos": " ".join(f"{v:.5f}" for v in bpos),
                "quat": " ".join(f"{v:.6f}" for v in normalize_quat(bquat)),
                "size": " ".join(f"{v:.5f}" for v in bsize),
                # 容器类物体允许在任务里单独指定密度：libero 素材默认 density=100
                # （相当于 0.1g/cm³ 的泡沫），木托盘只有 57g，被夹爪一碰就滑走
                # （用户实测"机械臂把托盘推开了"）。任务里给容器覆盖成 600(≈木头)。
                "density": f"{float(item.get('density', obj['density'])):g}",
                "friction": "1.0 0.02 0.001", "condim": "3",
                "solref": OBJ_SOLREF, "solimp": OBJ_SOLIMP,
                # 碰撞盒放到 group 3：界面默认不渲染（否则物体外面套着一堆灰方块），
                # 想看碰撞体时在查看器里打开 group 3 即可
                "group": "3", "rgba": "0.75 0.75 0.80 0.35"})
        return body

    @staticmethod
    def _attach_wrist_camera(worldbody: ET.Element) -> None:
        """把腕部相机挂到带 tool0_site 的那个 body 上。"""
        for body in worldbody.iter("body"):
            if any(site.get("name") == "tool0_site" for site in body.findall("site")):
                ET.SubElement(body, "camera", {
                    "name": "cam_wrist", "pos": "0 0 0.035",
                    "xyaxes": "1 0 0  0 1 0",      # 沿工具 -Z（接近方向）看向物体
                    "fovy": "75", "mode": "fixed"})
                return


def look_at_xyaxes(position, target) -> str:
    """由「相机位置 + 看向的目标点」算出 MuJoCo 相机需要的 ``xyaxes``。

    ⚠ 踩坑：手写 xyaxes 极易搞错方向——MuJoCo 相机沿自身 **-Z** 看，图片的上方是 **+Y**，
    之前 cam_front 手写的 xyaxes 叉乘出来正好背对桌子，界面里只能看到地板；
    cam_wrist 也写反了（朝天上）。这里统一用 look-at 计算，从构造上保证朝向正确。
    """
    pos = np.asarray(position, dtype=float)
    tgt = np.asarray(target, dtype=float)
    z_axis = pos - tgt                                    # 相机 -Z 指向目标
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z_axis, up_hint))) > 0.995:       # 正上/正下视角，换一个参考上方向
        up_hint = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    axes = np.concatenate([x_axis, y_axis])
    return " ".join(f"{v:.6f}" for v in axes)


def camera_spec(name: str, position, target, fovy: float | None = None) -> dict:
    spec = {"name": name, "pos": " ".join(f"{v:.4f}" for v in position),
            "xyaxes": look_at_xyaxes(position, target), "mode": "fixed"}
    if fovy is not None:
        spec["fovy"] = f"{fovy:g}"
    return spec


def default_cameras() -> list[dict]:
    """Qt 3D 视图用的几个机位（都看向桌面工作区，保证机械臂和物体都占足够画面）。"""
    return [
        # 斜前方：从桌子外侧斜上方看，工作区占画面主体
        camera_spec("cam_front", (0.52, -0.60, 1.16), (0.06, 0.0, 0.87), fovy=42),
        # 正上方俯视：看物体布局和落点
        camera_spec("cam_top", (0.02, 0.00, 1.72), (0.02, 0.0, 0.80), fovy=50),
        # 侧前方：从机械臂那一侧看（能看到臂的姿态和夹爪）
        camera_spec("cam_side", (-0.62, -0.70, 1.12), (0.06, 0.0, 0.86), fovy=45),
    ]
