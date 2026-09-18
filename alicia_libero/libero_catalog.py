#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LI-BERO 素材分析器：扫描 Datasets/libero_assets，生成可供场景拼装使用的物体目录。

对每个物体做四件事
------------------
1. 解析 ``<name>.xml``：取出 LIBERO 预烘焙的**凸盒分解**碰撞体（type=box）与其
   ``density``，算出物体在自身坐标系下的 AABB（用于精确放到桌面上、判成功等）
2. 找到视觉网格：XML 里写的是 ``visual/xxx_vis.msh``（本仓库未附带 .msh），
   按 ``xxx.obj`` 规则回退到目录里真实存在的 OBJ
3. 判断「摆正姿态」：这些扫描模型的局部轴向五花八门（番茄酱长轴是 Y、黄油长轴是 Z），
   默认规则 = 把最长轴转到 Z（立起来），并允许用 OVERRIDES 单独覆盖
4. 给出抓取提示：50mm 夹爪的最大开口约 45mm，据此判断/提示可行的抓取高度

用法
----
    python libero_catalog.py                # 生成 assets_catalog.json
    python libero_catalog.py --table        # 打印尺寸表格
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS_ROOT = HERE.parent / "Datasets" / "libero_assets"
CATALOG_PATH = HERE / "assets_catalog.json"

GRIP_MAX_OPENING = 0.045      # 50mm 夹爪可用开口（留一点余量）

# 分类（用于 UI 分组与配色）
CATEGORIES = {
    "stable_scanned_objects": "餐具/容器",
    "stable_hope_objects": "食品瓶罐",
    "turbosquid_objects": "日用杂项",
    "scenes": "场景道具",
    "articulated_objects": "可动家具",
}

# 建议颜色（MuJoCo 不读 OBJ 的贴图，用纯色让画面可读）
COLORS = {
    "plate": "0.92 0.92 0.92 1", "akita_black_bowl": "0.12 0.12 0.13 1",
    "red_bowl": "0.75 0.15 0.15 1", "white_bowl": "0.93 0.93 0.90 1",
    "basket": "0.35 0.55 0.30 1", "glazed_rim_porcelain_ramekin": "0.95 0.90 0.80 1",
    "chefmate_8_frypan": "0.20 0.20 0.22 1", "simple_rack": "0.6 0.6 0.62 1",
    "ketchup": "0.75 0.10 0.10 1", "tomato_sauce": "0.80 0.20 0.15 1",
    "alphabet_soup": "0.85 0.55 0.15 1", "orange_juice": "0.95 0.60 0.10 1",
    "bbq_sauce": "0.45 0.20 0.10 1", "butter": "0.95 0.85 0.45 1",
    "chocolate_pudding": "0.35 0.20 0.12 1", "cookies": "0.80 0.65 0.40 1",
    "cream_cheese": "0.90 0.90 0.85 1", "macaroni_and_cheese": "0.90 0.70 0.30 1",
    "milk": "0.90 0.92 0.95 1", "new_salad_dressing": "0.85 0.80 0.60 1",
    "popcorn": "0.90 0.85 0.60 1", "salad_dressing": "0.70 0.65 0.45 1",
    "porcelain_mug": "0.95 0.95 0.95 1", "red_coffee_mug": "0.70 0.15 0.15 1",
    "white_yellow_mug": "0.95 0.90 0.55 1", "wine_bottle": "0.25 0.35 0.25 1",
    "moka_pot": "0.75 0.75 0.78 1", "wooden_tray": "0.60 0.42 0.25 1",
    "wooden_shelf": "0.55 0.38 0.22 1", "wooden_two_layer_shelf": "0.55 0.38 0.22 1",
    "desk_caddy": "0.25 0.30 0.40 1", "bowl_drainer": "0.70 0.72 0.75 1",
    "white_storage_box": "0.90 0.90 0.92 1", "black_book": "0.15 0.15 0.20 1",
    "yellow_book": "0.85 0.70 0.15 1", "wine_rack": "0.50 0.35 0.20 1",
    "wine_rack_stand": "0.45 0.32 0.18 1",
}
DEFAULT_COLOR = "0.7 0.7 0.75 1"

# 姿态覆盖：实测/观察后手动指定（"roll_x/y/z" 为绕该轴旋转的角度，deg）
UPRIGHT_OVERRIDES = {
    # 长轴在局部 Y、需要立起来的瓶子/罐子，脚本会自动处理；
    # 这里只放自动规则不合适的情况：
    "butter": {"axis": "z->y"},          # 黄油块平躺（长轴水平）
    "plate": {"axis": "keep"},           # 盘子本来就是平放
    "chefmate_8_frypan": {"axis": "keep"},
    "basket": {"axis": "keep"},
    "akita_black_bowl": {"axis": "keep"},
    "red_bowl": {"axis": "keep"},
    "white_bowl": {"axis": "keep"},
    "glazed_rim_porcelain_ramekin": {"axis": "keep"},
    "porcelain_mug": {"axis": "keep"},
    "red_coffee_mug": {"axis": "keep"},
    "white_yellow_mug": {"axis": "keep"},
    "wooden_tray": {"axis": "keep"},
    "black_book": {"axis": "x->z"},
    "yellow_book": {"axis": "x->z"},
}


def quat_from_axis_angle(axis: str, angle_deg: float):
    """绕 x/y/z 轴旋转的 (w,x,y,z) 四元数。"""
    half = math.radians(angle_deg) / 2.0
    w, s = math.cos(half), math.sin(half)
    return {"x": (w, s, 0.0, 0.0), "y": (w, 0.0, s, 0.0), "z": (w, 0.0, 0.0, s)}[axis]


def quat_mul(a, b):
    """四元数乘法 a ⊗ b（先 b 后 a）。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_to_mat(quat) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def normalize_quat(quat) -> list:
    """归一化四元数。

    ⚠ 实测发现：libero_assets 里有相当一部分物体的碰撞盒 ``quat`` **没有归一化**
    （例如 butter 写成 ``0.00530 0.00000 0.00530 0.00000``，模长只有 0.0075）。
    直接拿去算旋转矩阵会得到一个近似全零的"旋转"，AABB 会退化成 0。
    因此无论内部计算还是写进场景 XML，都必须先归一化。
    """
    arr = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    return (arr / norm).tolist()


# ─────────────────── 单物体解析 ───────────────────
def parse_object_xml(xml_path: Path) -> dict | None:
    """解析物体 XML，返回碰撞盒 / 视觉网格 / 密度等原始信息。"""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    if root.find(".//body[@name='object']") is None:
        return None

    boxes, density = [], 100.0
    for geom in root.iter("geom"):
        if geom.get("type") != "box" or not geom.get("size"):
            continue
        boxes.append((
            [float(v) for v in (geom.get("pos") or "0 0 0").split()],
            normalize_quat([float(v) for v in (geom.get("quat") or "1 0 0 0").split()]),
            [float(v) for v in geom.get("size").split()],
        ))
        if geom.get("density"):
            density = float(geom.get("density"))
    if not boxes:
        return None

    mesh_file, mesh_scale = None, (1.0, 1.0, 1.0)
    for mesh in root.iter("mesh"):
        mesh_file = mesh.get("file")
        if mesh.get("scale"):
            mesh_scale = tuple(float(v) for v in mesh.get("scale").split())
    return {"boxes": boxes, "density": density, "mesh_file": mesh_file, "mesh_scale": mesh_scale}


MESH_CACHE_DIR = HERE / "mesh_cache"


def triangulate_obj(src: Path) -> tuple[Path | None, int]:
    """把 OBJ 里的**多边形面**拆成三角形，输出到 ``mesh_cache``。

    ⚠ 踩坑：MuJoCo 的 OBJ 加载器**只认三角形面**，遇到四边形/多边形会静默丢面。
    实测 ``black_book.obj`` 共 3540 个面、其中 3412 个是四边形，MuJoCo 只编译出
    1200 个面 —— 结果这本书的网格被啃掉大半，界面上基本看不见（只剩 68 个像素，
    而同等尺寸的盘子有 3771 个像素）。这里先离线三角化，再交给 MuJoCo。

    返回 ``(三角化后的文件路径|None, 被拆解的多边形面数)``；
    本来全是三角形时返回 ``(None, 0)``，表示无需替换。
    """
    try:
        text = src.read_text(errors="ignore")
    except OSError:
        return None, 0

    out_lines, polygons = [], 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("f ") or stripped.startswith("fo "):
            indices = stripped.split()[1:]
            if len(indices) > 3:
                polygons += 1
                for k in range(1, len(indices) - 1):      # 扇形三角化（凸面精确）
                    out_lines.append(f"f {indices[0]} {indices[k]} {indices[k + 1]}")
            else:
                out_lines.append(stripped)
        elif stripped.startswith(("mtllib", "usemtl", "o ", "g ", "s ")):
            continue                                       # 材质/分组对 MuJoCo 无意义
        else:
            out_lines.append(line)

    if polygons == 0:
        return None, 0
    MESH_CACHE_DIR.mkdir(exist_ok=True)
    dst = MESH_CACHE_DIR / f"{src.parent.name}_{src.stem}_tri.obj"
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return dst, polygons


def resolve_visual_mesh(obj_dir: Path, mesh_file: str | None) -> Path | None:
    """XML 里指向的 .msh 在本仓库中缺失，回退到真实存在的 .obj。"""
    candidates = []
    if mesh_file:
        stem = Path(mesh_file).name
        candidates.append(stem.replace("_vis.msh", ".obj"))
        candidates.append(Path(stem).with_suffix(".obj").name)
        stem2 = stem.replace("_vis.msh", "")
        candidates.append(f"{stem2}.obj")
    candidates.append(f"{obj_dir.name}.obj")
    for candidate in candidates:
        path = obj_dir / candidate
        if path.is_file():
            return path
    for path in sorted(obj_dir.glob("*.obj")):
        if not re.search(r"_(col|coll|ch|collision)\.obj$", path.name, re.I):
            return path
    return None


def rotate_boxes(boxes, quat_rot):
    """把碰撞盒整体旋转到目标姿态（返回新的 pos/quat/size 列表）。"""
    mat = quat_to_mat(quat_rot)
    out = []
    for pos, quat, size in boxes:
        out.append((
            (mat @ np.asarray(pos, dtype=float)).tolist(),
            list(quat_mul(tuple(quat_rot), tuple(quat))),
            list(size),
        ))
    return out


def boxes_aabb(boxes):
    lo = np.full(3, 1e9)
    hi = np.full(3, -1e9)
    for pos, quat, size in boxes:
        half = np.abs(quat_to_mat(quat)) @ np.asarray(size, dtype=float)
        lo = np.minimum(lo, np.asarray(pos) - half)
        hi = np.maximum(hi, np.asarray(pos) + half)
    return lo, hi


def choose_upright(name: str, extents) -> tuple[tuple, str]:
    """决定把物体摆到桌面上的姿态：默认让最长轴竖起来。"""
    override = UPRIGHT_OVERRIDES.get(name, {}).get("axis")
    order = np.argsort(extents)
    long_axis = "xyz"[int(order[-1])]
    ratio = float(extents[order[-1]] / max(float(extents[order[0]]), 1e-6))

    if override == "keep":
        return (1.0, 0.0, 0.0, 0.0), "keep(手动)"
    if override == "z->y":
        return quat_from_axis_angle("x", -90.0), "override(Z长轴→平躺)"
    if override == "x->z":
        # 书这类"薄片"：把**最薄的轴**转到竖直，才是真的平躺
        # （原来的 z->y 会让书以 134mm 的高边立着，看着别扭也不好推）
        return quat_from_axis_angle("y", 90.0), "override(薄轴X→平躺)"
    if long_axis == "z" or ratio < 1.35:
        # 素材是"Y 轴朝上"的约定：近等轴物体如果 Y 是最长边（番茄酱罐、汤罐这类
        # 圆柱），必须绕 X 转 90° 让 Y→Z，否则罐子会横躺在桌上（实测第 7 关的罐子）
        if ratio < 1.35 and float(extents[1]) >= max(float(extents[0]), float(extents[2])) * 0.98:
            return quat_from_axis_angle("x", 90.0), "近等轴Y最长→立起"
        rule = "keep(长轴已在 Z)" if long_axis == "z" else "keep(外形接近等轴)"
        return (1.0, 0.0, 0.0, 0.0), rule
    if long_axis == "y":
        return quat_from_axis_angle("x", 90.0), "长轴Y→立起"
    return quat_from_axis_angle("y", -90.0), "长轴X→立起"


def analyse_grasp(boxes, n_bands: int = 24) -> dict:
    """按高度切片估算「50mm 夹爪能不能夹、该夹多高」。

    做法：把物体沿 Z 切成 n_bands 层，每层取所有与之相交的碰撞盒的**联合水平轮廓**，
    轮廓的较小边就是该层夹爪需要张开的宽度；在所有层里找最窄的一层作为建议夹持高度。
    （注意不能用"最薄的单个盒子"，那些碎盒是凸分解的碎片，不代表能夹住的部位。）
    """
    lo, hi = boxes_aabb(boxes)
    z0, z1 = float(lo[2]), float(hi[2])
    if z1 - z0 < 1e-6:
        return {"grasp_width": 0.0, "graspable": True, "grasp_z": [z0, z1],
                "grasp_axis": "x", "max_width": 0.0, "width_profile": []}

    prepared = []
    for pos, quat, size in boxes:
        half = np.abs(quat_to_mat(quat)) @ np.asarray(size, dtype=float)
        prepared.append((np.asarray(pos, dtype=float), half))

    profile, candidates = [], []
    step = (z1 - z0) / n_bands
    for i in range(n_bands):
        band_lo = z0 + step * i
        band_hi = band_lo + step
        x_lo = y_lo = 1e9
        x_hi = y_hi = -1e9
        for pos, half in prepared:
            if pos[2] + half[2] < band_lo or pos[2] - half[2] > band_hi:
                continue
            x_lo, x_hi = min(x_lo, pos[0] - half[0]), max(x_hi, pos[0] + half[0])
            y_lo, y_hi = min(y_lo, pos[1] - half[1]), max(y_hi, pos[1] + half[1])
        if x_lo > x_hi:
            continue
        wx, wy = float(x_hi - x_lo), float(y_hi - y_lo)
        profile.append([round(band_lo, 4), round(wx, 4), round(wy, 4)])
        width, axis = (wx, "x") if wx <= wy else (wy, "y")
        candidates.append((width, band_lo, band_hi, axis))

    if not candidates:
        return {"grasp_width": 1e9, "graspable": False, "grasp_z": [z0, z1],
                "grasp_axis": "x", "max_width": 1e9, "width_profile": profile}

    # 取"最窄的一层"作为建议夹取高度：太靠近底面（<15% 高度）不好夹，先排除；
    # 多个同宽时取最靠下的那个（实测这样对细长瓶罐更稳，见 README 的实测小节）。
    # 注：可用 TCP 高度还与指爪接触区有关，扫描实测番茄酱为「原点 +30~+60mm」，
    #     而窄带中点 +40mm 正好落在区间中间，故本规则可作为可靠的启发值。
    safe = [c for c in candidates if c[1] >= z0 + 0.15 * (z1 - z0)] or candidates
    width, band_lo, band_hi, axis = min(safe, key=lambda c: c[0])
    return {
        "grasp_width": round(float(width), 4),
        "graspable": bool(width <= GRIP_MAX_OPENING),
        "grasp_z": [round(float(band_lo), 4), round(float(band_hi), 4)],
        "grasp_axis": axis,
        "max_width": round(float(max(c[0] for c in candidates)), 4),
        "width_profile": profile,
    }


# ─────────────────── 目录生成 ───────────────────
def build_catalog() -> dict:
    """扫描整个 libero_assets，返回物体目录 dict。"""
    objects: dict[str, dict] = {}
    skipped: list[str] = []

    for category_dir in sorted(ASSETS_ROOT.iterdir()):
        if not category_dir.is_dir() or category_dir.name in {".cache", "textures"}:
            continue
        for xml in sorted(category_dir.rglob("*.xml")):
            parsed = parse_object_xml(xml)
            if parsed is None:
                continue
            key = f"{category_dir.name}/{xml.parent.name}"
            if key in objects:
                continue

            name = xml.stem
            lo0, hi0 = boxes_aabb(parsed["boxes"])
            quat_up, rule = choose_upright(name, hi0 - lo0)
            boxes = rotate_boxes(parsed["boxes"], quat_up)
            lo, hi = boxes_aabb(boxes)
            mesh = resolve_visual_mesh(xml.parent, parsed["mesh_file"])
            mesh_polygons = 0
            if mesh is not None:
                triangulated, mesh_polygons = triangulate_obj(mesh)
                if triangulated is not None:               # 有多边形面 → 换用三角化副本
                    mesh = triangulated
            if mesh is None:
                skipped.append(key)
            grasp = analyse_grasp(boxes)
            volume = float(sum(8.0 * np.prod(size) for _, _, size in boxes))
            body_width = float(min(hi[0] - lo[0], hi[1] - lo[1]))   # 整体最窄水平尺寸

            objects[key] = {
                "key": key,
                "name": name,
                "group": category_dir.name,
                "group_label": CATEGORIES.get(category_dir.name, category_dir.name),
                "xml": str(xml),
                "mesh": str(mesh) if mesh else None,
                "mesh_scale": [float(v) for v in parsed["mesh_scale"]],
                "mesh_polygons": mesh_polygons,
                "boxes": [[[float(x) for x in pos], [float(x) for x in quat],
                           [float(x) for x in size]] for pos, quat, size in boxes],
                "size": [round(float(v), 4) for v in (hi - lo)],
                "bottom_offset": round(float(-lo[2]), 4),
                "top_offset": round(float(hi[2]), 4),
                "center_xy_offset": [round(float(-(lo[i] + hi[i]) / 2.0), 4) for i in (0, 1)],
                "upright_quat": [float(v) for v in quat_up],
                "upright_rule": rule,
                "color": COLORS.get(name, DEFAULT_COLOR),
                "density": parsed["density"],
                "mass_est": round(volume * parsed["density"], 4),
                "box_count": len(boxes),
                "body_width": round(body_width, 4),
                "graspable_strict": bool(body_width <= GRIP_MAX_OPENING),
                # 俯视抓取时 TCP(tool0_site) 应停在「物体原点 + 该高度」处。
                # 取"窄带中点"作为建议值：实测番茄酱可用区间为原点 +30~+60mm，
                # 其窄带中点 +40mm 正落在区间中间，可直接用。
                "grasp_tcp_offset": round(float((grasp["grasp_z"][0] + grasp["grasp_z"][1]) / 2.0), 4),
                **grasp,
            }

    catalog = {
        "assets_root": str(ASSETS_ROOT),
        "grip_max_opening": GRIP_MAX_OPENING,
        "object_count": len(objects),
        "missing_mesh": skipped,
        "objects": objects,
    }
    return catalog


def print_table(catalog: dict) -> None:
    print(f"素材根目录: {catalog['assets_root']}")
    print(f"物体总数: {catalog['object_count']} | 50mm 夹爪开口上限: {catalog['grip_max_opening']*1000:.0f} mm\n")
    print(f"{'分组/名称':44s} {'长×宽×高(mm)':>22s} {'质量(g)':>8s} {'盒':>3s} {'可夹':>4s} "
          f"{'最窄(mm)':>8s} {'姿态':>16s}")
    print("-" * 120)
    group = None
    for obj in sorted(catalog["objects"].values(), key=lambda o: (o["group"], o["name"])):
        if obj["group"] != group:
            group = obj["group"]
            print(f"[{obj['group_label']}  {group}]")
        size = "×".join(f"{v * 1000:.0f}" for v in obj["size"])
        print(f"  {obj['name']:42s} {size:>22s} {obj['mass_est'] * 1000:8.0f} {obj['box_count']:3d} "
              f"{'是' if obj['graspable'] else '否':>4s} {obj['grasp_width'] * 1000:8.1f} "
              f"{obj['upright_rule']:>16s}")
    if catalog["missing_mesh"]:
        print(f"\n[WARN] 找不到视觉网格的物体 {len(catalog['missing_mesh'])} 个: {catalog['missing_mesh']}")


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    """读取 catalog；不存在时自动生成一份。"""
    path = Path(path)
    if not path.is_file():
        catalog = build_catalog()
        path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        return catalog
    return json.loads(path.read_text(encoding="utf-8"))


def catalog_object(catalog: dict, key: str) -> dict:
    """按 ``分组/名称`` 或直接名称取物体，取不到就给出可用列表。"""
    objects = catalog["objects"]
    if key in objects:
        return objects[key]
    for candidate in objects.values():
        if candidate["name"] == key or candidate["key"].endswith("/" + key):
            return candidate
    raise KeyError(f"catalog 里没有物体 '{key}'，可用示例: {list(objects)[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LIBERO 素材分析器")
    parser.add_argument("--table", action="store_true", help="打印尺寸表格")
    parser.add_argument("--out", type=str, default=str(CATALOG_PATH), help="catalog 输出路径")
    args = parser.parse_args()

    if not ASSETS_ROOT.is_dir():
        print(f"[ERROR] 找不到素材目录: {ASSETS_ROOT}")
        return 1

    catalog = build_catalog()
    out = Path(args.out)
    out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"[OK] 已生成 {out}（{catalog['object_count']} 个物体）")
    if args.table:
        print()
        print_table(catalog)
    else:
        from collections import Counter

        counter = Counter(o["group_label"] for o in catalog["objects"].values())
        for label, count in counter.items():
            print(f"  {label}: {count}")
        graspable = [o["name"] for o in catalog["objects"].values() if o["graspable"]]
        print(f"  50mm 夹爪可夹的物体: {len(graspable)} 个 → {', '.join(sorted(graspable))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


