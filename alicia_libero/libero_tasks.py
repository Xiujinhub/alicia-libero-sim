#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务定义：基于 libero_assets 分析结果（assets_catalog.json）设计的 9 个桌面操作任务。

任务类型
--------
* 放进容器 (into)   ：目标物是"篮子/小碟"这类有开口的容器，把抓取物放进去
* 放到平面 (onto)   ：目标物是"盘子/托盘"，把抓取物稳稳放上去
* 叠放 (stack)      ：把物体叠到另一个物体正上方
* 从容器取出 (from) ：抓取物一开始就在容器里，先夹出来再放到指定位置
* 区域 (region)     ：只给桌面上一个目标区域（带标记），用于"空间指代"类任务

成功判据
--------
``success = {"xy": [x, y], "xy_tol": r, "z_ref": "table"|"target_top", "z_band": [lo, hi]}``

* ``xy``/``xy_tol``：目标区域中心 + 抓取物 AABB 中心的水平容差（米）
* ``z_ref``        ：高度基准 = 桌面 或 目标物顶面
* ``z_band``       ：抓取物**底面**相对基准的高度区间，用来区分"放好了"与"还举在半空"

坐标系：底座在 ``(-0.27, 0, 0.8)``、+X 指向桌子中心；桌面高 0.8 m，x,y ∈ [-0.4, 0.4]。
"""

from __future__ import annotations

from libero_catalog import catalog_object, load_catalog
from libero_scene import TABLE_TOP_Z, SceneBuilder

TASKS: list[dict] = [
    {
        "id": "t1_ketchup_into_basket",
        "name": "1) 番茄酱放进篮子（放进容器）",
        "kind": "into",
        "difficulty": "★",
        "task_text": "把番茄酱瓶放进绿色篮子里。",
        "hint": "番茄酱瓶身宽约 37mm，50mm 夹爪可整瓶夹住；提到篮子正上方再松开。",
        "objects": [
            {"key": "stable_hope_objects/ketchup", "xy": [0.02, -0.13], "yaw": 0.0},
            # 篮子单独配重：素材默认 density=100（泡沫级，仅 136g），机械臂经过时会被撞走
            {"key": "stable_scanned_objects/basket", "xy": [0.22, 0.10], "yaw": 0.0, "density": 1500},
        ],
        "grasp_object": "ketchup",
        "target_object": "basket",
        "success": {"xy": [0.22, 0.10], "xy_tol": 0.055, "z_ref": "table", "z_band": [0.0, 0.06]},
    },
    {
        "id": "t2_butter_onto_plate",
        "name": "2) 黄油放到盘子中间（放到平面）",
        "kind": "onto",
        "difficulty": "★",
        "task_text": "把黄油块放到盘子的中间。",
        "hint": "黄油厚约 17mm，很好夹；对准盘子中心再放下。",
        "objects": [
            {"key": "stable_hope_objects/butter", "xy": [0.02, -0.14], "yaw": 0.0},
            {"key": "stable_scanned_objects/plate", "xy": [0.20, 0.08], "yaw": 0.0},
        ],
        "grasp_object": "butter",
        "target_object": "plate",
        "success": {"xy": [0.20, 0.08], "xy_tol": 0.07, "z_ref": "table", "z_band": [0.0, 0.06]},
    },
    {
        "id": "t3_pudding_into_ramekin",
        "name": "3) 布丁盒放进小碟（放进容器）",
        "kind": "into",
        "difficulty": "★★",
        "task_text": "把巧克力布丁盒放进白色小碟里。",
        "hint": "小碟内径约 89mm、布丁盒 27mm 宽——对准碟口垂直放进去。",
        "objects": [
            {"key": "stable_hope_objects/chocolate_pudding", "xy": [0.03, -0.12], "yaw": 0.0},
            {"key": "stable_scanned_objects/glazed_rim_porcelain_ramekin",
             "xy": [0.21, 0.09], "yaw": 0.0},
        ],
        "grasp_object": "chocolate_pudding",
        "target_object": "glazed_rim_porcelain_ramekin",
        "success": {"xy": [0.21, 0.09], "xy_tol": 0.035, "z_ref": "table", "z_band": [0.0, 0.05]},
    },
    {
        "id": "t4_popcorn_onto_plate",
        "name": "4) 爆米花盒放到盘子上",
        "kind": "onto",
        "difficulty": "★★",
        "task_text": "把爆米花盒放到盘子上。",
        "hint": "盒子仅 20mm 厚、62mm 高，夹爪沿窄边（20mm）闭合；放的时候保持水平。",
        "objects": [
            {"key": "stable_hope_objects/popcorn", "xy": [0.02, -0.13], "yaw": 0.0},
            {"key": "stable_scanned_objects/plate", "xy": [0.22, 0.10], "yaw": 0.0},
        ],
        "grasp_object": "popcorn",
        "target_object": "plate",
        "success": {"xy": [0.22, 0.10], "xy_tol": 0.075, "z_ref": "table", "z_band": [0.0, 0.06]},
    },
    {
        "id": "t5_bottle_onto_tray",
        "name": "5) 细高瓶子搬到木托盘里（细高物体）",
        "kind": "onto",
        "difficulty": "★★★",
        "task_text": "把调味汁瓶立着搬进木托盘里。",
        "hint": "瓶身 36mm、高 146mm，重心高——搬运慢一点；托盘边高 82mm，放到托盘内部。",
        "objects": [
            {"key": "stable_hope_objects/new_salad_dressing", "xy": [0.02, -0.15], "yaw": 0.0},
            {"key": "turbosquid_objects/wooden_tray", "xy": [0.20, 0.05], "yaw": 0.0},
        ],
        "grasp_object": "new_salad_dressing",
        "target_object": "wooden_tray",
        "success": {"xy": [0.20, 0.05], "xy_tol": 0.10, "z_ref": "table", "z_band": [0.0, 0.06]},
    },
    {
        "id": "t6_book_push_to_plate",
        "name": "6) 把黑皮书推到盘子边上（推/滑，不用夹）",
        "kind": "push",
        "difficulty": "★★",
        "task_text": "把黑色封面的书推到盘子边上（书的前沿贴住盘沿），盘子必须留在原地。",
        "hint": "书 134×110×28.7mm：50mm 夹爪从上方夹不住（实测），而它也爬不上 19mm 高的盘沿"
                "（盘的碰撞体是 10 个台阶式方块、没有斜坡）；机械臂的推力又远大于盘子摩擦"
                "（实测即使盘子加重到 920g，继续顶照样被推走 100mm）。所以本关做成"
                "「把书推到盘沿前 3mm 停住」——判据点取实测停手位置 (117,-14)mm。",
        "objects": [
            {"key": "turbosquid_objects/black_book", "xy": [0.06, -0.12], "yaw": 0.0},
            # ⚠ 实测：libero 材质默认 density=100（≈泡沫），盘子只有 11.5g，书轻轻一碰就飞。
            #   只给 t6 的盘子覆盖成 2400（≈瓷盘 276g）；t2/t4 的盘子不受影响。
            {"key": "stable_scanned_objects/plate", "xy": [0.22, 0.10], "yaw": 0.0,
             "density": 2400},
        ],
        "grasp_object": "black_book",
        "target_object": "plate",
        # ⚠ 判据原来是"书心进盘心 85mm 内"，实测那等于要求书**爬上盘子**（书的投影半宽就 84mm）
        #   ——物理上做不到：要么书上不去，要么盘子被顶走。现改成"推到盘沿前停住"：
        #   实测停手时书心在 (117,-14)mm，重复性 ±1mm；±10mm 起始抖动下仍在 40mm 容差内。
        "success": {"xy": [0.117, -0.014], "xy_tol": 0.04, "z_ref": "table", "z_band": [0.0, 0.06]},
    },
    {
        "id": "t7_stack_pudding_on_can",
        "name": "7) 布丁盒叠放到番茄酱罐顶（叠放）",
        "kind": "stack",
        "difficulty": "★★★",
        "task_text": "把巧克力布丁盒稳稳叠到番茄酱罐的顶面上。",
        "hint": "罐顶面 62×62mm、离桌面 76mm（比布丁盒 27.4×46.3×80.2mm 还矮）——关键是"
                "**夹住布丁的 27mm 薄边**：实测闭合轴一旦偏掉就会夹到对角线，布丁会在爪子里转、"
                "抬起来就歪；夹正之后再对准罐口轻轻落下。",
        "objects": [
            {"key": "stable_hope_objects/chocolate_pudding", "xy": [0.03, -0.13], "yaw": 0.0},
            {"key": "stable_hope_objects/tomato_sauce", "xy": [0.21, 0.08], "yaw": 0.0},
        ],
        "grasp_object": "chocolate_pudding",
        "target_object": "tomato_sauce",
        "success": {"xy": [0.21, 0.08], "xy_tol": 0.05, "z_ref": "target_top", "z_band": [-0.02, 0.07]},
    },
    {
        "id": "t8_ketchup_to_region",
        "name": "8) 番茄酱放到盘子左侧的标记区（空间指代）",
        "kind": "region",
        "difficulty": "★★",
        "task_text": "把番茄酱放到桌面上那块蓝色标记圈里（盘子的左侧）。",
        "hint": "没有容器可依靠，要自己判断位置，平放在标记圈内。",
        "objects": [
            {"key": "stable_hope_objects/ketchup", "xy": [0.02, -0.13], "yaw": 0.0},
            {"key": "stable_scanned_objects/plate", "xy": [0.24, 0.12], "yaw": 0.0},
        ],
        "grasp_object": "ketchup",
        "target_object": "plate",
        "success": {"xy": [0.24, -0.06], "xy_tol": 0.09, "z_ref": "table", "z_band": [0.0, 0.05]},
    },
    {
        "id": "t9_ketchup_out_of_tray",
        "name": "9) 番茄酱从托盘里取出放到盘子上（从容器取出）",
        "kind": "from",
        "difficulty": "★★★",
        "task_text": "番茄酱现在立在木托盘里，先夹出来再放到盘子上。",
        "hint": "瓶高 146mm、托盘边高 82mm——直接从瓶子上部夹出来即可，手腕不用伸进托盘。",
        "objects": [
            # z_offset = 托盘内底面高度（实测碰撞盒顶面 0.8077）：瓶子要正好坐在托盘里，
            # 抬太高会"掉进托盘"砸倒自己（旧版 z_offset=0.03 就是这么翻的）
            # 位置贴近机械臂（x≈0.13）：腕部越竖直，机壳越不容易横扫到托盘沿
            {"key": "stable_hope_objects/ketchup", "xy": [0.13, 0.04], "yaw": 0.0, "z_offset": 0.0077},
            # 托盘单独配重：素材默认 density=100，托盘只有 57g，一碰就滑走
            {"key": "turbosquid_objects/wooden_tray", "xy": [0.13, 0.04], "yaw": 0.0, "density": 1500},
            {"key": "stable_scanned_objects/plate", "xy": [0.15, -0.12], "yaw": 0.0},
        ],
        "grasp_object": "ketchup",
        "target_object": "plate",
        "success": {"xy": [0.15, -0.12], "xy_tol": 0.07, "z_ref": "table", "z_band": [0.0, 0.06]},
    },
]


# ─────────────────── 场景生成 ───────────────────
def target_region_geom(success: dict) -> dict:
    """把成功区域画成一个半透明圆盘，让操作者看得见目标。"""
    return {
        "name": "target_region", "type": "cylinder",
        "pos": f"{success['xy'][0]:.4f} {success['xy'][1]:.4f} {TABLE_TOP_Z + 0.0015:.4f}",
        "size": f"{success['xy_tol']:.4f} 0.0015",
        "rgba": "0.15 0.55 1.0 0.35", "contype": "0", "conaffinity": "0",
        "group": "1", "condim": "1",
    }


def build_task_scene(task: dict, builder: SceneBuilder | None = None, catalog: dict | None = None):
    """生成任务场景，返回 (场景 XML 路径, SceneBuilder)。

    注意：这里**不做任何全局"统一调整"**（例如统一改密度/统一改位置）——
    那会连带改变其它任务的物理表现。哪一关有需要，就在该关的 ``objects`` 里
    单独写 ``density``／``xy``，改动范围一眼可见。
    """
    catalog = catalog or load_catalog()
    builder = builder or SceneBuilder(catalog=catalog)
    layout = {
        "name": task["id"],
        "objects": task["objects"],
        "extra_geoms": ([target_region_geom(task["success"])]
                        if task["kind"] in ("region", "stack", "into") else []),
    }
    return builder.build(layout), builder


# ─────────────────── 成功判定 ───────────────────
def object_world_aabb(model, data, catalog: dict, name: str):
    """用物体世界位姿 + catalog 的局部 AABB 算出世界系 AABB（不假设物体一定竖直）。"""
    import itertools

    import numpy as np

    from libero_catalog import boxes_aabb, quat_to_mat

    body_id = model.body(name).id
    lo, hi = boxes_aabb(catalog_object(catalog, name)["boxes"])
    rot = quat_to_mat(data.xquat[body_id])
    center = data.xpos[body_id]
    corners = np.array([center + rot @ np.array(corner)
                        for corner in itertools.product(*zip(lo, hi))])
    return corners.min(axis=0), corners.max(axis=0)


def check_success(task: dict, model, data, catalog: dict | None = None) -> tuple[bool, str]:
    """检查任务是否完成，返回 (是否成功, 说明文字)。"""
    import numpy as np

    catalog = catalog or load_catalog()
    spec = task["success"]
    lo, hi = object_world_aabb(model, data, catalog, task["grasp_object"])
    center_xy = (lo + hi)[:2] / 2.0
    error_xy = float(np.linalg.norm(center_xy - np.asarray(spec["xy"], dtype=float)))

    if spec["z_ref"] == "target_top":
        target = catalog_object(catalog, task["target_object"])
        z_ref = TABLE_TOP_Z + float(target["size"][2])
    else:
        z_ref = TABLE_TOP_Z
    bottom = float(lo[2] - z_ref)

    ok_xy = error_xy <= spec["xy_tol"]
    # 高度判据留 2mm 弹性：物体静止时可能因接触软约束略低于基准（实测 -0.4mm 会被判失败）
    ok_z = (spec["z_band"][0] - 0.002) <= bottom <= (spec["z_band"][1] + 0.002)
    message = (f"水平偏差 {error_xy * 1000:.0f}mm / 允许 {spec['xy_tol'] * 1000:.0f}mm；"
               f"底面相对基准 {bottom * 1000:+.0f}mm / 要求 "
               f"[{spec['z_band'][0] * 1000:+.0f}, {spec['z_band'][1] * 1000:+.0f}]mm")
    return bool(ok_xy and ok_z), message
