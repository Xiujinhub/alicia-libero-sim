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
* ``xy_ref``      ：可选。``"target"`` = 目标物**也被随机安放**（t1 的篮子）→
  目标 xy 取目标物**当前实际中心**（``xy`` 只当标称值/回退用），
  否则一律用写死的 ``xy``
* ``z_ref``        ：高度基准 = 桌面 或 目标物顶面
* ``z_band``       ：抓取物**底面**相对基准的高度区间，用来区分"放好了"与"还举在半空"

坐标系：底座在 ``(-0.27, 0, 0.8)``、+X 指向桌子中心；桌面高 0.8 m，x,y ∈ [-0.4, 0.4]。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from libero_catalog import (
    boxes_aabb,
    catalog_object,
    load_catalog,
    quat_from_axis_angle,
    quat_mul,
    rotate_boxes,
)
from libero_scene import TABLE_HALF, TABLE_TOP_Z, SceneBuilder

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
        # 任务多样化：抓取物**和篮子**每次随机安放（瓶子位置 + 姿态都随机），只在机械臂工作范围内
        # 的小矩形里抽。这是**运行时**行为（SimSession 抽完后写进物体的自由关节）；场景 XML 里仍写
        # 上面那两个标称位置（瓶子立着），以便 build_all_scenes / render_preview 产出的图固定可比。
        # 区域怎么定的见 README §8.9。
        # 抓取物上沿取到 -0.10（而不是 -0.06）：网格实测 (0.00,-0.06) 这个"离底座只有 277mm"的角点
        # 会让手指把瓶子碰倒（实时宽度 37→122mm、连续 4 档都夹空），其余 19 个格点全过。
        # 姿态："立着" 或 "平放"（各 50%）。平放还会**随机长轴朝向**，但生成前会把它
        #   适配到实测能过关的角度（`SPAWN_LYING_YAWS`：4 个"斜躺"角度，见那里的实测表）；
        #   瓶子长 145.6mm 比篮子内腔（122×111mm）还长，长轴与篮子边平行时必卡在沿口。
        #   位置与姿态都在**运行时**生效，机制见 README §8.9。
        "spawn_region": {
            "ketchup": {"x": [0.00, 0.18], "y": [-0.24, -0.10],
                        "poses": ["upright", "lying"],
                        # 平放还要单独收紧位置：躺姿的抓取点比立姿低约 90mm（TCP 要压到
                        # 810~830mm），而机械臂在远端压不到那么低（实测离底座 ≥400mm 时
                        # TCP 停在 ~870mm 且 xy 漂 5~20mm，手指会偏进瓶身）。
                        # 所以"平放"只出现在够得着的近处，"立着"仍用整个区域。
                        "pose_regions": {"lying": {"x": [0.00, 0.08], "y": [-0.24, -0.10]}}},
            "basket": {"x": [0.14, 0.28], "y": [0.02, 0.20]},
        },
        "grasp_object": "ketchup",
        "target_object": "basket",
        # ``xy_ref="target"``：目标篮子是随机安放的 → 判分的目标 xy 取**篮子当前实际中心**
        # （而不是写死的 0.22/0.10），跟 place_object"对准目标当前位置"保持一致。
        "success": {"xy": [0.22, 0.10], "xy_ref": "target", "xy_tol": 0.055,
                    "z_ref": "table", "z_band": [0.0, 0.06]},
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
        # 任务多样化（与 t1 同一套机制，见 README §8.9）：**黄油位置 + 姿态随机、盘子位置随机**。
        # 与 t1 的差别（每条都实测过）：
        #   ① 黄油是 76×17×40mm 的**小扁块**，盘面 137mm 宽 —— 任意朝向都放得下，
        #      所以不需要 t1 那种"为了落进容器而收窄角度"；收窄的原因变成了**夹得住**：
        #      平放时能夹的只有 39.5mm 那对侧面，而爪口 50mm，闭合轴必须准到 ~8° 以内。
        #      实测 8 个朝向里只有 θ∈{0°,135°,180°,315°} 在两个位置上全夹住，
        #      其余 4 个残余偏角太大（31°+ → 投影宽 72mm > 爪口）直接夹空 → 只保留这 4 个。
        #      （黄油左右对称，0/180 与 135/315 其实是同一条线，所以**等效于两种朝向各 50%**。）
        #   ② 黄油"平放"= 绕**长轴（局部 X）**躺下 → 厚边(17.4)朝上、大平面贴桌。
        #      t1 的瓶子是绕**薄轴（局部 Y）**躺下（薄边仍水平才夹得住），所以这里要写 lie_axis="x"；
        #      两种姿态各 50%。
        #   ③ 立着时抓取点 ≈ 807mm、平放时 = 底面 + 10mm ≈ 810mm —— 两种姿态的 TCP 都压得很低
        #      （和 t1 的躺姿一样），所以位置区域整体收紧，不放 t1 立姿那样远的 x=0.18。
        "spawn_region": {
            "butter": {"x": [0.00, 0.09], "y": [-0.21, -0.12],
                       "poses": ["upright", "lying"],
                       "lie_axis": "x",
                       "lying_yaws": [0.0, 135.0, 180.0, 315.0],
                       # 立着也随机朝向（绕 Z 转，站姿不变）：实测 8 个 yaw × 2 个位置
                       # **16/16 都夹得住**（立着夹的是 17.4mm 薄边、爪口 50mm，IK 偏角不影响），
                       # 所以这里放开成**整圆周任意朝向**（None = 不限制）
                       "upright_yaws": None,
                       # 平放再单独收一点：x≈0 那一列（离底座最近）夹得住但**搬运会滑落**
                       # —— 实测 flat_0 在 x=0 的 3 次里坏 1 次，x≥0.04 的 9 次全过（t1 同款手法）
                       "pose_regions": {"lying": {"x": [0.04, 0.09], "y": [-0.21, -0.12]}}},
            # 盘子区域比 t1 的篮子略收：实测失败的几局都是"黄油在 x≈0.10/y≈−0.22 的远角、盘子又在
            # x≈0.25/y≈0.03 的远角"，两点相距 ~300mm（最长），5g 的黄油长距离搬运途中会滑落
            # （判分水平偏 300mm+、黄油躺在桌面上）。收进 x≤0.24/y≤0.16 后这类极端组合不再出现。
            "plate": {"x": [0.14, 0.24], "y": [0.02, 0.16]},
        },
        "grasp_object": "butter",
        "target_object": "plate",
        # ``xy_ref="target"``：盘子随机安放 → 判分取**盘子当前实际中心**（与 t1 的篮子同理）
        "success": {"xy": [0.20, 0.08], "xy_ref": "target", "xy_tol": 0.07,
                    "z_ref": "table", "z_band": [0.0, 0.06]},
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
        # 任务多样化（与 t1/t2 同一套机制，见 README §8.9）：**布丁盒位置/朝向随机、小碟位置随机**。
        # 与 t2 的差别（每条都实测过，脚本在 ``E:\deepenv\_tools\diag_t3_poses.py``）：
        #   ① 布丁盒 27.4×46.3×80.2mm（又高又轻，`grasp_axis=x` = 夹 27.4mm 薄边）。
        #      立姿**没有第二种可选姿态** —— 躺下后最短水平投影是 80.2mm，而小碟内腔只有
        #      ~70mm（89mm 外径 − 两圈瓷壁），塞不进去：实测悬空落到碟心，最低点停在
        #      +37mm（骑在碟沿上），3 次里还有 1 次直接滑到桌面。所以这里只做
        #      "立着 + 随机朝向"（``upright_yaws``），不提供平放。
        #   ② 抓取点 = 物体中心 + catalog 的 ``grasp_tcp_offset``(−25.5mm) ≈ 814mm（和 t2 的
        #      807~810mm 一样低），所以位置区域同样收紧，不放 t1 立姿那样的 x=0.18。
        "spawn_region": {
            "chocolate_pudding": {"x": [0.00, 0.09], "y": [-0.20, -0.10],
                                  "upright_yaws": None},
            "glazed_rim_porcelain_ramekin": {"x": [0.14, 0.24], "y": [0.02, 0.16]},
        },
        "grasp_object": "chocolate_pudding",
        "target_object": "glazed_rim_porcelain_ramekin",
        # ``xy_ref="target"``：小碟随机安放 → 判分取**小碟当前实际中心**（与 t1/t2 同理）
        "success": {"xy": [0.21, 0.09], "xy_ref": "target", "xy_tol": 0.035,
                    "z_ref": "table", "z_band": [0.0, 0.05]},
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
        "task_text": "把调味汁瓶搬进木托盘里。",
        "hint": "瓶身 36mm、高 146mm，重心高——搬运慢一点；托盘边高 82mm，放到托盘内部。"
                "（本关瓶子可能立着也可能平放、位置随机，托盘位置也随机）",
        "objects": [
            {"key": "stable_hope_objects/new_salad_dressing", "xy": [0.02, -0.15], "yaw": 0.0},
            {"key": "turbosquid_objects/wooden_tray", "xy": [0.20, 0.05], "yaw": 0.0},
        ],
        # 任务多样化（与 t1 同一套机制，见 README §8.9）：**瓶子位置 + 姿态随机、木托盘位置随机**。
        # t1 的瓶子/篮子那一套在这里几乎逐条对应（脚本 `_tools\diag_t5_*.py`）：
        #   ① 瓶子 52.7×35.5×145.6mm —— 和 t1 的番茄酱瓶**同样高 145.6mm**，但更细（薄边 35.5mm）；
        #      "平放"要绕**薄轴（局部 Y）**躺（`lie_axis` 默认就是 "y"）：薄边仍水平才夹得住
        #      （实测绕世界 X 躺时闭合方向变成 52.7mm > 爪口 50mm，夹空）。
        #   ② 托盘内腔 267×135.6mm、内底 +8mm、沿高 82mm —— 比 t1 的篮子（122×111mm）**宽敞得多**，
        #      所以平放**不需要**按朝向筛角度：实测 5 个朝向 × 5 个落点 **25/25 全部落进托盘内底**
        #      （最低点 807.7mm、水平偏差 ≤39mm，容差 100mm）→ `lying_yaws: None`（整圆周随机）。
        #   ③ 立姿抓取点 = 中心 + 21.3mm ≈ 894mm（高，能到远处）；平放 = 底面 + 10mm = 810mm
        #      （低，只有近处压得下去）→ 平放**单独收紧位置**（`pose_regions`，与 t1 同款）。
        "spawn_region": {
            "new_salad_dressing": {"x": [0.00, 0.18], "y": [-0.24, -0.10],
                                   "poses": ["upright", "lying"],
                                   "upright_yaws": None,     # 立着：整圆周随机朝向
                                   "lying_yaws": None,       # 平放：整圆周随机（托盘腔大，实测都进得去）
                                   "pose_regions": {"lying": {"x": [0.00, 0.08], "y": [-0.24, -0.10]}}},
            # 托盘 297×166mm 很大：中心 x 收到 ≤0.24 才不越过桌沿（±0.4m），
            # 下沿 x ≥ 0.18 是为了跟瓶子区域留出"外接圆 74.9 + 170.5 + 25 ≈ 270mm"的占地间隙。
            "wooden_tray": {"x": [0.18, 0.24], "y": [0.00, 0.16]},
        },
        "grasp_object": "new_salad_dressing",
        "target_object": "wooden_tray",
        # ``xy_ref="target"``：托盘随机安放 → 判分取**托盘当前实际中心**（与 t1/t2/t3 同理）
        "success": {"xy": [0.20, 0.05], "xy_ref": "target", "xy_tol": 0.10,
                    "z_ref": "table", "z_band": [0.0, 0.06]},
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
        "hint": "瓶高 146mm、托盘边高 82mm——直接从瓶子上部夹出来即可，手腕不用伸进托盘。"
                "（本关瓶子在托盘里的位置随机、朝向随机，托盘与盘子的位置也都随机）",
        "objects": [
            # z_offset = 托盘内底面高度（实测碰撞盒顶面 0.8077）：瓶子要正好坐在托盘里，
            # 抬太高会"掉进托盘"砸倒自己（旧版 z_offset=0.03 就是这么翻的）
            # 位置贴近机械臂（x≈0.13）：腕部越竖直，机壳越不容易横扫到托盘沿
            {"key": "stable_hope_objects/ketchup", "xy": [0.13, 0.04], "yaw": 0.0, "z_offset": 0.0077},
            # 托盘单独配重：素材默认 density=100，托盘只有 57g，一碰就滑走
            {"key": "turbosquid_objects/wooden_tray", "xy": [0.13, 0.04], "yaw": 0.0, "density": 1500},
            {"key": "stable_scanned_objects/plate", "xy": [0.15, -0.12], "yaw": 0.0},
        ],
        "spawn_region": {
            # ① 盘子（**先抽**）：x/y 都给足（盘 137.6mm 圆、占地外接圆 97.3mm，全在桌面内）
            "plate": {"x": [0.04, 0.26], "y": [-0.30, -0.18]},
            # ② 木托盘（后抽，避让**已抽到的**盘子）：托盘 297×166mm、占地外接圆 170.5mm，
            #    跟盘子要隔开 170.5 + 97.3 + 25 ≈ 293mm —— 所以放在桌子靠里那半
            "wooden_tray": {"x": [0.14, 0.24], "y": [0.04, 0.16]},
            # ③ 番茄酱（**锚在托盘里**）：x/y 是相对托盘的偏移（瓶底坐在托盘内底、跟着托盘走），
            #    朝向整圆周随机；偏移范围按实测挑（见 README §8.9.4）
            "ketchup": {"anchor": "wooden_tray", "x": [-0.035, 0.035], "y": [-0.02, 0.02],
                        "upright_yaws": None},
        },
        "grasp_object": "ketchup",
        "target_object": "plate",
        # ``xy_ref="target"``：盘子随机安放 → 判分取**盘子当前实际中心**（与 t1/t2/t3/t5 同理）
        "success": {"xy": [0.15, -0.12], "xy_ref": "target", "xy_tol": 0.07,
                    "z_ref": "table", "z_band": [0.0, 0.06]},
    },
]


# ─────────────────── 任务多样化：抓取物随机安放（位置 + 姿态） ───────────────────
SPAWN_TRIES = 200
"""随机安放的拒绝采样上限（区域里可能被其它物体的占地挡住一部分）。"""
SPAWN_CLEARANCE = 0.025
"""随机安放位置与"其它物体占地"之间要留的间隙（米）。"""
SPAWN_TABLE_MARGIN = 0.06
"""随机安放区域距桌沿至少留的余量（米），免得瓶子被摆到桌子边缘外掉下去。"""

# 平放姿态的构造见 ``lying_quat``：绕**局部某条水平轴**转 90°（长轴从竖直变水平），
# 再绕**世界 Z** 转 θ 决定躺的方向。绕哪条轴由 `lie_axis` 决定 —— 瓶子绕薄轴（薄边仍水平，
# 平行夹爪还能水平闭合）、扁盒（黄油）绕长轴（大平面朝下）。


def lying_quat(theta_deg: float, axis: str = "y") -> tuple[float, float, float, float]:
    """平放姿态：绕**局部 ``axis``** 躺下、再绕世界 Z 转到长轴朝向 ``theta_deg``。

    * ``axis="y"``（默认，t1 的瓶子）：绕**薄轴**躺下 → 薄轴仍水平（夹爪还能水平闭合），
      长轴从竖直变水平；θ=225 就是以前写死的 ``laid_225``。
    * ``axis="x"``（t2 的黄油）：绕**长轴**躺下 → 厚边朝上、最大面贴桌（t1 的瓶子没有
      这么"扁"，绕薄轴躺才稳；黄油是 76×39mm 的大平面朝下最自然）。
    """
    return quat_mul(quat_from_axis_angle("z", float(theta_deg)),
                    quat_from_axis_angle(axis, 90.0))


SPAWN_LYING_YAWS = (45.0, 140.0, 215.0, 320.0)
"""**绕薄轴躺**（``axis="y"``）允许的长轴朝向（度）。**朝向随机**，但只在实测"能过关"的角度里抽。

⚠ 这张表是逐角度量出来的（脚本 ``_tools\\diag_t1_yaw_sweep.py``：把瓶子按长轴 θ 摆到
篮子中心正上方、瓶底在篮口上方 15mm 凌空松手，落定后看判据；每角度测 9 个落点 ——
中心 + 8 向 15mm 偏移，模拟真实放置误差）：

| 长轴朝向 θ | 全落点通过 | 说明 |
| --- | --- | --- |
| 40°, 45° | **9/9** | 45° 家族 ✓ |
| 140°, 150° | **9/9** | 145° 附近 ✓ |
| 205°, 210°, 225° | **9/9** | 斜对角家族 ✓ |
| 320° | **9/9** | 320° 附近 ✓ |
| 75°~105°、255°~285° | **0~2/9** | 长轴几乎与篮子边平行 —— 瓶子比内腔还长，必卡在沿口 |
| 其余（0°/15°/…/350°） | 2~8/9 | 对落点偏移太敏感，不用 |

四组角度大致相隔 90°（都在"斜躺"族里），少量偏差是瓶身重心偏在瓶体那端造成的。
``adapt_lying_yaw`` 就是把任意随机角度**折到这张表里最近的一个** —— 这就是"生成前
调整角度适配"的那一步。"""

SPAWN_BOX_YAWS = (0.0, 135.0, 180.0, 315.0)
"""**绕长轴躺**（``axis="x"``，t2 的黄油的默认表）：实测夹得住的 4 个朝向。

黄油 76×17×40mm，平放后能夹的只有 **39.5mm** 那对侧面，而爪口 50mm —— 闭合轴必须准到
~8° 以内，否则沿轴投影宽 = 39.5·cosθ + 76.2·sinθ 很快超过 50mm、手指直接夹空。

实测（每个朝向 2 个位置，脚本 ``_tools\\diag_t2_poses.py``）：

| 长轴朝向 θ | 抓取 | 说明 |
| --- | --- | --- |
| 0°, 180° | **2/2 ✓** | 闭合轴落在 yaw≈−90°（IK 咬得住） |
| 135°, 315° | **2/2 ✓** | 闭合轴落在 yaw≈−135° ✓ |
| 90°, 270° | 1/2 ✗ | 有时挑到 +90°（够不着），不稳 |
| 45°, 225° | **0/2 ✗** | 残余偏角 31°+ → 投影宽 72mm > 爪口，夹空 |

黄油左右对称，0/180 与 135/315 各是同一条线 —— 所以**等效于两种朝向各 50%**。
（不能像 t1 那样在整圈上均匀取角度：这里的约束是"IK 能不能咬准闭合轴"，只有特定几个
偏航角做得到。）"""

SPAWN_LIE_DEFAULTS = {"y": SPAWN_LYING_YAWS, "x": SPAWN_BOX_YAWS}
"""``lie_axis`` → 该姿态默认允许的朝向表（区域里可以再用 ``lying_yaws`` 覆盖）。"""


def adapt_lying_yaw(theta_deg: float, allowed=SPAWN_LYING_YAWS) -> float:
    """把随机抽到的长轴朝向**适配**到允许的角度（取最近的一个，考虑 0/360 环绕）。

    ``allowed=None`` 表示不限制（原样返回）—— 留给"任意朝向都能过"的物体。
    """
    if allowed is None:
        return float(theta_deg) % 360.0
    candidates = list(allowed)
    return min(candidates, key=lambda a: abs((float(theta_deg) - a + 180.0) % 360.0 - 180.0))


SPAWN_POSE_QUATS: dict[str, tuple[float, float, float, float]] = {
    "upright": (1.0, 0.0, 0.0, 0.0),                       # 立着（= XML 里的姿态）
    **{f"laid_{t:g}": lying_quat(t, "y") for t in SPAWN_LYING_YAWS},
    **{f"flat_{t:g}": lying_quat(t, "x") for t in SPAWN_BOX_YAWS},
}
"""姿态名 → 物体**局部位姿**要乘的四元数（作用在 catalog 的碰撞盒上）。

``lying`` / ``laid`` / ``flat`` 是**类别**，由 ``pick_spawn_pose`` 现场抽角度 + 适配
（见上面两张表）；直接写具体名字（如 ``laid_45``、``flat_0``）也可以。"""

SPAWN_POSE_LABELS = {"upright": "立着", "lying": "平放"}
"""界面上显示用的中文名。"""


def posed_boxes(key: str, quat, catalog: dict) -> list:
    """catalog 的碰撞盒**按给定姿态旋转后**的盒子（``quat=None`` = 原样）。"""
    obj = catalog_object(catalog, key)
    boxes = [([float(v) for v in pos], [float(v) for v in q], [float(v) for v in size])
             for pos, q, size in obj["boxes"]]
    return rotate_boxes(boxes, quat) if quat is not None else boxes


def footprint_radius_boxes(boxes) -> float:
    """一组（已带姿态的）盒子在世界 XY 上的**外接圆半径**（米）。"""
    lo, hi = boxes_aabb(boxes)
    return 0.5 * float(np.hypot(hi[0] - lo[0], hi[1] - lo[1]))


def placed_pose(key: str, xy, quat, catalog: dict) -> tuple[list, tuple]:
    """给定"姿态 + AABB 中心要落在的世界 xy"，返回物体 body 的 ``(位置, 四元数)``。

    和 ``SceneBuilder.placed_boxes`` 同一套算法（AABB 中心落在 xy、底面正好贴桌面），
    只是这里的姿态是任意四元数（不只是绕 Z 的偏航角）。平放时必须重新算 z：
    立着时底面在瓶子底部，平放时是**侧面**贴桌 —— 沿用 XML 的 z 会让瓶子悬空/穿桌。
    """
    boxes = posed_boxes(key, quat, catalog)
    lo, hi = boxes_aabb(boxes)
    pos = [float(xy[0]) - (lo[0] + hi[0]) / 2.0,
           float(xy[1]) - (lo[1] + hi[1]) / 2.0,
           TABLE_TOP_Z - float(lo[2])]
    return pos, (tuple(quat) if quat is not None else (1.0, 0.0, 0.0, 0.0))


def footprint_radius(key: str, yaw_deg: float, catalog: dict) -> float:
    """物体水平占地的**外接圆半径**（米）。随机安放时用它判断"会不会和旁边的东西叠在一起"。

    摆放约定（见 ``SceneBuilder.placed_boxes``）：物体 AABB 中心落在任务给的 ``xy`` 上，
    所以两个物体的世界 XY 距离 > ``r_a + r_b + 间隙`` 就一定不重叠。

    默认 = 物体**立着**（XML 里的姿态）、再绕 Z 转 ``yaw_deg``；随机姿态的占地用
    ``footprint_radius_boxes(posed_boxes(...))`` 算（平放的瓶子占地会大一圈：33.6 → 73mm）。
    """
    obj = catalog_object(catalog, key)
    boxes = [([float(v) for v in pos], [float(v) for v in quat], [float(v) for v in size])
             for pos, quat, size in obj["boxes"]]
    if abs(yaw_deg) > 1e-6:
        boxes = rotate_boxes(boxes, quat_from_axis_angle("z", yaw_deg))
    return footprint_radius_boxes(boxes)


class SpawnPose(NamedTuple):
    """本局某个物体的摆位：AABB 中心的世界 XY + 姿态四元数（``None`` = 沿用 XML 的立姿）。"""

    xy: tuple[float, float]
    quat: tuple[float, float, float, float] | None = None
    pose: str = "upright"                      # "upright" / "lying"（任务里写的姿态类别）
    detail: str = ""                           # 实际抽到的姿态名（如 "laid_45"），给界面显示

    @property
    def label(self) -> str:
        """界面上显示的姿态中文名。"""
        return SPAWN_POSE_LABELS.get(self.pose, self.pose)


def as_spawn_pose(value) -> SpawnPose:
    """把 ``(x, y)`` / ``SpawnPose`` 统一成 ``SpawnPose``（脚本、测试直接写位置时用）。

    允许 ``sess.spawn = {"ketchup": (x, y)}`` 这种旧写法继续能用 —— 那时按"立着"处理。
    """
    if isinstance(value, SpawnPose):
        return value
    if len(value) == 2 and not hasattr(value[0], "__len__"):
        return SpawnPose((float(value[0]), float(value[1])))
    xy, quat, pose = (list(value) + [None, "upright"])[:3]
    return SpawnPose((float(xy[0]), float(xy[1])),
                     tuple(quat) if quat is not None else None, pose)


def pick_spawn_pose(box: dict, rng) -> tuple[str, str, tuple | None]:
    """按区域配置抽一个姿态，返回 ``(姿态类别, 实际姿态名, 四元数)``。

    ``box["poses"]`` 写的是**姿态类别**（如 ``["upright", "lying"]``，各 50%）：

    * ``upright``：默认沿用 XML 的姿态与 z（**四元数返回 None**）→ 立姿局一个字节不变。
      区域里写了 ``upright_yaws`` 才会**给立姿再随机一个朝向**（绕世界 Z 转，不影响"站得住"）：
      ``None`` = 整圆周均匀随机（实测 t2 立着黄油 8 个 yaw × 2 个位置 **16/16 都夹得住**：
      立着夹的是 17.4mm 薄边、爪口 50mm，IK 偏角不影响）；给一个列表则随机后**适配**到最近的
      一个。姿态名给成 ``up_45`` 这样。
    * ``lying``（别名 ``laid`` / ``flat``）：**先随机抽长轴朝向，再适配到允许的角度**
      （``adapt_lying_yaw``），最后 ``lying_quat`` 转成四元数。

      - 绕哪条轴躺由 ``box["lie_axis"]`` 决定：默认 ``"y"``＝绕薄轴躺（t1 的瓶子），
        ``"x"``＝绕长轴躺（t2 的黄油：大平面朝下）；
      - 允许的朝向表默认取 ``SPAWN_LIE_DEFAULTS[轴]``，区域里可以用 ``lying_yaws`` 覆盖
        （``None`` = 不限制、真·任意朝向）；
      - 姿态名（``detail``）按轴给：绕薄轴 → ``laid_45``、绕长轴 → ``flat_45``。

    没写 ``poses`` 就一律"立着"。
    """
    tokens = list(box.get("poses") or ("upright",))
    token = tokens[int(rng.integers(len(tokens)))]
    if token in ("lying", "laid", "flat"):
        axis = str(box.get("lie_axis") or "y")
        allowed = box.get("lying_yaws", SPAWN_LIE_DEFAULTS.get(axis, SPAWN_LYING_YAWS))
        theta = adapt_lying_yaw(float(rng.uniform(0.0, 360.0)), allowed)
        prefix = "laid" if axis == "y" else "flat"
        return "lying", f"{prefix}_{theta:g}", lying_quat(theta, axis)
    if token == "upright":
        allowed = box.get("upright_yaws", 0.0)      # 缺省 0.0 = 不随机（老行为逐字不变）
        if allowed is None or allowed:
            yaw = adapt_lying_yaw(float(rng.uniform(0.0, 360.0)), allowed or None)
            return "upright", f"up_{yaw:g}", quat_from_axis_angle("z", yaw)
        return token, token, None
    return token, token, SPAWN_POSE_QUATS[token]


def sample_spawn(task: dict, rng=None, catalog: dict | None = None) -> dict:
    """按 ``task["spawn_region"]`` 随机抽物体的初始位姿，返回 ``{物体名: SpawnPose}``。

    位置只改 XY（z 由姿态和桌面重新算，见 ``placed_pose``）；姿态默认**不动**
    （沿用 XML 的立姿），只有区域里写了 ``poses`` 才会随机（t1：立着 / 平放）。
    抽到的位置要同时满足

    1. 落在任务里写的小矩形内（该矩形是按"抓取成功率"实测选出来的，见 README §8.9）；
    2. 与**其它物体占地**（外接圆 + ``SPAWN_CLEARANCE``）不重叠 —— 半径按**抽到的姿态**算
       （平放的瓶子占地 33.6 → 73mm，不按姿态算就会叠在一起）；
    3. 距桌沿留 ``SPAWN_TABLE_MARGIN``（区域被桌沿裁掉时自动收紧）。

    抽到姿态**之后**才算占地，所以"平放"会挤占更多空间、可放的点更少（拒绝采样自己会躲开）。

    ``spawn_region`` 里有多个物体时（t1：番茄酱 + 篮子）**按书写顺序依次抽**，
    后面的物体避让前面**已经抽到的实际位置**（而不是标称位置）—— 否则两个随机物
    会按标称位置算避让、实际却可能叠在一起。**书写顺序同时是"谁先抽"**：t9 把盘子写在
    托盘前面，托盘才能避让"已抽到的盘子"，而不是被盘子的**标称位置**卡死（§8.9.4）。

    区域里写 ``anchor`` 的物体（t9 的番茄酱锚在木托盘里）走另一条路：它的 ``x``/``y`` 是
    **相对锚点的偏移**，跟着锚点一起走，不参与外接圆避让（它本来就该在锚点的占地里面）；
    反过来，锚定的物体也不当别人的障碍 —— 它的"标称位置"没有参考价值。

    拒绝采样最多 ``SPAWN_TRIES`` 次；实在抽不到就退回任务里写死的 ``xy``（不会抛异常）。
    没有 ``spawn_region`` 的任务直接返回空字典 —— 完全不影响其它任务。
    """
    region = task.get("spawn_region")
    if not region:
        return {}
    rng = rng if rng is not None else np.random.default_rng()
    catalog = catalog or load_catalog()
    items = {item.get("name", item["key"].split("/")[-1]): item for item in task["objects"]}
    out: dict[str, SpawnPose] = {}
    for name, box in region.items():
        item = items[name]
        key, yaw = item["key"], float(item.get("yaw", 0.0))
        nominal = np.asarray(item.get("xy", (0.2, 0.0)), dtype=float)
        pose, detail, quat = pick_spawn_pose(box, rng)
        # 姿态可以有自己的矩形（``pose_regions``）：例如"平放"只在机械臂够得着的近处出现，
        # 因为躺姿的抓取点更低、远端压不下去（见 t1 的 spawn_region 注释）。
        # 键可以写姿态类别（``lying``）也可以写实际朝向名（``laid_225``）。
        pose_areas = box.get("pose_regions", {})
        area = pose_areas.get(detail) or pose_areas.get(pose) or box
        # ``anchor``：这个物体必须**待在另一个也被随机安放的物体里**（t9：番茄酱坐在木托盘里，
        # "从托盘里取物"要求瓶子永远在托盘内底面上）。指到谁，``x``/``y`` 就是**相对谁的偏移**，
        # 跟着锚点一起走（托盘摆到哪儿、瓶子就跟到哪儿），此时不再做避让/拒绝采样 ——
        # 瓶子本来就该"在托盘的占地里面"，拿外接圆去避让会把唯一合法的位置全否掉。
        anchor = box.get("anchor")
        if anchor:
            if anchor in out:
                base = np.asarray(out[anchor].xy, dtype=float)
            else:
                base = np.asarray(items[anchor].get("xy", (0.2, 0.0)), dtype=float)
            off = rng.uniform([area["x"][0], area["y"][0]], [area["x"][1], area["y"][1]])
            pick = np.clip(base + off, -TABLE_HALF + SPAWN_TABLE_MARGIN,
                           TABLE_HALF - SPAWN_TABLE_MARGIN)
            out[name] = SpawnPose((float(pick[0]), float(pick[1])), quat, pose, detail)
            continue
        r_self = (footprint_radius_boxes(posed_boxes(key, quat, catalog)) if quat is not None
                  else footprint_radius(key, yaw, catalog))
        # 其它物体：若它也在 region 里且已经抽过 → 用抽到的**姿态+位置**；否则用任务里写死的位置
        blocked = []
        for other_name, other in items.items():
            if other_name == name:
                continue
            if (region.get(other_name) or {}).get("anchor"):
                # **锚定在别人身上的物体**（t9：番茄酱坐在托盘里）跟着它的容器走，
                # 它的"标称位置"没有参考价值 —— 拿它当障碍只会把容器自己的合法位置全否掉
                # （托盘区域正好盖住番茄酱的标称点 → 200 次全被拒 → 退回标称 = 等于没随机）。
                continue
            other_self = out.get(other_name)
            if other_self is not None and other_self.quat is not None:
                r_other = footprint_radius_boxes(
                    posed_boxes(other["key"], other_self.quat, catalog))
                other_xy = other_self.xy
            else:
                r_other = footprint_radius(other["key"], float(other.get("yaw", 0.0)), catalog)
                other_xy = other_self.xy if other_self else other.get("xy", (0.2, 0.0))
            blocked.append((np.asarray(other_xy, dtype=float), r_other))
        lo = np.maximum(np.array([area["x"][0], area["y"][0]], dtype=float),
                        -TABLE_HALF + SPAWN_TABLE_MARGIN)
        hi = np.minimum(np.array([area["x"][1], area["y"][1]], dtype=float),
                        TABLE_HALF - SPAWN_TABLE_MARGIN)
        pick = nominal
        for _ in range(SPAWN_TRIES):
            cand = rng.uniform(lo, hi)
            if all(float(np.linalg.norm(cand - xy)) > r + r_self + SPAWN_CLEARANCE
                   for xy, r in blocked):
                pick = cand
                break
        out[name] = SpawnPose((float(pick[0]), float(pick[1])), quat, pose, detail)
    return out


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
    # 目标物**也被随机安放**的任务（``xy_ref="target"``，如 t1 的篮子）不画固定圆盘：
    # 那个圈只会停在 XML 里的标称位置，跟实际随机的目标对不上，反而误导操作者
    # （篮子本身就是最好的目标标记）。其它任务的场景与图片完全不变。
    show_region = task["kind"] in ("region", "stack", "into") and (
        task["success"].get("xy_ref") != "target")
    layout = {
        "name": task["id"],
        "objects": task["objects"],
        "extra_geoms": [target_region_geom(task["success"])] if show_region else [],
    }
    return builder.build(layout), builder


# ─────────────────── 成功判定 ───────────────────
def object_world_aabb(model, data, catalog: dict, name: str):
    """用物体世界位姿 + catalog 的局部 AABB 算出世界系 AABB（不假设物体一定竖直）。"""
    import itertools

    from libero_catalog import boxes_aabb, quat_to_mat

    body_id = model.body(name).id
    lo, hi = boxes_aabb(catalog_object(catalog, name)["boxes"])
    rot = quat_to_mat(data.xquat[body_id])
    center = data.xpos[body_id]
    corners = np.array([center + rot @ np.array(corner)
                        for corner in itertools.product(*zip(lo, hi))])
    return corners.min(axis=0), corners.max(axis=0)


def object_lowest_z(model, data, catalog: dict, name: str) -> float:
    """物体**碰撞盒角点**里最低的那个 z（"真正的最低材料点"）。

    和 ``object_world_aabb`` 的 ``lo[2]`` 的区别：AABB 用的是"物体**局部 AABB** 的 8 个角点"，
    对**斜放**的物体那个角点往往是**空的** —— 实测平放进篮子的瓶子：AABB 底面 806mm（相对
    桌面 +6mm，只差 6mm 就会被 z 判据判失败），而真实最低点 821mm 正压在篮底（+21mm）。
    轴对齐时两者完全相同（其余 8 关的姿态都是轴对齐的，判据一字不变）。
    """
    import itertools

    from libero_catalog import normalize_quat, quat_to_mat

    body_id = model.body(name).id
    obj = catalog_object(catalog, name)
    rot = quat_to_mat(data.xquat[body_id])
    org = data.xpos[body_id]
    z_min = np.inf
    for bpos, bquat, bsize in obj["boxes"]:
        brot = quat_to_mat(normalize_quat(bquat))
        half = np.asarray(bsize, dtype=float)
        for corner in itertools.product(*zip(-half, half)):
            point = org + rot @ (np.asarray(bpos, dtype=float) + brot @ np.asarray(corner))
            z_min = min(z_min, float(point[2]))
    return float(z_min)


def check_success(task: dict, model, data, catalog: dict | None = None) -> tuple[bool, str]:
    """检查任务是否完成，返回 (是否成功, 说明文字)。"""
    catalog = catalog or load_catalog()
    spec = task["success"]
    lo, hi = object_world_aabb(model, data, catalog, task["grasp_object"])
    center_xy = (lo + hi)[:2] / 2.0
    # 目标 xy：默认用任务里写死的；``xy_ref="target"``（目标物也被随机安放，如 t1 的篮子）
    # 改用**目标物当前实际中心** —— 与 place_object 的"对准目标当前位置"保持一致。
    target_xy = np.asarray(spec["xy"], dtype=float)
    live_target = spec.get("xy_ref") == "target" and task.get("target_object")
    if live_target:
        tlo, thi = object_world_aabb(model, data, catalog, task["target_object"])
        target_xy = (tlo + thi)[:2] / 2.0
    error_xy = float(np.linalg.norm(center_xy - target_xy))

    if spec["z_ref"] == "target_top":
        target = catalog_object(catalog, task["target_object"])
        z_ref = TABLE_TOP_Z + float(target["size"][2])
    else:
        z_ref = TABLE_TOP_Z
    # 用**真实最低点**（碰撞盒角点）而不是 AABB 的 lo[2]：斜放物体的 AABB 底面是空角点，
    # 会把"已经稳稳放在篮子/盘子里"的情况算低十几毫米（见 ``object_lowest_z``）。
    bottom = float(object_lowest_z(model, data, catalog, task["grasp_object"]) - z_ref)

    ok_xy = error_xy <= spec["xy_tol"]
    # 高度判据留 2mm 弹性：物体静止时可能因接触软约束略低于基准（实测 -0.4mm 会被判失败）
    ok_z = (spec["z_band"][0] - 0.002) <= bottom <= (spec["z_band"][1] + 0.002)
    message = (f"水平偏差 {error_xy * 1000:.0f}mm / 允许 {spec['xy_tol'] * 1000:.0f}mm；"
               f"底面相对基准 {bottom * 1000:+.0f}mm / 要求 "
               f"[{spec['z_band'][0] * 1000:+.0f}, {spec['z_band'][1] * 1000:+.0f}]mm")
    if live_target:
        message += f"（目标={task['target_object']}当前位置）"
    return bool(ok_xy and ok_z), message
