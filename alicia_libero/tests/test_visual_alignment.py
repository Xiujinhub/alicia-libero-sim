"""视觉/碰撞一致性测试：检查每个物体的**视觉网格**是否和**碰撞盒**重合。

作用：防止"瓶子躺在地上、旁边立着一个灰色方块"这类问题——libero 素材的碰撞盒套了
``upright_quat`` 旋转，如果视觉网格漏掉这个旋转，两者就会差 90°。
判定：网格 AABB 中心与碰撞盒 AABB 中心距离 < 30mm，且两者尺寸比例接近。

运行：python tests/test_visual_alignment.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from libero_catalog import load_catalog  # noqa: E402
from libero_tasks import TASKS, build_task_scene  # noqa: E402

MAX_CENTER_GAP = 0.030        # 30mm
MIN_SIZE_RATIO = 0.75         # 网格/盒 尺寸比例允许范围

catalog = load_catalog()


def geom_points(model, data, gid):
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        verts = model.mesh_vert[model.mesh_vertadr[mid]:
                                model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    else:
        s = model.geom_size[gid]
        verts = np.array([[sx, sy, sz] for sx in (-s[0], s[0])
                          for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
    rot = data.geom_xmat[gid].reshape(3, 3)
    return (rot @ verts.T).T + data.geom_xpos[gid]


failures = []
for task in TASKS:
    xml, _ = build_task_scene(task, catalog=catalog)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    for _ in range(80):
        mujoco.mj_step(model, data)
    for item in task["objects"]:
        name = item.get("name", item["key"].split("/")[-1])
        bid = model.body(name).id
        spans = {}
        for gid in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
            kind = "mesh" if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH else "box"
            pts = geom_points(model, data, gid)
            if kind not in spans:
                spans[kind] = [pts.min(axis=0), pts.max(axis=0)]
            else:
                spans[kind][0] = np.minimum(spans[kind][0], pts.min(axis=0))
                spans[kind][1] = np.maximum(spans[kind][1], pts.max(axis=0))
        if "mesh" not in spans:
            continue
        mesh_c = (spans["mesh"][0] + spans["mesh"][1]) / 2
        box_c = (spans["box"][0] + spans["box"][1]) / 2
        gap = float(np.linalg.norm(mesh_c - box_c))
        mesh_sz = spans["mesh"][1] - spans["mesh"][0]
        box_sz = spans["box"][1] - spans["box"][0]
        ratio = float(np.min(mesh_sz / np.maximum(box_sz, 1e-6)))
        ok = gap <= MAX_CENTER_GAP and ratio >= MIN_SIZE_RATIO
        if not ok:
            failures.append(f"{task['id']}/{name}")
        print(f"{task['id'][:26]:26s} {name:22s} 中心差={gap * 1000:6.1f}mm "
              f"尺寸比={ratio:4.2f}  网格(mm)={[round(v * 1000) for v in mesh_sz]} "
              f"{'PASS' if ok else 'FAIL'}")

print("-" * 78)
print(f"共 {len(failures)} 个物体不一致: {failures}" if failures else "全部物体视觉/碰撞一致")
sys.exit(1 if failures else 0)
