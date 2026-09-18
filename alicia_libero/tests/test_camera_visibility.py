"""相机可见性测试：用**分割渲染**数出各机位画面里属于机械臂的像素占比。

作用：防止"界面里只看到地板"这类相机朝向/位置写错的问题（cam_front 曾经正好背对桌子）。
运行：python tests/test_camera_visibility.py    （期望：每个机位都能看到机械臂与其他物体）
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from libero_catalog import load_catalog  # noqa: E402
from libero_tasks import TASKS, build_task_scene  # noqa: E402

CAMERAS = ["cam_front", "cam_top", "cam_side", "cam_wrist"]
ARM_BODIES = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6",
              "left_gripper", "right_gripper", "tool0")

catalog = load_catalog()
xml, _ = build_task_scene(TASKS[0], catalog=catalog)
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
for _ in range(60):
    mujoco.mj_step(model, data)

# 每个 geom 属于哪个 body，用来把像素归类
geom_body = np.array([model.geom_bodyid[g] for g in range(model.ngeom)])
arm_body_ids = {model.body(name).id for name in ARM_BODIES}
object_body_ids = {model.body(name).id for name in ("ketchup", "basket", "table")}

renderer = mujoco.Renderer(model, height=480, width=720)
renderer.enable_segmentation_rendering()
failures = []
print(f"{'机位':11s} {'机械臂':>8s} {'物体/桌子':>10s} {'地板/背景':>10s}   判定")
print("-" * 62)
for cam in CAMERAS:
    renderer.update_scene(data, camera=cam)
    seg = renderer.render()
    geom_ids = np.asarray(seg)[..., 0].astype(int)
    total = geom_ids.size
    is_arm = np.isin(geom_body[np.clip(geom_ids, 0, model.ngeom - 1)], list(arm_body_ids)) & (geom_ids >= 0)
    is_obj = np.isin(geom_body[np.clip(geom_ids, 0, model.ngeom - 1)], list(object_body_ids)) & (geom_ids >= 0)
    arm_pct = 100.0 * is_arm.sum() / total
    obj_pct = 100.0 * is_obj.sum() / total
    other = 100.0 - arm_pct - obj_pct
    # 腕部相机装在工具上、朝桌面看，画面里本来就不该有手臂本身
    ok = obj_pct >= 5.0 if cam == "cam_wrist" else (arm_pct >= 2.0 and obj_pct >= 0.5)
    if not ok:
        failures.append(cam)
    print(f"{cam:11s} {arm_pct:7.1f}% {obj_pct:9.1f}% {other:9.1f}%   "
          f"{'PASS 可见' if ok else 'FAIL 看不到机械臂/物体'}")
renderer.close()

print("-" * 62)
print("全部机位可见" if not failures else f"不可见的机位：{failures}")
sys.exit(1 if failures else 0)
