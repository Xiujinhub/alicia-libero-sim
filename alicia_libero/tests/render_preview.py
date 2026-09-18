"""渲染预览工具：把每个任务、每个相机的离屏渲染图存成 PNG，便于肉眼检查可见性。

运行：python tests/render_preview.py [任务序号...]
产物：alicia_libero/preview/<任务id>_<相机名>.png
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from libero_catalog import load_catalog  # noqa: E402
from libero_tasks import TASKS, build_task_scene  # noqa: E402
from PIL import Image  # noqa: E402

OUT = HERE.parent / "preview"
OUT.mkdir(exist_ok=True)
CAMERAS = ["cam_front", "cam_top", "cam_side", "cam_wrist"]

catalog = load_catalog()
picked = [int(v) for v in sys.argv[1:]] or [0]

for index in picked:
    task = TASKS[index]
    xml, _ = build_task_scene(task, catalog=catalog)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    for _ in range(60):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=640, width=900)
    for cam in CAMERAS:
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        path = OUT / f"{task['id']}_{cam}.png"
        Image.fromarray(frame).save(path)
        gray = frame.mean(axis=2)
        print(f"{task['id']:32s} {cam:11s} 亮度均值={gray.mean():6.1f} "
              f"方差={gray.std():6.1f} 最亮像素={frame.max()}  非背景占比="
              f"{float((np.abs(gray - gray.max()) > 12).mean()) * 100:5.1f}%  → {path.name}")
    del renderer
