"""生成全部任务场景 XML（也可作为回归测试：能加载 + 物体都精确落在桌面）。"""
import sys
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from libero_catalog import load_catalog  # noqa: E402
from libero_tasks import TASKS, build_task_scene, check_success  # noqa: E402
from libero_scene import TABLE_TOP_Z  # noqa: E402


def main() -> int:
    catalog = load_catalog()
    failures = []
    print(f"{'任务':34s} {'nq':>4s} {'nu':>3s} {'物体落桌偏差(m)':>16s} {'判据':>6s}")
    print("-" * 78)
    for task in TASKS:
        xml, _ = build_task_scene(task, catalog=catalog)
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(500):                      # 静置 1 秒
            mujoco.mj_step(model, data)
        from libero_tasks import object_world_aabb

        drops = []
        for item in task["objects"]:
            name = item["key"].split("/")[-1]
            lo, _ = object_world_aabb(model, data, catalog, name)
            drops.append(lo[2] - TABLE_TOP_Z)
        ok, _ = check_success(task, model, data, catalog)
        worst = max(drops)
        good = all(abs(v) < 0.085 for v in drops)   # 允许"放进容器里"导致的抬高
        if not good:
            failures.append(task["id"])
        print(f"  {task['id']:32s} {model.nq:4d} {model.nu:3d} "
              f"{worst * 1000:+13.1f}mm {'OK' if good else 'BAD':>5s} {'完成' if ok else '未完成':>5s}")
    print("-" * 78)
    print(f"共 {len(TASKS)} 个场景，{len(failures)} 个异常：{failures if failures else '无'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
