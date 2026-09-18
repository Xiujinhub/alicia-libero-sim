#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭环技能库的无头批量试验：每关跑 N 次（可加位置抖动），统计成功率。

用法
----
python tests/bench_tasks.py 1                 # 任务 1 跑 5 次（默认）
python tests/bench_tasks.py 1 10              # 任务 1 跑 10 次
python tests/bench_tasks.py 1 10 0.01 -v      # 加 ±10mm 位置抖动，并打印每步明细
python tests/bench_tasks.py --all 5           # 9 关每关 5 次
python tests/bench_tasks.py 1 10 0.01 --strict  # 成功率 <100% 时退出码非 0（当回归门槛用）

为什么不用界面：这里直接 ``SimSession(render=False)``，不建离屏渲染器，
所以一轮只要几秒；同一套 skills 代码在界面里由主循环逐帧驱动（见 MainWindow.tick）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from alicia_libero_app import SimSession  # noqa: E402
from libero_catalog import load_catalog  # noqa: E402
from libero_tasks import TASKS, check_success  # noqa: E402
import skills  # noqa: E402

MAX_FRAMES = 3000        # 每轮最多仿真帧数（16ms/帧 → 48s 仿真时间，远超正常需要）


def jitter_scene(sess: SimSession, rng: np.random.Generator, amp: float) -> None:
    """把物体初始位置随机挪一点（模拟"摆得没那么正"），用来验证鲁棒性。"""
    if amp <= 0:
        return
    for item in sess.task["objects"]:
        name = item.get("name", item["key"].split("/")[-1])
        joint = f"{name}_joint"
        try:
            adr = sess.model.jnt_qposadr[sess.model.joint(joint).id]
        except Exception:  # noqa: BLE001  物体不是自由关节就跳过
            continue
        sess.data.qpos[adr] += rng.uniform(-amp, amp)
        sess.data.qpos[adr + 1] += rng.uniform(-amp, amp)
    import mujoco

    mujoco.mj_forward(sess.model, sess.data)


def run_once(task: dict, catalog: dict, rng, jitter: float, verbose: bool = False):
    """跑一轮，返回 (是否完成, 帧数, 每步明细, 失败原因)。"""
    sess = SimSession(task, catalog, render=False)
    sess.reset()
    jitter_scene(sess, rng, jitter)
    runner = skills.SkillRunner(sess)
    runner.start()
    frames = 0
    t0 = time.perf_counter()
    while runner.active and frames < MAX_FRAMES:
        runner.step()
        sess.step(8)
        frames += 1
    wall = time.perf_counter() - t0
    ok, msg = check_success(task, sess.model, sess.data, sess.catalog)
    detail = runner.summary()
    if verbose:
        print(detail)
    sess.renderer = None
    reason = "" if ok else (runner.error or msg)
    return ok, frames, wall, reason


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("-")] or []
    verbose = "-v" in argv
    strict = "--strict" in argv
    catalog = load_catalog()
    if "--all" in argv:
        indices = list(range(len(TASKS)))
        repeats = int(args[0]) if args else 5
        jitter = float(args[1]) if len(args) > 1 else 0.0
    else:
        indices = [int(args[0]) - 1] if args else [0]
        repeats = int(args[1]) if len(args) > 1 else 5
        jitter = float(args[2]) if len(args) > 2 else 0.0

    total_ok = total_run = 0
    for idx in indices:
        task = TASKS[idx]
        rng = np.random.default_rng(20260918 + idx)
        results = []
        for run in range(repeats):
            ok, frames, wall, reason = run_once(task, catalog, rng, jitter, verbose)
            results.append((ok, frames, wall, reason))
            mark = "✅" if ok else "❌"
            print(f"  {mark} {task['id']} 第 {run + 1}/{repeats} 次："
                  f"{frames} 帧 / {wall:.1f}s" + (f"  ← {reason}" if reason else ""))
            if verbose:
                print()
        ok_n = sum(1 for r in results if r[0])
        total_ok += ok_n
        total_run += repeats
        print(f"→ {task['id']}（{task['kind']}，难度 {task['difficulty']}）"
              f" 成功率 {ok_n}/{repeats}，平均 {np.mean([r[1] for r in results]):.0f} 帧"
              f"（抖动 ±{jitter * 1000:.0f}mm）\n")
    print(f"===== 合计 {total_ok}/{total_run} =====")
    if strict and total_ok != total_run:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
