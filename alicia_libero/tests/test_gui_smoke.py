"""GUI 冒烟测试：真实创建窗口，自动切任务、模拟鼠标拖动/按键/滑块，并捕获所有异常。

运行：python tests/test_gui_smoke.py    （期望输出：全部 PASS、0 异常；结尾会打印"通过 N/N"）
"""
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ERRORS = []
def _hook(exc_type, exc, tb):
    ERRORS.append("".join(traceback.format_exception(exc_type, exc, tb)))
    sys.__excepthook__(exc_type, exc, tb)
sys.excepthook = _hook

import alicia_libero_app as app_mod  # noqa: E402
import libero_tasks as tasks_mod  # noqa: E402  （算"随机安放的占地间隙"用）
import skills as skills_mod  # noqa: E402  （量"闭合轴方向上的宽度"用）

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}  {detail}")


def main() -> int:
    app = QApplication(sys.argv)
    window = app_mod.MainWindow()
    window.resize(1400, 800)
    window.show()

    steps = []

    def step(fn):
        steps.append(fn)

    def run_next():
        if not steps:
            window.close()
            app.quit()
            return
        fn = steps.pop(0)
        try:
            fn()
        except Exception:  # noqa: BLE001
            ERRORS.append(traceback.format_exc())
            print("  FAIL  步骤抛异常:", traceback.format_exc().splitlines()[-1])
        QTimer.singleShot(700, run_next)

    # 1) 初始任务加载
    def s1():
        sess = window.session
        check("初始任务加载", sess is not None and sess.task["id"].startswith("t1"),
              f"{sess.task['id']} nq={sess.model.nq} nu={sess.model.nu}")
        # 瓶子可能立着（body 原点 ≈ 873mm）也可能平放（≈ 828mm，§8.9），两者都要认
        kz = float(sess.data.xpos[sess.model.body('ketchup').id][2])
        kp = sess.spawn["ketchup"]
        want = 0.873 if kp.quat is None else 0.828
        check("初始物体在桌面", abs(kz - want) < 0.02,
              f"ketchup z={kz:.3f}（{kp.label}，期望 {want:.3f}）")
        check("渲染出图", window.view.pixmap() is not None and not window.view.pixmap().isNull(),
              f"{window.view.pixmap().width()}x{window.view.pixmap().height()}")

    # 2) 鼠标拖动 → IK 移动
    def s2():
        before = window.session.ee_target.copy()
        for _ in range(40):
            window.drag_end_effector(5.0, 0.0, window.view.pixmap().width(),
                                     window.view.pixmap().height())
        after = window.session.ee_target.copy()
        moved = float(np.linalg.norm(after - before))
        check("鼠标拖动改变末端目标", moved > 0.02, f"移动 {moved*1000:.0f}mm")
        check("IK 解出的关节角非零", abs(window.session.joint_target).max() > 0.05,
              f"|q|max={abs(window.session.joint_target).max():.2f} rad")

    # 3) 滚轮升降 + 夹爪
    def s3():
        z0 = window.session.ee_target[2]
        window.nudge_end_effector(-3)
        check("滚轮降低末端目标", window.session.ee_target[2] < z0 - 0.01,
              f"z {z0:.3f} → {window.session.ee_target[2]:.3f}")
        window.grip_slider.setValue(100)
        window.nudge_gripper(-1.0)
        check("夹爪开合", window.session.gripper < 0.5, f"gripper={window.session.gripper:.2f}")

    # 4) 切任务（下拉框）
    def s4():
        window.task_combo.setCurrentIndex(8)     # 任务 9：从篮子里取出
        check("切换任务重建场景", window.session.task["id"].startswith("t9"),
              window.session.task["id"])
        check("任务文本已更新",
              window.task_text.text() == window.session.task["task_text"],
              window.task_text.text()[:26])
        check("素材分析已显示", "mm" in window.object_text.text(), window.object_text.text()[:40])

    # 5) 键盘虚拟示教臂
    def s5():
        window.mode_combo.setCurrentIndex(1)
        QTest.keyClick(window, Qt.Key_6)
        before = window.session.joint_target[5]
        for _ in range(5):
            QTest.keyClick(window, Qt.Key_Period)
        check("键盘微调 Joint6", abs(window.session.joint_target[5] - before) > 0.05,
              f"{np.degrees(before):.1f}° → {np.degrees(window.session.joint_target[5]):.1f}°")
        QTest.keyClick(window, Qt.Key_C)
        check("键盘闭合夹爪", window.session.gripper < 1.0, f"gripper={window.session.gripper:.2f}")

    # 6) 关节滑块模式
    def s6():
        window.mode_combo.setCurrentIndex(2)
        window.joint_sliders[0].setValue(25)
        check("滑块驱动关节", abs(np.degrees(window.session.joint_target[0]) - 25.0) < 0.6,
              f"J1={np.degrees(window.session.joint_target[0]):.1f}°")

    # 7) 相机 / 复位
    def s7():
        window.camera_combo.setCurrentIndex(1)
        check("切相机", window.session.camera == "cam_top", window.session.camera)
        window.joint_sliders[0].setValue(60)
        window.callback_reset()
        grasp = window.session.task["grasp_object"]
        z = float(window.session.data.xpos[window.session.model.body(grasp).id][2])
        check("复位场景", abs(window.session.joint_target[0]) < 1e-6 and z > 0.80,
              f"J1={np.degrees(window.session.joint_target[0]):.1f}° {grasp} z={z:.3f}")

    # 8) 自动执行（闭环技能库）已接好：界面/会话里应有对应接口，且不再有旧的 libero_plan
    def s8():
        import importlib.util

        check("旧规划模块仍未复活", importlib.util.find_spec("libero_plan") is None,
              "libero_plan 不存在")
        check("技能库可导入", importlib.util.find_spec("skills") is not None,
              "alicia_libero/skills.py 可用")
        check("会话带闭环技能所需接口",
              all(hasattr(window.session, name) for name in
                  ("set_ee_target", "tcp_position", "gripper", "step", "task", "home_tcp")),
              "set_ee_target/tcp_position/gripper/step/task/home_tcp 齐备")
        check("界面有自动执行控件",
              all(hasattr(window, name) for name in
                  ("auto_button", "auto_step_button", "auto_status", "auto_result")),
              "▶ 自动完成本关 / 单步 / 状态 / 明细")
        check("SkillRunner 已挂到会话",
              window.auto is not None and window.auto.sess is window.session,
              "window.auto 指向当前 session")
        import skills as skills_mod

        check("技能库带回程（return_home）",
              hasattr(skills_mod, "return_home") and hasattr(window.session, "home_tcp"),
              f"return_home + session.home_tcp={np.round(window.session.home_tcp * 1000, 1)}")
        # 单步一次：应能推进技能状态且不抛异常
        window.auto.start()
        window.callback_auto_single()
        check("自动执行可单步推进", window.auto.status not in ("", "就绪"),
              f"status={window.auto.status}")
        window.auto.stop()

    # 10) t1 抓取物（位置+姿态）与篮子随机安放（任务多样化：每次复位换一局新摆位）
    def s10():
        window.task_combo.setCurrentIndex(0)          # 回到 t1
        regions = window.session.task.get("spawn_region", {})
        region = regions.get("ketchup")
        check("t1 抓取物带随机安放区域", region is not None,
              f"x={region['x']} y={region['y']}" if region else "没有 spawn_region")
        check("t1 抓取物带姿态随机（立着/平放）",
              list(region.get("poses", [])) == ["upright", "lying"] if region else False,
              f"poses={region.get('poses') if region else None}")
        check("t1 篮子也带随机安放区域（目标物随机）",
              regions.get("basket") is not None,
              f"x={regions['basket']['x']} y={regions['basket']['y']}"
              if regions.get("basket") else "篮子没有 spawn_region")
        if region is None:
            return
        seen, bad, seen_basket, poses = set(), [], set(), set()
        for _ in range(6):
            window.callback_reset()                   # 复位 = 换一局新摆位
            sess = window.session
            kp = sess.spawn["ketchup"]
            x, y = kp.xy
            poses.add(kp.pose)
            seen.add((round(x, 4), round(y, 4)))
            if not (region["x"][0] <= x <= region["x"][1] and region["y"][0] <= y <= region["y"][1]):
                bad.append(f"({x:.3f},{y:.3f}) 抽到了区域外")
            adr = sess.model.jnt_qposadr[sess.model.joint("ketchup_joint").id]
            q, q0 = sess.data.qpos[adr:adr + 7], sess.model.qpos0[adr:adr + 7]
            if kp.quat is None:                       # 立着：沿用 XML 的 z 与姿态
                off = sess.spawn_offset["ketchup"]
                if abs(q[0] - off[0] - x) > 1e-6 or abs(q[1] - off[1] - y) > 1e-6:
                    bad.append(f"物体没落到抽到的位置（qpos x={q[0]:.4f}，应为 {x + off[0]:.4f}）")
                if abs(q[2] - q0[2]) > 1e-9 or not np.allclose(q[3:7], q0[3:7], atol=1e-9):
                    bad.append("立着那局 z 或姿态被改了")
            else:                                     # 平放：姿态要写进去、底面要重新贴桌
                if not np.allclose(q[3:7], kp.quat, atol=1e-6):
                    bad.append("平放那局的四元数没写进自由关节")
                lo, _ = tasks_mod.object_world_aabb(sess.model, sess.data, sess.catalog, "ketchup")
                low = tasks_mod.object_lowest_z(sess.model, sess.data, sess.catalog, "ketchup")
                if abs(low - tasks_mod.TABLE_TOP_Z) > 0.002:
                    bad.append(f"平放那局最低点 {low * 1000:.1f}mm 没贴在桌面上")
            # 闭合轴要跟着姿态走：立着 +90°；平放则取"沿物体薄边"的等价方向并折算到负半圈
            yaw = sess.grasp_yaw
            if kp.quat is None:
                if abs(yaw - 90.0) > 1.0:
                    bad.append(f"立着那局的闭合轴 {yaw:.0f}° 应为 +90°")
            else:
                if not (-180.0 < yaw <= 0.0):
                    bad.append(f"平放那局的闭合轴 {yaw:.0f}° 没折算到负半圈")
                # 更有意义的检查：闭合方向上量到的宽度必须是**薄边**（瓶子 36.8mm）
                thin = skills_mod.thin_axis_world(sess, "ketchup")
                wide = skills_mod.width_along_dir(sess, "ketchup", thin) * 1000
                if abs(wide - 36.8) > 1.5:
                    bad.append(f"{kp.label}那局闭合轴方向量到的宽度 {wide:.1f}mm ≠ 薄边 36.8mm（夹错方向）")
            # 篮子：同样要落进它自己的区域、真写进仿真、不歪；并与抓取物保持占地间隙
            box = regions.get("basket")
            if box is None:
                continue
            bp = sess.spawn["basket"]
            bx, by = bp.xy
            seen_basket.add((round(bx, 4), round(by, 4)))
            if not (box["x"][0] <= bx <= box["x"][1] and box["y"][0] <= by <= box["y"][1]):
                bad.append(f"篮子 ({bx:.3f},{by:.3f}) 抽到了区域外")
            badr = sess.model.jnt_qposadr[sess.model.joint("basket_joint").id]
            boff = sess.spawn_offset["basket"]
            bq, bq0 = sess.data.qpos[badr:badr + 7], sess.model.qpos0[badr:badr + 7]
            if abs(bq[0] - boff[0] - bx) > 1e-6 or abs(bq[1] - boff[1] - by) > 1e-6:
                bad.append("篮子没落到抽到的位置")
            if abs(bq[2] - bq0[2]) > 1e-9 or not np.allclose(bq[3:7], bq0[3:7], atol=1e-9):
                bad.append("篮子的 z 或姿态被改了")
            # 占地间隙要按**本局姿态**算（平放的瓶子占地 33.6→73mm）
            r_k = (tasks_mod.footprint_radius_boxes(
                tasks_mod.posed_boxes("ketchup", kp.quat, sess.catalog))
                if kp.quat is not None else tasks_mod.footprint_radius("ketchup", 0.0, sess.catalog))
            need = r_k + tasks_mod.footprint_radius("basket", 0.0, sess.catalog) \
                + tasks_mod.SPAWN_CLEARANCE
            gap = float(np.hypot(bx - x, by - y))
            if gap < need - 1e-9:
                bad.append(f"抓取物与篮子间距 {gap * 1000:.0f}mm < 需要的 {need * 1000:.0f}mm")
        check("复位会重新随机摆位（多次复位位置不同）", len(seen) >= 5,
              f"{len(seen)} 种：" + " ".join(f"({px * 1000:+.0f},{py * 1000:+.0f})"
                                            for px, py in sorted(seen)))
        check("复位也会随机姿态（6 局里立着/平放都出现过）", len(poses) == 2,
              f"抽到 {sorted(poses)}")
        check("篮子复位也会重新随机摆位", len(seen_basket) >= 5,
              f"{len(seen_basket)} 种：" + " ".join(f"({px * 1000:+.0f},{py * 1000:+.0f})"
                                                   for px, py in sorted(seen_basket)))
        check("摆位（位置+姿态+闭合轴+贴桌+不重叠）都合规", not bad,
              "；".join(bad) if bad else "6 局都合规")

    # 11) 让主循环再跑一会，统计 FPS 并报告
    def s9():
        check("主循环无异常", not ERRORS, f"{len(ERRORS)} 个异常")

    for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10):
        step(fn)

    QTimer.singleShot(900, run_next)
    app.exec()

    print("\n===== 汇总 =====")
    failed = [name for name, ok, _ in results if not ok]
    print(f"通过 {len(results) - len(failed)}/{len(results)}")
    if failed:
        print("失败项:", failed)
    if ERRORS:
        print(f"\n捕获到 {len(ERRORS)} 个异常，第一个：\n{ERRORS[0]}")
    return 1 if (failed or ERRORS) else 0


import numpy as np  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
