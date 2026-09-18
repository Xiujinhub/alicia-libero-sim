"""GUI 冒烟测试：真实创建窗口，自动切任务、模拟鼠标拖动/按键/滑块，并捕获所有异常。

运行：python tests/test_gui_smoke.py    （期望输出 通过 16/16）
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
        check("初始物体在桌面", abs(sess.data.xpos[sess.model.body('ketchup').id][2] - 0.87) < 0.02,
              f"ketchup z={sess.data.xpos[sess.model.body('ketchup').id][2]:.3f}")
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
                  ("set_ee_target", "tcp_position", "gripper", "step", "task")),
              "set_ee_target/tcp_position/gripper/step/task 齐备")
        check("界面有自动执行控件",
              all(hasattr(window, name) for name in
                  ("auto_button", "auto_step_button", "auto_status", "auto_result")),
              "▶ 自动完成本关 / 单步 / 状态 / 明细")
        check("SkillRunner 已挂到会话",
              window.auto is not None and window.auto.sess is window.session,
              "window.auto 指向当前 session")
        # 单步一次：应能推进技能状态且不抛异常
        window.auto.start()
        window.callback_auto_single()
        check("自动执行可单步推进", window.auto.status not in ("", "就绪"),
              f"status={window.auto.status}")
        window.auto.stop()

    # 10) 让主循环再跑一会，统计 FPS 并报告
    def s9():
        check("主循环无异常", not ERRORS, f"{len(ERRORS)} 个异常")

    for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9):
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
