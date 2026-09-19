"""GUI 冒烟测试：真实创建窗口，自动切任务、模拟鼠标拖动/按键/滑块，并捕获所有异常。

运行：python tests/test_gui_smoke.py    （期望输出：全部 PASS、0 异常；结尾会打印"通过 N/N"）
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# 数据集采集（s12）写到本项目的 Datasets/_smoke 沙盒里（跑完自动清理，别脏了真数据集）
REC_ROOT = HERE.parents[1] / "Datasets" / "_smoke"
shutil.rmtree(REC_ROOT, ignore_errors=True)
REC_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["ALICIA_DATASETS_DIR"] = str(REC_ROOT)

from PySide6.QtCore import QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ERRORS = []
def _hook(exc_type, exc, tb):
    ERRORS.append("".join(traceback.format_exception(exc_type, exc, tb)))
    sys.__excepthook__(exc_type, exc, tb)
sys.excepthook = _hook

import alicia_libero_app as app_mod  # noqa: E402
import dataset_recorder as rec_mod  # noqa: E402
import libero_tasks as tasks_mod  # noqa: E402  （算"随机安放的占地间隙"用）
import skills as skills_mod  # noqa: E402  （量"闭合轴方向上的宽度"用）

# 保险：冒烟测试**只准**往沙盒目录写数据集，绝不碰用户的 Datasets/（s12 会真的录一小段）
assert rec_mod.datasets_dir() == REC_ROOT, \
    f"冒烟测试必须只写沙盒目录，实际 = {rec_mod.datasets_dir()}"

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
        # 连续任务面板的**构造默认值**（放在这里查最稳：后面的步骤会切模式/勾选表）
        check("连续任务默认单任务模式、勾选表不可点、采集未开始",
              window.cont_mode_combo.currentIndex() == 0
              and not window.cont_task_list.isEnabled()
              and not window.rec_check.isChecked() and not window.recorder.active,
              f"模式={window.cont_mode_combo.currentText()}、"
              f"勾选表 enabled={window.cont_task_list.isEnabled()}")

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

    # 12) 数据集采集（LeRobot 格式）：勾选即采 / 一局=复位边界 / 每相机一个 mp4 / 断点续采
    def s12():
        import json
        import os
        import dataset_recorder as dsmod

        root = Path(os.environ["ALICIA_DATASETS_DIR"])
        window.rec_name.setText("smoke_ds")
        window.rec_fps.setValue(60)                   # 60Hz：一 tick 一帧，测试里攒帧最快
        window.rec_only_ok.setChecked(False)          # 这一局不判成功也要存，方便验证
        window.rec_resume.setChecked(False)
        window.rec_check.setChecked(True)             # 勾选 = 只是"要不要采"的开关
        check("勾选「数据采集」只是开关：不建库、不记录",
              (not window.recorder.active) and window.recorder.path is None
              and window.recorder.wanted,
              window.rec_status.text().replace("\n", " "))
        check("状态行提示已就绪（等着点连续任务）",
              "就绪" in window.rec_status.text(),
              window.rec_status.text().replace("\n", " "))

        # 点「开始连续任务」→ 才真正开录（用假技能流走完这一局，不真跑物理，秒级）
        window.cont_mode_combo.setCurrentIndex(0)
        window.cont_count_spin.setValue(1)
        window.task_combo.setCurrentIndex(0)
        window.start_continuous()
        rec = window.recorder
        first = rec.path
        check("点「开始连续任务」才开始采集（后台建库）", rec.active and first is not None,
              f"path={first.name if first else None}")
        check("不勾断点续采 → 用带时间戳的新名字",
              first is not None and first.name.startswith("smoke_ds_") and first.parent == root,
              f"{first.name if first else None} @ {root.name}")

        end = time.perf_counter() + 240               # 等后台把 lerobot 建库打开（首次导入较慢）
        while time.perf_counter() < end and not window.recorder.error:
            if window.recorder._dataset is not None:  # noqa: SLF001（测试里允许看内部）
                break
            window.tick()
            app.processEvents()
            time.sleep(0.02)
        check("数据集在后台打开成功", not window.recorder.error, window.recorder.error or "ok")

        for _ in range(12):                           # 攒几帧（60Hz + 每 tick 16ms 仿真 ≈ 一 tick 一帧）
            window.tick()
            app.processEvents()
            QTest.qWait(10)
        check("连续任务进行中才记录", rec.frames + rec.buffered > 0,
              f"{rec.frames + rec.buffered} 帧")

        def slow_gen(ok=True):                        # 假技能流：多 yield 几下，够攒几帧
            for _ in range(6):
                yield "假执行"
            return [skills_mod.Step(skills_mod.JUDGE_LABEL, ok, "假判分")]

        window.auto.gen = slow_gen(True)              # 换掉真技能，秒级走完这一局
        end = time.perf_counter() + 120
        while time.perf_counter() < end and window.cont_active:
            window.tick()
            app.processEvents()
        end = time.perf_counter() + 180               # 等这一局写完（视频编码在后台）
        while time.perf_counter() < end:
            app.processEvents()
            if rec.episodes >= 1 and rec.queued == 0:
                break
            time.sleep(0.02)
        check("连跑 1 局 = 数据集里 1 个 episode", rec.episodes == 1 and rec.frames >= 3,
              f"{rec.episodes} 局 / {rec.frames} 帧")
        check("连跑结束后数据集仍打开（下次连跑继续追加）",
              rec.active and not window.cont_active, f"active={rec.active}")
        frames_after = rec.frames
        for _ in range(10):                           # 不在连续任务里 → 不应再记录
            window.tick()
            app.processEvents()
        check("不在连续任务里就不记录", rec.frames == frames_after and rec.buffered == 0,
              f"{frames_after} → {rec.frames} 帧")

        window.rec_check.setChecked(False)            # 取消勾选 = 收尾 + 关闭数据集
        check("取消勾选后收尾、无错误", (not rec.active) and not rec.error, rec.error or "ok")

        info = json.loads((first / "meta" / "info.json").read_text(encoding="utf-8"))
        video_keys = sorted(k for k, v in info["features"].items() if v["dtype"] == "video")
        check("LeRobot 元数据：四路相机都是 video 特征",
              video_keys == sorted(f"observation.images.{c}" for c in app_mod.CAMERAS),
              f"fps={info['fps']} 帧={info['total_frames']} 局={info['total_episodes']}")
        mp4s = {c: len(list((first / "videos" / f"observation.images.{c}"
                              / "chunk-000").glob("*.mp4"))) for c in app_mod.CAMERAS}
        check("一个相机一个 mp4", set(mp4s.values()) == {1}, str(mp4s))
        check("录像尺寸 = 320×228（h264 要偶数边、和声明一致）",
              tuple(info["features"][video_keys[0]]["shape"]) == (*dsmod.record_shape(), 3),
              str(tuple(info["features"][video_keys[0]]["shape"])))
        check("parquet 帧数 = 落盘帧数",
              len(pd.read_parquet(first / "data" / "chunk-000" / "file-000.parquet")) == rec.frames,
              f"{rec.frames} 帧")
        top = sorted(p.name for p in first.iterdir() if p.is_dir())
        check("数据集里只留 data/meta/videos（编视频的临时 images/ 已清）",
              top == ["data", "meta", "videos"] and not (first / "images").exists(),
              f"顶层目录={top}")

        # 断点续采：勾上 → 再点一次连跑 → 回到同名目录继续追加
        window.rec_resume.setChecked(True)            # 名称框里已经是实际生效的名字
        window.rec_check.setChecked(True)
        window.start_continuous()
        check("断点续采 → 回到同名目录、且识别为续采",
              window.recorder.path == first and window.recorder.resumed,
              f"{window.recorder.path.name} resumed={window.recorder.resumed}")
        window.auto.gen = slow_gen(True)
        end = time.perf_counter() + 120
        while time.perf_counter() < end and window.cont_active:
            window.tick()
            app.processEvents()
        window.rec_check.setChecked(False)
        check("续采收尾无错误", not window.recorder.error, window.recorder.error or "ok")

        # 视角勾选决定录哪几路：只勾前两路 → 数据集里就只有两路视频（没勾的连目录都没有）
        window.view_checks["cam_side"].setChecked(False)
        window.view_checks["cam_wrist"].setChecked(False)
        check("取消勾选的视角不参与采集（渲染名单同步）",
              window.checked_cameras() == ["cam_front", "cam_top"]
              or window.visible_cameras() == ["cam_front", "cam_top"],
              f"勾选={window.checked_cameras()}")
        window.rec_resume.setChecked(False)
        window.rec_name.setText("smoke_cams")
        window.rec_check.setChecked(True)
        window.start_continuous()
        two = window.recorder.path
        check("建库时按勾选集录（2 路）", window.recorder.cameras == ["cam_front", "cam_top"],
              f"{two.name if two else None} ← {window.recorder.cameras}")
        window.view_checks["cam_top"].setChecked(False)     # 中途改勾选：只提示，不换名单
        check("录的中途改勾选 → 状态行提示仍按开录名单（重开才生效）",
              "视角勾选已改" in window.rec_status.text(),
              window.rec_status.text().replace("\n", " ")[-40:])
        window.view_checks["cam_top"].setChecked(True)
        window.auto.gen = slow_gen(True)
        end = time.perf_counter() + 120
        while time.perf_counter() < end and window.cont_active:
            window.tick()
            app.processEvents()
        end = time.perf_counter() + 180
        while time.perf_counter() < end:
            app.processEvents()
            if window.recorder.episodes >= 1 and window.recorder.queued == 0:
                break
            time.sleep(0.02)
        window.rec_check.setChecked(False)
        info2 = json.loads((two / "meta" / "info.json").read_text(encoding="utf-8"))
        keys2 = sorted(k for k, v in info2["features"].items() if v["dtype"] == "video")
        dirs2 = sorted(p.name for p in (two / "videos").iterdir())
        check("数据集里只有勾选的两路视频特征",
              keys2 == ["observation.images.cam_front", "observation.images.cam_top"],
              str(keys2))
        check("没勾的视角连 videos 目录/mp4 都没有", dirs2 == keys2, str(dirs2))
        for cam in app_mod.CAMERAS:                         # 复原勾选，免得影响后面的步骤
            window.view_checks[cam].setChecked(True)

    # 7) 多视角显示（2×2 网格 / 逐格放大 / 逐格勾选）+ 复位
    def s7():
        cams = app_mod.CAMERAS
        check("默认四个视角全部显示（4 个勾选框）",
              [c for c in cams if window.view_checks[c].isChecked()] == cams,
              f"{len(window.view_checks)} 个勾选框："
              + "、".join(c.replace("cam_", "") for c in window.visible_cameras()))
        window.tick()
        check("2×2 拼图：四格、每格占 1/4（含 2px 分隔线）",
              len(window.view_cells) == 4
              and all(w * h == app_mod.VIEW_W * app_mod.VIEW_H
                      for _c, (_x, _y, w, h) in window.view_cells)
              and window.view_image_size == (2 * app_mod.VIEW_W + 2,
                                             2 * app_mod.VIEW_H + 2),
              f"拼图 {window.view_image_size[0]}×{window.view_image_size[1]}、"
              f"{len(window.view_cells)} 格、单格 {app_mod.VIEW_W}×{app_mod.VIEW_H}")
        pixmap = window.view.pixmap()
        image_w, image_h = window.view_image_size
        ox = (window.view.width() - pixmap.width()) / 2.0
        oy = (window.view.height() - pixmap.height()) / 2.0
        centers = [window.view_camera_at(QPointF(
            ox + (x + w / 2) * pixmap.width() / image_w,
            oy + (y + h / 2) * pixmap.height() / image_h))
            for _c, (x, y, w, h) in window.view_cells]
        check("点哪一格就是哪一路机位（拖动按格换算）", centers == cams, str(centers))
        window.toggle_view_maximize("cam_top")
        check("放大一格：它占满、其它三格隐藏",
              window.visible_cameras() == ["cam_top"] and len(window.view_cells) == 1,
              f"visible={window.visible_cameras()}")
        window.toggle_view_maximize("cam_top")
        check("再操作一次还原成 2×2 网格",
              window.view_maximized is None and len(window.visible_cameras()) == 4,
              str(window.visible_cameras()))
        window.view_checks["cam_wrist"].setChecked(False)
        check("取消勾选后该视角不再渲染显示",
              "cam_wrist" not in window.visible_cameras(),
              str(window.visible_cameras()))
        window.view_checks["cam_wrist"].setChecked(True)
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
        for _ in range(20):        # 20 局：概率判据（立着/平放都出现过）要稳
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
        check("复位也会随机姿态（20 局里立着/平放都出现过）", len(poses) == 2,
              f"抽到 {sorted(poses)}")
        check("篮子复位也会重新随机摆位", len(seen_basket) >= 5,
              f"{len(seen_basket)} 种：" + " ".join(f"({px * 1000:+.0f},{py * 1000:+.0f})"
                                                   for px, py in sorted(seen_basket)))
        check("摆位（位置+姿态+闭合轴+贴桌+不重叠）都合规", not bad,
              "；".join(bad) if bad else "各局都合规")

    # 11) t2 抓取物（黄油：位置+姿态）与盘子随机安放（与 t1 同一套机制，见 README §8.9）
    def s11():
        window.task_combo.setCurrentIndex(1)          # 切到 t2
        regions = window.session.task.get("spawn_region", {})
        region = regions.get("butter")
        check("t2 黄油带随机安放区域", region is not None,
              f"x={region['x']} y={region['y']}" if region else "没有 spawn_region")
        check("t2 黄油带姿态随机（立着/平放）",
              list(region.get("poses", [])) == ["upright", "lying"] if region else False,
              f"poses={region.get('poses') if region else None}  lie_axis="
              f"{region.get('lie_axis') if region else None}")
        check("t2 盘子也带随机安放区域（目标物随机）",
              regions.get("plate") is not None,
              f"x={regions['plate']['x']} y={regions['plate']['y']}"
              if regions.get("plate") else "盘子没有 spawn_region")
        check("t2 平放只在这些实测夹得住的朝向上随机",
              tuple(region.get("lying_yaws") or tasks_mod.SPAWN_BOX_YAWS)
              == tasks_mod.SPAWN_BOX_YAWS if region else False,
              f"lying_yaws={region.get('lying_yaws') if region else None}")
        if region is None:
            return
        seen, bad, seen_plate, poses, flats = set(), [], set(), set(), set()
        for _ in range(20):        # 20 局：概率判据（立着/平放都出现过）要稳
            window.callback_reset()
            sess = window.session
            bp = sess.spawn["butter"]
            x, y = bp.xy
            poses.add(bp.pose)
            seen.add((round(x, 4), round(y, 4)))
            if not (region["x"][0] <= x <= region["x"][1] and region["y"][0] <= y <= region["y"][1]):
                bad.append(f"黄油 ({x:.3f},{y:.3f}) 抽到了区域外")
            adr = sess.model.jnt_qposadr[sess.model.joint("butter_joint").id]
            q, q0 = sess.data.qpos[adr:adr + 7], sess.model.qpos0[adr:adr + 7]
            if bp.pose == "upright":                  # 立着：z 不变；朝向可以是（随机出来的）纯 yaw
                if bp.quat is None:
                    if abs(q[2] - q0[2]) > 1e-9 or not np.allclose(q[3:7], q0[3:7], atol=1e-9):
                        bad.append("黄油立着那局 z 或姿态被改了")
                else:
                    if not np.allclose(q[3:7], bp.quat, atol=1e-6):
                        bad.append("黄油立着的随机朝向没写进自由关节")
                    if abs(q[2] - q0[2]) > 1e-9:
                        bad.append("黄油立着那局的 z 变了（纯绕 Z 转不该改高度）")
            else:                                     # 平放：姿态写进去、大平面贴桌
                if not np.allclose(q[3:7], bp.quat, atol=1e-6):
                    bad.append("黄油平放那局的四元数没写进自由关节")
                if bp.detail:
                    flats.add(bp.detail)
                low = tasks_mod.object_lowest_z(sess.model, sess.data, sess.catalog, "butter")
                if abs(low - tasks_mod.TABLE_TOP_Z) > 0.002:
                    bad.append(f"黄油平放那局最低点 {low * 1000:.1f}mm 没贴在桌面上")
            # 闭合轴要跟着姿态走：立着沿 17.4mm 薄边、平放沿 39.5mm 那对侧面
            thin = skills_mod.thin_axis_world(sess, "butter")
            wide = skills_mod.width_along_dir(sess, "butter", thin) * 1000
            want = 17.4 if bp.pose == "upright" else 39.5
            if abs(wide - want) > 1.5:
                bad.append(f"{bp.label}那局闭合轴方向量到的宽度 {wide:.1f}mm ≠ {want}mm（夹错方向）")
            box = regions.get("plate")
            if box is None:
                continue
            pl = sess.spawn["plate"]
            px_, py_ = pl.xy
            seen_plate.add((round(px_, 4), round(py_, 4)))
            if not (box["x"][0] <= px_ <= box["x"][1] and box["y"][0] <= py_ <= box["y"][1]):
                bad.append(f"盘子 ({px_:.3f},{py_:.3f}) 抽到了区域外")
            padr = sess.model.jnt_qposadr[sess.model.joint("plate_joint").id]
            poff = sess.spawn_offset["plate"]
            pq, pq0 = sess.data.qpos[padr:padr + 7], sess.model.qpos0[padr:padr + 7]
            if abs(pq[0] - poff[0] - px_) > 1e-6 or abs(pq[1] - poff[1] - py_) > 1e-6:
                bad.append("盘子没落到抽到的位置")
            if abs(pq[2] - pq0[2]) > 1e-9 or not np.allclose(pq[3:7], pq0[3:7], atol=1e-9):
                bad.append("盘子的 z 或姿态被改了")
        check("t2 复位会重新随机摆位（黄油与盘子位置都变）", len(seen) >= 5 and len(seen_plate) >= 5,
              f"黄油 {len(seen)} 种 / 盘子 {len(seen_plate)} 种")
        check("t2 复位也会随机姿态（20 局里立着/平放都出现过）", len(poses) == 2,
              f"抽到 {sorted(poses)}")
        # 立着的朝向也随机：用固定种子的采样器直查（不受"6 局里刚好几局立着"的随机性影响）
        rng = np.random.default_rng(7)
        up_yaws = {p.detail for p in (tasks_mod.sample_spawn(sess.task, rng, sess.catalog)["butter"]
                                      for _ in range(20)) if p.pose == "upright"}
        check("t2 立着的朝向也随机（20 局抽到多个不同 yaw）", len(up_yaws) >= 8,
              f"{len(up_yaws)} 个：" + " ".join(sorted(up_yaws)[:6]) + " …")
        check("t2 摆位（位置+姿态+闭合轴+贴桌）都合规", not bad, "；".join(bad) if bad else "各局都合规")

    # 11b) t3 抓取物（布丁盒：位置+朝向）与小碟随机安放（照 t2 同一套机制，见 README §8.9.2）
    def s11b():
        window.task_combo.setCurrentIndex(2)          # 切到 t3
        regions = window.session.task.get("spawn_region", {})
        pud = regions.get("chocolate_pudding")
        check("t3 布丁盒带随机安放区域", pud is not None,
              f"x={pud['x']} y={pud['y']}" if pud else "没有 spawn_region")
        check("t3 布丁盒朝向整圆周随机（upright_yaws=None）", pud is not None and "upright_yaws" in pud
              and pud["upright_yaws"] is None,
              f"upright_yaws={pud.get('upright_yaws') if pud else None}")
        check("t3 不做平放（躺下 80.2mm 塞不进小碟内腔，见 §8.9.2）",
              pud is not None and "lying" not in list(pud.get("poses") or ()),
              f"poses={list(pud.get('poses') or ()) if pud else None}")
        check("t3 小碟也带随机安放区域（目标物随机）",
              regions.get("glazed_rim_porcelain_ramekin") is not None,
              f"x={regions['glazed_rim_porcelain_ramekin']['x']} "
              f"y={regions['glazed_rim_porcelain_ramekin']['y']}"
              if regions.get("glazed_rim_porcelain_ramekin") else "小碟没有 spawn_region")
        check("t3 判分取小碟当前位置（xy_ref=target）",
              window.session.task["success"].get("xy_ref") == "target")
        if pud is None:
            return
        seen, seen_dish, bad, poses = set(), set(), [], set()
        for _ in range(20):        # 20 局：概率判据（立着/平放都出现过）要稳
            window.callback_reset()
            sess = window.session
            bp = sess.spawn["chocolate_pudding"]
            x, y = bp.xy
            poses.add(bp.pose)
            seen.add((round(x, 4), round(y, 4)))
            if not (pud["x"][0] <= x <= pud["x"][1] and pud["y"][0] <= y <= pud["y"][1]):
                bad.append(f"布丁盒 ({x:.3f},{y:.3f}) 抽到了区域外")
            adr = sess.model.jnt_qposadr[sess.model.joint("chocolate_pudding_joint").id]
            q, q0 = sess.data.qpos[adr:adr + 7], sess.model.qpos0[adr:adr + 7]
            # 立着 + 纯绕世界 Z 的朝向：高度不变、四元数只有 z 分量与 w
            if abs(q[2] - q0[2]) > 1e-9:
                bad.append("布丁盒那局的 z 变了（纯绕 Z 转不该改高度）")
            if bp.quat is not None and not np.allclose(q[3:7], bp.quat, atol=1e-6):
                bad.append("布丁盒的随机朝向没写进自由关节")
            if abs(float(q[4])) > 1e-6 or abs(float(q[5])) > 1e-6:
                bad.append(f"布丁盒朝向不是纯绕 Z（quat={np.round(q[3:7], 3)}）")
            # 立着才该夹 27.4mm 薄边
            thin = skills_mod.thin_axis_world(sess, "chocolate_pudding")
            wide = skills_mod.width_along_dir(sess, "chocolate_pudding", thin) * 1000
            if abs(wide - 27.4) > 1.5:
                bad.append(f"布丁盒闭合轴方向量到的宽度 {wide:.1f}mm ≠ 27.4mm（夹错方向）")
            dish = sess.spawn.get("glazed_rim_porcelain_ramekin")
            if dish is not None:
                seen_dish.add((round(dish.xy[0], 4), round(dish.xy[1], 4)))
        check("t3 复位会重新随机摆位（布丁盒与小碟位置都变）",
              len(seen) >= 5 and len(seen_dish) >= 5,
              f"布丁盒 {len(seen)} 种 / 小碟 {len(seen_dish)} 种")
        rng = np.random.default_rng(7)
        yaws = {p.detail for p in (tasks_mod.sample_spawn(sess.task, rng, sess.catalog)["chocolate_pudding"]
                                   for _ in range(20))}
        check("t3 朝向整圆周随机（20 局抽到多个不同 yaw）", len(yaws) >= 8,
              f"{len(yaws)} 个：" + " ".join(sorted(yaws)[:6]) + " …")
        check("t3 摆位（位置+朝向+高度+闭合轴）都合规", not bad, "；".join(bad) if bad else "各局都合规")

    # 11c) t5 抓取物（瓶子：位置+姿态）与木托盘随机安放（照 t1 同一套机制，见 README §8.9.3）
    def s11c():
        window.task_combo.setCurrentIndex(4)          # 切到 t5
        regions = window.session.task.get("spawn_region", {})
        bot = regions.get("new_salad_dressing")
        check("t5 瓶子带随机安放区域", bot is not None,
              f"x={bot['x']} y={bot['y']}" if bot else "没有 spawn_region")
        check("t5 瓶子带姿态随机（立着/平放，与 t1 同款）",
              list(bot.get("poses", [])) == ["upright", "lying"] if bot else False,
              f"poses={bot.get('poses') if bot else None}  lie_axis="
              f"{bot.get('lie_axis') if bot else None}（默认 y = 绕薄轴躺）")
        check("t5 立着/平放的朝向都整圆周随机",
              bot is not None and "upright_yaws" in bot and bot["upright_yaws"] is None
              and "lying_yaws" in bot and bot["lying_yaws"] is None,
              f"upright_yaws={bot.get('upright_yaws') if bot else None} / "
              f"lying_yaws={bot.get('lying_yaws') if bot else None}")
        check("t5 木托盘也带随机安放区域（目标物随机）",
              regions.get("wooden_tray") is not None,
              f"x={regions['wooden_tray']['x']} y={regions['wooden_tray']['y']}"
              if regions.get("wooden_tray") else "托盘没有 spawn_region")
        check("t5 判分取托盘当前位置（xy_ref=target）",
              window.session.task["success"].get("xy_ref") == "target")
        if bot is None:
            return
        seen, seen_tray, bad, poses = set(), set(), [], set()
        for _ in range(20):        # 20 局：概率判据（立着/平放都出现过）要稳
            window.callback_reset()
            sess = window.session
            bp = sess.spawn["new_salad_dressing"]
            x, y = bp.xy
            poses.add(bp.pose)
            seen.add((round(x, 4), round(y, 4)))
            if not (bot["x"][0] <= x <= bot["x"][1] and bot["y"][0] <= y <= bot["y"][1]):
                bad.append(f"瓶子 ({x:.3f},{y:.3f}) 抽到了区域外")
            adr = sess.model.jnt_qposadr[sess.model.joint("new_salad_dressing_joint").id]
            q, q0 = sess.data.qpos[adr:adr + 7], sess.model.qpos0[adr:adr + 7]
            if bp.pose == "upright":                  # 立着：z 不变 + 纯绕世界 Z 的朝向
                if abs(q[2] - q0[2]) > 1e-9:
                    bad.append("瓶子立着那局的 z 变了（纯绕 Z 转不该改高度）")
                if abs(float(q[4])) > 1e-6 or abs(float(q[5])) > 1e-6:
                    bad.append(f"瓶子立着朝向不是纯绕 Z（quat={np.round(q[3:7], 3)}）")
            else:                                     # 平放：姿态写进去、侧面贴桌
                if not np.allclose(q[3:7], bp.quat, atol=1e-6):
                    bad.append("瓶子平放那局的四元数没写进自由关节")
                low = tasks_mod.object_lowest_z(sess.model, sess.data, sess.catalog,
                                                "new_salad_dressing")
                if abs(low - tasks_mod.TABLE_TOP_Z) > 0.002:
                    bad.append(f"瓶子平放那局最低点 {low * 1000:.1f}mm 没贴在桌面上")
            # 两种姿态都该沿 35.5mm 薄边闭合（绕薄轴躺才不会变成 52.7mm > 爪口）
            thin = skills_mod.thin_axis_world(sess, "new_salad_dressing")
            wide = skills_mod.width_along_dir(sess, "new_salad_dressing", thin) * 1000
            if abs(wide - 35.5) > 1.5:
                bad.append(f"{bp.label}那局闭合轴方向量到的宽度 {wide:.1f}mm ≠ 35.5mm（夹错方向）")
            tray = sess.spawn.get("wooden_tray")
            if tray is not None:
                seen_tray.add((round(tray.xy[0], 4), round(tray.xy[1], 4)))
        check("t5 复位会重新随机摆位（瓶子与托盘位置都变）",
              len(seen) >= 5 and len(seen_tray) >= 5,
              f"瓶子 {len(seen)} 种 / 托盘 {len(seen_tray)} 种")
        rng = np.random.default_rng(7)
        yaws = {p.detail for p in (tasks_mod.sample_spawn(sess.task, rng, sess.catalog)["new_salad_dressing"]
                                   for _ in range(20))}
        check("t5 朝向整圆周随机（20 局抽到多个不同 yaw）", len(yaws) >= 8,
              f"{len(yaws)} 个：" + " ".join(sorted(yaws)[:6]) + " …")
        check("t5 摆位（位置+姿态+高度+闭合轴）都合规", not bad, "；".join(bad) if bad else "各局都合规")

    # 11d) t9 三件物体都随机：瓶子（锚在托盘里 + 整圆周朝向）、木托盘、盘子（见 README §8.9.4）
    def s11d():
        window.task_combo.setCurrentIndex(8)          # 切到 t9
        regions = window.session.task.get("spawn_region", {})
        bot, tray, plate = (regions.get("ketchup"), regions.get("wooden_tray"),
                            regions.get("plate"))
        check("t9 瓶子带随机安放区域", bot is not None,
              f"相对托盘的偏移 x={bot['x']} y={bot['y']}" if bot else "没有 spawn_region")
        check("t9 瓶子**锚在木托盘里**（anchor，跟着托盘走）",
              bot is not None and bot.get("anchor") == "wooden_tray",
              f"anchor={bot.get('anchor') if bot else None}")
        check("t9 瓶子朝向整圆周随机（upright_yaws=None）",
              bot is not None and "upright_yaws" in bot and bot["upright_yaws"] is None,
              f"upright_yaws={bot.get('upright_yaws') if bot else None}")
        check("t9 木托盘与盘子都带随机安放区域",
              tray is not None and plate is not None,
              f"托盘 x={tray['x']} y={tray['y']}；盘子 x={plate['x']} y={plate['y']}"
              if tray and plate else "托盘或盘子没有 spawn_region")
        check("t9 判分取盘子当前位置（xy_ref=target）",
              window.session.task["success"].get("xy_ref") == "target")
        if bot is None or tray is None or plate is None:
            return
        # 三件物体的占地：托盘 297×166mm（外接圆 170.5）、盘 137.6mm（97.3）——随机摆位不能重叠
        seen, seen_tray, seen_plate, bad = set(), set(), set(), []
        for _ in range(6):
            window.callback_reset()
            sess = window.session
            bp, tp, pp = (sess.spawn["ketchup"], sess.spawn["wooden_tray"],
                          sess.spawn["plate"])
            seen.add((round(bp.xy[0], 4), round(bp.xy[1], 4)))
            seen_tray.add((round(tp.xy[0], 4), round(tp.xy[1], 4)))
            seen_plate.add((round(pp.xy[0], 4), round(pp.xy[1], 4)))
            # ① 瓶子必须落在"托盘位置 + 允许的偏移"里（锚定：x/y 是相对量，不是绝对坐标）
            dx, dy = bp.xy[0] - tp.xy[0], bp.xy[1] - tp.xy[1]
            if not (bot["x"][0] - 1e-9 <= dx <= bot["x"][1] + 1e-9
                    and bot["y"][0] - 1e-9 <= dy <= bot["y"][1] + 1e-9):
                bad.append(f"瓶子相对托盘的偏移 ({dx * 1000:+.0f},{dy * 1000:+.0f})mm 超出区域")
            # ② 托盘也在自己的区域里
            if not (tray["x"][0] <= tp.xy[0] <= tray["x"][1]
                    and tray["y"][0] <= tp.xy[1] <= tray["y"][1]):
                bad.append(f"托盘 ({tp.xy[0]:.3f},{tp.xy[1]:.3f}) 抽到了区域外")
            # ③ 托盘 ↔ 盘子 不能重叠（外接圆之和 + 采样间隙，和 sample_spawn 的判据一致）
            r_sum = (tasks_mod.footprint_radius("turbosquid_objects/wooden_tray", 0.0, sess.catalog)
                     + tasks_mod.footprint_radius("stable_scanned_objects/plate", 0.0, sess.catalog))
            gap = float(np.hypot(*(np.asarray(tp.xy) - np.asarray(pp.xy))))
            if gap <= r_sum + tasks_mod.SPAWN_CLEARANCE - 1e-9:
                bad.append(f"托盘与盘子只隔 {gap * 1000:.0f}mm"
                           f"（≤ 外接圆 {r_sum * 1000:.0f} + 间隙 {tasks_mod.SPAWN_CLEARANCE * 1000:.0f}mm）")
            # ④ 瓶子要**坐在托盘内底**上（内底 = 桌面 +8mm），不能悬空也不能穿地板
            low = tasks_mod.object_lowest_z(sess.model, sess.data, sess.catalog, "ketchup")
            if not (tasks_mod.TABLE_TOP_Z + 0.004 <= low <= tasks_mod.TABLE_TOP_Z + 0.012):
                bad.append(f"瓶子最低点 {low * 1000:.1f}mm 不在托盘内底（桌面 +8mm）上")
            # ⑤ 托盘、盘子也要贴桌面
            for name, key in (("wooden_tray", "托盘"), ("plate", "盘")):
                lo, hi = skills_mod.aabb(sess, name)
                if abs(lo[2] - tasks_mod.TABLE_TOP_Z) > 0.004:
                    bad.append(f"{key}子底面 {lo[2] * 1000:.1f}mm 没贴桌面")
        check("t9 复位会重新随机摆位（三件物体位置都变）",
              len(seen) >= 5 and len(seen_tray) >= 5 and len(seen_plate) >= 5,
              f"瓶子 {len(seen)} 种 / 托盘 {len(seen_tray)} 种 / 盘子 {len(seen_plate)} 种")
        rng = np.random.default_rng(7)
        yaws = {p.detail for p in (tasks_mod.sample_spawn(sess.task, rng, sess.catalog)["ketchup"]
                                   for _ in range(20))}
        check("t9 朝向整圆周随机（20 局抽到多个不同 yaw）", len(yaws) >= 8,
              f"{len(yaws)} 个：" + " ".join(sorted(yaws)[:6]) + " …")
        check("t9 摆位（锚定+区域+不重叠+贴桌面）都合规", not bad,
              "；".join(bad) if bad else "各局都合规")

    # 11e) 连续任务：单任务连跑 N 局 / 多任务轮转 R 轮（每局重制场景，见 README §4、§8.12）
    def s11e():
        # 构造时的默认（单任务、勾选表不可点）在 s1 里查过；这里显式回到单任务再验行为，
        # 免得受前面步骤影响（曾经偶发地在"默认"检查上翻车）
        window.cont_mode_combo.setCurrentIndex(0)
        check("界面有连续任务控件",
              all(hasattr(window, n) for n in
                  ("cont_mode_combo", "cont_count_spin", "cont_task_list", "cont_button",
                   "cont_status")),
              "模式 / 次数 / 勾选表 / 开始按钮 / 一行状态")
        check("连续任务没有日志栏（不打印逐局运行日志）",
              not any(hasattr(window, n) for n in
                      ("cont_detail", "cont_plan_label", "cont_log")),
              f"cont_status={window.cont_status.text()!r}")
        check("连续任务勾选表覆盖全部 9 关",
              window.cont_task_list.count() == len(tasks_mod.TASKS),
              f"{window.cont_task_list.count()} 项；默认模式 = "
              f"{window.cont_mode_combo.currentText()}")
        check("单任务模式：勾选表不可点（多任务模式才可点）",
              window.cont_mode_combo.currentIndex() == 0
              and not window.cont_task_list.isEnabled(),
              f"模式={window.cont_mode_combo.currentText()}、"
              f"勾选表 enabled={window.cont_task_list.isEnabled()}")
        orig_dwell = app_mod.CONT_DWELL_FRAMES
        app_mod.CONT_DWELL_FRAMES = 0        # 测试里不要"结束停留"，省帧（真跑那局会还原）

        def fake_gen(ok):
            """下一帧就出判分结果的假技能流水线（不跑物理，只验证调度/记账）。"""
            yield "假执行"
            return [skills_mod.Step(skills_mod.JUDGE_LABEL, ok, "假判分")]

        def wait_run(limit=40):
            for _ in range(limit):
                if window.auto.active and not window.auto.finished:
                    return True
                window.tick()
            return False

        def drive(ok=True):
            wait_run()
            window.auto.gen = fake_gen(ok)
            done = window.cont_done
            for _ in range(40):
                window.tick()
                if window.cont_done != done:
                    break

        # ① 单任务连跑 3 局（t1：抓取物 + 篮子都随机）
        window.cont_mode_combo.setCurrentIndex(0)
        window.cont_count_spin.setValue(3)
        window.task_combo.setCurrentIndex(0)
        check("单任务计划 = 同一关 N 局", window._cont_build_plan() == [(0, 0)] * 3,
              f"plan={window._cont_build_plan()}")
        window.callback_cont_toggle()
        spawns, baskets = [], []
        for _ in range(3):
            wait_run()
            kp, kb = window.session.spawn["ketchup"], window.session.spawn["basket"]
            spawns.append((round(kp.xy[0], 5), round(kp.xy[1], 5)))
            baskets.append((round(kb.xy[0], 5), round(kb.xy[1], 5)))
            drive(True)
        check("单任务连跑 3 局逐局记账（3/3 成功）",
              window.cont_ok == 3 and window.cont_done == 3,
              f"ok={window.cont_ok} done={window.cont_done}")
        check("每局都重制场景：抓取物摆位 3 局全不同（随机化每局生效）",
              len(set(spawns)) == 3,
              " ".join(f"({x * 1000:+.0f},{y * 1000:+.0f})" for x, y in spawns))
        check("目标物（篮子）也每局重新随机", len(set(baskets)) == 3,
              " ".join(f"({x * 1000:+.0f},{y * 1000:+.0f})" for x, y in baskets))
        check("跑完自动收尾（按钮复位 + 一行汇总）",
              not window.cont_active and "跑完" in window.cont_status.text()
              and "\n" not in window.cont_status.text(),
              window.cont_status.text())

        # ② 多任务轮转 2 轮 × 勾选 t1/t3（一轮跑完再下一轮）
        window.cont_mode_combo.setCurrentIndex(1)
        window.callback_cont_clear_all()
        for idx in (0, 2):
            window.cont_task_list.item(idx).setCheckState(Qt.Checked)
        window.cont_count_spin.setValue(2)
        check("多任务计划按轮展开（轮内按列表顺序）",
              window._cont_build_plan() == [(0, 0), (0, 2), (1, 0), (1, 2)],
              f"plan={window._cont_build_plan()}")
        window.callback_cont_toggle()
        order = []
        for run in range(4):
            wait_run()
            order.append(window.session.task["id"].split("_")[0])
            drive(run != 1)                  # 故意让第 2 局判分不过，验证记账
        check("多任务依次跑完一轮再进下一轮", order == ["t1", "t3", "t1", "t3"],
              " → ".join(order))
        check("成绩按局记账（3 成功 / 1 失败）",
              window.cont_ok == 3 and window.cont_done == 4,
              f"ok={window.cont_ok} done={window.cont_done}；"
              f"{window.cont_status.text().replace(chr(10), ' ')}")
        check("分任务统计", window.cont_counts.get("t1_ketchup_into_basket") == [2, 2]
              and window.cont_counts.get("t3_pudding_into_ramekin") == [1, 2],
              f"{window.cont_counts}")
        check("汇总一行带分任务、且没有逐局日志",
              "分任务" in window.cont_status.text()
              and "\n" not in window.cont_status.text(),
              window.cont_status.text())

        # ③ 打断路径：手动接管 → 暂停；复位 / 切关 → 结束
        window.cont_mode_combo.setCurrentIndex(0)
        window.cont_count_spin.setValue(2)
        window.task_combo.setCurrentIndex(0)
        window.callback_cont_toggle()
        window.nudge_gripper(-0.3)
        check("手动操作 → 连跑暂停（按钮变\"继续连跑\"）",
              window.cont_active and window.auto.paused
              and "继续" in window.cont_button.text(), window.cont_button.text())
        window.callback_cont_toggle()
        check("再点按钮 → 继续连跑",
              window.cont_active and not window.auto.paused, window.cont_button.text())
        window.callback_reset()
        check("手动复位 → 连跑结束",
              not window.cont_active and "已停止" in window.cont_status.text(),
              window.cont_status.text())

        # ④ 真跑一局 t8（真实技能库 + 真的"结束停留"），验证与技能库的集成
        app_mod.CONT_DWELL_FRAMES = orig_dwell
        window.cont_count_spin.setValue(1)
        window.task_combo.setCurrentIndex(7)          # t8：帧数最少的一关
        window.callback_cont_toggle()
        frames = 0
        while window.cont_active and frames < 2000:
            window.tick()
            frames += 1
        check("真跑一局（t8）走完并记账", window.cont_done == 1 and window.cont_ok == 1,
              window.cont_status.text())

    # 11f) 夹爪上的"小球"全部隐藏：① ik_marker（红）② tool0 绿色球 geom ③ tool0_site（灰）
    #      真机夹爪中间没有这些球，都是仿真里的可视化标记（物理上都不参与）
    def s11f():
        import mujoco

        sess = window.session
        model, data = sess.model, sess.data
        left = [f"geom:{model.geom(i).name or i}" for i in range(model.ngeom)
                if model.geom(i).type == mujoco.mjtGeom.mjGEOM_SPHERE
                and model.geom(i).rgba[3] > 0]
        left += [f"site:{model.site(i).name or i}" for i in range(model.nsite)
                 if model.site(i).rgba[3] > 0]
        check("模型里所有球/site 都已全透明（夹爪上三个球全隐藏）", not left,
              f"geom {model.ngeom} 个 / site {model.nsite} 个；还看得见的："
              + ("、".join(left) if left else "无"))
        # 渲染级验证：把 ik_marker 挪到地下 5m，画面应当看不出任何差别
        # （用 Δ>8 计数：同状态渲两次也有 1~2 个 Δ≤1 的噪声像素，见 _tools\diag_marker.py）
        before = sess.render()
        data.mocap_pos[0] = [0.10, -0.10, -5.0]
        mujoco.mj_forward(model, data)
        after = sess.render()
        delta = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2)
        check("标记球挪到地下画面也看不出差别（真的看不见）", int((delta > 8).sum()) == 0,
              f"Δ>8 的像素 {int((delta > 8).sum())} 个（噪声级 {int((delta > 0).sum())} 个）")

    # 12) 让主循环再跑一会，统计 FPS 并报告
    def s9():
        check("主循环无异常", not ERRORS, f"{len(ERRORS)} 个异常")

    for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s11b, s11c, s11d, s11e, s11f):
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
    if not failed and not ERRORS:
        shutil.rmtree(REC_ROOT, ignore_errors=True)      # 冒烟写的数据集只是验证用
    else:
        print(f"（保留冒烟数据集供排查：{REC_ROOT}）")
    return 1 if (failed or ERRORS) else 0


import numpy as np  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
