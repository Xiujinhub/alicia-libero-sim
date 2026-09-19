"""示教臂（手摇跟随）链路单测：**不接硬件、不开窗口**，只测映射与适配层。

覆盖：
* 夹爪映射方向（SDK 1000=张开 → 界面 1=张开，并与 ``alicia_ik.finger_targets`` 对得上）；
* 相对/绝对同步、使能瞬间不跳变、每帧 8° 限速、死人开关（左键）、未使能不下发；
* 方向系数 / 偏置、参数解析的中文报错；
* ``LeaderLink`` 模拟源（连/断/快照/按键/push_virtual）与真机源的失败提示。

运行：python tests/test_leader_arm.py
      python tests/test_leader_arm.py --skip-hardware   # 不跑"连不上真机"那一项（更快）
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import leader_arm  # noqa: E402
from alicia_ik import finger_targets  # noqa: E402
from libero_catalog import load_catalog  # noqa: E402
from libero_tasks import TASKS, build_task_scene  # noqa: E402

failures = []


def check(name, condition, detail=""):
    if not condition:
        failures.append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}  {detail}")


# ────────────── 1) 夹爪映射方向（最容易被抄错的一处）──────────────
print("[1] 夹爪：SDK 值 → 界面张开度 → MuJoCo 滑轨")
check("SDK 1000（张开）→ 1.0", leader_arm.gripper_sdk_to_unit(1000) == 1.0)
check("SDK 0（闭合）→ 0.0", leader_arm.gripper_sdk_to_unit(0) == 0.0)
check("SDK 500 → 0.5", abs(leader_arm.gripper_sdk_to_unit(500) - 0.5) < 1e-9)
check("读数缺失时兜底为张开", leader_arm.gripper_sdk_to_unit(None) == 1.0)

model = None
try:
    xml, _ = build_task_scene(TASKS[0], catalog=load_catalog())
    import mujoco  # noqa: PLC0415

    model = mujoco.MjModel.from_xml_path(str(xml))
except Exception as exc:  # noqa: BLE001
    print(f"  ⚠ 场景加载失败（跳过与 finger_targets 的对照）：{exc}")
if model is not None:
    travel = float(model.jnt_range[model.joint("left_finger").id][1])
    open_ctrl = finger_targets(model, leader_arm.gripper_sdk_to_unit(1000))[0]
    close_ctrl = finger_targets(model, leader_arm.gripper_sdk_to_unit(0))[0]
    check("SDK 1000（张开）→ 滑轨 ctrl=0（两指分开）", abs(open_ctrl) < 1e-9,
          f"ctrl={open_ctrl:.4f}")
    check("SDK 0（闭合）→ 滑轨 ctrl=+行程（两指并拢）", abs(close_ctrl - travel) < 1e-9,
          f"ctrl={close_ctrl:.4f}（行程 {travel * 1000:.1f}mm）")

# ────────────── 2) 参数解析 ──────────────
print("[2] 参数解析（方向 / 偏置）")
check("逗号分隔解析", np.allclose(leader_arm.parse_six("1,-1,1,1,1,1", "方向"), [1, -1, 1, 1, 1, 1]))
check("中文逗号也认", np.allclose(leader_arm.parse_six("1，1，1，1，1，1", "方向"), [1] * 6))
check("数字间的空格被忽略", np.allclose(leader_arm.parse_six(" 1, 1, 1, 1, 1, 1 ", "方向"), [1] * 6))
for bad in ("1,1,1", "a,b,c,d,e,f"):
    try:
        leader_arm.parse_six(bad, "方向")
        check(f"非法输入要报中文错：{bad}", False, "没抛异常")
    except leader_arm.LeaderError as exc:
        check(f"非法输入要报中文错：{bad}", "方向" in str(exc), str(exc)[:34])

# ────────────── 3) 相对模式：使能不跳变 + 增量镜像 ──────────────
print("[3] 相对模式（默认，手摇推荐）")
mapper = leader_arm.LeaderFollowMapper()
follower = np.array([0.10, -0.05, 0.20, 0.00, 0.30, -0.10])
snap = {"joint_angles": [0.60] * 6, "gripper_value": 1000.0, "button1": False}
cmd = mapper.step(snap, follower, manual_enable=True)
check("使能那一帧不跳变（目标 = 机械臂当前目标）",
      np.allclose(cmd.target, follower), f"{np.round(cmd.target, 3)}")
check("使能帧被标成 just_enabled（界面用它暂停自动执行）", cmd.just_enabled)
snap["joint_angles"] = [0.80] * 6                       # 示教臂再转 0.2 rad
cmd = mapper.step(snap, follower, manual_enable=True)
check("增量被镜像，且每帧限速 8°",
      np.allclose(cmd.target - follower, np.radians(8.0), atol=1e-9),
      f"本帧 +{np.degrees(cmd.target[0] - follower[0]):.2f}°")
for _ in range(5):
    cmd = mapper.step(snap, follower, manual_enable=True)
check("连续几帧后到位（+0.2 rad）",
      np.allclose(cmd.target - follower, 0.2, atol=1e-9), f"{np.round(cmd.target, 3)}")

# ────────────── 4) 绝对模式：完全镜像 + 限速爬升 ──────────────
print("[4] 绝对模式")
absolute = leader_arm.LeaderFollowMapper(mode="absolute")
snap = {"joint_angles": [0.50] * 6, "gripper_value": 1000.0}
cmd = absolute.step(snap, np.zeros(6), manual_enable=True)
check("首次使能从当前姿态限速爬升（不会瞬移）",
      np.allclose(cmd.target, np.radians(8.0)), f"{np.round(cmd.target, 4)}")
for _ in range(10):
    cmd = absolute.step(snap, np.zeros(6), manual_enable=True)
check("最终完全镜像示教臂姿态", np.allclose(cmd.target, 0.5), f"{np.round(cmd.target, 3)}")

# ────────────── 5) 死人开关 / 未使能 ──────────────
print("[5] 死人开关与未使能")
gate = leader_arm.LeaderFollowMapper()                 # 新状态机（对应界面里刚连上时）
snap = {"joint_angles": [0.30] * 6, "gripper_value": 1000.0, "button1": True}
cmd = gate.step(snap, follower, manual_enable=False)
check("真机左键按下 = 使能（不用勾选框）", cmd.enabled and cmd.just_enabled)
check("使能那一帧先对齐（示教臂姿态与机械臂差多远都不跳）",
      np.allclose(cmd.target, follower), f"{np.round(cmd.target, 3)}")
snap["button1"] = False
cmd = gate.step(snap, follower, manual_enable=False)
check("左键松开 = 未使能（界面不下发，机械臂保持）",
      not cmd.enabled and cmd.frozen and np.allclose(cmd.target, follower))
snap["joint_angles"] = [1.00] * 6                      # 未使能时把示教臂挪很远
cmd = gate.step(snap, follower, manual_enable=False)
check("未使能时目标冻结在上一帧（示教臂乱动也不跟）",
      np.allclose(cmd.target, follower), f"{np.round(cmd.target, 3)}")
snap["button1"] = True                                 # 再按左键：重新对齐（不跳回过远处）
cmd = gate.step(snap, follower, manual_enable=False)
check("再次按下左键会重新对齐（不会跳变）",
      cmd.just_enabled and np.allclose(cmd.target, follower))

# ────────────── 6) 方向系数 / 偏置 / 夹爪 ──────────────
print("[6] 方向、偏置与夹爪")
signed = leader_arm.LeaderFollowMapper(mode="absolute", max_step_deg=1000.0)
signed.set_calibration("-1,1,1,1,1,1", "0,10,0,0,0,0")
snap = {"joint_angles": [0.20] * 6, "gripper_value": 0.0}
cmd = signed.step(snap, np.zeros(6), manual_enable=True)
check("方向系数 -1 让该轴反向", abs(cmd.target[0] + 0.20) < 1e-6, f"{np.round(cmd.target, 3)}")
check("偏置 10° 叠加上去", abs(cmd.target[1] - (0.20 + np.radians(10))) < 1e-6,
      f"{np.round(cmd.target, 3)}")
check("夹爪 SDK 0 → 界面 0（闭合）", cmd.gripper == 0.0)
snap["gripper_value"] = 850.0                          # 扳机也按下（< 900）
cmd = signed.step(snap, np.zeros(6), manual_enable=True)
check("扳机阈值（<900）与 SDK 判据一致",
      snap["gripper_value"] < leader_arm.TRIGGER_THRESHOLD)
check("夹爪值 850 → 界面 0.85（连续跟随）", abs(cmd.gripper - 0.85) < 1e-9)
probe = {"joint_angles": [0.05] * 6, "gripper_value": 0.0}
relative = leader_arm.LeaderFollowMapper()
check("相对模式对齐 = 记偏置（机械臂原地不动）",
      np.allclose(relative.align(probe, np.zeros(6)), [0.05] * 6))
far = {"joint_angles": [0.50] * 6, "gripper_value": 0.0}     # 离当前位置 28.6°（超过每帧 8°）
absolute = leader_arm.LeaderFollowMapper(mode="absolute")
check("绝对模式对齐：不留偏置（目标本来就一直追示教臂姿态，每帧限速 8°）",
      np.allclose(absolute.align(far, np.zeros(6)), 0.0)
      and np.allclose(absolute.step(far, np.zeros(6), manual_enable=True).target,
                      np.radians(8.0)))

# ────────────── 7) 模拟源链路 ──────────────
print("[7] LeaderLink 模拟源（无硬件）")
link = leader_arm.LeaderLink(source="virtual")
link.connect()
check("模拟源连上（connected + 描述）", link.connected and "模拟" in link.describe(),
      link.describe())
snapshot = link.snapshot()
check("快照字段齐备",
      set(leader_arm.EMPTY_SNAPSHOT) <= set(snapshot) and len(snapshot["joint_angles"]) == 6,
      "/".join(sorted(snapshot))[:58])
check("模拟源 age()=0、不会误报掉线", link.age() == 0.0 and not link.stale())
check("按下 3 被虚拟示教臂消费（选关节）", link.virtual_key("3"))
check("按下 z 不被消费（交回界面）", not link.virtual_key("z"))
link.push_virtual(angles=[0.4] * 6, gripper=1000.0, enabled=True)
snapshot = link.snapshot()
check("push_virtual 写进快照",
      abs(snapshot["joint_angles"][0] - 0.4) < 1e-9 and snapshot["button1"],
      leader_arm.describe_snapshot(snapshot))
link.close()
check("close 幂等且回到空快照",
      not link.connected and link.age() == 0.0
      and np.allclose(link.snapshot()["joint_angles"], 0.0))
link.close()
try:
    link.push_virtual(angles=[0.1] * 6)
    check("断开后不能程序化驱动（要报中文错）", False, "没抛异常")
except leader_arm.LeaderError as exc:
    check("断开后不能程序化驱动（要报中文错）", "模拟示教臂" in str(exc), str(exc)[:30])

# ────────────── 8) 真机源连不上时的提示（不碰硬件：用一个不存在的串口）──────────────
if "--skip-hardware" in sys.argv:
    print("[8] 跳过真机连接失败测试（--skip-hardware）")
else:
    print("[8] 真机源：串口不存在时给中文排查提示（不碰任何真机）")
    real = leader_arm.LeaderLink(source="leader", port="COM91", disable_torque=False)
    try:
        real.connect()
        check("不存在的串口要报 LeaderError", False, "居然连上了？")
        real.close()
    except leader_arm.LeaderError as exc:
        check("不存在的串口要报 LeaderError", "示教臂连接失败" in str(exc), str(exc)[:36])
    check("连不上后的链路是干净的", not real.connected and real.robot is None)

print("-" * 62)
if failures:
    print(f"失败 {len(failures)} 项：{failures}")
else:
    print("全部通过 ✅")
sys.exit(1 if failures else 0)
