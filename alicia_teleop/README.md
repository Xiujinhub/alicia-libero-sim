# Alicia-D 虚拟遥操作实操教程（示教臂 → MuJoCo）

> 对应官方文档：[虚拟遥操作 MuJoCo](https://docs.sparklingrobo.com/docs/alicia-d-series/leader-force-control/doc_05_mujoco_leader)
> 官方参考实现：[Synria-Robotics/Teleoperation-SDK](https://github.com/Synria-Robotics/Teleoperation-SDK) 里的 `02_demo_mujoco_follower.py`

「虚拟遥操作」= **真机示教臂（Leader）** 通过 USB 串口把关节角/扳机送进电脑，
**MuJoCo 里的虚拟操作臂（Follower）** 实时复现它的姿态。本文给出一套**在本机（Windows + conda `lerobot` 环境）
已经跑通**的实现：`alicia_virtual_teleop.py`，以及把官方模型变成「可遥操作模型」的转换脚本。

---

## 1. 环境与前置（本机已确认）

| 项 | 本机情况 |
| --- | --- |
| 操作系统 | Windows 10/11 |
| Python 环境 | conda 环境 **`lerobot`**（`D:\Anaconda\envs\lerobot`，Python 3.10） |
| MuJoCo | **3.13.0**（`lerobot` 环境里的 `mujoco` 包）+ `bin\simulate.exe`（MuJoCo 3.12 GUI） |
| 机器人 SDK | `alicia_d_sdk 6.1.0rc4`、`synria-robocore 2.0.0a1`、`pyserial 3.5` |
| 模型 | `Synria-Robot-Descriptions-main/synriard/`（MJCF + mesh） |
| 可选 | `matplotlib`（官方 Demo 01 画曲线用）、`opencv`（腕部相机预览用） |

装依赖（如果换机器）：

```powershell
conda activate lerobot
pip install mujoco alicia-d-sdk pyserial
```

---

## 2. 三步跑起来

### 第 0 步（只需一次）：生成「可遥操作」的模型

官方遥操作脚本要求模型里有一套**位置伺服 actuator**（`pos1..pos6` / `pos_grip_l` / `pos_grip_r`），
而 `Synria-Robot-Descriptions` 里的模型**不满足**这个条件：

| 原始模型 | 问题 |
| --- | --- |
| `Alicia_D_v5_6_gripper_50mm.xml` | **完全没有** `<actuator>`（`nu = 0`），无法下发任何指令 |
| `Alicia_D_v5_6_gripper_100mm.xml` | 只有 `actuator1..actuator8`，其中 6 个臂关节是**纯力矩**（ctrl = 力矩），不是目标角 |

所以先跑一次转换脚本，它在原 MJCF 旁边生成 `*_teleop.xml`：

```powershell
conda activate lerobot
cd E:\deepenv\mujoco\alicia_teleop
python make_follower_xml.py
```

输出（已实测）：

```
[OK] Alicia_D_v5_6_gripper_50mm.xml → Alicia_D_v5_6_gripper_50mm_teleop.xml  (8 actuators)
     自检: nq=8 nu=8 [('pos1','pos'),...('pos_grip_l','pos'),('pos_grip_r','pos')]
[OK] Alicia_D_v5_6_gripper_100mm.xml → Alicia_D_v5_6_gripper_100mm_teleop.xml  (8 actuators)
```

转换脚本做了 3 件事（都在生成的 XML 里看得到）：

1. `<actuator>` 换成位置伺服，kp/kv 取官方遥操作模型的值：J1–J3 `kp=800 kv=40`、J4 `400/20`、
   J5–J6 `300/15`、夹爪 `200/10`
2. 补 `<option timestep="0.002" integrator="implicit">` 和 `<default><joint damping="1.0" armature="0.02"/>`
   —— **没有这两项，高 kp 位置伺服在仿真里会抖振/发散**（原模型没有任何关节阻尼与 armature）
3. 不改 `meshdir`，所以生成文件与原文件同目录即可直接加载

其他机型也能转：

```powershell
python make_follower_xml.py --list                       # 看仓库里所有可转换的 MJCF
python make_follower_xml.py --src "..\Synria-Robot-Descriptions-main\synriard\mjcf\Alicia_M_v1_2\Alicia_M_v1_2_follower.xml"
```

### 第 1 步：先在没有硬件的情况下自测（强烈建议）

```powershell
python alicia_virtual_teleop.py --source virtual
```

会弹出 MuJoCo 窗口，用**键盘当"虚拟示教臂"**：

1. 按 `t` 使能遥操作（窗口左下角提示变成 `TELEOP ACTIVE`）
2. 按 `1` 选中 Joint1，按 `.` 几次 → 虚拟机械臂第 1 轴应该跟着转（每按一次 2°）
3. 按 `2` 再按 `.` → 第 2 轴动；按 `o`/`c` → 夹爪开/合
4. 按 `a` 可以把当前示教臂姿态对齐到虚拟机械臂；按 `0` 复位虚拟示教臂

这一步能验证：模型能加载、控制通道正确、夹爪映射正确、窗口/热键正常。
官方文档建议先跑 Demo 01 再看仿真，本脚本用 `--source virtual` 起到同样的作用，而且不需要硬件。

### 第 2 步：接真机示教臂做虚拟遥操作

先确认串口（**本机现在有两个 CH343 串口，务必分清哪个是示教臂**）：

```powershell
python alicia_virtual_teleop.py --list-ports
```

```
检测到 2 个串口：
  COM6     USB-Enhanced-SERIAL CH343 (COM6)
  COM5     USB-Enhanced-SERIAL CH343 (COM5)
```

再做一次**只读**自检（只读关节角，不发任何运动指令）：

```powershell
python alicia_leader_probe.py --port COM5
```

```
     t       status   grip  deadman  lock   fps  关节(deg)
   0.2         idle   1000 released released  49.8   +0.12   -0.05  ...
```

确认 `fps` 在 40–50、角度随手动示教臂变化 → 串口链路 OK，然后启动遥操作：

```powershell
python alicia_virtual_teleop.py --source leader --port COM5
```

- 窗口打开后**按住示教臂左键**（死人开关）→ `TELEOP ACTIVE`，虚拟机械臂开始跟随
- 松开左键 → 虚拟机械臂**冻结**在当前姿态（安全设计，官方同款）
- 扣扳机 → 夹爪开合；按住右键 → 锁定示教臂
- 想用手拖着示教臂走：加 `--disable-torque`（官方同款参数）

> 快捷方式：也可以直接双击 `run_teleop.bat`，或用
> `run_teleop.bat virtual`、`run_teleop.bat list`、`run_teleop.bat probe`、`run_teleop.bat --port COM5`。

### 本机实测结果（供对照）

在 `--port COM6 --duration 8 --record logs/example_leader_run.csv` 下实测：

| 项 | 结果 |
| --- | --- |
| 串口 | **COM6 是示教臂**（能应答协议）；COM5 能打开但不应答（同一颗 CH343，但不是本机的示教臂口） |
| 采集帧率 | 示教臂读取 ~62 Hz，整个遥操作循环 ~45 fps |
| 状态字 | `run_status_text = sync_locked`，`_run_status = 0x11` → 判定左键(死人开关)+右键(锁定)按下 |
| 跟随精度（稳态） | 6 轴最大误差 **0.059°**（位置伺服模型） |
| 夹爪 | SDK `1000` → MuJoCo 目标 `0 mm` → 实际间隙 `0.15 mm`（张开） |
| CSV | 8 秒 358 行，样例见 `logs/example_leader_run.csv` |


---

## 3. 窗口里的按键（全部实测可用）

MuJoCo 自己占用了 `空格` `[` `]` `-` `=`，所以脚本刻意避开了这些键。

| 按键 | 作用 |
| --- | --- |
| `t` | 使能 / 断开遥操作（= 官方的"按住左键"，键盘没有按住状态所以做成开关） |
| `a` | 对齐/清零：absolute 模式=把 follower 目标对齐到示教臂当前姿态；relative 模式=偏置清零 |
| `p` | 在终端打印一次状态（示教臂姿态、使能状态、follower 关节角） |
| `r` | 重新加载 XML（改完模型不用重启，官方同款） |
| `v` | 开关腕部相机预览（需要模型里有 `<camera>` + opencv） |
| `q` | 退出（直接关窗口也一样） |

`--source virtual` 时额外：

| 按键 | 作用 |
| --- | --- |
| `1`..`6` | 选择要操作的关节（终端会提示 `选中 JointN`） |
| `,` / `.` | 选中关节 −2° / +2° |
| `o` / `c` | 夹爪 张开 / 闭合（每次 50，范围 0–1000） |
| `0` | 虚拟示教臂复位到零位 |

窗口左上角 HUD（`mujoco>=3.13` 用 `Handle.set_texts()`，3.12 及以前用 `mjr_overlay`，脚本两套都兼容）：

```
SOURCE            leader (COM5)
Leader FPS        49.8
Deadman (left btn) PRESSED
Right btn (lock)   released
Trigger            OFF
Gripper SDK        1000 / 1000
Gripper MuJoCo     50.0 mm
Run status         sync
TELEOP             ACTIVE
J1                 +12.34 deg
...
```

---

## 4. 参数速查

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--xml` | `...\Alicia_D_v5_6_gripper_50mm_teleop.xml` | follower 模型 |
| `--source` | `leader` | `leader` 真机 / `virtual` 键盘 / `auto` 连不上自动退回 |
| `--port` | 空=自动搜索 | 示教臂串口，如 `COM5`（**有两个 CH343 时一定要显式指定**） |
| `--variant` | `leader` | 示教臂变体：`leader` / `leader_ur` / `gripper_50mm` / `gripper_100mm` / `vertical_50mm` |
| `--fps` | `50` | 示教臂轮询频率（官方同款；串口带宽有限不要盲目调高） |
| `--mode` | `absolute` | `absolute` 完全镜像姿态；`relative` 只镜像增量（首次使能不跳变，更安全） |
| `--signs` | `1,1,1,1,1,1` | 6 轴方向系数，**方向反了就改正负号** |
| `--offsets` | `0,0,0,0,0,0` | 6 轴角度偏置（度），零点不一致时粗校准 |
| `--disable-torque` | 关 | 启动即关示教臂力矩（用手拖拽示教臂时加） |
| `--record` | 空 | 写 CSV，如 `logs/teleop.csv` |
| `--camera` | 空 | 腕部相机名，如 `wrist_realsense` |
| `--kp` / `--kd` | `400` / `10` | **仅力矩型模型**用到的内部 PD 增益 |
| `--duration` | `0`（不限） | 跑 N 秒后自动退出，自动化测试用 |

---

## 5. 原理拆解（4 个环节）

### 5.1 输入：把示教臂状态读成一个线程安全快照

后台线程按 `--fps` 轮询 SDK，写进 `LeaderState`；主线程只管读快照，互不阻塞：

```python
raw = robot.get_robot_state("joint_gripper")      # JointState(angles, gripper, timestamp, run_status_text)
angles   = list(raw.angles)[:6]                   # 6 轴关节角，弧度
grip_val = raw.gripper                            # 0=夹爪闭合, 1000=张开
run_text = raw.run_status_text                    # "idle"/"sync"/"locked"/"sync_locked"
run_raw  = robot.data_parser._run_status          # 原始状态字，用来解按键位
```

按键解码沿用官方逻辑（`decode_handle_inputs()`）：

| 输入 | 判定 |
| --- | --- |
| 左键 = 死人开关 | `run_status_text in ("sync","sync_locked")` 或 状态字 `& 0x10` |
| 右键 = 锁定 | `run_status_text in ("locked","sync_locked")` 或 状态字 `& 0x01` |
| 扳机 | `gripper < 900` |

### 5.2 模型：把关节映射到 actuator，并按类型决定怎么下发

`FollowerBundle` 不用 actuator 名字硬编码，而是**按关节名找驱动它的 actuator**，再看
actuator 的 `biastype/biasprm` 判断类型：

```python
if bias == mjBIAS_AFFINE and biasprm[1] < 0:   # <position kp kv> 或 general biastype=affine
    mode = "pos"      # ctrl 是目标位置，直接写
else:
    mode = "frc"      # ctrl 是力矩，脚本内做 PD：
                      # tau = kp*(target - q) - kd*qvel，并夹到 ctrlrange
```

所以同一份脚本既能跑官方 `alicia_d_follower.xml`（位置伺服），也能跑
`Synria-Robot-Descriptions` 的原始力矩模型（自动 PD 兜底），不用改代码。

**踩过的坑（重要）**：原始力矩模型没有关节阻尼/armature，舵机反射惯量只有 `1e-4` 量级，
直接上 PD 会立刻发散（实测 `Nan, Inf or huge value in QACC`）。脚本检测到这种情况会自动补
`armature=0.02 / damping=1.0` 并把积分器切成 `implicit`（与官方模型一致），之后就稳了：

| 模型 / 控制方式 | 目标 30° 时的稳态误差（实测） |
| --- | --- |
| 位置伺服（`*_teleop.xml`） | **0.05°** |
| 力矩 + PD `kp=400 kd=10`（自动补 armature） | ~0.1° |
| 力矩 + PD `kp=30 kd=1.5`（不补，会发散→补后可跑但很软） | 1.34° |

### 5.3 映射：目标角 + 夹爪行程

```python
target = leader_angles * signs + offsets          # 方向系数 + 偏置
grip   = (1 - clamp(sdk/1000)) * travel            # SDK 1000=张开 → MuJoCo 0=张开
bundle.command(target, grip)                      # 6 轴 + 左右手指（符号自动从滑轨 range 推导）
```

夹爪行程 `travel` 直接从模型里读（`left_finger` 滑轨 range）：50mm 夹爪 → 25mm，100mm → 50mm，
所以换夹爪不用改代码。SDK `0`（闭合）对应 MuJoCo 内侧 `+travel`，`1000`（张开）对应 `0`。

### 5.4 时序：让仿真时间跟上真实时间

官方做法（本脚本沿用）：`n_substeps = round(1/(60·dt))`，每帧走 `n_substeps` 步，
并把渲染间隔对齐到 `n_substeps·dt`，最后用 `precise_sleep` 精确补时：

```python
n_substeps = max(1, round(1.0 / (VIEWER_FPS * model.opt.timestep)))   # dt=0.002 → 8 步
render_interval = n_substeps * model.opt.timestep                     # ≈ 16.7 ms
...
for _ in range(n_substeps):
    mujoco.mj_step(model, data)
viewer.sync()
如果这一帧还没用完 16.7 ms，就 precise_sleep 补齐
```

**注意**：如果机器渲染不过来（本机是笔记本集显 + 高面数 STL，实测约 25–30 fps），
真实耗时 > 16.7 ms 时仿真会变成"慢动作"——这是固定步长方案的固有取舍，
想要严格实时需要降面数/关阴影/换 `--xml` 用的简化模型。

---

## 6. 常见扩展

### 6.1 换机型 / 换夹爪

```powershell
python make_follower_xml.py --src "..\Synria-Robot-Descriptions-main\synriard\mjcf\Alicia_D_v5_6\Alicia_D_v5_6_gripper_100mm.xml"
python alicia_virtual_teleop.py --xml "..\Synria-Robot-Descriptions-main\synriard\mjcf\Alicia_D_v5_6\Alicia_D_v5_6_gripper_100mm_teleop.xml"
```

夹爪行程会自己按 100mm 变成 50 mm，不用改代码。

### 6.2 加一张桌子/一个方块（改完按 `r` 热重载）

在你正在用的 `*_teleop.xml` 的 `<worldbody>` 里加：

```xml
<body name="table" pos="0.4 0 0.2">
  <geom type="box" size="0.3 0.4 0.02" rgba="0.6 0.5 0.4 1"/>
</body>
<body name="cube" pos="0.4 0 0.25">
  <freejoint/>
  <geom type="box" size="0.02 0.02 0.02" rgba="0.9 0.2 0.2 1"/>
</body>
```

然后在窗口里按 `r`，模型立即重载（不用重启程序）。这就有了"虚拟抓取"的场景。

### 6.3 录数据看跟随效果

```powershell
python alicia_virtual_teleop.py --source leader --port COM5 --record logs/teleop.csv
```

CSV 列：`t, enabled, leader_j1..j6, leader_gripper_sdk, follower_j1..j6, follower_gripper_m`
（实测每帧一行，8 秒约 199 行）。画图：

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("logs/teleop.csv")
plt.plot(df.t, df.leader_j1, label="leader")
plt.plot(df.t, df.follower_j1, label="follower")
plt.legend(); plt.xlabel("t (s)"); plt.ylabel("Joint1 (rad)"); plt.show()
```

### 6.4 让它跟着控制真机（进阶）

把 `bundle.command()` 换成 SDK 的 `robot.set_robot_state(...)`（示教臂→操作臂真机遥操作），
模型读数换成 `goal = data.qpos[:6]` 即可 —— 这也正是 `Robot/Alicia-D-SDK-6.1.0/examples/`
里真机遥操作例程的思路。本脚本的 `LeaderState` 与按键解码可以原样复用。

---

## 7. 排错清单

| 现象 | 原因 / 解决 |
| --- | --- |
| `PermissionError(13, '拒绝访问')` 打开串口失败 | 端口被占用：关掉 **Synria Desk 上位机**、串口助手、其他正在跑的脚本（本机实测就是这个原因） |
| 连接成功但角度一直是 0 | `--port` 选错设备：本机有两个 CH343（`COM5`/`COM6`），必须显式指定示教臂那一个 |
| `找不到关节 ['Joint1', ...]` 或 `关节 'Joint1' 没有对应的 actuator` | 用的是原始 MJCF，先跑 `python make_follower_xml.py` 生成 `*_teleop.xml` |
| 按左键没反应 | 看 HUD 的 `Deadman (left btn)`；也可以直接按 `t` 手动使能 |
| 关节转的方向相反 | `--signs "1,-1,1,1,1,1"` 之类，把对应轴改成正负号 |
| 到位后一直抖 / 一使能就飞 | 用位置伺服模型（`*_teleop.xml`）；力矩模型把 `--kp/--kd` 调小（如 `--kp 200 --kd 6`） |
| 动作有延迟、像慢动作 | 渲染帧率低于 60（集显 + 高面数 STL）。关掉 MuJoCo 的阴影/反射，或换简化模型 |
| 终端中文乱码 | 用 Windows Terminal / PowerShell 直接运行；被管道重定向时按 UTF-8 保存 |
| 只想看模型、不想接硬件 | `--source virtual` |
| `ModuleNotFoundError: alicia_d_sdk` | `pip install alicia-d-sdk`（本机已在 `lerobot` 环境装好 6.1.0rc4） |
| 加了 `--debug` 反而连不上，报 `'SerialComm' object has no attribute '_print_hex_frame'` | **SDK 6.1.0rc4 的 debug 模式 bug**（本机实测）。脚本已自动降级为 `debug_mode=False` 重试；也可以直接不加 `--debug` |
| 某个轴到极限后不动 | 该轴超出 follower 关节范围（leader 与 follower 零点不一致），用 `--offsets` 粗校准该轴 |
| `--source leader` 直接退出并提示"无法连接示教臂" | 改用 `--source auto`（连不上自动退回 virtual），或先按第 2 步自检串口 |

---

## 8. 文件清单与和官方示例的差异

```
e:\deepenv\mujoco\alicia_teleop\
├── alicia_virtual_teleop.py     # 主脚本：真机/虚拟输入 → MuJoCo follower
├── make_follower_xml.py         # 把 Robot-Descriptions 的 MJCF 转成位置伺服版 *_teleop.xml
├── alicia_leader_probe.py       # 示教臂只读自检（串口 + 按键 + 实时角度）
├── run_teleop.bat               # Windows 启动器（双击 / list / virtual / probe）
├── README.md                    # 本文
└── logs/                        # --record 的 CSV 输出

..\Synria-Robot-Descriptions-main\synriard\mjcf\Alicia_D_v5_6\
├── Alicia_D_v5_6_gripper_50mm_teleop.xml     # make_follower_xml.py 生成
└── Alicia_D_v5_6_gripper_100mm_teleop.xml    # 同上
```

与官方 `02_demo_mujoco_follower.py` 的差异：

| 能力 | 官方示例 | 本脚本 |
| --- | --- | --- |
| 输入源 | 只能真机（串口） | 真机 **或 键盘虚拟示教臂**（`--source virtual`） |
| 模型 | 只认官方 `alicia_d_follower.xml`（`pos1..pos8`） | 按关节名自动匹配 actuator，位置伺服/力矩都能跑 |
| 夹爪行程 | 硬编码 `GRIPPER_MJ_MAX = 0.025` | 从模型滑轨 range 自动读（50mm/100mm 通用） |
| 画面叠加 | `mjr_overlay(... viewer.ctx)`（3.13 已移除） | `set_texts()` / `mjr_overlay` 双兼容 |
| 易用性 | 需要自己接硬件才能看到效果 | `--source virtual` 可无硬件自检；`--record` 出 CSV |
| 腕部相机 | 默认开（`wrist_realsense`） | 默认关，`--camera` 按需开 |
| 死区/安全 | 左键死人开关 | 同款左键 + `t` 键 + `--mode relative` 防首帧跳变 |

## 9. 参考

- 官方文档：<https://docs.sparklingrobo.com/docs/alicia-d-series/leader-force-control/doc_05_mujoco_leader>
- 官方 Teleoperation-SDK：<https://github.com/Synria-Robotics/Teleoperation-SDK>
- 模型仓库：`E:\deepenv\mujoco\Synria-Robot-Descriptions-main`（本机已下载）
- 官方 SDK 文档：`E:\deepenv\lerobot-6.1.0\Robot\Alicia-D-SDK-6.1.0\docs\`
- MuJoCo 模型查看：`E:\deepenv\mujoco\synria_sim.ps1`（见 `SYNRIA_MUJOCO_使用说明.md`）


