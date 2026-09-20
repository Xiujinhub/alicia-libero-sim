# RevA1-sim —— TB6-R5-RevA1 机械臂的 MuJoCo 仿真

把 `RevA1-sim/TB6-R5-RevA1/urdf/7260501-000000-001 TB6-R5-RevA1-URDF.urdf`（随机型包一起拷进来的那份；
找不到时会回退到仓库根 `RobotSDK/TB6-R5-RevA1/`）变成 **自包含的 MuJoCo 模型**，
并附上自检、可视化、演示三件套。

```bash
conda activate lerobot          # 本机已有 mujoco / numpy / imageio / PySide6 6.7.3
cd RevA1-sim

python convert_urdf_to_mjcf.py  # ① 生成 assets/revA1_arm.xml + revA1_scene.xml + meshes/
python check_model.py           # ② 自检：结构/质量/奇异位形/稳定性/工作空间/IK 精度
python revA1_gui.py             # ③ 交互控制台（PySide6）：关节 −/+ 微调 / 笛卡尔直线点到点 / IK（推荐）
python viewer.py                # ④ 开 MuJoCo 原生窗口玩（拖滑条 / 空格暂停 / 1-4 预设姿态）
python demo_trajectory.py       # ⑤ 自动跑一段点到点轨迹
python demo_trajectory.py --cartesian --video runs/cart.mp4   # ⑥ 笛卡尔画方+圆并录视频
```

## 目录里有什么

| 文件 | 作用 |
|---|---|
| `revA1_spec.py` | **机型参数表**（唯一事实来源）：URDF 路径、工具长度、kp/kv、home 姿态、地面高度 |
| `convert_urdf_to_mjcf.py` | URDF → MJCF 转换的命令行入口（在 `assets/` 下生成场景三件套） |
| `model_import.py` | 转换用的工具库：`package://` 路径修复、mesh 拷贝、MJCF 文本后处理、场景模板 |
| `check_model.py` | 模型自检脚本，结果打印并写入 `check_model.log` |
| `arm_core.py` | **控制核心**（不依赖界面框架）：`Camera` 轨道相机、`Motion` 运动插值、`ArmSim` 仿真/伺服/IK、多初值 IK、笛卡尔直线规划 |
| `revA1_gui.py` | **交互控制台（PySide6，主推）**：关节 −/+ 微调、末端目标/IK、示教点位（点到点）、伺服参数、日志；自带 `--selftest` / `--ui-test` |
| `interactive_control.py` | 交互控制台入口（转发到 `revA1_gui.py`，让老的命令行继续可用） |
| `interactive_control_tk.py` | 旧版 tkinter 界面（保留作对照/回退，逻辑同源，命令行不变） |
| `sim_ik.py` | 末端位姿数值 IK（阻尼最小二乘）+ `drive_to` 平滑运动 / `reset_home`，也可单独当命令行 IK 用 |
| `viewer.py` | 交互可视化（含 `--headless` 离屏出图，无显示器也能跑） |
| `demo_trajectory.py` | 路点演示 / 笛卡尔轨迹演示，可 `--view` 边跑边看或 `--video` 录 mp4 |
| `gl_backend.py` | 自动挑 `MUJOCO_GL` 后端（WSL/无头服务器免手配） |
| `TB6-R5-RevA1/` | 机型包（URDF + mesh + RDM），`revA1_spec.py` 默认就用它 |
| `assets/revA1_arm.xml` | **生成物**：机器人本体（mesh 已拷到 `assets/meshes/`，自包含） |
| `assets/revA1_scene.xml` | **生成物**：场景（本体 + 地面 + 灯光 + 3 个相机 + `home` keyframe） |
| `runs/demo.mp4` | 示例产物（`demo_trajectory.py --video` 录的，可删） |

## 环境

本机验证环境：Python 3.10.21 + mujoco 3.13.0 + numpy 2.2.6 + PySide6 6.7.3（conda 环境 `lerobot`）。

```bash
pip install -r requirements.txt          # mujoco / numpy / PySide6 / imageio / imageio-ffmpeg
```

* 想用别的环境：只要 `python -c "import mujoco"` 能用就行，脚本本身不依赖 ROS。
* 渲染后端：`gl_backend.py` 会自动选（有 `DISPLAY`/`WAYLAND_DISPLAY` 用 `glfw`，否则 `egl`）。
  手动指定：`MUJOCO_GL=egl python viewer.py --headless`。
* `revA1_gui.py` 只需要 **PySide6**（不依赖 tkinter / Pillow）；旧界面
  `interactive_control_tk.py` 才需要 tkinter + Pillow（Linux 上 tkinter 通常要单独装系统包，
  例如 `sudo apt install python3-tk`）。
* 没有显示器时用 `--headless`（离屏渲染成 PNG / mp4），不要开窗口；
  界面自检也能在无头机器上跑：`QT_QPA_PLATFORM=offscreen python revA1_gui.py --ui-test`。

## 交互控制台 `revA1_gui.py`（PySide6）

左边是 MuJoCo 实时画面（离屏渲染 → QImage → 等比缩放居中显示），右边是可滚动的控制面板，
底下一行状态栏。旧版是 tkinter，画面贴 Canvas、控件用 ttk 默认皮肤，**现已整体换成 PySide6**：
深色主题、卡片式布局、按钮/进度条/数值框都是自绘样式。老的入口
`python interactive_control.py` 保留为转发（跑的还是这套新界面），旧 tkinter 版本另存
`interactive_control_tk.py`。

```bash
python revA1_gui.py                            # 开窗口（= python interactive_control.py）
python revA1_gui.py --width 960 --height 600 --fps 30   # 渲染分辨率 / 帧率上限
python revA1_gui.py --selftest                 # 无窗口：控制逻辑 + 渲染通路全跑一遍并断言
python revA1_gui.py --ui-test                  # 真建窗口 → 脚本化点一遍控件 → 存图 → 退出
python revA1_gui.py --exit-after 10            # 开窗口跑 10 秒自动退出（自动化 / 截图）
```

**五张卡片**

| 卡片 | 做什么 | 怎么用 |
|---|---|---|
| 关节微调 | 每个关节一行：`−` `目标角（大号字）` `+` `(实测角)` | **点一下动一点**；步长可选 0.5°/1°/5°/15°；按住 0.4 s 后自动连点；还有「同步实测 / 全部归零 / 回 home」 |
| 末端目标 / IK | 工具尖目标 `x y z`（±5 mm / 1 cm / 5 cm 步进）+ 工具轴方向（保持当前朝向 / 竖直朝下 / 自定义 / 不约束） | 「求解 IK」只算不动 → 结果区显示**位置误差、姿态误差、耗时、解出的 6 个关节角**；「求解并沿直线运动」再让工具尖沿直线过去；预设按钮 home / 前伸 / 侧向 / 低位 |
| 点位（示教 / 点到点） | 列表 + 「记录当前位姿 / 走到选中点 / 删除 / 清空」 | 先用手动（关节 ± 或 IK）摆到位 → 记录 → 以后双击列表项就能**沿直线复现**（这就是最直观的点到点） |
| 运行 / 伺服 | 运动时长、进度条、kp 缩放、暂停物理、重力、急停、复位、存图、视角按钮 | 急停 = 就地保持（指令钉在实测角上）；视角 4 个预设（斜视/俯视/侧视/近看末端） |
| 日志 | 每个动作 + **实测**到位精度 | 终端里也会打印同样的 `[gui] ...` |

**这次改掉的三件事**

1. **单关节：滑条 → `−` / `+` 按钮**。旧版 6 根滑条既占地方又拖不准（尤其 0.5° 微调）；
   现在一行 `− 目标角 + (实测角)`，点一下就是一个步长，按住就连点。
2. **点到点：关节空间插值 → 笛卡尔直线**。旧版 6 个关节各自 smoothstep，工具尖走的是条弧线，
   肉眼看不出"从 A 到 B"；现在 `arm_core.ArmSim.plan_line()` 沿"当前工具尖 → 目标点"的直线
   每 1 cm 解一次 IK，再按 smoothstep 回放时间——工具尖就是**沿直线**过去的（等同真机 MoveL），
   画面里还会画出这条青色路径 + 起点处的工具尖小球。
3. **IK：单初值 → 多初值 + 明确的成败反馈**。旧版 IK 其实是能算的，但结果只写日志：
   解出来看不出差多少、解不出来（偏置腕在奇异/局部极小处很容易）界面也毫无提示，用起来就像"按了没反应"。
   现在换成多初值（当前姿态 → home → 两个肩肘大扰动，命中即停），并且：
   * 结果区直接写「IK 可解 / IK 不可用（位置差 xx mm（超出工作空间或顶到关节限位））」+ 误差 + 耗时 + 关节角；
   * 目标点在画面里是**绿球（可达）/ 红球（不可达）** + 黄色工具轴指示；
   * 到不了就**明确拒绝执行**（不会偷偷动、也不会卡在奇怪的姿态上）。

**鼠标/键盘**：左键拖动转视角、右键拖动平移、滚轮推拉、双击复位视角；
`空格` 暂停、`H` 回 home、`R` 复位、`G` 重力、`1..6` 选关节、`−`/`=` 给选中关节 ± 一个步长、
`Ctrl+S` 存图、`ESC` 退出。（输入框里打字时快捷键不抢键，负号、数字都能正常输入。）

**动作到点后会自己报告精度**（不是只看指令，是量实际状态）：

```
[gui] 直线规划：[0.4, -0.0, 0.179] → [0.52, 0.18, 0.34]，28 个路点（间隔 10 mm），用时 23 ms，直线度 0.000 mm
[gui] 到位（cartesian）：工具尖 [0.52, 0.18, 0.334] / 目标 [0.52, 0.18, 0.34]，位置误差 6.40 mm，姿态误差 0.79°，最大关节跟踪误差 0.377°
```

### 写这个窗口踩的坑（都已修，并留在自检里）

1. **`mujoco.Renderer` 反复创建不 `close()` 会耗尽 GL 上下文 → 段错误**（本机软件 GL 下 12 个左右就崩）。
   所以窗口缩放**绝不重建渲染器**：渲染分辨率固定成 `--width/--height`，窗口变大只是把画面
   等比缩放居中（`Qt.KeepAspectRatio`，不变形），`close()` 时显式 `renderer.close()`。
2. **PySide6 6.11 的 `Qt6Core.dll` 加载失败**（WinError 127）→ 用 `PySide6==6.7.3`。
3. **快捷键不能抢输入框**：`1..6`、`−`、`=` 这些在 `QDoubleSpinBox` 里就是正常输入（比如打 `-0.05`），
   所以按键处理先看焦点是不是数值框/日志框，是就原样交给控件；另外一次动作（H/R/G/ESC）挡掉自动重复。
4. **QSS 的 `[prop="true"]` 选择器改完要 `unpolish/polish` 一次**才生效（例如关节高亮、IK 失败变红）。

### 自检结果（`--selftest`，29 项全过）

```
home 姿态 0.16 mm / 无自碰撞        单关节 5 下「+」J4 实测 +4.92°（其余关节 <0.27°），限位自动夹住
IK 预设 4 个点最差 0.09 mm / 0.000°  不可达点 (1.30,0,0.30) → 明确「不可用」且拒绝运动
直线规划 28 点，直线度 0.000000 mm   直线执行实际轨迹偏离直线最大 4.88 mm、到位 6.40 mm / 姿态 0.79°
关节空间 P2P 到位 0.896°（含重力下垂） 急停：指令定格、停下后 0.5 s 漂移 0.01 mm
离屏渲染 320x480 正常               目标标记目标球 + 路径胶囊 ngeom 9 → 11
相机 拖动/平移/推拉/预设视角 + 工具轴插值（含反向 180°）
```

`--ui-test` 会真开窗口把**每个卡片**都点一遍（30 项，全过；加 `--no-render` 是 26 项，跳过渲染相关）：
关节 ± 按钮、键盘选关节/微调、功能键不误触发、「求解 IK」结果文案、「求解并沿直线运动」到位、
不可达点提示、记录/走到/删除点位、急停、复位、kp 滑块、窗口缩放不重建渲染器、截图；
产物在 `runs/gui_ui_test.png`（窗口截图）、`runs/gui_ui_test_view.png`（3D 画面）
和 `runs/gui_selftest.png`（自检渲染帧）。

> 关于中文：界面文字是中文，`pick_font()` 会从「Microsoft YaHei UI / PingFang SC / Noto Sans CJK」
> 里挑一个系统里真有的字体；都没有才会退回默认字体。终端里的 `[gui] ...` 日志同样是中文。

## 生成物与坐标系约定

* 世界原点 = **机械臂安装面中心**（URDF 原点），z 向上；
* `base_link` 底面在 `z = -0.1712`，地面就铺在这个高度（机器人用地脚螺栓坐在车间地板上）；
  这个数值是转换脚本从 `base_link.STL` 自动量出来的，不是拍的；
* 六轴全 0 时手臂沿 **-x** 方向平伸（`joint2/3/4` 轴平行于 -y，是肩/肘/腕一轴）；
* `qpos` / `ctrl` 顺序：`joint1..joint6`（弧度）；
* 末端有两个 site：`ee_site` = J6 法兰中心（机械接口原点），`tool_site` = 工具尖
  （= `ee_site` 沿工具轴 +42.7 mm，量自 `ee_Link.STL` 包围盒）；
* `home` keyframe：工具轴**竖直朝下**、离地 0.35 m、无自碰撞（详见下节实测数据）；
* 场景自带 3 个固定相机：`overview_cam` / `top_cam` / `side_cam`，录视频用 `--camera <名字>`。

## URDF → MJCF：踩过的 6 个坑都补掉了

| # | 现象 | 处理（`model_import.py`） |
|---|---|---|
| 1 | URDF 里 mesh 写的是 `package://7260501-...-URDF/meshes/x.STL`，但本机没有这层"包名目录" | 依次尝试 `包根/包名/...`、`包根/...`，找到真实文件再改写；生成 MJCF 时把 mesh 拷到 `assets/meshes/` 并用相对路径（目录可整体搬走） |
| 2 | URDF 没有 `<actuator>` → 编译出来 `nu=0`，关节根本不会动 | 自动为每个 hinge 补 `<position kp kv ctrlrange forcerange>`，kp/kv 按关节受力分档 |
| 3 | MuJoCo 会把 root link（base_link）并进 `worldbody`，于是 base_link/Link1 的贴合面**一直报 4 对接触** | 把 root geom 重新包进固定 body `base_link`，并把 URDF 里被丢掉的 root 惯量写回去（总质量 95.85 kg 才对得上） |
| 4 | MuJoCo 只自动过滤"通过关节相连的父子连杆"，**父连杆是固定 body 时不过滤** | 显式补 `<contact><exclude>`：树距离 ≤ 2 的连杆对全部排除（关节处外观件本来就互相插入） |
| 5 | 导入后 mesh 顶点被重表达到各连杆的**惯量主轴坐标系**（geom 带上了 pos/quat） | 量地面高度这类计算必须套上 geom 变换（`geom_world_min_z()`），直接读 `model.mesh_vert` 会算错 244 mm |
| 6 | URDF 的 `effort=0 velocity=0` 在仿真里没意义 | `forcerange` 按"最大重力力矩"实测值估（见下表），接真机时换成伺服手册数值 |

## 实测数据（`python check_model.py` 的输出，可复现）

**质量与力矩**

* 总质量 **95.85 kg**（URDF 各连杆求和，含 base_link 5.05 kg）；大臂 Link2 一个就 46.8 kg；
* 随机姿态下的最大重力力矩：J2 **318 Nm**、J3 83.8 Nm、J4 5.8 Nm（J1/J5/J6 ≈ 0）；
* 所以 J2 的 kp 给到 20000（kp 太小会直接塌下来：kp=5 时稳态误差 ≈ 195°，手臂撑不住）；
* `home` 姿态下重力力矩 = `[0, 69.9, 56.2, 5.7, 0, 0]` Nm。

**home 姿态**

```
qpos      = [0.3152, -1.6288, -2.3803, -0.7033, 1.5708, -2.3152] rad
tool_site = [0.400, 0.000, 0.179]  → 离地 0.350 m，工具轴 = [0, 0, -1]（正朝下）
ee_site   = [0.400, 0.000, 0.222]
雅可比条件数 = 8.8（不接近奇异）
```

**稳定性 / 精度**

* 静置 3 s：最大漂移 **0.27°**（J3），无振动、无自碰撞；
* IK 求解精度：**0.06 ~ 0.10 mm / 0.00°**（阻尼最小二乘，300 次迭代内收敛）；
* 控制器实际到位（IK 解 → 位置伺服跟踪）：**3.4 ~ 6.3 mm**。这个残差主要是位置伺服没有积分项
  带来的稳态误差 ≈ 力矩/kp，不是 IK 的锅；
* 笛卡尔连续轨迹（61 个路点画方+圆）：平均 **4.0 mm**、最大 6.7 mm。

**工作空间**（随机采样 20000 组关节角）

* `tool_site` 可达范围 x∈[-0.96, 0.96]、y∈[-0.96, 0.97]、z∈[-0.83, 1.09]（离地 -0.66 ~ 1.26 m）；
* 水平半径可达 0.97 m；**推荐作业区：r ∈ [0.30, 0.65] m、离地 0.15 ~ 0.60 m**；
* 随机姿态里 43.9% 会撞地面（随机采样当然会往地板里钻，正常）、**9.8% 会自碰撞**。

**运动学结构（和教科书上的球形腕不太一样，值得注意）**

* qpos=0 时各轴世界方向：`J1 = z`、`J2 = J3 = J4 = -y`、`J5 = -z`、`J6 = +y`；
* 即 **J2/J3/J4 三个轴恒平行**（肩 + 肘 + 腕一轴，运动在同一平面内），J5 恒垂直于它们；
* 腕部是**偏置腕**：`J4` 与 `J6` 的轴线在 `q6=0` 时平行，且两轴线**不交于一点**
  （qpos=0 时两锚点 x 相同、z 相差 105 mm）；
* `J5 = 0° 或 ±180°` 是**奇异位形**（雅可比最小奇异值直接掉到 0.0000），
  `J5 = ±90°` 才正常（0.20）。实测：让工具轴竖直朝下时，IK 总是落在 `J5 = ±90°`；
* 结论：**没有简洁的解析逆解**，本目录统一用 `sim_ik.solve_ik_pose()`（数值 DLS + 零空间偏置）。

## 这些脚本的边界（别踩）

1. **没有夹爪**：这个 URDF 的 `ee_Link` 只是法兰盘，抓取要自己加（可参考
   `lerobot_codeit/sim/assets/kinova/` 里给 Kinova 加两指的做法）。
2. **碰撞体 = 外观件**：SolidWorks 导出的碰撞 mesh 就是外壳本身，相邻件在关节处本来就重叠，
   所以自碰撞比例偏高（偏保守）。要做避障 / RL，建议换成简化碰撞体（圆柱/球包络）。
3. **摩擦/阻尼/armature 是估的**（`assets/revA1_arm.xml` 的 `<default>` 那几行），
   接真机前需要辨识；想让手臂"更硬/更软"，用 `viewer.py` 的 `[` `]` 键现场体会。
4. **位置伺服无积分项** → 稳态误差 ≈ 力矩/kp（现在 3~7 mm）。要更准就提 kp（注意别震），
   或者在 `drive_to()` 外面加前馈/积分补偿。
5. **不是用来做精确动力学辨识的**：URDF 只给了质量/惯量，没有关节刚度、齿隙、摩擦模型。

## 常用命令

```bash
# 换 home 姿态后重新生成模型
python convert_urdf_to_mjcf.py --home 0.3 -1.6 -2.4 -0.7 1.57 -2.3

# 只用 IK 求一个"工具尖到 (0.5, 0, 0.3) 且朝下"的解
python sim_ik.py --target 0.5 0 0.3 --axis 0 0 -1

# 自检（采样次数可调，默认 20000）
python check_model.py --samples 50000

# 交互控制台（PySide6）：开窗口 / 无窗口自检 / 真窗口 UI 自检
python revA1_gui.py                    # 等价于 python interactive_control.py
python revA1_gui.py --selftest
python revA1_gui.py --ui-test --width 960 --height 600

# 无显示器：离屏出图 / 录视频
python viewer.py --headless --seconds 3 --out runs/h.png --camera overview_cam
python demo_trajectory.py --cartesian --video runs/cart.mp4 --camera overview_cam
```

## 想改东西改哪里

| 想改 | 去哪 |
|---|---|
| 换 URDF / 换机器人 | `revA1_spec.py` 的 `URDF` / `PACKAGE_ROOT`（默认用本目录 `TB6-R5-RevA1/`），然后重跑 `convert_urdf_to_mjcf.py` |
| 工具长度（TCP） | `revA1_spec.py` 的 `TOOL_TIP_LOCAL_Z`（换成自己夹爪的尺寸） |
| 伺服软硬 | `revA1_spec.py` 的 `GAINS`（kp/kv/forcerange） |
| 地面高度 | 默认自动量 `base_link` 网格最低点；要覆盖用 `--floor-z` |
| 场景布局（地面材质/相机/灯光/目标标记） | `assets/revA1_scene.xml`（生成物，可手改；重跑转换会覆盖） |
| 界面配色 / 控件样式 | `revA1_gui.py` 顶部的 `QSS`（一处改全局） |
| IK 预设点（home/前伸/侧向/低位） | `revA1_gui.py` 顶部的 `POSE_PRESETS` |
| 相机预设视角 | `arm_core.py` 顶部的 `VIEW_PRESETS` |
| IK 判定阈值 / 直线点距 | `arm_core.py` 顶部的 `IK_TOL_POS_M` / `IK_TOL_AXIS_RAD` / `LINE_STEP_M` |
| 多初值 IK 的初值策略 | `arm_core.py` 的 `ik_seeds()` |
| 界面的窗口/渲染分辨率/帧率 | 命令行 `--width/--height/--fps`（默认 960x600 @ 30 fps） |
| 关节阻尼/摩擦/armature | `model_import.py` 的 `DEFAULT_XML`，或直接改 `assets/revA1_arm.xml` |

## 下一步可以做什么

* 接 `lerobot`：用 `assets/revA1_scene.xml` 当环境 xml，把 `tool_site` 当 TCP、
  `ee_site` 当抓取参考点（`lerobot_codeit/sim/mujoco_env.py` 就是按这个思路写的）。
* 加夹爪 + 工件，做 pick & place（参考 `lerobot_codeit/sim/assets/kinova/kinova_scene.xml`）。
* 加相机渲染做视觉策略输入（场景里已经有 3 个相机，`mujoco.Renderer` 直接出图）。

