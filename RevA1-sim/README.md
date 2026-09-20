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
python revA1_gui.py --follow    # ③′ 同上，并直接开「真机跟随」：真机 UDP 广播 / HTTP 状态 → 仿真实时跟
python revA1_gui.py --point-cloud point_cloud/urinal_o2e_stride5.json   # ③″ 把深度相机点云接进场景
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
| `arm_core.py` | **控制核心**（不依赖界面框架）：`Camera` 轨道相机、`Motion` 运动插值、`ArmSim` 仿真/伺服/IK/`follow()`（真机跟随）、多初值 IK、笛卡尔直线规划 |
| `robot_link.py` | **真机 ↔ 仿真 姿态链路**：状态源（HTTP 轮询 / UDP 广播接收）× 报文解析 × 映射/平滑/限速/看门狗；界面里的「真机跟随」卡片就用它，另有 `--probe / --listen / --emit-demo / --selftest` |
| `point_cloud.py` | **点云 → 机械臂场景**：读点云自带坐标系（本机是 `o2e`＝末端系；也支持相机光学系 / 16 位深度图，含自解 PNG 与畸变校正）+ 拍摄姿态 → 换算到基座系 → 生成「机械臂 + 点云 + 小车」场景（`--report / --build-scene / --png / --selftest`） |
| `revA1_gui.py` | **交互控制台（PySide6，主推）**：关节 `−`/`+` 微调、**真机跟随（真机姿态实时映到模型）**、末端目标/IK、示教点位（点到点）、伺服参数、日志；自带 `--selftest` / `--ui-test` |
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
* `robot_link.py`（真机跟随）只用 **标准库 + numpy**，不 import mujoco / Qt：既能被界面调用，
  也能单独当命令行工具跑（`--probe / --listen / --emit-demo / --selftest`）。
* `point_cloud.py`（点云 → 场景）同样只用 **标准库 + numpy**：16 位深度 PNG 是自己解的、
  畸变校正是自己迭代的，所以不需要 OpenCV；只有 `--png` / 界面里出图那一步要 mujoco（+ imageio）。
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
python revA1_gui.py --follow                   # 开窗口并直接开「真机跟随」（UDP 6001 + HTTP 8080）
python revA1_gui.py --width 960 --height 600 --fps 30   # 渲染分辨率 / 帧率上限
python revA1_gui.py --selftest                 # 无窗口：控制逻辑 + 渲染通路全跑一遍并断言
python revA1_gui.py --ui-test                  # 真建窗口 → 脚本化点一遍控件 → 存图 → 退出
python revA1_gui.py --exit-after 10            # 开窗口跑 10 秒自动退出（自动化 / 截图）
```

**七张卡片**

| 卡片 | 做什么 | 怎么用 |
|---|---|---|
| 关节微调 | 每个关节一行：`−` `目标角（大号字）` `+` `(实测角)` | **点一下动一点**；步长可选 0.5°/1°/5°/15°；按住 0.4 s 后自动连点；还有「同步实测 / 全部归零 / 回 home」 |
| **真机跟随** | 真机在广播/开机时，把它的 **6 个关节角实时搬到模型上**（数据源：UDP 广播 / HTTP 状态 / 自动）；卡片上实时显示**延迟、包率、真机 ↔ 仿真的 TCP 位置与工具轴误差** | 填/确认地址端口 → 「启动跟随」；「读一次」只读一帧看通不通；「停止跟随」就地保持姿态（不会掉下来）。详见下文「真机跟随」一节 |
| **点云（相机 → 机械臂）** | 把深度相机点云（本机是 **o2e**＝已在末端系）按**拍摄姿态**搬到机械臂基座系，生成「机械臂 + 点云 + 小车」场景并加载；卡片上显示体检报告（坐标系判定依据、相机位置、包围盒、离地高度） | 选点云文件 → 「加载点云」；「清除点云」回干净场景；「取真机姿态」把拍摄姿态填成真机当前 pose；`坐标系 / 小车高 / 点数上限 / 点大小 / 深度分色 / 微调` 现场调。详见下文「点云」一节 |
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
`空格` 暂停、`H` 回 home、`R` 复位、`G` 重力、`F` 真机跟随（切启动/停止）、`1..6` 选关节、`−`/`=` 给选中关节 ± 一个步长、
`Ctrl+S` 存图、`ESC` 退出。（输入框里打字时快捷键不抢键，负号、数字都能正常输入。）

**动作到点后会自己报告精度**（不是只看指令，是量实际状态）：

```
[gui] 直线规划：[0.111, -0.004, 0.394] → [0.25, 0.1, 0.3]，21 个路点（间隔 10 mm），用时 26 ms，直线度 0.000 mm
[gui] 到位（cartesian）：工具尖 [0.25, 0.101, 0.297] / 目标 [0.25, 0.1, 0.3]，位置误差 3.05 mm，姿态误差 0.36°，最大关节跟踪误差 0.539°
[gui] 直线规划：[0.25, 0.101, 0.297] → [0.52, 0.18, 0.34]，4 个路点（间隔 10 mm），用时 585 ms，直线度 0.000 mm
[gui] 直线运动取消：直线走到第 4/30 个路点就走不过去了（这一支到不了：顶到限位或要换姿态）
      （画面里目标点会标成红色。本支走不过去时：先用手/关节按钮把姿态转过去，或把目标改到这条直线走得到的位置）
```

第 4 行就是"换初始姿态之后"的那个副作用：从**前倾**的 home 出发，把工具轴摆成竖直朝下去那个点
需要换解支，规划器**拒绝**（而不是偷偷换支把工具甩出去）。

### 写这个窗口踩的坑（都已修，并留在自检里）

1. **`mujoco.Renderer` 反复创建不 `close()` 会耗尽 GL 上下文 → 段错误**（本机软件 GL 下 12 个左右就崩）。
   所以窗口缩放**绝不重建渲染器**：渲染分辨率固定成 `--width/--height`，窗口变大只是把画面
   等比缩放居中（`Qt.KeepAspectRatio`，不变形），`close()` 时显式 `renderer.close()`。
2. **PySide6 6.11 的 `Qt6Core.dll` 加载失败**（WinError 127）→ 用 `PySide6==6.7.3`。
3. **快捷键不能抢输入框**：`1..6`、`−`、`=` 这些在 `QDoubleSpinBox` 里就是正常输入（比如打 `-0.05`），
   所以按键处理先看焦点是不是数值框/日志框，是就原样交给控件；另外一次动作（H/R/G/F/ESC）挡掉自动重复。
4. **QSS 的 `[prop="true"]` 选择器改完要 `unpolish/polish` 一次**才生效（例如关节高亮、IK 失败变红）。

### 自检结果（`--selftest`，40 项全过）

```
home 姿态 0.44 mm / 无自碰撞        单关节 5 下「+」J4 实测 +4.92°（其余关节 <0.27°），限位自动夹住
IK 预设 4 个点最差 0.10 mm / 0.047°  不可达点 (1.30,0,0.30) → 明确「不可用」且拒绝运动
直线规划 21 点，直线度 0.000000 mm   直线执行实际轨迹偏离直线最大 5.75 mm、到位 3.25 mm / 姿态 0.34°
本支走不到的直线 → 明确拒绝 + 给原因，且不开始运动（不许偷偷换解支把工具甩出去）
关节空间 P2P 到位 0.339°            急停：指令定格、停下后 0.5 s 漂移 0.01 mm
离屏渲染 320x480 正常               目标标记目标球 + 路径胶囊 ngeom 9 → 11
相机 拖动/平移/推拉/预设视角 + 工具轴插值（含反向 180°）
真机跟随（假真机 HTTP + 正运动学 pose → JointLink → ArmSim）：关节最大差 0.325°、TCP 差 4.04 mm、
        数据率 21.3 Hz、模式 follow、停止后线程退出
点云（点云 → 场景 → ArmSim）：6 轴 + 点云几何 + 小车都在、地面 z = -1.1000 m、离地高度按场景地面算
```

`--ui-test` 会真开窗口把**每个卡片**都点一遍（40 项，全过；加 `--no-render` 是 36 项，跳过渲染相关）：
关节 ± 按钮、键盘选关节/微调、功能键不误触发、「求解 IK」结果文案、「求解并沿直线运动」到位、
**本支到不了时拒绝且工具尖不动**、不可达点提示、记录/走到/删除点位、急停、复位、kp 滑块、
窗口缩放不重建渲染器、截图；
**真机跟随卡片与点云卡片各启停/加载清除一遍**（点云那步真的写网格、换场景、再换回来，
用没人发的 UDP 端口，不碰真机）；
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
* `home` keyframe = **真机实测姿态**（就是拍点云那时的姿态）：工具轴**前倾 38.2°**、
  工具尖离地 0.567 m、无自碰撞、静置漂移 0.34°（见下节实测数据）；
* 三个姿态常量别搞混（都在 `revA1_spec.py`）：
  `HOME_QPOS` = 当前初始姿态（真机读数）、`NEUTRAL_QPOS` = 工具轴**竖直朝下**的中性姿态
  （多初值 IK 的种子，也是历史上那组 home）、`WORK_POSE` = 地面绿点标记 / demo 路径中心；
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
* `home` 姿态下重力力矩 = `[0, 100.7, -71.0, 4.1, 0.1, 0]` Nm。

**home 姿态**（= 真机实测 = 拍摄点云时的姿态，`spec.CAPTURE_POSE`）

```
qpos      = [1.7284, -2.3904, 2.1147, 1.1962, 1.7300, -1.7549] rad  (度: [99.0 -137.0 121.2 68.5 99.1 -100.5])
tool_site = [0.1109, -0.0055, 0.3959] → 离地 0.567 m，工具轴 = [-0.063, -0.615, -0.786]
            （工具轴前倾 38.2°；与 spec.CAPTURE_POSE 的工具轴差 0.004°）
ee_site   = [0.1135, 0.0208, 0.4295]
自碰撞 0 对 · 静置 3 s 漂移 0.339° · 雅可比条件数 73.7 · 最小限位余量 42.8°
```

> 这组关节角是真机状态接口的读数（`runs/probe.log`：把 joints 灌进模型后工具尖差 0.50 mm、
> 工具轴差 0.00°）。独立验算：多初值 IK 搜到的同支解与它只差 0.04°。
> 想换成别的初始姿态：`python convert_urdf_to_mjcf.py --home <6 个关节角(rad)>`，或先用
> `sim_ik.py` 对目标位姿求逆解。

**换了初始姿态之后的一个副作用（值得知道）**

home 现在是**前倾**姿态（工具轴前倾 38.2°），所以：

* 「直线规划」只保证**同一解支**内的直线：从当前姿态到"工具轴竖直朝下"的那类目标，往往需要
  换个解支（机械臂得先转身），这时规划器会**明确拒绝**并说明原因（"直线走到第 k 个路点就走不过去了"），
  而不是偷偷换支把工具甩出去（换支实测能让工具尖偏离直线 0.64 m）；想干"朝下"的活就先转身再走直线；
* 多初值 IK 的种子集里加了 `NEUTRAL_QPOS`（工具轴竖直朝下的中性姿态），"朝下"类目标依然解得又快又准；
* IK 的选解规则改成"**精度都够好时挑离初值最近的解**"——避免为了 0.02 mm 的精度差跳到另一个解支上
  （关节要翻 100°+），这也是笛卡尔直线能贴住直线的原因。

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

## 真机跟随（真机 → 仿真，实时互通）

真机（TB6-R5-RevA1）一直在往外发状态，本目录用 `robot_link.py` 收下来，**把 6 个关节角逐帧下发给模型**，
于是仿真里的手臂跟着真机同步动。两条数据路都支持，界面里选「自动」就都用：

| 路 | 真机怎么发 | 本机实测 | 说明 |
|---|---|---|---|
| **UDP 广播** | 往 `255.255.255.255:6001` 广播 JSON | 源 `192.168.66.169:44558`，约 **96 Hz**，一包 ~1.1 kB | 最实时；另有约 1 Hz 的**心跳包**（没有关节角，已单独计数、不算错误） |
| **HTTP 状态** | `http://192.168.66.169/` 是 nginx 上的 H5，状态接口在 **8080** | `GET http://192.168.66.169:8080/api/state`，往返 ~34 ms | 方便命令行 / 远程查；字段与广播一致 |

两条路的报文里都带着 `joints`（6 个关节角，**弧度**）、`pose`（工具尖 x/y/z[m] + 外旋 XYZ 欧拉角[rad]）、
`vel`、`torque`、`enabled`、`running`、`speed`、`ts`：

```json
{"data": {"joints": [1.7285, -2.3903, 2.1147, 1.1962, 1.7299, -1.7549],
          "pose":   [0.1109, -0.0055, 0.3964, 3.4320, 0.6090, 1.9507],
          "vel": [...], "torque": [...], "enabled": [...], "running": true, "speed": "v25"},
 "type": "state", "ver": 1}
```

**关节约定已现场核对**（这是能直接镜像的前提）：把真机 `joints` 灌进模型 qpos，模型 FK 出的
`tool_site` 与真机 `pose[:3]` 差 **0.5 mm**、工具轴差 **0.00°** —— 也就是**同号同零位，不用改符号/偏置**。
（`python robot_link.py --probe` 随时可以复现这两行。）

### 怎么用

```bash
# 界面：开窗口 → 「真机跟随」卡片 → 「启动跟随」（或直接下面这条，等效）
python revA1_gui.py --follow
python revA1_gui.py --follow --follow-source udp --follow-port 6001        # 只用 UDP 广播
python revA1_gui.py --follow --follow-source http \
                    --follow-url http://192.168.66.169:8080/api/state     # 只用 HTTP 状态

# 命令行联调（不用开界面）
python robot_link.py --probe                    # 读一次真机状态 + 复核它与模型的约定
python robot_link.py --listen --port 6001       # 只听 UDP 广播，逐包打印
python robot_link.py --listen --source http     # 或听 HTTP 状态接口
python robot_link.py --selftest                 # 全链路自检（含"真机可达"，读不到只 SKIP 不算失败）
python robot_link.py --emit-demo --target 127.0.0.1:6001   # 模拟真机发报文（离线联调）
```

卡片上的参数（命令行里没有的就改卡片默认值，或改 `robot_link.py` 顶部常量）：

| 参数 | 默认 | 作用 |
|---|---|---|
| 数据源 | 自动 | `自动` = HTTP 与 UDP 同时开，每帧用最新那一路；也可只用其中之一 |
| HTTP / UDP 端口 | `…:8080/api/state` / `6001` | 真机地址；换机型 / 换网段改这里 |
| 报文 / 单位 | 自动 / 自动 | UDP 报文格式（真机是 JSON）；单位自动判：绝对值 > 7 当**度**，否则当**弧度** |
| 映射 | 绝对（完全镜像） | `相对` = 启动瞬间记下"真机基准 / 仿真基准"，之后只镜像增量（两边本来就不同姿时用） |
| 平滑 | 0.08 s | 一阶低通时间常数；0 = 最跟手，真机信号抖就调大 |
| 限速 | 180 °/s | 每秒最多跟多少度，防真机跳变/毛刺把仿真甩出去；0 = 不限 |
| 看门狗 | 1.0 s | 超过这么久没有新包 → 判掉线：**保持不动**并把「掉线！」写在卡片上（也可改成回 home） |

### 跟得准不准（本机实测）

真机保持不动 8 s，`source=auto`（HTTP 30 Hz + UDP 96 Hz 同时收）：

```
数据：21.4 + 96.4 Hz · 延迟 1 ms · 包 928 / 错 0
真机：q [99.03, -136.96, 121.17, 68.54, 99.12, -100.55]°
仿真：q [99.03, -137.25, 121.50, 68.48, 99.12, -100.55]° · 关节跟踪 0.339°
误差：TCP 2.84 mm · 工具轴 0.01°（真机 pose vs 模型 tool_site）
```

* 关节角是**直接镜像**的（不走 IK），所以真机 ↔ 仿真的差只来自位置伺服（无积分项，稳态误差 ≈ 重力力矩/kp）
  和跟踪带宽，量级是**零点几度 / 几毫米**；
* 想更紧：把「运行 / 伺服」里的 **kp 缩放**调大（界面滑条），或把「平滑」调到 0、「限速」放宽；
* 真机跳变不会带飞仿真：报文先过**关节范围合理性检查**（绝对值 ≤ 7 rad），再受**限速**与**看门狗**约束。

### 排错

| 现象 | 先看 |
|---|---|
| 卡片一直「还没收到数据」 | ① `ping 192.168.66.169`；② 电脑是否和真机同网段（本机 `192.168.66.88/24`）；③ 真机是否在广播（`python robot_link.py --listen --port 6001`）；④ 端口被占了就换端口或改用 HTTP |
| 「错」一直在涨 | 看卡片/日志里的**原始包预览**，对着报文改「报文 / 单位」；真机的心跳包不算错 |
| 仿真手势和真机反着来 | 机型换了符号/零位：用 `JointLink(signs=..., offsets=...)` 改（本机同号，不用改） |
| 掉线后手臂不动了 | 这是**故意**的（看门狗：保持不动比乱动安全）；恢复广播后会自动继续跟 |

## 点云（相机 → 机械臂）

深度相机拍到的点云，按它**自带的坐标系** + **拍摄姿态**换算到机械臂基座系，再当作**静态几何**
放进世界场景里，于是机械臂和点云同框、相对位置真实（**深度 z 越小越靠近相机**），
视角可以随便绕、和机械臂正常遮挡。

::

    点云（o2e = 已经在**末端系**，mm）--------------------------------┐
                                                                    ├─► 基座系点云（m）┬─► assets/meshes/pc_cloud_*.obj
    拍摄姿态（TCP 在基座系：mm + 外旋 XYZ 欧拉角[deg]）----------------┘                  └─► assets/revA1_pc_scene.xml
    （相机光学系的点云才会多一步：p_末端 = R·p_相机 + t，R/t 来自手眼标定）

**本机这份数据是 `o2e`：光学→末端的转换已经做过了，所以直接搬、不再乘一次手眼标定。**
判断依据不是猜的（见下面「坐标系判定」）。

**输入**

| 文件 | 内容 | 说明 |
|---|---|---|
| `xml/calib_extrinsic.xml` | `R`(3×3) + `t`(3×1) | OpenCV `calibrateHandEye` 输出：**相机 → 末端**，mm。实测 `|t| = 245.56 mm`、相机光轴与工具轴同向（eye-in-hand） |
| `xml/calib_intrinsic_1st.xml` | `K` + `distortion` | 相机内参 + 畸变；读深度图 / 判坐标系时用 |
| `point_cloud/*.json` | ① `depth_o2e: [[x,y,z],…]` + `"frame": "o2e"` + `box`；② `depth.png_base64` 16 位深度图 | 深度图那路**自己解 PNG + 按内参反投影**（含畸变校正），不依赖 OpenCV；也认 `*.npy`（N×3） |

**拍摄姿态**：`110.862 -5.474 396.436 196.64 34.89 111.77`（mm + 度，外旋 XYZ）—— 就是真机 `pose`
的 6 个数；**模型的初始姿态（home）也换成了它**（见上文实测数据），所以打开界面就是"拍点云那个姿态"。
界面里点「取真机姿态」可直接从真机读当前值。

### 坐标系判定（为什么不会"重复乘手眼"）

`depth_o2e` 里的点**已经在末端系里**（文件自己写着 `"frame": "o2e"`），于是有三种可能做法，
模块用**数据自己带的 ROI** 当裁判（把点当成某个坐标系 → 反解回相机系 → 用内参投影成像素 →
看命中当初那个裁剪框的比例）：

```
当成相机光学系：49.5% 的点反投影落在 ROI x[512,920] y[1,756]
当成末端系（o2e）：100.0% 的点反投影落在 ROI x[512,920] y[1,756]   ← 四边严丝合缝
```

还有一个**独立**的硬证据：JSON 里 `det_centers` 这类字段同时给了同一个点的 `cam` 与 `end` 坐标，
8 组地标全部满足 `end = R·cam + t`（误差 **0.00 mm**）—— 手眼标定的方向、以及"末端系"的约定就此定死。
界面上「坐标系」下拉框默认 **自动判定**（= 上面这套），也可以强制指定；命令行是 `--frame auto|cam|end`。

### 怎么用

```bash
python point_cloud.py --list                 # 看看有哪些点云文件
python point_cloud.py                        # 默认那份 → 生成场景 + 打印体检报告
python point_cloud.py --png runs/pc.png      # 顺便离屏渲一张预览图
python point_cloud.py --selftest             # 全链路自检（标定/点云/坐标系/变换/建模/渲染）

python revA1_gui.py --scene assets/revA1_pc_scene.xml                   # 用带点云的场景开界面
python revA1_gui.py --point-cloud point_cloud/urinal_o2e_stride5.json   # 或启动时自动接进去

# 换点云 / 换姿态 / 微调（点云没对准时用）
python point_cloud.py --cloud point_cloud/urinal-1.json --points 60000 --point-mm 4 --bands 6
python point_cloud.py --pose "110.9 -5.5 396.4 196.6 34.9 111.8" --cart-height 1.10
python point_cloud.py --frame cam                        # 强制按"相机光学系"处理（会先乘手眼）
python point_cloud.py --ref flange                       # 标定的"末端"若是法兰盘，就退回工具长 42.7 mm
python point_cloud.py --offset 0 0 -150 --yaw 1.5        # 基座系平移[mm] / 绕 z 转[deg]
```

界面第 ③ 张卡片「点云（相机 → 机械臂）」是同一套：选文件 → **加载点云**（把世界换成
"机械臂 + 点云 + 小车"并重建模型）→ **清除点云** 回到干净场景；卡片上还能现场改
`坐标系 / 拍摄姿态 / 手眼标定 / 参考点(TCP 或法兰) / 小车高度 / 点数上限 / 点大小 / 深度分色 / 微调`，
下面直接显示体检报告（坐标系判定依据、相机位置、包围盒、离地高度、标定正交性）。

`python point_cloud.py --selftest`（31 项全过）会把这条链整条验一遍：标定正交性 `|RᵀR−I| = 3.4e-8`、
`|t| = 245.56 mm`、三个点云文件都读得进来（含深度图自己解码 + 反投影）、
**坐标系判定（o2e → 末端系 100% vs 相机系 49.5%；深度图那路反过来 100% vs 38.6%）**、
相机位置 (0.1197, -0.2013, 0.2486) m、点云整片在前下方、末端系点云 z 全为正、
**深度小的那批点离机械臂更近**（最近 10% 平均 1.09 m ＜ 最远 10% 平均 1.86 m）、
离地高度峰值落在小便池高度、场景能被 MuJoCo 加载、**渲染出来的图里数得到点云像素**、
抽稀 / 分带 / 参考点切换 / 微调都符合预期。

### 这份数据算出来是什么样（实测）

```
点云    : urinal_o2e_stride5.json（json 3D 点）78721 点
坐标系  : 末端系（o2e）· 文件声明 o2e · 反投影命中 ROI：末端系 100.0% vs 相机系 60.8%
相机位置: 基座系 (0.120, -0.201, 0.249) m · 离地 1.349 m
基座系  : x[-0.209,0.353] y[-0.872,-0.493] z[-1.174,-0.121] m
离地高度: -0.07 ~ 0.98 m（地面按小车高度 1.10 m 铺）· 埋到地下 7.9%
```

* 点云整片落在**机械臂前方（y < 0）偏下（z < 0）**、离相机 0.64 ~ 1.50 m ——
  和"相机装在工具尖、朝前下方看"完全一致；
* 把"离地高度"做直方图，峰值在 **0.80 ~ 0.95 m**（小便池顶沿的高度），0.4~0.8 m 一段占一半 ——
  `--selftest` 里就把"峰值落在 0.3~1.2 m"当断言；
* 最低那部分点略低于按 1.1 m 车高算的地面（埋地下 7.9%）：深度相机在地面**掠射角**上测距偏大、
  车高也是"大概 1.1 m"，属正常残差 —— 想要更贴，改 `--cart-height`（或卡片上的「小车高」）
  再用 `--offset` 微调一下。

### 实现说明（为什么不是"每帧画点"）

MuJoCo 没有"点云"图元。每帧往场景里塞小几何实测很慢
（`mjv_initGeom` 从 Python 循环：2000 点 = **24 ms/帧**、10000 点 = 139 ms/帧），
所以这里一次性把点云写成 **OBJ 静态网格**（每个点一个小八面体，6 顶点/8 面，按深度分 6 带颜色），
交给模型渲染：整云 78721 点（分 6 带）**载入 0.5 s、之后每帧 2.8 ms**。
生成物都在 `assets/` 下（和基场景同目录，`<include>` 与 `meshdir` 才不会错）：

* `assets/meshes/pc_cloud_*.obj` —— 每个深度带一个网格（`assets/meshes/` 是场景的 `meshdir`）；
* `assets/revA1_pc_scene.xml` —— 基场景 + 点云几何 + 小车 + **地面下移到小车脚下**（`-小车高度`）。

顺带一个小改进：`ArmSim.tip_height()` / 状态栏的「离地」现在读**场景里那个 `floor` geom** 的高度，
所以换成点云场景（地面在 -1.1）之后，"离地"依然是真离地，而不是相对安装面。

### 注意

1. **坐标系别选错**：本机数据是 `o2e`（**末端系**，光学→末端已经做过）→ 直接搬。
   若某份点云其实是**相机光学系**（比如你自己用深度图新算的），要选「相机光学系」/`--frame cam`，
   否则会重复乘一次手眼（差 245.6 mm 平移 + 绕工具轴 76°）；
   自动判定靠"反投影回 ROI 的命中率"，没有 `box` 元数据时会退回文件声明；
2. **参考点（42.7 mm 的模糊）**：真机 `pose` 报的是**工具尖**（`tool_site`）。数据里写的"末端"
   若是**法兰盘**，就选 `flange`/`--ref flange`（沿工具轴退回 `revA1_spec.TOOL_TIP_LOCAL_Z = 42.7 mm`）。
   两种解释只差工具轴上 42.7 mm，用卡片上的微调 / 小车高就能对上；
3. **深度图那路**默认做畸变校正（`--no-undistort` 可关掉对比）；深度 PNG 是裁剪图时，
   本模块会把 ROI 左上角加回像素坐标再用 `cx/cy` 反投影；
4. 点云是**纯视觉几何**（`contype="0" conaffinity="0"`），不参与碰撞、不会挡机械臂运动；
5. 点数别一上来拉满：`--points 60000` + `--point-mm 4` 已经能把小便池看得清清楚楚，
   真要全量 40 万点，网格会到几十 MB、加载几秒。

## 常用命令

```bash
# 换 home 姿态后重新生成模型（6 个关节角，弧度）
python convert_urdf_to_mjcf.py --home 1.72836 -2.39037 2.11474 1.19617 1.72995 -1.75492

# 只用 IK 求一个"工具尖到 (0.5, 0, 0.3) 且朝下"的解
python sim_ik.py --target 0.5 0 0.3 --axis 0 0 -1

# 自检（采样次数可调，默认 20000）
python check_model.py --samples 50000

# 交互控制台（PySide6）：开窗口 / 无窗口自检 / 真窗口 UI 自检
python revA1_gui.py                    # 等价于 python interactive_control.py
python revA1_gui.py --selftest
python revA1_gui.py --ui-test --width 960 --height 600

# 真机跟随：开窗口并直接开跟随 / 只用某一路 / 命令行联调
python revA1_gui.py --follow
python revA1_gui.py --follow --follow-source udp --follow-port 6001
python robot_link.py --probe
python robot_link.py --listen --port 6001
python robot_link.py --selftest

# 点云：生成"机械臂 + 点云 + 小车"场景 / 出预览图 / 自检
python point_cloud.py --list
python point_cloud.py --png runs/pc.png
python point_cloud.py --selftest
python revA1_gui.py --point-cloud point_cloud/urinal_o2e_stride5.json

# 无显示器：离屏出图 / 录视频
python viewer.py --headless --seconds 3 --out runs/h.png --camera overview_cam
python demo_trajectory.py --cartesian --video runs/cart.mp4 --camera overview_cam
```

## 想改东西改哪里

| 想改 | 去哪 |
|---|---|
| 换 URDF / 换机器人 | `revA1_spec.py` 的 `URDF` / `PACKAGE_ROOT`（默认用本目录 `TB6-R5-RevA1/`），然后重跑 `convert_urdf_to_mjcf.py` |
| 真机地址 / 端口 | `robot_link.py` 顶部 `DEFAULT_HOST` / `DEFAULT_HTTP_URL` / `DEFAULT_UDP_PORT`（界面「真机跟随」卡片里也能直接改） |
| 跟随手感（跟多紧 / 多平滑 / 限速 / 断线策略） | 卡片上的 平滑 / 限速 / 看门狗；或 `JointLink(smooth=…, max_speed_deg=…, timeout=…, timeout_policy=…)` |
| 真机符号 / 零位（机型不同才要动） | `JointLink(signs=…, offsets=…)`；先用 `python robot_link.py --probe` 看模型复核那两行 |
| 点云的小车高度 / 点数 / 点大小 / 分色 | `point_cloud.py` 顶部 `DEFAULT_CART_HEIGHT_M` / `DEFAULT_MAX_POINTS` / `DEFAULT_POINT_R_MM` / `DEFAULT_BANDS`（界面「点云」卡片上也能直接改） |
| **初始姿态 / home**（= 拍摄点云时的姿态） | `revA1_spec.py` 的 `HOME_QPOS`（连带 `CAPTURE_POSE` / `TARGET_POSE` / `NEUTRAL_QPOS` / `WORK_POSE`），改完 `python convert_urdf_to_mjcf.py` 重生成场景（也可 `--home <6 个关节角(rad)>` 覆盖） |
| 点云自带的坐标系（o2e = 末端系） | 界面「点云」卡片的「坐标系」或 `--frame auto\|cam\|end`；判定逻辑在 `point_cloud.detect_cloud_frame()`（ROI 反投影命中率 + 文件声明） |
| IK 选哪个解 / 直线能不能换解支 | `arm_core.solve_ik_best()`（精度都够好时挑离初值最近的解）、`LINE_MAX_BRANCH_JUMP_DEG`（超过就判"这条直线走不过去"） |
| 点云颜色（近红 → 远紫） | `point_cloud.py` 的 `DEPTH_COLORS`（每带一个 rgba） |
| 点云和机械臂对不齐 | ① 先看「参考点」该选 TCP 还是法兰；② 再用卡片上的 `微调 Δx/Δy/Δz/绕 z`（= `--offset / --yaw`）；③ 改 `小车高`（= `--cart-height`） |
| 点云场景的地面 / 小车外形 | `point_cloud.build_scene()`（地面挪到 `-小车高度`、小车 `pc_cart` 两个 box，可改尺寸/颜色） |
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
* 真机跟随还能往前一步：现在只**读**真机（安全第一），要双向就接真机的写接口
  （`POST /api/command`，`cmd=move_joint` / `move_c` 等，界面 H5 里用的就是它）——
  那属于"远程遥控真机"，请先在真机侧做好限速与急停。

