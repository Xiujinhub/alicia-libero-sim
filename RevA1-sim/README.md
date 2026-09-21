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
| `arm_core.py` | **控制核心**（不依赖界面框架）：`Camera` 轨道相机、`Motion` 运动插值、`ArmSim` 仿真/伺服/IK/`follow()`（真机跟随）/`set_joint_limits()`、工具向量箭头 `draw_tool_arrows()` + 工具尖轨迹 `Trail`/`draw_trail()`、多初值 IK、笛卡尔直线规划 |
| `robot_link.py` | **真机 ↔ 仿真 链路**：姿态（状态源 HTTP 轮询 / UDP 广播接收 × 报文解析 × 映射/平滑/限速/看门狗）+ **任务信号（UDP 6501 的 `motion: start/over` → `TaskListener`）**；界面里的「真机跟随」「任务信号」两张卡片就用它，另有 `--probe / --listen / --task-listen / --task-demo / --emit-demo / --selftest` |
| `point_cloud.py` | **点云 → 机械臂场景**：读点云自带坐标系（本机是 `o2e`＝末端系；也支持相机光学系 / 16 位深度图，含自解 PNG 与畸变校正）+ 拍摄姿态 → 换算到基座系 → 生成「机械臂 + 点云 + 小车」场景（`--report / --build-scene / --png / --selftest`） |
| `revA1_gui.py` | **交互控制台（PySide6，主推）**：关节 `−`/`+` 微调、**关节限位（可保存为默认 → `config/joint_limits.json`）**、**工具 TCP 向量箭头**、**任务信号 6501 → 工具尖轨迹（勾选要画的工具，默认全不勾）**、**真机跟随（真机姿态实时映到模型）**、末端目标/IK、示教点位（点到点）、伺服参数、日志；自带 `--selftest` / `--ui-test` |
| `revA1_config.py` | **用户配置**（只依赖标准库）：读写 `config/joint_limits.json`（按关节的限位覆盖，删文件即回 URDF 默认；`REVA1_CONFIG_DIR` 可换目录） |
| `config/` | 用户配置目录（首次点「保存为默认」才会出现 `joint_limits.json`） |
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
| **点云（相机 → 机械臂）** | 把深度相机点云（本机是 **o2e**＝已在末端系）按**拍摄姿态**搬到机械臂基座系，生成「机械臂 + 点云 + 小车」场景并加载；卡片上显示体检报告（坐标系判定依据、相机位置、包围盒、离地高度） | 选点云文件 → 「加载点云」；「清除点云」回干净场景；「取真机姿态」把拍摄姿态填成真机当前 pose；`坐标系 / 小车高 / 小车中心 / 点数上限 / 点大小 / 深度分色 / 微调` 现场调；**「作业点标记」勾上才画那个绿圆盘（默认隐藏）**。详见下文「点云」一节 |
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
5. **数字控件的上/下三角 → `+` / `−`**：QSS 的箭头子控件只能引用**图片**，所以在启动时用
   `QPainter` 画两张 12 px 的图标（`runs/ui_icons/`，是生成物）再由 `spin_buttons_qss()` 挂上
   `::up-arrow / ::down-arrow`。只改外观：步进仍是 `stepUp()/stepDown()`（单击 = 一个 `singleStep`、
   滚轮/键盘上下键/长按都一样），布局不用动。
   ⚠️ 这里踩过一个 CSS 坑：写成 `QSpinBox, QDoubleSpinBox::up-button { … }` 时，`::up-button`
   **只作用于最后一个选择器**，前一个被静默忽略 —— "改了没反应"。得逐个展开成
   `QSpinBox::up-button, QDoubleSpinBox::up-button`。
   ⚠️ 另一个坑：`grab()` 出来的图是**设备像素**（本机屏幕 150% → 逻辑 126×30 抓到 189×45），
   量按钮区得乘 `devicePixelRatio()`，不然会"看起来没画"。
5. **控件"显示不全"的几种成因（都已修）**：
   * `setMaximumWidth(96)` 写到比控件自己的 `minimumSizeHint()`（跟字体、后缀、小数位有关）还小
     → Qt 把控件压扁，数字被裁。现在统一走 `cap_width(w, cap)`：上限取
     `max(想给的上限, 最小需求)`；
   * 面板内容比可视区宽（本项目实测 **578 px > 视口 450 px**），而横向滚动条被设成 **AlwaysOff**
     → 每行右边一截**看不见也滚不到**。现在：面板最小宽度由 `panel.minimumSizeHint()` 定，
     `QSplitter` 尊重它；横向滚动条改成 `AsNeeded` 兜底；
   * **窗口比屏幕大**（默认 1420×900 在 1366×768 笔记本上超出屏幕）→ 右侧/状态栏跑到屏幕外。
     现在 `_fit_to_screen()` 按 `screen.availableGeometry()` 收一下；
   * **行里放了弹性空格（`add_row(..., None, ...)`）时，Qt 会把输入框压到它自己的"最小尺寸"**
     —— 实测「深度分色」那个 `QSpinBox` 只剩 **43 px**（Windows 风格下它的 `sizeHint` 会跟着
     数值范围缩：0~8 这种一位数就只要 43 px），里面那格 13 px，**数字直接看不见**。
     现在 `ControlPanel.floor_input_widths()` 按类型给下限（spin 84 / combo 96 / 输入行 110 px，
     且跳过 spin box 内建的那个编辑框 —— 硬加宽它反而会把上下箭头挤出盒子）；
   * 状态栏 7 段文字一行放不下 → 现在是"按宽度省略成 `…` + 全文在 tooltip"
     （`_set_status()`），窗口一改大小立刻重算（`resizeEvent`）。
   这些都由 `layout_report()` 在 `--ui-test` 里量：**任何控件宽度 < 自身最小需求 = 失败**
   （默认尺寸 / 760×520 / 1366×768 三种尺寸各量一遍）。

### 自检结果（`--selftest`，72 项全过）

```
home 姿态 0.44 mm / 无自碰撞        单关节 5 下「+」J4 实测 +4.92°（其余关节 <0.27°），限位自动夹住
home 完整姿态（3×3）与 CAPTURE_POSE 差 0.0097° ← J6 轴写反这种错只有这一项看得见
IK 预设 4 个点最差 0.10 mm / 0.047°  不可达点 (1.30,0,0.30) → 明确「不可用」且拒绝运动
直线规划 21 点，直线度 0.000000 mm   直线执行实际轨迹偏离直线最大 5.75 mm、到位 3.25 mm / 姿态 0.34°
本支走不到的直线 → 明确拒绝 + 给原因，且不开始运动（不许偷偷换解支把工具甩出去）
关节空间 P2P 到位 0.339°            急停：指令定格、停下后 0.5 s 漂移 0.01 mm
离屏渲染 320x480 正常               目标标记目标球 + 路径胶囊 ngeom 9 → 11
三个工具 TCP 向量：|TCP−法兰| = |v|（294.8/241.7/243.5 mm，误差 <1e-12）、都在地面上
        箭头几何：起点 = 安装点 (vx,vy,0)、方向 = 末端工具轴、size[2] = 2·vz = 589.599 mm
        **三根箭头互相平行**：两两夹角 0.00000°、与末端工具轴 0.00733°（改之前是 13.6°~14.9° 的扇形）
        箭头尖仍落在各自 TCP 上（数值差 0.000019 mm；画面上离 TCP 球心 0.7 / 1.6 / 1.1 px）
        画面上差 5142 像素（只画一根 1735）；J1 转 0.30 rad → 重心位移 19.5 px；三个都关 → 差 0 像素
关节限位（config/joint_limits.json）：写文件 → 新场景自动加载 → 三处一起生效（J3 到 180°）
        → follow(175°) 不再截断（URDF 那套仍截到 163.998°）→ 坏配置明确报错 → 恢复 URDF 默认删文件
工具尖轨迹（6501 任务信号驱动）：采样间距过滤 / 点数封顶滚动丢最老的 / 清空 / 只记勾上的工具 /
        记的就是箭头尖（= 该工具 TCP）；任务中随机械臂长出来（39 点 · 63.5 mm）、任务结束后不再加点；
        绘制 = 相邻两点一段胶囊（38 段 · pos=中点、size[2]=半长 0.762 mm）、几何只在画面里（DECOR）
相机 拖动/平移/推拉/预设视角 + 工具轴插值（含反向 180°）
真机跟随（假真机 HTTP + 正运动学 pose → JointLink → ArmSim）：关节最大差 0.325°、TCP 差 4.04 mm、
        数据率 21.3 Hz、模式 follow、停止后线程退出
点云（点云 → 场景 → ArmSim）：6 轴 + 点云几何 + 小车都在、地面 z = -1.1000 m、离地高度按场景地面算
```

`--ui-test` 会真开窗口把**每个卡片**都点一遍（77 项，全过；加 `--no-render` 是 71 项，跳过渲染相关）：
关节 ± 按钮、键盘选关节/微调、功能键不误触发、「求解 IK」结果文案、「求解并沿直线运动」到位、
**本支到不了时拒绝且工具尖不动**、不可达点提示、记录/走到/删除点位、急停、复位、kp 滑块、
**数字控件的上/下按钮是 `+` / `−`（图标内容 + 真控件上的像素都量：上按钮竖跨 14 px、下按钮 4 px）
且步进逻辑不变**、
**「关节限位」应用/保存为默认/恢复 URDF 默认（临时目录，不碰你本机的 config/）**、
**「任务信号」卡片：默认全不勾 → 假真机发 `motion: start` → 勾上夹爪后轨迹随臂长出来（35 点）
→ 手动「清空轨迹」后又接着记（12 点）→ 发 `motion: over` → 轨迹立刻清空（0 点）→ 「停止监听」线程退出**、
**「作业点标记」勾选（绿圆盘藏/显）**、**「工具 TCP 向量」三个勾选框（画面里 `ngeom` 12→9→10）与箭头粗细**、
窗口缩放不重建渲染器、截图；
**布局体检 6 项**（默认尺寸 / 拖到 760×520 / 1366×768 笔记本尺寸各量一遍：没有控件被压到小于自身最小需求、
面板内容不超视口、状态栏文字按「…」省略而不是切半个字）—— 这条是"控件显示不全"的回归网，
实现见 `revA1_gui.layout_report()` / `cap_width()`。
**真机跟随卡片与点云卡片各启停/加载清除一遍**（点云那步真的写网格、换场景、再换回来，
用没人发的 UDP 端口，不碰真机）；
产物在 `runs/gui_ui_test.png`（窗口截图）、`runs/gui_ui_test_view.png`（3D 画面）
和 `runs/gui_selftest.png`、`runs/tool_arrows*.png`（自检渲染帧 / 三根工具向量箭头）。

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
            （工具轴前倾 38.2°；与 spec.CAPTURE_POSE 的工具轴差 0.004°、完整姿态差 0.010°）
ee_site   = [0.1135, 0.0208, 0.4295]
自碰撞 0 对 · 静置 3 s 漂移 0.339° · 雅可比条件数 73.7 · 最小限位余量 42.8°
```

> 这组关节角是真机状态接口的读数（`runs/probe.log`：把 joints 灌进模型后工具尖差 0.50 mm、
> 工具轴差 0.00°、**完整姿态差 0.00°**——第三项才是能发现"J6 轴写反"的那一项，
> 详见「真机跟随」里那个 J6 的坑）。独立验算：多初值 IK 搜到的同支解与它只差 0.04°。
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
`tool_site` 与真机 `pose[:3]` 差 **0.5 mm**、工具轴差 **0.00°**、**完整姿态（3×3）差 0.00°**
—— 也就是**同号同零位，不用改符号/偏置**。

> ⚠️ **第三项（完整姿态）不能省，这里踩过一次坑**：工具尖恰在 J6 轴上、工具轴又是 J6 的旋转轴，
> 所以 **J6 的轴/符号写反了，前两项照样全过**。真机 URDF 里 `joint1..5` 的 axis 是 `0 0 1`，
> **唯独 `joint6` 写成 `0 0 -1`**（与固件相反）→ 表现就是**法兰上装的三个工具方位左右颠倒**
> （位置/工具轴都看不出来）。实测（`runs/j6_before_fix.log`，修复前；`runs/probe_after_fix.log`，修复后）：
>
> | 真机 J6 | 模型 vs 真机的完整姿态差 | `360° − 2·|J6|` |
> | --- | --- | --- |
> | −107.779° | 144.443° | 144.442° ✓ |
> | −100.550°（拍摄姿态） | 158.892° | 158.900° ✓ |
>
> 两次都精确满足 `q6_model = −q6_robot` → 修正是把 `joint6` 的轴翻正
> （`revA1_spec.JOINT_AXIS_FIX`，`convert_urdf_to_mjcf.py` 生成模型时自动应用，见 `model_import.fix_joint_axis`）。
> 修完再量：**当前姿态 0.000°、拍摄姿态 0.010°**，三个工具的方位（现在是三根**平行**箭头）
> 也跟着对了。
> `check_model.py` / `revA1_gui.py --selftest` / `robot_link.py --probe` 现在都盯着这一项。
（`python robot_link.py --probe` 随时可以复现这两行。）

### 跟随到限位附近"卡住"？—— 关节限位（`config/joint_limits.json`）

跟随链路里唯一会**截断**目标角的地方就是关节限位（`ArmSim.follow()` 按范围 clip，
MuJoCo 也会按 `ctrlrange` 再夹一次）。而模型范围来自 URDF：**`joint1/2/4/5/6 = ±180°，
只有 `joint3 = ±164.0°`** —— 真机 J3 一旦超出 164°，仿真就会卡在限位上（表现：真机继续转、
仿真姿态不动/看着更"竖直"，量级 = 超出的度数，见下表）。

| 真机 J3 | 模型截断到 | 末端工具轴差 | 工具尖差 |
| --- | --- | --- | --- |
| 165° | 164° | 1.0° | 7 mm |
| 170° | 164° | 6.0° | 43 mm |
| 175° | 164° | 11.0° | 79 mm |
| 180° | 164° | 16.0° | 114 mm |

**改法（界面，两下点完）**：「关节限位」卡片 → 把 J3 改成 `-180 ~ 180` → 点「保存为默认」。

* 「应用」= 立刻生效（同时改三处：`sim.lo/hi`、`model.jnt_range`、`model.actuator_ctrlrange`
  —— 少改一处 MuJoCo 就会把 `ctrl` 再按旧范围静默夹回去）；
* 「保存为默认」= 再写进 **`config/joint_limits.json`**（只写与 URDF 不同的项），**下次启动自动加载**；
* 「恢复 URDF 默认」= 回 URDF 值并**删掉该文件**；文件损坏/不合法会在卡片上报错并退回 URDF（不会崩）；
* 换场景（例如接点云）时限位会跟着带过去；`check_model.py` / `--selftest` 也会应用并打印它。

```bash
python revA1_gui.py --config-dir D:/my_cfg      # 换配置目录（= 环境变量 REVA1_CONFIG_DIR）
cat config/joint_limits.json                    # 就是这样一个文件（手改也行）
```

```json
{
  "_note": "关节限位（度）。界面「关节限位」卡片写入；删掉本文件即恢复 URDF 默认。",
  "_urdf": "revA1_spec.LIMITS（joint1/2/4/5/6 = ±180°，joint3 = ±164°）",
  "unit": "deg",
  "limits": { "joint3": [-180.0, 180.0] }
}
```

> 实测（`save_joint_limits` 落盘 → 重开一个 `ArmSim` 当"下次启动"）：
> URDF 下 `follow(175°)` 的 J3 = **163.998°**；保存后 = **175.000°**，且三处都是 180.0°；
> 点「恢复 URDF 默认」→ 文件消失、又回到 163.998°。
>
> ⚠️ **别把模型放得比真机还宽**：放宽只是让仿真跟得上真机；如果真机自己也到不了，
> 那它永远不会超限，也就不是这个原因（先看示教器上的关节范围）。

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

# 小车怎么摆 / 那个"绿色圆盘"
python point_cloud.py --cart-center 150 150     # 小车中心在基座系里的位置[mm]（默认：基座靠车头左前缘）
python point_cloud.py --cart-center 0 0        # 让基座回到小车正中
python point_cloud.py --show-target-pad        # 保留基场景那个"名义作业点"绿圆盘（默认去掉）
```

界面第 ③ 张卡片「点云（相机 → 机械臂）」是同一套：选文件 → **加载点云**（把世界换成
"机械臂 + 点云 + 小车"并重建模型）→ **清除点云** 回到干净场景；卡片上还能现场改
`坐标系 / 拍摄姿态 / 手眼标定 / 参考点(TCP 或法兰) / 小车高度 / 小车中心 / 点数上限 / 点大小 / 深度分色 / 微调`，
下面直接显示体检报告（坐标系判定依据、相机位置、包围盒、离地高度、标定正交性）。

### 小车怎么摆 + 那个"绿色圆盘"

* **小车**：默认画成一个 `0.60 × 0.52 × 1.10 m` 的长方体（`CART_SIZE_M`）+ 一块顶板。
  真机上机械臂是**贴着小车前缘**装的，所以默认把小车中心放在基座系的 `(+150, +150) mm`
  ——等价于"基座沿 −x 挪 150 mm、朝前 −y 挪 150 mm"，此时基座离**前缘 110 mm**、离**左缘 150 mm**
  （报告里会写"基座离车沿：前/后/左/右"四个数）。想改成别的摆位：界面「小车中心」两个框，
  或 `--cart-center X Y`；想让基座回到车正中就填 `0 0`。
* **作业点标记**（就是那个绿色圆盘）：基场景里的 `target_pad` = **名义作业点**标记，
  圆柱 ⌀100×4 mm、材质 `rgba 0.15 0.80 0.35`，正常贴在地面上。
  点云场景里地面会被挪到小车脚下，它就会被挪到车顶平面，于是看起来像"机械臂旁多出来的一块"
  ——所以**点云场景默认不画它**（`--show-target-pad` 可以要回来），
  界面上的「作业点标记」勾选框对**任何场景**都立刻生效（默认不勾 = 隐藏，勾上就回原位）。
* 场景里**其他偏绿**的东西（如果你说的不是上面那个圆盘，大概就是这两个）：
  ① 点云的**第 3 个深度带**是黄绿色（`DEPTH_COLORS[3] = 0.52 0.92 0.42`，`--bands 1` 可改单色）；
  ② 机械臂**底座** `base_link` 是 URDF 里给的淡青绿色（改材质要去 `TB6-R5-RevA1/urdf/*.urdf`
  或 `assets/revA1_arm.xml` 里 `base_link_geom` 的 `rgba`，改完重跑 `convert_urdf_to_mjcf.py`）。

`python point_cloud.py --selftest`（34 项全过）会把这条链整条验一遍：标定正交性 `|RᵀR−I| = 3.4e-8`、
`|t| = 245.56 mm`、三个点云文件都读得进来（含深度图自己解码 + 反投影）、
**坐标系判定（o2e → 末端系 100% vs 相机系 49.5%；深度图那路反过来 100% vs 38.6%）**、
相机位置 (0.1197, -0.2013, 0.2486) m、点云整片在前下方、末端系点云 z 全为正、
**深度小的那批点离机械臂更近**（最近 10% 平均 1.09 m ＜ 最远 10% 平均 1.86 m）、
离地高度峰值落在小便池高度、**小车按「小车中心」摆放（基座离前缘 110 mm）**、
**那个绿圆盘默认不写进场景**、场景能被 MuJoCo 加载、**渲染出来的图里数得到点云像素**、
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

## 三个工具的 TCP 向量（画面里的三根箭头）

真机上同一片法兰装了 **三个工具**，TCP 标定给的是**末端系**（法兰中心为原点、+z = 工具轴 =
模型里的 `ee_site`）里的三个向量（`revA1_spec.TOOLS`，单位 m）：

| 工具 | TCP 向量 v（末端系，m） | \|v\| | 颜色（画面箭头） | home 姿态下的世界坐标 | 离地 |
| --- | --- | --- | --- | --- | --- |
| **夹爪**  | `-0.0023439,  0.0004872,  0.2947993` | **294.8 mm** | 蓝 | `(0.096, -0.162, 0.199)` m | 0.370 m |
| **喷嘴1** | ` 0.0410224,  0.0454661,  0.2337835` | **241.7 mm** | 橙 | `(0.130, -0.083, 0.212)` m | 0.383 m |
| **喷嘴2** | ` 0.0588531, -0.0091195,  0.2361271` | **243.5 mm** | 品红 | `(0.072, -0.082, 0.212)` m | 0.384 m |

（这三个世界坐标是 **joint6 轴翻正之后**的数；翻正前整片扇形会绕工具轴镜像到另一边，
而三个 TCP 之间的相对间距 57.5 / 85.3 / 87.3 mm 不变 —— 详见下面「真机跟随」里 J6 那个坑。）

画面上每个工具一根箭头，**三根互相平行、而且都平行于末端姿态的工具轴**（不是"从法兰中心
直着画到 TCP"的那种扇形）：

* 起点 = **安装点** = 法兰平面上的 `(vx, vy, 0)` —— 三个工具并排装在同一片法兰上，侧偏
  2.4 / 61.2 / 59.6 mm 就体现在这儿；
* 方向 = **法兰 +z（末端工具轴）**；长度 = `vz` = 该工具的**轴向长度**（294.8 / 233.8 / 236.1 mm）；
* 箭头尖 = 起点 + 方向·`vz` = `法兰 + R·v` = **该工具的 TCP**（所以尖端仍精确落在 TCP 上）；
* 每帧按当前末端位姿重算，所以**跟着机械臂实时动**（三个 TCP 两两间距 57.5 / 85.3 / 87.3 mm 不变）。

> ⚠️ 早先的版本是"**从法兰中心直接画到 TCP**"（= 把 `v` 的侧偏也算进了方向）：那样三根箭头两两
> 夹角 **13.6°~14.9°**，画面里是**一把扇形**，跟真机（三个工具并排、**平行**装在同一片法兰上）
> 对不上。现在按物理装法画：两两夹角 **0.00000°**、与末端工具轴 **0.00733°**（自检里有一项盯着它）。

* 界面：新卡片 **「工具 TCP 向量（跟随机械臂）」**（在「关节微调」下面）
  —— 三个勾选框（文字颜色 = 箭头颜色，可单独开关）、「箭头粗细」（杆半径 1~12 mm，
  头部 = 2.2 倍）、下面三行实时读数（TCP 世界坐标 / 离法兰多远 / 离地）；
* 代码：`spec.TOOLS`（数据）→ `arm_core.draw_tool_arrows()` / `ArmSim.tool_lines()` /
  `ArmSim.tool_points()` / `ArmSim.add_tool_arrows()`；
* **不动模型**：箭头只在 `MjvScene` 里（不进 `mjModel`、不参与碰撞/物理、也不改质量与 IK），
  归到 `mjCAT_DECOR` 所以**不投阴影**（否则小臂/腕部会多出一片跟着动的色斑）；
* 换成别的工具 / 工具改版：改 `spec.TOOLS` 里的数就行，界面读数与箭头自动跟着变。

> ⚠️ **坑：MuJoCo 画 `mjGEOM_ARROW` 只画 `size[2]` 的一半**（沿本地 +z 从 `pos` 起）。
> 一开始直接 `mjv_connector(法兰, TCP)` 得到 `size[2] = |v|`，画出来只有工具长度的**一半**
> （空场景对照实验，箭长 300 mm：`size[2]` = 150/300/600 mm → 画出 75/150/300 mm，起点都在 `pos`）。
> 所以代码里把终点放到 **2 倍**处，画出来正好是 **安装点 → TCP**；`size[2]` 因此是 `2·vz`（有注释）。
> 自检里那三项「箭头尖离 TCP 球心 < 8 px」就是这个坑的守门人（修之前是 144~209 px）。

> ⚠️ **|v| 和 URDF 里那个 42.7 mm 是两回事**：`TOOL_TIP_LOCAL_Z = 42.7 mm` 只是
> `ee_Link.STL` 这块**法兰盘本身**的长度（旧 `tool_site` 用它）；真机装的是上表这三根
> **长杆工具**（24~30 cm），所以 `tool_site`（IK/点云用的那个默认 TCP）与这三个 TCP
> 差着 20~25 cm —— 要用哪个当"末端"，看你的作业定义。

验证（都在自检里）：

| 检查 | 结果 |
| --- | --- |
| TCP = 法兰原点 + R·v（三个都算一遍） | 距离 = \|v\|，误差 **< 1e-12**；home 下三个都离地 > 0.3 m、在工作中 |
| 箭头几何 | `pos` = 安装点 `(vx, vy, 0)`、`mat[:,2]` = **末端工具轴**（法兰 +z）、`size[2]` = 2·`vz` = **589.599 mm**（MuJoCo 只画一半 → 画出来正是 294.799 mm） |
| **三根箭头互相平行** | 两两夹角 **0.00000°**、与末端工具轴 **0.00733°**（改之前是从法兰中心画的扇形：两两 13.6°~14.9°） |
| **箭头尖真的落在 TCP 上** | 数值上尖与 TCP 差 **1.9e-5 mm**；画面上把 TCP 用白球标出来量像素：**0.7 / 1.6 / 1.1 px**（修"只画一半"那个坑之前是 144~209 px） |
| 画面上真的画出来了 | 与"不放箭头"那帧差 **5142 像素**；只画一根时 1735 像素 |
| 跟随机械臂 | J1 转 0.30 rad → 箭头像素重心位移 **19.5 px** |
| 单独开关 | 三个都关 → 画面与基线一致（差 0 像素）；只画一根 → 1735 < 5142 |
| 三个 TCP 与真机一致 | 靠 `joint6` 轴翻正（见「真机跟随」里的 J6 坑）：完整姿态与真机差 **0.00°**，三个 TCP 间距 57.5/85.3/87.3 mm |
| 出图 | `runs/tool_arrows.png`（全身）、`runs/tool_arrows_close.png`（近看末端） |


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

# 任务信号（6501）+ 工具尖轨迹：启动就监听 / 假真机发 start-stop 试
python revA1_gui.py --task-listen --point-cloud point_cloud/urinal_o2e_stride5.json
python robot_link.py --task-listen
python robot_link.py --task-demo start

# 点云：生成"机械臂 + 点云 + 小车"场景 / 出预览图 / 自检
python point_cloud.py --list
python point_cloud.py --png runs/pc.png
python point_cloud.py --selftest
python revA1_gui.py --point-cloud point_cloud/urinal_o2e_stride5.json

# 无显示器：离屏出图 / 录视频
python viewer.py --headless --seconds 3 --out runs/h.png --camera overview_cam
python demo_trajectory.py --cartesian --video runs/cart.mp4 --camera overview_cam
```

## 任务信号 6501 → 工具尖轨迹

真机**开始 / 结束作业**时会往 **UDP 6501** 广播一个 JSON：

```json
{"motion": "start"}      // 开始作业
{"motion": "over"}       // 结束作业（⚠️ 结束信号是 over）
```

界面卡片「任务信号 6501 → 工具尖轨迹」收到 **start** 就开始把**勾上的**工具尖位置连成轨迹
（随机械臂实时长出来），收到 **over** 就**立刻清空轨迹** —— 画面上不留（想手动擦也可以点「清空轨迹」）。

* **勾选框默认全不勾**：一个都不勾 → 只监听，不记也不画（这是默认行为）；
* 轨迹颜色 = 该工具箭头的颜色（蓝/橙/品红），和画面上那三根工具向量箭头对得上；
* 记的是**工具箭头尖**（= 该工具的 TCP，见上一节）—— 所以轨迹就是"末端实际走过哪儿"；
* 采样：两点间距 < `spec.TRAIL_MIN_STEP_M`（默认 **1.5 mm**）不记 → 工具尖停着不动时不会灌点；
  每根最多 `spec.TRAIL_MAX_POINTS`（默认 **6000** 点 ≈ 9 m 路程），超了**滚动丢最老的**
  （卡片读数会写"已滚动丢掉 N 点"，路程照累计）；
* 只在画面里（`mjCAT_DECOR`：不投阴影、不进 `mjModel`、不参与碰撞/物理/IK）；
* 换场景（接点云）时轨迹跟着带过去。

### 怎么用

```bash
# 界面：卡片上「开始监听」→ 勾上要画的工具 → 真机一开始作业就会长轨迹
python revA1_gui.py --task-listen                    # 启动就监听（默认 6501）
python revA1_gui.py --task-listen --task-port 6551   # 换个端口试

# 没有真机也能试：命令行发一条任务信号（假真机）
python robot_link.py --task-listen                   # 只听任务信号并打印 start/over
python robot_link.py --task-demo start               # 往 127.0.0.1:6501 发 3 包 start
python robot_link.py --task-demo over                # 结束（真机的结束信号就是 over）
python robot_link.py --task-demo start --task-target 127.0.0.1:6551
```

> 报文写法宽松一点（真机改版不至于把链路断掉）：大小写/前后空格、`state`/`data` 外壳、
> `{"motion": {"state": "start"}}` 都认；**结束**除了 `over`，`stop / done / finish / end / idle`、
> `true-false`、`1-0` 这类等价写法也一并当"结束"（见 `robot_link.MOTION_END_WORDS`）。
> 认不出来的包只记一条错误（卡片上会写出来），**不会**乱改任务状态，也不会把界面搞崩。
>
> 实测（`--ui-test` + 一次**跨进程**联调）：假真机发 `{"motion": "start"}` → 卡片显示「任务中」、
> 日志记一条"任务信号：开始任务"；勾上夹爪后机械臂走一段 → 夹爪轨迹 35 点；「清空轨迹」→ 归零后
> 又接着记（12 点）；发 `{"motion": "over"}` → **轨迹立刻清空（0 点）**、日志记一条"任务结束 → 已清空"；
> 「停止监听」→ 线程退出。

## 想改东西改哪里

| 想改 | 去哪 |
|---|---|
| 换 URDF / 换机器人 | `revA1_spec.py` 的 `URDF` / `PACKAGE_ROOT`（默认用本目录 `TB6-R5-RevA1/`），然后重跑 `convert_urdf_to_mjcf.py` |
| 真机地址 / 端口 | `robot_link.py` 顶部 `DEFAULT_HOST` / `DEFAULT_HTTP_URL` / `DEFAULT_UDP_PORT`（界面「真机跟随」卡片里也能直接改） |
| 跟随手感（跟多紧 / 多平滑 / 限速 / 断线策略） | 卡片上的 平滑 / 限速 / 看门狗；或 `JointLink(smooth=…, max_speed_deg=…, timeout=…, timeout_policy=…)` |
| 真机符号 / 零位（机型不同才要动） | `JointLink(signs=…, offsets=…)`；先用 `python robot_link.py --probe` 看模型复核那三行（**完整姿态差也要 ≈0°**，只有它能发现 J6 这类"位置上看不出来"的错） |
| 某个关节的轴写反了（URDF 的坑） | `revA1_spec.JOINT_AXIS_FIX`（当前：`joint6 → 0 0 1`），改完重跑 `convert_urdf_to_mjcf.py`；实现见 `model_import.fix_joint_axis` |
| **每个关节的限位**（真机与 URDF 范围不一致时） | 界面「关节限位」卡片 →「保存为默认」写进 **`config/joint_limits.json`**（删掉即回 URDF）；换目录用 `--config-dir` / 环境变量 `REVA1_CONFIG_DIR`；读写逻辑在 `revA1_config.py`，写进模型在 `arm_core.apply_joint_limits()`（`lo/hi` + `jnt_range` + `ctrlrange` 三处） |
| 界面里**控件/按钮显示不全** | `revA1_gui.cap_width()`（宽度上限绝不低于控件最小需求）、`ControlPanel.floor_input_widths()`（行里有弹性空格时输入框的下限）、`layout_report()`（`--ui-test` 的布局体检）、`_fit_to_screen()`（窗口不超屏幕）、`RevA1Window._set_status()`（状态栏按宽度省略）、`scroll.setMinimumWidth(...)`（面板最小宽度）+ 横向滚动条 `AsNeeded` |
| 点云的小车高度 / 点数 / 点大小 / 分色 | `point_cloud.py` 顶部 `DEFAULT_CART_HEIGHT_M` / `DEFAULT_MAX_POINTS` / `DEFAULT_POINT_R_MM` / `DEFAULT_BANDS`（界面「点云」卡片上也能直接改） |
| 小车外形 / 基座在小车上怎么摆 | `point_cloud.py` 的 `CART_SIZE_M` / `CART_TOP_SIZE_M` / `DEFAULT_CART_CENTER_MM`（界面「小车中心」= `--cart-center`） |
| 那个"绿色圆盘"（作业点标记） | 显隐：界面「作业点标记」勾选框（运行时对任何场景生效）/ 生成时 `point_cloud.py --show-target-pad`；要挪位置改基场景 `assets/revA1_scene.xml` 的 `target_pad`（或 `model_import.SCENE_TEMPLATE`） |
| **三个工具的 TCP 向量**（画面箭头） | `revA1_spec.py` 的 `TOOLS`（末端法兰系，m）——箭头、界面读数自动跟着变；颜色 = 每项的 `rgba`；默认粗细 = `TOOL_ARROW_R_M`（界面「箭头粗细」也能调）；要先看数就跑 `python check_model.py` 的「三个工具的 TCP」那一段 |
| 三根箭头**怎么画**（平行 / 起点 / 长度） | `arm_core.draw_tool_arrows()`：起点 = 安装点 `(vx, vy, 0)`、方向 = 法兰 +z（末端工具轴）、长度 = `vz` —— 所以**三根互相平行、且都平行于末端姿态**，箭头尖仍落在各自 TCP；自检里「三根箭头互相平行」那一项盯着它 |
| **任务信号**（6501 的 `motion: start/over`） | 端口 = 界面「任务信号」卡片的端口框 / `--task-port`（`robot_link.DEFAULT_TASK_PORT` = 6501）；解析在 `robot_link.parse_task_payload()`（宽容写法在 `MOTION_TRUE`/`MOTION_END_WORDS`）；接收线程 `robot_link.TaskListener`；联调用 `--task-listen` / `--task-demo` |
| **工具尖轨迹**（采样密度 / 点数上限 / 线粗细 / 颜色） | `revA1_spec.TRAIL_MIN_STEP_M`（默认 1.5 mm）、`TRAIL_MAX_POINTS`（6000）、`TRAIL_R_M`（2 mm）；颜色 = 该工具 `TOOLS[...]["rgba"]`；实现 `arm_core.Trail` / `draw_trail()` / `ArmSim.record_trails()`，界面接线在 `revA1_gui` 的 `_build_task()` / `task_tick()` |
| **初始姿态 / home**（= 拍摄点云时的姿态） | `revA1_spec.py` 的 `HOME_QPOS`（连带 `CAPTURE_POSE` / `TARGET_POSE` / `NEUTRAL_QPOS` / `WORK_POSE`），改完 `python convert_urdf_to_mjcf.py` 重生成场景（也可 `--home <6 个关节角(rad)>` 覆盖） |
| 点云自带的坐标系（o2e = 末端系） | 界面「点云」卡片的「坐标系」或 `--frame auto\|cam\|end`；判定逻辑在 `point_cloud.detect_cloud_frame()`（ROI 反投影命中率 + 文件声明） |
| IK 选哪个解 / 直线能不能换解支 | `arm_core.solve_ik_best()`（精度都够好时挑离初值最近的解）、`LINE_MAX_BRANCH_JUMP_DEG`（超过就判"这条直线走不过去"） |
| 点云颜色（近红 → 远紫） | `point_cloud.py` 的 `DEPTH_COLORS`（每带一个 rgba） |
| 点云和机械臂对不齐 | ① 先看「参考点」该选 TCP 还是法兰；② 再用卡片上的 `微调 Δx/Δy/Δz/绕 z`（= `--offset / --yaw`）；③ 改 `小车高`（= `--cart-height`） |
| 点云场景的地面 / 小车外形 | `point_cloud.build_scene()`（地面挪到 `-小车高度`、小车 `pc_cart` 两个 box，可改尺寸/颜色/位置） |
| 工具长度（TCP） | `revA1_spec.py` 的 `TOOL_TIP_LOCAL_Z`（换成自己夹爪的尺寸） |
| 伺服软硬 | `revA1_spec.py` 的 `GAINS`（kp/kv/forcerange） |
| 地面高度 | 默认自动量 `base_link` 网格最低点；要覆盖用 `--floor-z` |
| 场景布局（地面材质/相机/灯光/目标标记） | `assets/revA1_scene.xml`（生成物，可手改；重跑转换会覆盖） |
| 界面配色 / 控件样式 | `revA1_gui.py` 顶部的 `QSS`（一处改全局）+ `spin_buttons_qss()`（数字控件上/下按钮的 `+`/`−` 图标，`make_app()` 里拼上去） |
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

## 已知问题

### 笛卡尔直线：从「前倾」home 出发、工具轴要求竖直朝下时，会被判为「走不过去」

**现象**：`python revA1_gui.py --selftest` 的「直线规划 / 直线执行」5 项、`--ui-test` 的
「求解并沿直线运动后工具尖到位」「日志实测到位精度」2 项会 FAIL。

**复现**：目标 `(0.25, 0.10, 0.30)` + 工具轴 `(0,0,-1)`（竖直朝下）。此时
`ArmSim.plan_line()` 沿直线约走到第 9/21 个路点就顶到关节限位 / 需要换解支，
`plan.ok=False`、末端位置 `(0,0,0)`、到位误差 `nan`：

```python
p = sim.plan_line([0.25, 0.10, 0.30], (0, 0, -1))
# p.ok=False, p.n=9，reason="直线走到第 9/21 个路点就走不过去了（这一支到不了：顶到限位或要换姿态）"
```

**原因**：`HOME_QPOS` 已换成真机实测的「前倾」姿态（工具轴离竖直约 38°），工具轴本来就不
竖直朝下；强行要求它沿直线转到竖直朝下，这一支中途会顶到限位 —— `plan_line()` 的换解支保护
（`LINE_MAX_BRANCH_JUMP_DEG`）会**明确拒绝**（这是设计内行为，不是崩溃）。

**状态**：暂不处理。不影响真机跟随、点云、三个工具 TCP 向量、任务信号轨迹等功能。

**若要修**：要么放宽/改进 `arm_core.plan_line()` 的路点与换解支策略，要么调整自检里直线
目标点（选一个从当前 home 这一支沿直线可达、且工具轴不需要大幅转向的点）。

