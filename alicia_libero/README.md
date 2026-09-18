# Alicia-D × LIBERO 桌面操作仿真台（PySide6）

用 **Alicia_D_v5_6_gripper_50mm** 机械臂在 MuJoCo 里做桌面操作任务：
界面用 **PySide6**，任务素材来自 ``Datasets/libero_assets``（LIBERO 官方素材，402 MB / 60 个物体），
操作方式支持 **鼠标拖拽（IK 跟随）**、**键盘虚拟示教臂**、**关节滑块**。

> ⚠ **动作规划已整体删除（等待重写）**：自动接近/夹取/搬运/释放、辅助按钮 ①②③④⑤、「▶ 执行动作计划」都不再存在；当前只保留 **手动操作 + 每帧实时判分**。

```
e:\deepenv\mujoco\alicia_libero\
├── libero_catalog.py      # 素材分析器：扫描 libero_assets → assets_catalog.json（尺寸/碰撞盒/可夹性/姿态）
├── assets_catalog.json    # 自动生成：60 个物体的分析结果
├── libero_scene.py        # 场景拼装：Alicia 手臂 + LIBERO 标准桌面 + 物体 → MJCF
├── libero_tasks.py        # 9 个任务定义（布局 + 成功判据）与判分逻辑
├── alicia_ik.py           # 正/逆运动学（阻尼最小二乘 + 零空间姿态优化）、夹爪映射
├── alicia_libero_app.py   # PySide6 主程序（3D 视图 + 控制面板）
├── build_all_scenes.py    # 一键生成全部任务场景（兼作回归测试）
├── run_app.bat            # Windows 启动器（也支持 catalog / scenes 子命令）
├── scenes/                # 生成的任务场景 XML（9 个）
├── preview/               # tests/render_preview.py 生成的各机位 PNG（肉眼检查用）
└── tests/
    ├── test_gui_smoke.py         # 界面冒烟测试（切任务/拖拽/按键/滑块/复位/相机 + 规划已删除确认）
    ├── test_camera_visibility.py # 相机可见性（分割渲染数机械臂像素，防"只看到地板"）
    ├── test_visual_alignment.py  # 物体与桌面/相机的对齐检查
    <!-- 规划相关测试已随规划移除，备份见 ..\_archive_motion_plan_20260918\tests\ -->
    └── render_preview.py         # 把各机位画面存成 PNG 便于肉眼检查
```

## 1. 快速开始

```powershell
conda activate lerobot
cd E:\deepenv\mujoco\alicia_libero

python libero_catalog.py --table          # ① 分析素材（生成 assets_catalog.json，可看尺寸表）
python alicia_libero_app.py               # ② 启动界面
```

或者直接双击 `run_app.bat`（等价于上面第 ②步；`run_app.bat catalog` / `run_app.bat scenes` 分别是 ① 和场景生成）。

> 依赖：`mujoco 3.13`、`numpy`、`PySide6`。
> ⚠ 本机实测 **PySide6 6.11 的 Qt6Core.dll 加载失败**（WinError 127），换成 **6.7.3** 后正常：
> `pip install "PySide6==6.7.3"`。

## 2. 素材分析结论（回答"能构建几个任务"）

`libero_catalog.py` 把 ``libero_assets`` 全部扫了一遍，得到 **60 个可用物体**（另外的场景/贴图不计）：

| 分组 | 数量 | 代表物体 |
| --- | --- | --- |
| 餐具/容器 | 10 | plate（138×138×19）、akita_black_bowl（107mm）、篮子和木托盘、平底锅 |
| 食品瓶罐 | 14 | ketchup（56×37×146）、new_salad_dressing、popcorn、butter、chocolate_pudding |
| 日用杂项 | 17 | porcelain_mug、wine_bottle、black_book、wooden_tray、desk_caddy |
| 可动家具 | 17 | microwave 门/抽屉等（本工程未用于任务，可动部件需另外建模） |
| 场景道具 | 2 | desk、living_room_table |

分析器对每个物体做四件事（都能在 UI 里看到）：

1. **碰撞几何**：直接沿用 LIBERO 预烘焙的**凸盒分解**（`<name>.xml` 里的 `type=box`），
   解析后得到物体自身坐标系下的 AABB → 用于精确摆放到桌面、判分
2. **视觉网格**：XML 指向的 `visual/*_vis.msh` 在本仓库里**并不存在**，按规则回退到同目录的 `.obj`
   （MuJoCo 不读 OBJ 的贴图，所以用 catalog 里的纯色材质代替）
3. **俯视抓取高度**：把物体沿 Z 切成 24 层，每层取所有相交碰撞盒的**联合水平轮廓**，
   轮廓较窄的那条边就是夹爪要张开的宽度 → 给出「最窄一层」的高度区间 + 该层的夹取宽度
4. **可夹判断**：50mm 夹爪的实际开口约 **45mm**，据此判断哪些物体能夹。
   实测 **36/60** 可夹；盘子（138mm）、碗（81~107mm）、马克杯（125mm）、篮子（170mm）
   这类"宽口"物体只能当**目标容器**，不能当抓取物

**实测物理规律（本工程最关键的三条，都写进了代码注释）**

| 规律 | 结论 | 依据 |
| --- | --- | --- |
| 夹爪闭合方向 | **必须沿物体的窄边闭合** | 番茄酱 56×37mm，若沿 56mm 侧闭合，50mm 开口夹不上，搬运途中掉落（掉到桌下 533mm 外） |
| 夹取高度 | **TCP（tool0_site）= 物体原点 + 窄带中点** | 物理扫描：番茄酱可用区间为「原点 +30~+60mm」，窄带中点 +40mm 正在区间中间 |
| 释放高度 | **TCP = 目标面 + 物体底面偏移 + 夹取偏移** | 深放会撞容器、浅放会挂在篮沿上，公式算出后 t1/t3/t4 的落点误差 <30mm |

## 3. 九个任务

| # | 任务 | 类型 | 抓取物（最窄处） | 目标 | 难度 |
| --- | --- | --- | --- | --- | --- |
| 1 | 番茄酱放进篮子 | 放进容器 | ketchup（37mm） | 篮子内 | ★ |
| 2 | 黄油放到盘子中间 | 放到平面 | butter（17mm） | 盘子 | ★ |
| 3 | 布丁盒放进小碟 | 放进容器 | chocolate_pudding（27mm） | 小碟内 | ★★ |
| 4 | 爆米花盒放到盘子上 | 放到平面 | popcorn（20mm） | 盘子 | ★★ |
| 5 | 细高瓶子搬进木托盘 | 放置（重心高） | new_salad_dressing（36mm） | 木托盘内 | ★★★ |
| 6 | 黑皮书推到盘子上 | **推/滑（不用夹）** | black_book（110~134mm，夹不住） | 盘子 | ★★ |
| 7 | 布丁盒叠到番茄酱罐顶 | 叠放 | chocolate_pudding | 罐顶面 62×76mm | ★★★ |
| 8 | 番茄酱放到盘子左侧标记区 | 空间指代 | ketchup | 桌面指定区域（半透明圆盘标记） | ★★ |
| 9 | 奶油奶酪盒从托盘取出放盘子 | 从容器取出 | cream_cheese（18mm） | 盘子 | ★★★ |

每个任务的判据都是**可量化**的（`libero_tasks.py` 里 `success` 字段）：

```python
{"xy": [0.22, 0.10], "xy_tol": 0.055,      # 抓取物 AABB 中心与目标的水平容差
 "z_ref": "table" | "target_top",           # 高度基准：桌面 / 目标物顶面
 "z_band": [0.0, 0.06]}                     # 抓取物“底面”相对基准的高度区间（判断是否放下而不是举着）
```

界面每帧都会算并显示：`水平偏差 30mm / 允许 55mm；底面相对基准 +17mm / 要求 [+0, +60]mm`。

## 4. 界面使用

左侧面板从上到下：

1. **任务下拉框**：切换任务 = 重新生成并加载 MuJoCo 场景（9 个任务）
   - 任务描述、操作提示、**素材分析摘要**（抓取物尺寸/最窄处/50mm 夹爪能否夹）
   - 实时判据（黄字）与完成状态（绿字）
2. **操作方式**：鼠标拖拽（IK 跟随）/ 键盘虚拟示教臂 / 关节滑块
   - 关节滑块（J1~J6）在"滑块模式"下可直接拖动；其他模式下作为实时回显
   - 夹爪滑块：0 = 闭合，100 = 张开
   - ~~辅助按钮 ① 对准夹取位 / ② 夹取 / ③ 对准放置位 / ④ 松开~~ —— 已随动作规划删除
3. **视图/场景**：相机切换（斜前方 / 正上方 / 侧前方 / 腕部相机）、重新开始（复位场景）

鼠标 / 键盘：

| 操作 | 作用 |
| --- | --- |
| 鼠标左键拖动 | 移动夹爪目标点（IK 实时求解，按当前视角的屏幕方向换算成世界位移） |
| 滚轮 | 升降夹爪（±12mm/格） |
| Shift + 滚轮 | 夹爪开合 |
| `1`..`6` | 选择要微调的关节 |
| `,` / `.`（或 ←/→） | 选中关节 ∓2° |
| `O` / `C` | 夹爪张开 / 闭合 |
| `R` | 复位场景（物体回到初始位姿） |
| `T` | 切换相机 |

推荐的操作流程（以任务 1 为例）：

```
选中任务 → 鼠标拖动（或滚轮 / 关节滑块 / 键盘微调）把夹爪挪到物体旁并夹住
        → 拖到目标上方 → 张开夹爪放下 → 看左上判据变绿「任务完成」
```

## 5. 技术实现要点

**场景拼装**（`libero_scene.py`）：以 `Alicia_D_v5_6_gripper_50mm_teleop.xml`（位置伺服版）为骨架，
用 ElementTree 把地面、LIBERO 标准桌面、物体、相机、灯光插进它的 worldbody，
保留原有的 actuator 与 contact 排除；手臂的 `meshdir` 改成绝对路径，物体网格用相对该 `meshdir` 的
相对路径引用 —— 这样生成的场景 XML 放到任何目录都能加载。

**为什么必须先跑 `make_follower_xml.py`**：Robot-Descriptions 里的原始 MJCF
（`gripper_50mm.xml`）**没有任何 actuator**（nu=0），无法下发指令；
位置伺服版才能直接往 `data.ctrl` 写目标关节角。

**IK**（`alicia_ik.py`）：
* 从模型零位姿态反推「接近轴 / 夹紧轴在 tool 帧下的方向」（不能想当然认为某个轴就是接近轴，
  实测 tool 帧 Y 才是夹紧轴、Z 是接近轴）
* **两阶段求解**：先纯位置收敛（位置优先），再在**位置雅可比的零空间**里调整姿态
  —— 这样姿态能改、位置严格不掉（桌面网格可达率 93%，位置误差 ~0.0mm）
* 多种子重试：从"上一帧解"做种子会陷局部极小（实测差 46mm），失败时用零位种子重试
* 姿态约束是**软**的：强行要求"严格竖直向下"会把 J5 顶到 ±90° 限位，远处目标直接解不出来

**渲染**：`mujoco.Renderer` 离屏渲染 → `QImage` → `QLabel`（不另开 MuJoCo 窗口），
主循环 `QTimer(16ms)` 驱动：IK/控制 → `mj_step`×8（dt=2ms，≈1 帧 1/60s）→ 渲染 → 判分。

## 6. 实测结果

| 验证项 | 结果 |
| --- | --- |
| 素材分析 | 60 个物体解析成功，36 个判定为 50mm 夹爪可夹 |
| 场景生成 | 9 个场景全部可加载，物体静置后**底面精确落在 800.0mm**（桌面）|
| 界面自动化测试 | `python tests/test_gui_smoke.py` → **20/20 通过**（切任务、拖拽 IK、滚轮、键鼠、滑块、复位、切相机 + 4 项「规划已删除」确认），0 异常 |
| 相机可见性 | `python tests/test_camera_visibility.py` → 4 个机位都可见：机械臂占 **7.7% / 4.7% / 9.3%** 像素，工作区占 40~53% |
| 任务初始状态 | 9/9 都正确地判为"未完成" |


## 7. 踩坑记录（都已在代码里注释）

| 坑 | 现象 | 处理 |
| --- | --- | --- |
| LIBERO 物体 XML 的 `quat` **未归一化** | 如 butter 写成 `0.0053 0 0.0053 0`（模长 0.0075），旋转矩阵退化、AABB 算成 0 | 解析与写场景时统一归一化 |
| 物体 XML 指向的 `visual/*_vis.msh` 缺失 | MuJoCo 报找不到文件 | 回退到同目录 `.obj`（按 `_vis.msh → .obj` 规则）|
| 夹爪"张开度"映射反了 | `opening=1` 实际把手指推到 0.025（**闭合**）→ 想夹紧却全张开，抓取全失败 | `finger_targets()` 改成 `(1-opening)*travel` |
| 夹爪闭合方向没对齐窄边 | 番茄酱搬到一半掉落，飞到桌下 | 按 catalog 的 AABB 较短边自动选闭合方向（yaw 0 或 90°）|
| 判据 z 下界太严 | 物体静止时底面 -0.4mm 被判失败 | 判据留 2mm 弹性 |
| **相机 `xyaxes` 手写错方向** | 界面里**只看得到地板**：`cam_front` 的 -Z 轴正好背对桌子、`cam_wrist` 朝天 | 改用 `look_at_xyaxes(位置, 看向点)` 自动计算；再加**分割渲染可见性测试**兜底 |
| `mujoco 3.13` 的 viewer 没有 `ctx` | 官方示例的 `mjr_overlay(viewer.ctx)` 直接 AttributeError | 改用 `Handle.set_texts()`（旧版走 mjr_overlay 分支）|
| `"" in "123456"` 恒为真 | 空文本按键走进数字分支，`int("")` 崩 | 先判 `if text and text in ...`，并从 keycode 兜底取字符 |
| 换任务后滑块与仿真不同步 | 滑块显示 0 而仿真以为全张开，按 C 无反应 | 加 `_sync_controls()` 同步控件与 session |

## 8. 动作规划：已整体删除（等待重写）

原来这一节记录的是任务 1 由几何反算出来的四个动作点（接近 0.9656 / 抓取 0.9122 /
搬运 1.0787 / 释放 0.9312）、`execute_plan()` 的动作时序，以及 `PLAN_DRIVEN_TASKS` 白名单机制。

这一整套已经从工程里删除，删掉的东西包括：

| 被删除的内容 | 原来在哪 |
| --- | --- |
| 动作点反算（接近/抓取/搬运/释放高度、容器剖面、抓取高度逐档实测） | `libero_plan.py` |
| 动作队列执行器（分段横移、按物体实际落点收位、滑脱重夹、两段慢开松手） | `libero_plan.py` |
| 辅助按钮 ① 对准夹取位 ② 夹取 ③ 对准放置位 ④ 松开 ⑤ 归位 | `alicia_libero_app.py` |
| 「▶ 执行动作计划」按钮、计划文案、计划驱动的多子种子 IK 分支 | `alicia_libero_app.py` |
| `SimSession.plan / plan_mode / script / grasp_status / home_ee / move_via_safe_height ...` | `alicia_libero_app.py` |
| 动作计划/任务播放测试与动作点清单 | `tests/test_task_plans.py`、`tests/test_task_playback.py`、`tests/dump_task_plans.py`、`MOTION_PLAN.md` |

**备份**（不在软件里，只是留档，重写时可以对照，也可直接删掉）：
`..\_archive_motion_plan_20260918\` —— 含 `libero_plan.py`、`MOTION_PLAN.md`、
改动前的 `alicia_libero_app.py.before_strip` 以及三个测试脚本。

现在的界面 = **手动操作 + 实时判分**。`tests/test_gui_smoke.py` 里有一组断言专门检查
"规划确实已经不在"：`libero_plan` 不可导入、session 没有 plan/script/grasp_status/home_ee、
没有 assisted_grasp/assisted_place/move_via_safe_height 等方法、界面上没有那些按钮。

## 9. 已删除规划的踩坑留档（重写时参考）

"把两个动作点合成一步斜着走会扫倒物体""IK 姿态权重太小会刮着物体走""抓瓶颈横移太快会甩脱"
"释放高度留 10mm 会弹""正运动学量手腕高度要注意 mesh scale 已含缩放""托盘内底会被侧壁盒误判成
0.8609""`from` 类任务的搬运高度必须看**出发**容器的口沿" —— 这些坑都属于**已被删除的动作规划**。

原文完整保留在 `..\_archive_motion_plan_20260918\MOTION_PLAN.md` 与
`..\_archive_motion_plan_20260918\libero_plan.py` 的注释里，本 README 不再复述，
以免和"当前代码里根本没有规划"这件事打架。

## 10. 常见问题与扩展

| 问题 | 解决 |
| --- | --- |
| 界面起不来 / `ImportError: DLL load failed` | 装 `PySide6==6.7.3`（本机 6.11 与系统 Qt 6.7.2 冲突）|
| 想换机型号/夹爪 | 改 `libero_scene.ARM_XML` 指向其它 `*_teleop.xml`（先用 `make_follower_xml.py` 生成）|
| 想加新任务 | 在 `libero_tasks.TASKS` 里加一条（`objects` 布局 + `success` 判据），UI 下拉框会自动出现 |
| 想换物体 | `objects[].key` 填 catalog 的 `分组/名称`（用 `python libero_catalog.py --table` 查看）|
| 想加可动家具（微波炉门等） | 素材里的 `articulated_objects` 需要自己建关节（本工程未做）|
| 想接真机示教臂 | 见上一个工程 `..\alicia_teleop\`（真机 → MuJoCo 虚拟遥操作），本工程的 `SimSession` 可直接复用其 `LeaderState` |
| 渲染慢 | `VIEW_W/VIEW_H` 调小；或关掉阴影（`renderer.update_scene(..., scene_option=...)`）|
| 画面里看不到机械臂/物体 | 先跑 `python tests/test_camera_visibility.py` 定位是哪个机位不对，再用 `tests/render_preview.py` 出图看；机位定义在 `libero_scene.default_cameras()`（用 `camera_spec(名, 位置, 看向点)` 写，别手写 `xyaxes`）|

