# Alicia-D × MuJoCo 桌面操作仿真 / 遥操作

本仓库包含两个相互独立、共享同一套 Alicia 机器人模型的工程：

| 目录 | 内容 | 说明 |
| --- | --- | --- |
| [`alicia_libero/`](alicia_libero/README.md) | **LIBERO 桌面操作仿真台**（PySide6 界面） | 9 个桌面任务；鼠标拖拽 IK、键盘虚拟示教臂、关节滑块、**示教臂（手摇跟随：真机 leader → 仿真 follower，见 `leader_arm.py`）**；每帧实时判分 |
| [`alicia_teleop/`](alicia_teleop/README.md) | **虚拟示教臂 / 遥操作** | `alicia_virtual_teleop.py`（真机 leader ↔ MuJoCo follower 同步 + 相机 + CSV 录制）、`make_follower_xml.py`（给原始 MJCF 补 actuator） |
| [`RevA1-sim/`](RevA1-sim/README.md) | **TB6-R5-RevA1 六轴工业臂仿真 + 交互控制台**（PySide6 界面，自带 URDF/mesh，可独立拷走） | 关节 `−`/`+` 微调、**真机跟随（真机 UDP 6001 广播 / HTTP 状态 → 模型实时同步动，见 `robot_link.py`）**、**笛卡尔直线点到点**、多初值 IK（可解/不可解 + 误差 + 目标点红绿标记）、示教点位；`viewer.py` / `demo_trajectory.py` 可视化与轨迹演示 |

## 1. 快速开始

```powershell
conda activate lerobot
pip install mujoco numpy PySide6==6.7.3

# ① LIBERO 桌面操作仿真台
cd alicia_libero
python libero_catalog.py --table       # 扫描素材，生成 assets_catalog.json（首次必做）
python alicia_libero_app.py            # 或双击 run_app.bat

# ② 虚拟示教臂
cd ..\alicia_teleop
python alicia_virtual_teleop.py --help
```

> ⚠ 本机实测 PySide6 6.11 的 `Qt6Core.dll` 加载失败（WinError 127），**装 6.7.3 正常**。

## 2. 外部依赖（不在本仓库里，需自行准备）

仓库只收录代码与文档，下面这些第三方目录按需自行下载，**放在本仓库同级目录**（代码里用的是相对路径）：

| 目录 | 体积 | 用途 | 缺失后果 |
| --- | --- | --- | --- |
| `Datasets/libero_assets/` | 约 403 MB | LIBERO 官方物体素材（60 个物体） | `libero_catalog.py` 无法分析素材、物体无法渲染 |
| `Synria-Robot-Descriptions-main/` | 约 494 MB | Alicia 机械臂 MJCF + 网格（本工程用 `synriard/mjcf/Alicia_D_v5_6/` 与 `synriard/meshes/Alicia_D_v5_6/follower_standard/`） | 无法生成/加载任何场景 |

`alicia_libero/mesh_cache/`、`alicia_libero/scenes/`、`alicia_libero/preview/` 都是**脚本自动生成**的（分别由 `libero_catalog.py`、`build_all_scenes.py`、`tests/render_preview.py` 产出），所以也不入库。

## 3. 测试

```powershell
cd alicia_libero
python tests\test_gui_smoke.py          # 界面冒烟：切任务/拖拽 IK/键鼠/滑块/复位/相机（20 项）
python tests\test_camera_visibility.py  # 各机位可见性（防止"只看到地板"）
python tests\test_visual_alignment.py   # 物体视觉网格与碰撞盒对齐
python build_all_scenes.py              # 9 个场景生成回归（物体应精确落在桌面）
```

## 4. 当前状态

- ✅ 素材分析、场景拼装、手眼标定式的 6 轴 IK、手动操作界面、实时判分均可用
- 🚧 **动作规划（自动接近/夹取/搬运/释放）已整体移除，等待重写**；
  界面目前只提供手动操作，详见 [`alicia_libero/README.md`](alicia_libero/README.md) 第 8 节
