# 在 MuJoCo 中查看 Synria (Alicia) 机械臂

## 结论：不需要任何转换

`Synria-Robot-Descriptions-main\synriard\mjcf\` 下**已经自带** MJCF（`.xml`），
而且每个 XML 的 `<compiler meshdir="../../meshes/...">` 都是相对路径，指回同级的
`synriard\meshes\`，所以直接把 XML 丢给 MuJoCo 就能加载，无需 URDF→MJCF 转换。

已验证：20 个 MJCF 全部能被 MuJoCo 3.13.0 正常加载（见 `synria_check_models.py`）。

## 最快方式（一键脚本）

```powershell
cd E:\deepenv\mujoco

.\synria_sim.ps1 -List                                          # 列出全部可用模型
.\synria_sim.ps1                                                # 打开默认：Alicia_D_v5_6_gripper_100mm
.\synria_sim.ps1 -Model Alicia_D_v5_6_gripper_50mm              # 50mm 夹爪版
.\synria_sim.ps1 -Model Alicia_M_v1_2_follower                  # Alicia-M v1.2
.\synria_sim.ps1 -Model Alicia_D_v5_6_gripper_100mm -Viewer python   # 用 lerobot 环境的 python viewer
.\synria_sim.ps1 -Model Alicia_M_v1_2_follower -Viewer studio   # 用 MuJoCo Studio
```

也可以直接双击 `synria_sim.bat`（等价于打开默认模型，脚本内部已用 `-ExecutionPolicy Bypass`，不受策略限制）。

如果直接运行 `.ps1` 报“禁止运行脚本”，用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\synria_sim.ps1 -List
```

`-Viewer` 三种取值：

| 取值 | 使用的程序 | 说明 |
| --- | --- | --- |
| `simulate`（默认） | `bin\simulate.exe` (MuJoCo 3.12.0) | 经典官方模拟器 |
| `studio` | `bin\mujoco_studio.exe` | 新版 Studio GUI，带可视化编辑 |
| `python` | `conda run -n lerobot python -m mujoco.viewer` | 走 Python 绑定（mujoco 3.13.0） |

## 手动命令（等价）

```powershell
# 1) 官方 simulate
E:\deepenv\mujoco\bin\simulate.exe `
  "E:\deepenv\mujoco\Synria-Robot-Descriptions-main\synriard\mjcf\Alicia_D_v5_6\Alicia_D_v5_6_gripper_100mm.xml"

# 2) lerobot 环境里的 Python viewer
conda run --no-capture-output -n lerobot python -m mujoco.viewer `
  --mjcf="E:\deepenv\mujoco\Synria-Robot-Descriptions-main\synriard\mjcf\Alicia_D_v5_6\Alicia_D_v5_6_gripper_100mm.xml"
```

Python 里直接加载：

```python
import mujoco

XML = r"E:\deepenv\mujoco\Synria-Robot-Descriptions-main\synriard\mjcf\Alicia_D_v5_6\Alicia_D_v5_6_gripper_100mm.xml"
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
print(model.nq, model.nu)  # 8 8  -> 6 关节 + 2 个夹爪滑轨，带 8 个 actuator
```

## 可用的 Alicia 模型

| 分组 | 文件（`synriard/mjcf/<分组>/`） | nq |
| --- | --- | --- |
| Alicia_D_v5_6 | `Alicia_D_v5_6_gripper_100mm` | 8 |
| Alicia_D_v5_6 | `Alicia_D_v5_6_gripper_50mm` | 8 |
| Alicia_D_v5_6 | `Alicia_D_v5_6_vertical_50mm` | 8 |
| Alicia_D_v5_6 | `Alicia_D_v5_6_leader` / `_leader_arx` / `_leader_ur` | 6 |
| Alicia_D_v6_1_2 | `Alicia_D_v6_1_2_leader_ffb` | 6 |
| Alicia_M_v1_1 | `Alicia_M_v1_1_follower` / `_follower_interactive` | 8 |
| Alicia_M_v1_1 | `Alicia_M_v1_1_vertical` | 8 |
| Alicia_M_v1_1 | `Alicia_M_v1_1_bi_vertical_interactive`（双臂） | 16 |
| Alicia_M_v1_2 | `Alicia_M_v1_2_follower` / `_follower_ARX` / `_follower_ARX_PX6D` / `_follower_ARX_deprecated` | 8 |

另有 `Bessica_D_v1_1`、`Bessica_M_v1_0`、`Corina_v1_2`。

## 注意事项

1. **Alicia_D_v5_5 没有现成 MJCF**：该版本只有 URDF。好在美国 MuJoCo 能直接读 URDF，
   `mujoco.MjModel.from_xml_path("...Alicia_D_v5_5_gripper_100mm.urdf")` 可正常加载
   （实测 nq=8、ngeom=19），只是 URDF 导入的碰撞体是凸包、且没有 actuator，不如 MJCF 完善。
2. **3 个 URDF 有路径 bug**（mesh 写成 `../../meshes/meshes/...`，多了一层 `meshes/`）：
   `Alicia_M_v1_0_follower.urdf`、`Alicia_M_v1_1_vertical.urdf`、`Bessica_D_v1_1_covered.urdf`。
   它们对应的 MJCF 版本是好的，请直接用 MJCF。
3. `simulate.exe` 读 `bin\.mujoco.ini` 保存窗口布局，和上面的模型加载无关。
4. 目前环境：conda 环境 `lerobot`（`D:\Anaconda\envs\lerobot`）里是 mujoco 3.13.0；
   目录里预编译的 GUI 是 MuJoCo 3.12.0。两者加载同一批 XML 均正常。
