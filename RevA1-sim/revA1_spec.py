"""TB6-R5-RevA1 的"机型参数表"——转换脚本 / 自检脚本 / 演示脚本共用的唯一事实来源。

数值来源都写在注释里，方便以后换 URDF 或接真机时按实际数据替换。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# ------------------------------------------------------------------ 输入 / 输出
NAME = "revA1"                       # 生成文件前缀：assets/revA1_arm.xml / revA1_scene.xml
# URDF / mesh 的来源：**优先用本目录里自带的那份**（RevA1-sim/TB6-R5-RevA1，
# 于是整个 RevA1-sim 目录可以独立拷到别的机器上跑），找不到才回退到仓库根的 RobotSDK。
_LOCAL_PKG = HERE / "TB6-R5-RevA1"
PACKAGE_ROOT = _LOCAL_PKG if (_LOCAL_PKG / "urdf").is_dir() else (
    REPO_ROOT / "RobotSDK" / "TB6-R5-RevA1")
URDF = PACKAGE_ROOT / "urdf" / "7260501-000000-001 TB6-R5-RevA1-URDF.urdf"
# URDF 里 mesh 写的是 package://7260501-000000-001 TB6-R5-RevA1-URDF/meshes/x.STL，
# 但本机并没有这层"包名目录"（mesh 直接放在包根下的 meshes/ 里），所以这里给包根目录。
ASSETS = HERE / "assets"
ARM_XML = ASSETS / f"{NAME}_arm.xml"
SCENE_XML = ASSETS / f"{NAME}_scene.xml"

# ------------------------------------------------------------------ 关节 / 末端
JOINTS = [f"joint{i}" for i in range(1, 7)]
LIMITS = {                            # 来自 URDF（弧度）
    "joint1": (-3.1415, 3.1415),
    "joint2": (-3.1415, 3.1415),
    "joint3": (-2.8623, 2.8623),
    "joint4": (-3.1415, 3.1415),
    "joint5": (-3.1415, 3.1415),
    "joint6": (-3.1415, 3.1415),
}
# 工具尖端：ee_Link.STL 在自身坐标系里沿 +z 长到 42.7 mm（用 mesh 包围盒量出来的）。
# 接真机换成自己的夹爪/工具时，改这里即可。
TOOL_TIP_LOCAL_Z = 0.0427

SITES = {
    "ee_Link": (
        '<!-- J6 法兰中心（机械接口原点）：插工具/相机都用它 -->\n'
        '<site name="ee_site" pos="0 0 0" size="0.012" rgba="1 0.55 0.1 1" group="3"/>\n'
        '<!-- 工具尖端 / TCP：= ee_site 沿工具轴 +42.7mm -->\n'
        f'<site name="tool_site" pos="0 0 {TOOL_TIP_LOCAL_Z}" size="0.012" '
        'rgba="0.2 1 0.3 1" group="3"/>'
    ),
}

# ------------------------------------------------------------------ 位置伺服增益
# 为什么这么"猛"：整条手臂 ~90kg，J2 要撑住 18kg 的大臂 + 20kg 的小臂 + 腕部，
# 满伸姿态实测重力力矩 ~310 Nm；kp 太小会稳态偏差几度甚至直接塌下来。
# 稳态偏差 ≈ 重力力矩 / kp：J2 在 100 Nm 下想压到 0.3° 就需要 kp ≈ 20000。
# forcerange 是按各关节受力大小估的（不是伺服手册值），够撑住自重还有余量。
GAINS: dict[str, dict] = {
    "joint1": dict(kp=4000, kv=300, forcerange=300),
    "joint2": dict(kp=20000, kv=1000, forcerange=600),
    "joint3": dict(kp=12000, kv=600, forcerange=400),
    "joint4": dict(kp=4000, kv=200, forcerange=120),
    "joint5": dict(kp=3000, kv=150, forcerange=100),
    "joint6": dict(kp=1500, kv=80, forcerange=60),
}

# ------------------------------------------------------------------ 世界 / 场景
# base_link.STL 的 z 最小值 = -0.1712 m（底座安装面到底面），地面就铺在这个高度：
# 相当于机器人用地脚螺栓固定在地板上。convert 脚本会自动量这个值。
FLOOR_Z = -0.1712

# 拍摄姿态（真机 pose）：点云就是在这个姿态下拍的，现在也把它当作**初始姿态**。
# 格式 = 与真机 pose 一致：x y z [mm] + rx ry rz [deg]（**外旋 XYZ** 欧拉角）。
CAPTURE_POSE = (110.862, -5.474, 396.436, 196.64, 34.89, 111.77)

# home / ready 姿态（弧度）= **真机实测关节角**，就是上面这个拍摄姿态：
# 工具轴前倾 38.2°（离竖直）、工具尖离地 0.568 m、0 自碰撞、静置漂移 < 0.1°、限位余量 ≥ 42.8°。
# 关节角取自真机状态接口读数（见 runs/probe.log：模型 FK 复核工具尖差 0.50 mm、工具轴差 0.00°）；
# 独立验算：IK 多分支搜索里同一分支的解与它只差 0.04°（见 convert_urdf_to_mjcf.py --home-from-pose）。
# ⚠️ 改完记得重跑 convert_urdf_to_mjcf.py 和 check_model.py。
TARGET_POSE = (0.110860, -0.005473, 0.395935)  # = home 下 tool_site 的世界系位置（模型 FK，离地 0.567 m）
HOME_QPOS = (1.72836, -2.39037, 2.11474, 1.19617, 1.72995, -1.75492)

# 世界系**名义作业点**：地面上的绿点标记 + 笛卡尔 demo 的路径中心（工具轴竖直朝下时的常用作业区，
# 与 home 无关 —— 换初始姿态不该把 demo 的路径一起搬走）。
WORK_POSE = (0.40, 0.0, 0.179)

# 工具轴**竖直朝下**的中性姿态（弧度）：多初值 IK 的备用初值。
# 以前它就是 home；home 换成真机实测的前倾姿态后，"让工具轴朝下"这类目标（demo / 自检里全是）
# 从它出发一次就能解出来，所以留作种子用。数值 = 历史上那组 home：
# 对 (0.40, 0, 0.179) 求逆解得到的（位置误差 0.06 mm、姿态误差 0.00°），静置漂移 < 0.3°。
NEUTRAL_QPOS = (0.3152, -1.6288, -2.3803, -0.7033, 1.5708, -2.3152)


# ------------------------------------------------------------------ 位姿约定
# 真机 pose / 拍摄姿态统一是 [x, y, z, rx, ry, rz]：位置 mm、角度 deg，旋转部分为
# **外旋 XYZ 欧拉角**（R = Rz(rz) @ Ry(ry) @ Rx(rx)）。全仓只在这里实现一次。
def euler_xyz_matrix(rx_rad: float, ry_rad: float, rz_rad: float) -> np.ndarray:
    """外旋 XYZ 欧拉角（弧度）→ 旋转矩阵。"""
    cx, sx = np.cos(rx_rad), np.sin(rx_rad)
    cy, sy = np.cos(ry_rad), np.sin(ry_rad)
    cz, sz = np.cos(rz_rad), np.sin(rz_rad)
    return (np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            @ np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]))


def pose_matrix(pose) -> tuple[np.ndarray, np.ndarray]:
    """``[x,y,z(mm), rx,ry,rz(deg)]`` → ``(R 3×3, p 3[m])``。"""
    v = [float(x) for x in np.asarray(pose, dtype=float).ravel()]
    if len(v) != 6:
        raise ValueError(f"位姿要 6 个数（x y z mm + rx ry rz deg）：{pose!r}")
    return euler_xyz_matrix(*np.radians(v[3:6])), np.asarray(v[:3], dtype=float) / 1000.0


def pose_tool_axis(pose) -> np.ndarray:
    """位姿的工具轴方向（旋转矩阵第三列 = 末端 +z 在世界系的方向）。"""
    return pose_matrix(pose)[0][:, 2]
