"""TB6-R5-RevA1 的"机型参数表"——转换脚本 / 自检脚本 / 演示脚本共用的唯一事实来源。

数值来源都写在注释里，方便以后换 URDF 或接真机时按实际数据替换。
"""

from __future__ import annotations

from pathlib import Path

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

# home / ready 姿态（弧度）：工具轴**竖直朝下**、无自碰撞、重力力矩小（J2 70 Nm / J3 56 Nm）。
# 由 sim_ik.py 对目标点 (0.40, 0, 0.179) 求逆解得到（位置误差 0.06mm、姿态误差 0.00°），
# 再用 check_model.py 复核过静置漂移 < 0.3°、雅可比条件数 8.8。
# ⚠️ 改完记得重跑 convert_urdf_to_mjcf.py 和 check_model.py。
TARGET_POSE = (0.40, 0.0, 0.179)     # 世界系名义作业点 = home 下 tool_site 的位置（场景里的绿点标记）
HOME_QPOS = (0.3152, -1.6288, -2.3803, -0.7033, 1.5708, -2.3152)
