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

# ------------------------------------------------------------------ 三个工具（TCP 向量）
# 每个工具在**末端法兰系**里的 TCP 向量[m]：坐标系 = ``ee_Link`` 的坐标系
# （原点 = 法兰中心，+z = 工具轴；也就是模型里的 ``ee_site``）。
# 来源：真机 TCP 标定（用户提供）。三个工具装在同一片法兰上，画面上每个工具画一根箭头
# （见 ``arm_core.draw_tool_arrows``）：
#   * 起点 = **安装点** = 法兰平面上的 ``(vx, vy, 0)``（三个工具是并排装的，侧偏就体现在这儿）；
#   * 方向 = **法兰 +z（工具轴）**，长度 = ``vz``（该工具的轴向长度）；
#   * 于是三根箭头**互相平行、都平行于末端姿态的工具轴**，箭头尖正好落在各自 TCP 上。
# 所以 v 的用途有两个：``v`` 本身 = TCP 位置（数值/读数/`tool_points`），
# ``(vx, vy)`` + ``vz`` = 那根平行箭头的摆放与长度。
#
# ⚠️ |v| 比 URDF 里的 ``TOOL_TIP_LOCAL_Z`` 长一个量级：42.7 mm 只是 ``ee_Link.STL``
# 这块法兰盘自己的长度，真机上装的是下面这几种长杆工具（29.5 / 24.2 / 24.4 cm）。
# 换工具、工具改版，就改这里的数（界面上的箭头/读数会自动跟着变）。
TOOLS: dict[str, dict] = {
    "夹爪": dict(v=(-0.002343855654993024, 0.00048717676677189833, 0.2947992870405087),
                 rgba=(0.20, 0.50, 1.00, 1.0)),        # 蓝
    "喷嘴1": dict(v=(0.04102235917199713, 0.04546610134883331, 0.23378347492299675),
                  rgba=(1.00, 0.45, 0.05, 1.0)),        # 橙
    "喷嘴2": dict(v=(0.05885311314786616, -0.00911954858245911, 0.23612713092268872),
                  rgba=(0.90, 0.15, 0.80, 1.0)),        # 品红
}
# 画面里箭头的默认杆半径[m]（箭头头部 = 2.2 倍杆半径）。三种颜色都避开了机械臂自身的
# 配色（青绿/藕荷/土黄/淡红），在点云场景里也看得清。
TOOL_ARROW_R_M = 0.0035

# ------------------------------------------------------------------ 工具尖轨迹（6501 任务信号驱动）
# 真机开始/结束作业时往 6501 广播 ``motion: start|stop``（见 ``robot_link.TaskListener``）；
# 界面收到 start 就把**勾上的**工具尖位置连成轨迹（随机械臂实时长），收到 stop 就停住。
# 下面三个数控制"记多密 / 留多久 / 画多粗"（界面「任务信号 → 轨迹」卡片）。
TRAIL_MIN_STEP_M = 0.0015    # 采样最小间距[m]：工具尖移动不到这个距离就不记点（防抖动、防爆点数）
TRAIL_MAX_POINTS = 6000      # 每根轨迹最多存多少点（超了丢最老的；6000 点 ≈ 9 m 路程）
TRAIL_R_M = 0.0020           # 轨迹线半径[m]（画成一串短胶囊）



def tool_vector(name: str) -> np.ndarray:
    """某个工具的 TCP 向量（末端法兰系，m）。名字不认识抛 ``KeyError``。"""
    return np.asarray(TOOLS[name]["v"], dtype=float)


def tool_names() -> list[str]:
    """三个工具的名字（顺序 = 上面 ``TOOLS`` 的书写顺序）。"""
    return list(TOOLS)

# ------------------------------------------------------------------ 关节轴修正（URDF 的坑）
# URDF 里 joint1..joint5 的 axis 都是 ``0 0 1``，**唯独 joint6 写成 0 0 -1** —— 也就是说
# URDF 的 J6 正方向与真机固件（状态接口读数）**相反**。这个错很隐蔽：
# 工具尖恰好落在 J6 轴上、工具轴方向又是 J6 的旋转轴，所以**位置/工具轴都查不出来**，
# 只有"绕工具轴的滚转"会镜像 —— 表现出来就是**法兰上装的东西方位左右颠倒**
# （本工程就是那三个工具：夹爪/喷嘴1/喷嘴2 的扇形朝向）。
#
# 实测（真机在线，两次不同姿态；比的是模型 FK 的 3×3 姿态与真机 pose 的 3×3 姿态之差，
# 误差轴正好是末端 z 轴 = 纯滚转。修复前见 runs/j6_before_fix.log，修复后 runs/probe_after_fix.log）：
#   真机 J6 = −107.779° → 完整姿态差 144.443° ｜ 360° − 2·|J6| = 144.442° ✓
#   真机 J6 = −100.550°（拍摄姿态）→ 158.892° ｜ 360° − 2·|J6| = 158.900° ✓
# 两次都精确满足 ``q6_model = −q6_robot``（残差 < 0.01°），所以修正就是**把 joint6 的轴翻正**：
# 之后"真机关节角 = 模型关节角"（同号同零位，不需要 signs/offsets）；修完再量 = 0.000° / 0.010°。
# 改完记得重跑 ``convert_urdf_to_mjcf.py`` 重新生成模型；
# ``robot_link.py --probe`` 现在会把"完整姿态差"也打出来，真机在任何姿态下都应 ≈0.00°。
JOINT_AXIS_FIX: dict[str, tuple[float, float, float]] = {"joint6": (0.0, 0.0, 1.0)}

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
