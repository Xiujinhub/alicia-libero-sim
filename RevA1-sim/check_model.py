#!/usr/bin/env python3
"""模型自检：编译场景 -> 打印结构 -> home 稳定性 -> 工作空间 -> IK 到位精度。

跑法::

    python check_model.py                 # 结果同时打印并写入 check_model.log
    python check_model.py --samples 50000 # 更细的工作空间采样

检查项都是"换了 URDF / 改了增益之后最容易踩的坑"：
  · 总质量对不对（URDF 里 root link 的惯量很容易被丢掉）
  · 静置时会不会漂移 / 抖（位置伺服 kp、armature、damping 没调好就会）
  · 静止姿态有没有虚假自碰撞（相邻连杆贴面）
  · 工具能不能按预期"指哪打哪"（IK + 控制器一起验证）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gl_backend import configure_gl

configure_gl()
import mujoco  # noqa: E402

import revA1_spec as spec  # noqa: E402
import sim_ik as ik  # noqa: E402
from arm_core import ik_seeds, solve_ik_pose  # noqa: E402  （多初值 IK，和界面用的是同一套）

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
OUT: list[str] = []
WARN: list[str] = []


def log(msg: str = "") -> None:
    OUT.append(msg)
    print(msg)


def warn(msg: str) -> None:
    WARN.append(msg)
    log(f"  [WARN] {msg}")


def jid(model, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))


def fmt(v, nd: int = 3) -> str:
    return np.array2string(np.round(np.asarray(v, dtype=float), nd))


# ============================================================================ 1
def dump_structure(model: mujoco.MjModel) -> None:
    log(f"mujoco 版本   : {mujoco.__version__}")
    log(f"模型          : {model.nq=} {model.nv=} {model.nu=} {model.nbody=} "
        f"{model.ngeom=} {model.nmesh=} {model.nsite=}")
    log(f"积分器        : timestep={model.opt.timestep} integrator={model.opt.integrator} "
        f"iterations={model.opt.iterations} cone={model.opt.cone}")

    log("\n--- 连杆质量 ---")
    total, urdf_sum = 0.0, 0.0
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        log(f"  [{i:2d}] {str(name):10s} mass={model.body_mass[i]:8.4f} kg "
            f"ipos={fmt(model.body_ipos[i])}")
        total += model.body_mass[i]
    log(f"  合计 {total:.3f} kg")
    log("  (URDF 里 base_link 5.0487 + Link1 13.203 + Link2 46.802 + Link3 20.24"
        " + Link4 4.6142 + Link5 4.7679 + ee_Link 1.1773 = 95.853 kg)")
    if abs(total - 95.853) > 0.05:
        warn(f"总质量 {total:.3f} kg 与 URDF 求和 95.853 kg 不一致")

    log("\n--- 关节（世界系轴向取 qpos=0 姿态）---")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for j in spec.JOINTS:
        i = jid(model, j)
        lo, hi = model.jnt_range[i]
        log(f"  {j:8s} qpos[{int(model.jnt_qposadr[i])}] dof[{int(model.jnt_dofadr[i])}] "
            f"range=[{lo:+.4f},{hi:+.4f}] rad  damping={model.dof_damping[int(model.jnt_dofadr[i])]:g} "
            f"armature={model.dof_armature[int(model.jnt_dofadr[i])]:g} "
            f"世界轴@0={fmt(data.xaxis[i])} 锚点={fmt(data.xanchor[i])}")

    log("\n--- 驱动器（位置伺服）---")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        log(f"  {name:14s} kp={model.actuator_gainprm[i][0]:8.1f} "
            f"kv={-model.actuator_biasprm[i][2]:8.1f} "
            f"ctrlrange={fmt(model.actuator_ctrlrange[i], 4)} "
            f"forcerange={fmt(model.actuator_forcerange[i], 1)}")

    log("\n--- site / camera ---")
    for i in range(model.nsite):
        log(f"  site  : {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")
    for i in range(model.ncam):
        log(f"  camera: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)}")


# ============================================================================ 2
def check_home(model: mujoco.MjModel, data: mujoco.MjData, floor_z: float) -> None:
    log("\n--- home 姿态（keyframe 'home'）---")
    if not ik.reset_home(model, data):
        warn("场景里没有 home keyframe，用全 0 姿态代替")
        mujoco.mj_forward(model, data)
    q = ik.arm_qpos(model, data)
    log(f"  qpos      = {fmt(q, 4)} rad  (度: {fmt(np.degrees(q), 1)})")
    log(f"  ee_site   = {fmt(ik.site_pos(model, data, 'ee_site'), 4)}")
    log(f"  tool_site = {fmt(ik.site_pos(model, data, 'tool_site'), 4)}  "
        f"离地 {ik.site_pos(model, data, 'tool_site')[2] - floor_z:.3f} m")
    log(f"  工具轴    = {fmt(ik.tool_axis(model, data, 'tool_site'))}")
    expect_axis = spec.pose_tool_axis(spec.CAPTURE_POSE)
    got_axis = ik.tool_axis(model, data, "tool_site")
    dev = float(np.degrees(np.arccos(np.clip(np.dot(expect_axis, got_axis), -1, 1))))
    log(f"  期望工具轴= {fmt(expect_axis)}（spec.CAPTURE_POSE）偏差 {dev:.3f}°")
    # ⚠️ 只看工具尖 + 工具轴**不够**：它们对 J6 的符号/零位不敏感（工具尖恰在 J6 轴上、
    #    工具轴又是 J6 的旋转轴）。这里把完整 3×3 姿态也对一遍 —— 当年 URDF 把 joint6 的轴
    #    写成 0 0 -1（与真机固件相反）就是这一项抓出来的（差值 158.9°，而前两项全过）。
    R_home = np.array(data.site_xmat[ik.site_id(model, "ee_site")], dtype=float).reshape(3, 3)
    R_cap = spec.pose_matrix(spec.CAPTURE_POSE)[0]
    full = float(np.degrees(np.arccos(np.clip(
        (np.trace(R_home.T @ R_cap) - 1.0) / 2.0, -1.0, 1.0))))
    log(f"  完整姿态偏差= {full:.4f}°（含绕工具轴滚转；与 spec.CAPTURE_POSE 的 3×3 比）")
    if full > 1.0:
        warn(f"home 与 CAPTURE_POSE 的完整姿态差 {full:.2f}°：多半有某个关节的轴/零位写反了"
             "（J6 这类错在工具尖/工具轴上都看不出来）→ 看 revA1_spec.JOINT_AXIS_FIX")
    log(f"  重力力矩  = {fmt(data.qfrc_bias[:6], 1)} Nm")
    log(f"  自碰撞对数= {data.ncon}")

    # 雅可比条件数：判断这个姿态离奇异有多近（越大越容易"卡住"）
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr,
                      ik.site_id(model, 'tool_site'))
    J = np.vstack([jacp[:, :6], jacr[:, :6]])
    sv = np.linalg.svd(J, compute_uv=False)
    log(f"  tool_site 雅可比奇异值 = {fmt(sv, 4)}  条件数 = {sv[0] / sv[-1]:.1f}")
    if sv[-1] < 0.02:
        warn(f"home 姿态接近奇异（最小奇异值 {sv[-1]:.4f}），笛卡尔运动可能解不开")

    # 奇异位形：这套腕是"偏置腕"（J4 与 J6 轴线在 q5=0 时平行且不相交），
    # 于是 q5≈0 或 ±180° 时腕部丢一个自由度，笛卡尔运动到那里会解不动。
    log("  奇异位形探测（只改 joint5，看雅可比最小奇异值）:")
    for q5 in (-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi):
        ik.set_arm_qpos(model, data, q)
        data.qpos[ik.qpos_adr(model, 'joint5')] = q5
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, ik.site_id(model, 'tool_site'))
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        mn = np.linalg.svd(J, compute_uv=False)[-1]
        log(f"    joint5 = {np.degrees(q5):+7.1f}° -> 最小奇异值 {mn:.4f}"
            f"{'   <-- 奇异!' if mn < 0.02 else ''}")
    ik.set_arm_qpos(model, data, q)
    mujoco.mj_forward(model, data)
    if data.ncon:
        warn("home 姿态有自碰撞接触，检查 <contact><exclude> 是否漏了连杆对")

    # 满伸姿态能撑住吗（worst case 重力力矩 vs 驱动器出力上限）
    log("\n--- 最坏情况：重力力矩 vs 驱动器出力 ---")
    worst = np.zeros(6)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        ik.set_arm_qpos(model, data, rng.uniform(model.jnt_range[:6, 0],
                                                 model.jnt_range[:6, 1]))
        mujoco.mj_forward(model, data)
        worst = np.maximum(worst, np.abs(data.qfrc_bias[:6]))
    for k, j in enumerate(spec.JOINTS):
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}")
        lim = model.actuator_forcerange[act][1]
        log(f"  {j:8s} 最大重力力矩 {worst[k]:7.1f} Nm / forcerange {lim:6.1f} Nm"
            f"  {'OK' if lim >= worst[k] else '<-- 撑不住!'}")
        if lim < worst[k]:
            warn(f"{j} 的 forcerange 小于最坏重力力矩，满伸姿态会塌")


# ============================================================================ 3
def check_stability(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    log("\n--- 稳定性：home 姿态静置 3 s（看漂移/振动）---")
    if not ik.reset_home(model, data):
        mujoco.mj_forward(model, data)
    q0 = ik.arm_qpos(model, data)
    n = int(3.0 / model.opt.timestep)
    hist = np.zeros((n, 6))
    for k in range(n):
        mujoco.mj_step(model, data)
        hist[k] = ik.arm_qpos(model, data)
    drift = np.degrees(np.abs(hist[-1] - q0))
    peak = np.degrees(np.abs(hist - q0).max(axis=0))
    log(f"  末态偏差(度) = {fmt(drift, 3)}")
    log(f"  过程峰值(度) = {fmt(peak, 3)}   自碰撞对数 = {data.ncon}")
    if drift.max() > 0.5:
        warn(f"静置漂移 {drift.max():.2f}° > 0.5°，建议加大 kp 或减小负载")
    if peak.max() > 2.0:
        warn(f"过程中抖了 {peak.max():.2f}°，检查 kv/armature/damping")


# ============================================================================ 3b
def check_tools(model: mujoco.MjModel, data: mujoco.MjData, floor_z: float) -> None:
    """三个工具（夹爪 / 喷嘴1 / 喷嘴2）的 TCP 向量：在 home 姿态下算一遍世界坐标。

    TCP 向量定义在 ``revA1_spec.TOOLS``（末端法兰系，m），画面上的箭头 = 法兰原点 + R·v。
    """
    log("\n--- 三个工具的 TCP（home 姿态，末端法兰系向量 → 世界坐标）---")
    if not ik.reset_home(model, data):
        mujoco.mj_forward(model, data)
    sid = ik.site_id(model, "ee_site")
    p = np.array(data.site_xpos[sid], dtype=float)
    R = np.array(data.site_xmat[sid], dtype=float).reshape(3, 3)
    log(f"  法兰原点 ee_site = {fmt(p, 4)} m   （|RᵀR−I| = "
        f"{float(np.abs(R.T @ R - np.eye(3)).max()):.1e}）")
    pts = {}
    for name in spec.tool_names():
        v = spec.tool_vector(name)
        q = p + R @ v
        pts[name] = q
        log(f"  {name:5s} v = {fmt(v, 4)} m  |v| = {float(np.linalg.norm(v)) * 1000:6.1f} mm"
            f"  →  世界 {fmt(q, 4)} m  离地 {q[2] - floor_z:.3f} m")
        if abs(float(np.linalg.norm(q - p)) - float(np.linalg.norm(v))) > 1e-12:
            warn(f"{name} 的 TCP 离法兰距离与 |v| 不一致（旋转矩阵不正交？）")
        if q[2] <= floor_z:
            warn(f"{name} 的 TCP 在 home 姿态下低于地面（{q[2] - floor_z:.3f} m），检查标定向量")
    names = spec.tool_names()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            log(f"  {a} ↔ {b} 间距 = {float(np.linalg.norm(pts[a] - pts[b])) * 1000:.1f} mm")
    log("  说明：箭头/小球只在**画面**上（mjGEOM_ARROW，不进 mjModel、不参与物理）；"
        "界面「工具 TCP 向量」卡片可单独开关、调粗细。")


# ============================================================================ 4
def check_workspace(model: mujoco.MjModel, data: mujoco.MjData, floor_z: float,
                    samples: int) -> None:
    log(f"\n--- 工作空间探测（随机采样 {samples} 组关节角做 FK）---")
    rng = np.random.default_rng(1)
    lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    world_bid = 0
    pts = np.zeros((samples, 3))
    down = floor_hit = self_hit = 0
    for i in range(samples):
        ik.set_arm_qpos(model, data, rng.uniform(lo, hi))
        mujoco.mj_forward(model, data)
        pts[i] = ik.site_pos(model, data, 'tool_site')
        if ik.tool_axis(model, data, 'tool_site')[2] < -0.95:
            down += 1
        if data.ncon:
            # 区分"撞地面"和"自己撞自己"：地面那个 geom 挂在 world 上
            if any(int(model.geom_bodyid[data.contact[k].geom1]) == world_bid
                   or int(model.geom_bodyid[data.contact[k].geom2]) == world_bid
                   for k in range(data.ncon)):
                floor_hit += 1
            else:
                self_hit += 1
    log(f"  tool_site 范围: x[{pts[:, 0].min():+.3f},{pts[:, 0].max():+.3f}] "
        f"y[{pts[:, 1].min():+.3f},{pts[:, 1].max():+.3f}] "
        f"z[{pts[:, 2].min():+.3f},{pts[:, 2].max():+.3f}]")
    log(f"          离地高度: [{pts[:, 2].min() - floor_z:+.3f},"
        f"{pts[:, 2].max() - floor_z:+.3f}] m")
    log(f"  工具轴朝下(>18°内)的采样占比: {down / samples * 100:.1f}%")
    log(f"  撞到地面的采样占比          : {floor_hit / samples * 100:.1f}%"
        f"（随机采样当然会往地板里钻，正常）")
    log(f"  自碰撞的采样占比            : {self_hit / samples * 100:.1f}%"
        f"（碰撞网格=外观件，含外壳，比真机保守）")
    reach = np.hypot(pts[:, 0], pts[:, 1])
    log(f"  水平半径范围: [{reach.min():.3f},{reach.max():.3f}] m，"
        f"常用作业区建议取 r ∈ [0.30, 0.65]、离地 0.15~0.60 m")


def ik_clear(model: mujoco.MjModel, data: mujoco.MjData, target, axis, seeds,
             *, tol_p: float = 1e-3, tol_a: float = 0.02):
    """多初值 IK，但**优先挑"精度够 + 不碰东西"的解**；真找不到就退回精度最好的那个。

    这一步要回答的是"能不能把工具尖送到目标、而且不杵到东西"，所以不能因为某个解支正好
    穿进地面就把模型判成不合格。返回 ``(q, 位置误差, 姿态误差, 尝试初值数, 目标姿态接触对数)``。
    """
    best: tuple | None = None
    clear: tuple | None = None
    tries = 0
    for sd in seeds:
        tries += 1
        q, ep, ea = solve_ik_pose(model, data, 'tool_site', target, axis, seed=sd)
        mujoco.mj_forward(model, data)               # solve 后 qpos 就在解上，直接数接触
        record = (float(ep) + 0.05 * float(ea), q, float(ep), float(ea), int(data.ncon))
        if best is None or record[0] < best[0]:
            best = record
        if ep < tol_p and ea < tol_a and data.ncon == 0 and (clear is None or record[0] < clear[0]):
            clear = record
    chosen = clear if clear is not None else best
    assert chosen is not None
    return chosen[1], chosen[2], chosen[3], tries, chosen[4]


# ============================================================================ 5
def check_reach(model: mujoco.MjModel, data: mujoco.MjData, floor_z: float) -> None:
    """IK + 控制器联调：让工具尖按给定点/朝向走，量真实到位误差。

    这一步同时验证三件事：逆解精度、位置伺服能不能把手臂拖到位、目标姿态会不会自碰撞。
    """
    targets = [
        ("正前方 r=0.40 h=0.30", (0.40, 0.0, floor_z + 0.30), (0, 0, -1)),
        ("正前方 r=0.55 h=0.45", (0.55, 0.0, floor_z + 0.45), (0, 0, -1)),
        ("左前   r=0.45 h=0.35", (0.32, 0.32, floor_z + 0.35), (0, 0, -1)),
        ("斜向下 45°", (0.45, 0.0, floor_z + 0.35), (-0.7071, 0, -0.7071)),
        ("水平前伸", (0.55, 0.0, floor_z + 0.45), (1, 0, 0)),
    ]
    log("\n--- IK + 控制器到位精度（每个目标都从 home 出发，IK 用多初值且优先无碰撞解）---")
    ok = 0
    for label, target, axis in targets:
        if not ik.reset_home(model, data):
            mujoco.mj_forward(model, data)
        seed = ik.arm_qpos(model, data)
        q, e_ik_p, e_ik_a, tries, ncon_ik = ik_clear(model, data, target, axis, ik_seeds(seed))
        if not ik.reset_home(model, data):           # IK 求解会改 qpos，跑之前先把模型复位
            ik.set_arm_qpos(model, data, seed)
            mujoco.mj_forward(model, data)
        ik.drive_to(model, data, q, seconds=2.5, settle=1.0, q_start=seed)
        p = ik.site_pos(model, data, 'tool_site')
        z = ik.tool_axis(model, data, 'tool_site')
        e_p = float(np.linalg.norm(p - np.array(target)))
        e_a = float(np.degrees(np.arccos(np.clip(np.dot(z, np.array(axis)
                                                 / np.linalg.norm(axis)), -1, 1))))
        good = e_p < 0.01 and e_a < 2.0 and data.ncon == 0
        ok += bool(good)
        log(f"  {label:22s} 位置误差 {e_p * 1000:6.2f} mm  姿态误差 {e_a:5.2f}°  "
            f"接触 {data.ncon} 对  IK {e_ik_p * 1000:5.2f}mm/{e_ik_a:4.2f}°"
            f"（{tries} 初值，解姿态接触 {ncon_ik} 对）  {'OK' if good else 'FAIL'}")
    if ok < len(targets):
        warn(f"到位精度只有 {ok}/{len(targets)} 个目标达标，可能要调 kp/kv 或目标点不可达")


# ============================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TB6-R5-RevA1 MuJoCo 模型自检")
    ap.add_argument("--scene", default=str(spec.SCENE_XML), help="场景 xml")
    ap.add_argument("--samples", type=int, default=20000, help="工作空间采样次数")
    ap.add_argument("--log", default=str(HERE / "check_model.log"), help="日志输出")
    args = ap.parse_args(argv)

    log("=" * 78)
    log("TB6-R5-RevA1 MuJoCo 模型自检")
    log("=" * 78)
    log(f"场景        : {args.scene}")
    try:
        model = mujoco.MjModel.from_xml_path(args.scene)
    except Exception as exc:  # noqa: BLE001
        log(f"!!! 场景编译失败: {type(exc).__name__}: {exc}")
        Path(args.log).write_text("\n".join(OUT), encoding="utf-8")
        return 1
    data = mujoco.MjData(model)
    log("场景编译成功 [OK]")

    # 用户配置里的关节限位（config/joint_limits.json）也要生效 —— 否则这里报的
    # "工作空间 / 限位余量" 和界面里看到的不是一回事（界面「关节限位」卡片可改）。
    try:
        import revA1_config as cfg
        saved = cfg.load_limits()
        if saved:
            eff = cfg.effective(saved)                      # 度 → 弧度写进模型
            for jn in spec.JOINTS:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    model.jnt_range[jid] = (np.radians(eff[jn][0]), np.radians(eff[jn][1]))
            log(f"关节限位    : 已应用 {cfg.limits_path()} —— {cfg.describe(saved)}")
        else:
            log(f"关节限位    : URDF 默认（没有 {cfg.limits_path()}）"
                f"：joint3 ±164°，其余 ±180°")
    except Exception as exc:  # noqa: BLE001
        log(f"关节限位    : 配置文件读不了（{exc}）→ 用 URDF 默认")

    # 地面高度直接问场景里的 floor geom（比 spec 里的常数更准）
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_z = float(model.geom_pos[gid][2]) if gid >= 0 else spec.FLOOR_Z
    log(f"地面高度    : z = {floor_z:.4f}"
        f"{'（读自场景 floor geom）' if gid >= 0 else '（revA1_spec.FLOOR_Z）'}")

    dump_structure(model)
    check_home(model, data, floor_z)
    check_tools(model, data, floor_z)
    check_stability(model, data)
    check_workspace(model, data, floor_z, args.samples)
    check_reach(model, data, floor_z)

    log("\n" + "=" * 78)
    if WARN:
        log(f"结论：{len(WARN)} 条警告，请逐条确认")
        for w in WARN:
            log(f"  - {w}")
    else:
        log("结论：全部检查通过 [OK]")
    log("=" * 78)
    Path(args.log).write_text("\n".join(OUT), encoding="utf-8")
    log(f"\n日志已写入 {args.log}")
    return 1 if WARN else 0


if __name__ == "__main__":
    sys.exit(main())

