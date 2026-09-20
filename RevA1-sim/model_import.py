"""把 URDF 机器人模型导入 MuJoCo（TB6-R5-RevA1 / RevA1-sim 版）。

和 `lerobot_codeit/sim/model_import.py` 是同一套思路，但这里针对 SolidWorks 导出的
工业臂 URDF 做了额外处理。

URDF -> MJCF 的 4 个坑，本模块逐个补掉
--------------------------------------
1. mesh 路径是 ``package://<包名>/meshes/x.STL``，而本机的包名目录并不存在
   （``RobotSDK/TB6-R5-RevA1/meshes`` 直接放在包根下）-> 自动解析成本地绝对路径，
   并在生成 MJCF 时把 mesh 拷到 ``assets/meshes/``，改写成相对路径（自包含、可搬移）。
2. URDF 里没有 ``<actuator>`` -> 编译出来 nu=0，关节在仿真里不会动
   -> 自动为每个 hinge 关节补 ``<position>`` 驱动器（kp/kv 按关节受力大小分档）。
3. URDF 的第一个 link（base_link）会被 MuJoCo 并进 worldbody，于是
   相邻的 base_link/Link1 贴合面会一直报虚假接触
   -> 把 root link 的 geom 重新包进一个固定 body（``base_link``），
      这样"父子连杆不参与接触"的默认规则才能生效。
4. SolidWorks 导出的碰撞 mesh 就是可视化 mesh 本身，相邻连杆的贴合面本来就会互相插入
   -> 自动补 ``<contact><exclude>``：排除"树距离 <= 2"的连杆对（爷爷-孙子）。

另外还顺手做了：给 geom 起名、补 ``<option>``/``<default>``、加末端 site。

生成物
------
    <out_dir>/<name>_arm.xml     机器人本体（自包含 MJCF）
    <out_dir>/<name>_scene.xml   带地面/灯光/相机/keyframe 的场景（可直接跑 viewer/demo）
    <out_dir>/meshes/*.STL       从 URDF 包里拷过来的网格

用命令行最省事::

    python convert_urdf_to_mjcf.py            # 生成上面三个
    python check_model.py                     # 自检
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
try:  # Windows 控制台默认 GBK，这里强制 UTF-8，避免中文打印崩掉
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


# =============================================================================
# 1) URDF 预处理：把 package:// 变成真实存在的本地路径
# =============================================================================
_PKG_RE = re.compile(r'package://([^/"]+)/([^"]*)')


def fix_urdf_package_paths(
    urdf_path: str | Path,
    package_root: str | Path | None = None,
) -> tuple[str, dict[str, str]]:
    """把 URDF 里的 ``package://pkg/...`` 改写成本地绝对路径。

    ``package_root`` 指"包名那层"所在的目录（例如 ``.../RobotSDK``），也允许直接给
    包目录本身（例如 ``.../RobotSDK/TB6-R5-RevA1``）——TB6-R5 就属于后者，
    因为它的 mesh 直接放在包根目录下。

    返回 ``(改写后的 urdf 文本, {package:// 里的包名: 实际使用的目录})``。
    """
    urdf_path = Path(urdf_path)
    text = urdf_path.read_text(encoding="utf-8", errors="replace")
    if "package://" not in text:
        return text, {}

    root = Path(package_root) if package_root else urdf_path.parent.parent
    root = root.resolve()

    used: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        pkg, rest = m.group(1), m.group(2)
        # 依次尝试：root/pkg/rest（标准 catkin 布局）、root/rest（包名是虚拟前缀）
        for candidate_dir in (root / pkg, root):
            if (candidate_dir / rest).exists():
                used[pkg] = str(candidate_dir).replace("\\", "/")
                return f"{used[pkg]}/{rest}"
        # 都不存在：退回 root/pkg/，让 MuJoCo 报出清晰的错误
        used[pkg] = str(root / pkg).replace("\\", "/")
        return f"{used[pkg]}/{rest}"

    return _PKG_RE.sub(repl, text), used


def load_model(
    path: str | Path,
    package_root: str | Path | None = None,
    tmp_dir: str | Path | None = None,
) -> mujoco.MjModel:
    """加载 MJCF 或 URDF（URDF 会自动修 ``package://`` 路径）。"""
    path = Path(path)
    if path.suffix.lower() != ".urdf":
        return mujoco.MjModel.from_xml_path(str(path))

    text, _ = fix_urdf_package_paths(path, package_root)
    tmp_dir = Path(tmp_dir) if tmp_dir else path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fixed = tmp_dir / f"{path.stem}_fixed.urdf"
    fixed.write_text(text, encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(fixed))


def save_mjcf(model: mujoco.MjModel, out_path: str | Path) -> Path:
    """把编译好的模型存成 MJCF 文本（mesh 路径由调用方再处理）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(out_path), model)
    return out_path


# =============================================================================
# 2) MJCF 文本后处理小工具（mj_saveLastXML 出来的文本很规整，够用且不引入依赖）
# =============================================================================
_TAG_MESH = re.compile(r"<mesh\b[^>]*/>")
_TAG_GEOM = re.compile(r"<geom\b[^>]*/>")
_TAG_BODY = re.compile(r"<body\b[^>]*?(/?)>|</body>")
_ATTR_FILE = re.compile(r'file="([^"]*)"')
_ATTR_NAME = re.compile(r'name="([^"]*)"')


def rewrite_mesh_paths(text: str, meshdir: str = "meshes") -> tuple[str, dict[str, Path]]:
    """把 ``<mesh file="/abs/.../Link1.STL"/>`` 改写成 ``file="Link1.STL"``。

    返回 ``(新文本, {文件名: 原始绝对路径})``。重名文件会加序号避免互相覆盖。
    """
    files: dict[str, Path] = {}

    def repl(m: re.Match) -> str:
        tag = m.group(0)
        fm = _ATTR_FILE.search(tag)
        if not fm:
            return tag
        src = Path(fm.group(1))
        base = src.name
        if base in files and files[base] != src:
            base = f"{src.parent.name}_{base}"
        files[base] = src
        return _ATTR_FILE.sub(f'file="{base}"', tag, count=1)

    return _TAG_MESH.sub(repl, text), files


def copy_mesh_files(files: dict[str, Path], dst_dir: str | Path) -> list[Path]:
    """把 mesh 拷到 ``dst_dir``（内容相同就跳过），返回已就位的文件列表。"""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for name, src in files.items():
        dst = dst_dir / name
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            shutil.copy2(src, dst)
        out.append(dst)
    return out


def set_compiler(text: str, **attrs: str) -> str:
    """给 ``<compiler/>`` 增补/覆盖属性（angle / meshdir / autolimits ...）。"""
    m = re.search(r"<compiler\b([^>]*?)/>", text)
    if not m:
        return re.sub(r"(<mujoco\b[^>]*>)",
                      lambda mm: mm.group(1) + "\n  <compiler/>", text, count=1)
    body = m.group(1)
    for key, val in attrs.items():
        if re.search(rf'\b{key}="[^"]*"', body):
            body = re.sub(rf'\b{key}="[^"]*"', f'{key}="{val}"', body, count=1)
        else:
            body = f'{body} {key}="{val}"'
    return text[:m.start()] + f"<compiler{body}/>" + text[m.end():]


def insert_after_compiler(text: str, snippet: str) -> str:
    """把 ``snippet`` 插到 ``<compiler/>`` 之后（schema 要求 option/default/asset 在此之后）。"""
    m = re.search(r"<compiler\b[^>]*?/>", text)
    if not m:
        raise ValueError("找不到 <compiler/>，无法插入内容")
    return text[:m.end()] + "\n" + snippet.rstrip("\n") + text[m.end():]


def append_before_close(text: str, snippet: str) -> str:
    """把 ``snippet``（contact / actuator / keyframe 等顶层标签）插到 ``</mujoco>`` 之前。"""
    idx = text.rfind("</mujoco>")
    if idx < 0:
        raise ValueError("找不到 </mujoco>，无法插入内容")
    return text[:idx] + snippet.rstrip("\n") + "\n" + text[idx:]


def set_model_name(text: str, name: str) -> str:
    return re.sub(r'(<mujoco\s+model=")[^"]*(")', rf"\g<1>{name}\g<2>", text, count=1)


def name_geoms(text: str, suffix: str = "_geom") -> str:
    """给匿名 geom 起名（用 mesh 名 + 后缀），方便调试/写奖励函数。"""

    def repl(m: re.Match) -> str:
        tag = m.group(0)
        if "name=" in tag:
            return tag
        nm = re.search(r'mesh="([^"]+)"', tag)
        if not nm:
            return tag
        return tag.replace("<geom ", f'<geom name="{nm.group(1)}{suffix}" ', 1)

    return _TAG_GEOM.sub(repl, text)


def insert_into_body(text: str, body_name: str, snippet: str) -> str:
    """把 ``snippet`` 插进名为 ``body_name`` 的 body 内部（缩进自动对齐）。"""
    stack: list[str] = []
    for m in _TAG_BODY.finditer(text):
        tag = m.group(0)
        if tag == "</body>":
            if not stack:
                continue
            name = stack.pop()
            if name == body_name:
                line_start = text.rfind("\n", 0, m.start()) + 1
                indent = text[line_start:m.start()] + "  "   # 子元素比 </body> 深 2 格
                body = "\n".join((indent + ln) if ln.strip() else ln
                                 for ln in snippet.strip("\n").splitlines())
                return text[:line_start] + body + "\n" + text[line_start:]
        elif not m.group(1):
            nm = _ATTR_NAME.search(tag)
            stack.append(nm.group(1) if nm else "")
    raise ValueError(f"找不到 body {body_name!r}")


def wrap_worldbody_geom(text: str, mesh: str, body_name: str,
                        inertial_xml: str = "") -> str:
    """把 worldbody 里第一个 geom（root link 的 mesh）包进一个固定 body。

    为什么必须这么做：MuJoCo 的 URDF 导入器会把 root link 的 geom 直接塞进 worldbody，
    于是 ``base_link`` 和 ``Link1`` 变成"同一个 body"之外的兄弟，贴合面上的虚假接触
    就被保留下来了（实测量到 4 个）。包成独立 body 后，父子连杆不参与接触的默认规则生效。
    """
    m = re.search(rf'(?P<ind>[ \t]*)<geom\b[^>]*mesh="{re.escape(mesh)}"[^>]*/>', text)
    if not m:
        raise ValueError(f"worldbody 里找不到 mesh={mesh!r} 的 geom")
    if "<body" in text[:m.start()]:
        raise ValueError(f"mesh={mesh!r} 的 geom 不在 worldbody 顶层，无法包装")

    ind, inner = m.group("ind"), m.group("ind") + "  "
    parts = [f"{ind}<body name=\"{body_name}\">"]
    if inertial_xml:
        parts.append(f"{inner}{inertial_xml.strip()}")
    parts.append(f"{inner}{m.group(0).strip()}")
    parts.append(f"{ind}</body>")
    return text[:m.start()] + "\n".join(parts) + text[m.end():]


# =============================================================================
# 3) 模型信息 / 根连杆惯量 / 接触排除对
# =============================================================================
def list_robot_info(model: mujoco.MjModel) -> dict:
    """汇总关节、驱动器、site、相机、连杆质量——自检脚本和日志都用它。"""
    def jname(i):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint{i}"

    def bname(i):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"

    joints, actuators, sites, cameras, bodies = [], [], [], [], []
    for i in range(model.njnt):
        jtype = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(int(model.jnt_type[i]), "?")
        joints.append({
            "name": jname(i),
            "type": jtype,
            "axis": model.jnt_axis[i].tolist(),
            "range": model.jnt_range[i].tolist(),
            "qposadr": int(model.jnt_qposadr[i]),
            "dofadr": int(model.jnt_dofadr[i]),
        })
    for i in range(model.nu):
        actuators.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}")
    for i in range(model.nsite):
        sites.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site{i}")
    for i in range(model.ncam):
        cameras.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}")
    for i in range(model.nbody):
        bodies.append({"name": bname(i), "mass": float(model.body_mass[i]),
                       "parent": int(model.body_parentid[i])})
    return {
        "nq": model.nq, "nv": model.nv, "nu": model.nu, "nbody": model.nbody,
        "ngeom": model.ngeom, "nmesh": model.nmesh,
        "total_mass": float(model.body_mass.sum()),
        "joints": joints, "actuators": actuators, "sites": sites,
        "cameras": cameras, "bodies": bodies,
    }


def geom_world_min_z(model: mujoco.MjModel, geom_name: str) -> float:
    """量某个 mesh geom 在世界系里的最低点 z（qpos=0 时）。

    用途：URDF 原点在安装面上，底座还往下伸 171mm，所以"车间地板"并不在 z=0，
    地面该铺在哪个高度可以直接从 base_link 的网格量出来。

    注意：MuJoCo 的 URDF 导入器会把 mesh 顶点重表达到该连杆的**惯量主轴坐标系**里
    （geom 带上了 pos/quat），所以直接读 ``model.mesh_vert`` 是错的，必须套上 geom 变换。
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise KeyError(f"模型里没有 geom {geom_name!r}")
    mid = int(model.geom_dataid[gid])
    if mid < 0:
        raise ValueError(f"geom {geom_name!r} 不是 mesh")
    adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    verts = model.mesh_vert[adr:adr + num]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, model.geom_quat[gid])
    world = verts @ mat.reshape(3, 3).T + model.geom_pos[gid]
    return float(world[:, 2].min())


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF 的 rpy（固定轴 XYZ，等价 Rz*Ry*Rx）转成 MuJoCo 的 (w, x, y, z) 四元数。"""
    import math
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def root_inertial_xml(urdf_text: str, link_name: str) -> str:
    """把 root link（base_link）的 URDF 惯量转成一行 MJCF ``<inertial>``。

    MuJoCo 的 URDF 导入器会把 root link 的质量丢掉（它被并进 worldbody，不动所以无所谓），
    但既然我们要把 root link 包成独立 body，就把惯量一起写回去，让模型质量守恒。
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf_text)
    for link in root.iter("link"):
        if link.get("name") != link_name:
            continue
        ine = link.find("inertial")
        if ine is None:
            return ""
        mass = float(ine.find("mass").get("value"))
        origin = ine.find("origin")
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        it = ine.find("inertia")
        full = " ".join(it.get(k, "0") for k in
                        ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        skip = all(abs(float(v)) < 1e-12 for v in rpy.split())
        quat = "" if skip else ' quat="{:.6g} {:.6g} {:.6g} {:.6g}"'.format(
            *rpy_to_quat(*[float(v) for v in rpy.split()]))
        return (f'<inertial pos="{xyz}"{quat} mass="{mass:g}" fullinertia="{full}"/>')
    return ""


def serial_chain(model: mujoco.MjModel, root_name: str) -> list[str]:
    """按 body id 顺序取出串联链上的 body 名，并把 root link 放在最前面。

    仅对串联机械臂成立（本项目就是），分支结构请手写 exclude 列表。
    """
    chain = [root_name]
    for i in range(1, model.nbody):
        chain.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}")
    return chain


def contact_excludes_xml(chain: list[str], max_distance: int = 2) -> str:
    """生成 ``<contact><exclude>``：排除"树距离 <= max_distance"的所有连杆对。

    为什么要排除
    ------------
    · 工业臂的碰撞 mesh 就是外观件本身，相邻连杆在关节处本来就会互相插入，
      不排除的话静止姿态就一直在报接触、还会被互相推开（实测 base_link/Link1 之间 4 对）；
    · MuJoCo 只会自动过滤"通过关节相连的父子连杆"，当父连杆是**固定** body
      （例如被焊死的 base_link）时不会过滤，必须显式写 exclude。
    """
    pairs = [(chain[i], chain[j]) for i in range(len(chain))
             for j in range(i + 1, min(i + 1 + max_distance, len(chain)))]
    if not pairs:
        return ""
    lines = [f'  <!-- 树距离 <= {max_distance} 的连杆对不参与自碰撞（关节处外观件本来就叠在一起） -->',
             "  <contact>"]
    lines += [f'    <exclude body1="{a}" body2="{b}"/>' for a, b in pairs]
    lines.append("  </contact>")
    return "\n".join(lines)


def position_actuators_xml(info: dict, gains: dict[str, dict]) -> str:
    """按 ``gains`` 生成 position 驱动器块（URDF 里没有 actuator）。"""
    lines = [
        "  <!-- 自动补的驱动器：URDF 本身没有 <actuator>，不补的话关节在仿真里不会动。",
        "       kp/kv 按关节受力大小分档：J2/J3 要撑起整条 90kg 的手臂，增益必须比小臂大一个量级。",
        '       forcerange 是"能出多大力"的粗略估计，接真机时请换成伺服手册上的数值。 -->',
        "  <actuator>",
    ]
    for j in info["joints"]:
        if j["type"] not in ("hinge", "slide"):
            continue
        g = gains.get(j["name"])
        if not g:
            continue
        lo, hi = j["range"]
        frc = g.get("forcerange")
        attrs = [f'name="act_{j["name"]}"', f'joint="{j["name"]}"',
                 f'kp="{g["kp"]:g}"', f'kv="{g["kv"]:g}"',
                 f'ctrlrange="{lo:.6g} {hi:.6g}"']
        if frc:
            attrs.append(f'forcerange="{-frc:g} {frc:g}"')
        lines.append("    <position " + " ".join(attrs) + "/>")
    lines.append("  </actuator>")
    return "\n".join(lines)


# =============================================================================
# 4) 组装机器人 MJCF
# =============================================================================
OPTION_XML = """  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"
          iterations="50" tolerance="1e-9" gravity="0 0 -9.81"/>"""

DEFAULT_XML = """  <default>
    <!-- damping/frictionloss：静止时不抖；armature（电机+减速器折算惯量）对高 kp 位置伺服的
         稳定性非常关键，太小会震、太大会钝。 -->
    <joint damping="0.8" armature="0.5" frictionloss="0.5"/>
    <geom friction="1.0 0.005 0.0001" condim="4" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <position kp="8000" kv="400"/>
  </default>"""


def build_arm_mjcf(
    urdf_path: str | Path,
    out_arm_xml: str | Path,
    *,
    package_root: str | Path | None = None,
    meshdir: str = "meshes",
    model_name: str | None = None,
    root_body_name: str = "base_link",
    gains: dict[str, dict] | None = None,
    sites: dict[str, str] | None = None,
    exclude_distance: int = 2,
    option_xml: str = OPTION_XML,
    default_xml: str = DEFAULT_XML,
    verbose: bool = False,
) -> dict:
    """URDF -> 自包含的机器人 MJCF（mesh 拷进 ``<out_dir>/<meshdir>/``）。

    返回 ``{"arm_xml", "meshes", "info", "model"}``。
    """
    import tempfile

    urdf_path = Path(urdf_path)
    out_arm_xml = Path(out_arm_xml)
    out_arm_xml.parent.mkdir(parents=True, exist_ok=True)
    name = model_name or urdf_path.stem

    fixed_text, pkg_roots = fix_urdf_package_paths(urdf_path, package_root)

    with tempfile.TemporaryDirectory(prefix="revA1_import_") as td:
        fixed = Path(td) / f"{name}_fixed.urdf"
        fixed.write_text(fixed_text, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(fixed))
        mujoco.mj_saveLastXML(str(Path(td) / f"{name}_raw.xml"), model)
        text = (Path(td) / f"{name}_raw.xml").read_text(encoding="utf-8")

    # --- mesh 路径：绝对路径 -> 相对 meshdir，并把文件拷过来 ---
    text, mesh_files = rewrite_mesh_paths(text, meshdir)
    meshes = copy_mesh_files(mesh_files, out_arm_xml.parent / meshdir)

    # --- 结构性的修补 ---
    text = set_model_name(text, name)
    text = name_geoms(text)
    root_mesh = re.search(r'<worldbody>\s*<geom\b[^>]*mesh="([^"]+)"', text)
    text = wrap_worldbody_geom(
        text, mesh=root_mesh.group(1), body_name=root_body_name,
        inertial_xml=root_inertial_xml(fixed_text, root_body_name))
    text = set_compiler(text, angle="radian", meshdir=meshdir,
                        autolimits="true", inertiafromgeom="false")
    text = insert_after_compiler(text, option_xml + "\n\n" + default_xml)

    # --- 末端 site ---
    for body_name, snippet in (sites or {}).items():
        text = insert_into_body(text, body_name, snippet)

    # --- actuator + contact exclude ---
    info = list_robot_info(model)
    blocks = []
    if gains:
        blocks.append(position_actuators_xml(info, gains))
    excludes = contact_excludes_xml(serial_chain(model, root_body_name),
                                    exclude_distance)
    if excludes:
        blocks.append(excludes)
    if blocks:
        text = append_before_close(text, "\n".join(blocks))

    out_arm_xml.write_text(text, encoding="utf-8")

    # --- 编译一次，确保生成物真的是有效 MJCF ---
    compiled = mujoco.MjModel.from_xml_path(str(out_arm_xml))
    compiled_info = list_robot_info(compiled)
    if verbose:
        print(f"URDF    : {urdf_path}")
        print(f"包路径  : {pkg_roots or '(无需改写)'}")
        print(f"机器人  : {out_arm_xml}   (mesh {len(meshes)} 个 -> {meshdir}/)")
        print(f"编译结果: nq={compiled_info['nq']} nv={compiled_info['nv']} "
              f"nu={compiled_info['nu']} nbody={compiled_info['nbody']} "
              f"质量={compiled_info['total_mass']:.2f} kg")
    return {"arm_xml": out_arm_xml, "meshes": meshes, "info": compiled_info,
            "model": compiled}


# =============================================================================
# 5) 场景（地面/灯光/相机/keyframe），可以直接跑 viewer / demo
# =============================================================================
SCENE_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!-- {name}_scene.xml —— 由 convert_urdf_to_mjcf.py 自动生成，可以随意手改。
     机器人本体在 {arm_xml}（include 进来），这里只放"世界"相关的东西。

坐标系约定
----------
· 世界原点 = 机械臂安装面中心（URDF 原点），z 向上；
· base_link 底面在 z = {floor_z}，地面就铺在这个高度（= 机器人坐在车间地板上）；
· 六轴全 0 时手臂沿 -x 方向平伸；正常工作区大约 x ∈ [0.3, 0.9]、|y| < 0.6、z > 0；
· qpos / ctrl 顺序：joint1..joint6（弧度）。
-->
<mujoco model="{name}_scene">
  <include file="{arm_xml}"/>

  <statistic center="0 0 {stat_center_z}" extent="1.6"/>

  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-18" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture name="{name}_sky" type="skybox" builtin="gradient"
             rgb1="0.28 0.40 0.56" rgb2="0.04 0.06 0.09" width="256" height="256"/>
    <texture name="{name}_floor_tex" type="2d" builtin="checker"
             rgb1="0.26 0.27 0.29" rgb2="0.20 0.21 0.23" width="256" height="256"/>
    <material name="{name}_floor_mat" texture="{name}_floor_tex" texrepeat="16 16"
              reflectance="0.03"/>
    <material name="{name}_target_mat" rgba="0.15 0.80 0.35 0.55"/>
  </asset>

  <worldbody>
    <light name="key_light" pos="1.2 -1.2 2.4" dir="-0.4 0.4 -1" directional="true"
           diffuse="0.6 0.6 0.6" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill_light" pos="-1.4 1.4 1.6" dir="0.45 -0.45 -1" directional="true"
           diffuse="0.3 0.3 0.35"/>

    <!-- 地面：正好在机器人的安装底面高度 -->
    <geom name="floor" type="plane" pos="0 0 {floor_z}" size="4 4 0.1"
          material="{name}_floor_mat" condim="4" friction="1.0 0.005 0.0001"/>

    <!-- 名义作业点标记（纯视觉，不参与碰撞）= home 姿态下 tool_site 的地面投影 -->
    <geom name="target_pad" type="cylinder" size="0.05 0.002" pos="{target_xy} {pad_z}"
          material="{name}_target_mat" contype="0" conaffinity="0"/>

    <!-- 三个固定相机，方便离线渲染/录视频 -->
    <camera name="overview_cam" pos="1.6 -1.6 1.1" xyaxes="0.7071 0.7071 0 -0.21 0.21 0.958"
            fovy="45"/>
    <camera name="top_cam" pos="0 0 2.2" xyaxes="1 0 0 0 1 0" fovy="50"/>
    <camera name="side_cam" pos="2.2 0 0.7" xyaxes="0 1 0 0 0 1" fovy="50"/>
  </worldbody>

  <keyframe>
    <!-- home：真机实测姿态（拍摄点云时的姿态）——无自碰撞、工具尖离地面 {home_tip_z:.3f} m，
         可以直接作为训练/演示的初始位姿。改姿态后请重跑 check_model.py 复核。 -->
    <key name="home" qpos="{qpos}" ctrl="{qpos}"/>
  </keyframe>
</mujoco>
"""


def build_scene(
    arm_xml: str | Path,
    out_scene: str | Path,
    *,
    name: str | None = None,
    floor_z: float = 0.0,
    target_xy: tuple[float, float] = (0.0, 0.0),
    home_qpos: list[float] | tuple[float, ...] = (),
    home_tip_z: float = 0.0,
    stat_center_z: float = 0.5,
) -> Path:
    """生成场景文件（机器人 include + 地面/灯光/相机/home keyframe）。"""
    arm_xml = Path(arm_xml)
    out_scene = Path(out_scene)
    name = name or arm_xml.stem.replace("_arm", "")
    qpos = " ".join(f"{v:.6g}" for v in home_qpos)
    out_scene.write_text(
        SCENE_TEMPLATE.format(
            name=name, arm_xml=arm_xml.name, floor_z=f"{floor_z:.6g}",
            stat_center_z=f"{stat_center_z:.6g}",
            target_xy=" ".join(f"{v:.4g}" for v in target_xy),
            pad_z=f"{floor_z + 0.002:.6g}",
            home_tip_z=home_tip_z, qpos=qpos),
        encoding="utf-8")
    return out_scene






