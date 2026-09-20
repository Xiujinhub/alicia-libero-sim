#!/usr/bin/env python3
"""点云 ↔ 机械臂：把深度相机拍到的点云按（手眼标定 + 拍摄姿态）放进 MuJoCo 场景里。

它在整个系统里的位置
--------------------
::

    深度相机点云（相机光学系，mm）     手眼标定 R,t（相机 → 末端，mm）    拍摄姿态（TCP 在基座系）
              │                                  │                            │
              └──────────────► Cloud ────────────┴────────────► transform_to_base(...)
                                                                      │  基座系点云（m）
                                                                      ▼
                                            build_scene() → assets/meshes/pc_cloud_*.obj
                                                          → assets/revA1_pc_scene.xml
                                                          → MuJoCo 里和机械臂同框显示

数据与约定（本机实测那份数据）
------------------------------
**① 手眼标定** ``xml/calib_extrinsic.xml``：OpenCV ``calibrateHandEye`` 的输出 ``R`` / ``t``，
把**相机光学系**（x 向右、y 向下、z 向前/深度）变换到**末端/法兰系**，单位 mm
（实测 ``t = [-67.28, 3.35, 236.13] mm``，即相机沿工具轴前方约 24.6 cm）。

**② 拍摄姿态**：真机当时的 TCP 位姿，``[x, y, z, rx, ry, rz]`` = mm + **外旋 XYZ 欧拉角[deg]**
（和真机广播 ``pose`` 同一套，见 ``robot_link.py``）。镜头朝哪，就看工具轴
``Rz(rz)Ry(ry)Rx(rx)`` 的第三列。

**③ 点云** ``point_cloud/*.json``，两种写法都认：

* ``{"depth_o2e": [[x,y,z], ...], "box": {...}, "depth_n": N}`` —— 已经在相机系里的 3D 点（mm）；
* ``{"depth": {"png_base64": ..., "encoding": "16UC1", "depth_scale": s, "box": {...}}}``
  —— 16 位深度图，本模块自己解码 PNG + 用内参反投影（含畸变校正），所以不需要 OpenCV。

坐标链
------
``p_基座 = T_基座←TCP(拍摄姿态) · T_TCP←相机(R,t) · p_相机``，再加上：
小车高度（真机是装在小车上的，基座离地约 1.1 m）、可选的法兰/TCP 参考点切换（±工具长度）、
以及手动微调（沿基座 xyz 平移 + 绕基座 z 旋转）。

渲染为什么不用"每帧塞点"
------------------------
``mjv_initGeom`` 每帧从 Python 循环塞点很慢（实测 2000 点 = 24 ms/帧、10000 点 = 139 ms/帧），
所以这里**一次性把点云写成静态网格**（每个点一个小八面体，按深度分几带颜色），交给模型渲染：
整云 78721 点（分 6 带）载入 0.5 s、之后**每帧 2.8 ms**，还能被相机随便绕、和机械臂正常遮挡。

命令行
------
::

    python point_cloud.py --report                      # 只打印标定/点云的体检报告
    python point_cloud.py --build-scene                 # 生成 assets/revA1_pc_scene.xml（带点云）
    python point_cloud.py --png runs/pc.png             # 再离屏渲染一张预览图
    python point_cloud.py --selftest                    # 全链路自检（解析/变换/建模/渲染）

界面入口：``revA1_gui.py`` 的「点云（相机 → 机械臂）」卡片。
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import revA1_spec as spec

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
CALIB_DIR = HERE / "xml"
CLOUD_DIR = HERE / "point_cloud"
ASSETS = HERE / "assets"
MESH_DIR = ASSETS / "meshes"          # 必须和场景的 <compiler meshdir="meshes"> 对上
BASE_SCENE = ASSETS / "revA1_scene.xml"
OUT_SCENE = ASSETS / "revA1_pc_scene.xml"
RUNS = HERE / "runs"                  # 预览图 / 自检产物（.gitignore 里忽略掉了）

DEFAULT_EXTRINSIC = CALIB_DIR / "calib_extrinsic.xml"
DEFAULT_INTRINSIC = CALIB_DIR / "calib_intrinsic_1st.xml"
# 这份数据拍摄时的真机 TCP 位姿：mm + 外旋 XYZ 欧拉角[deg]（= 真机 pose 字段）
DEFAULT_POSE = (110.862, -5.474, 396.436, 196.64, 34.89, 111.77)
DEFAULT_CART_HEIGHT_M = 1.10          # 机械臂装在小车上，基座离地约 1.1 m
DEFAULT_MAX_POINTS = 60000            # 超过就抽稀（整云 78721 点也能跑，只是文件大）
DEFAULT_POINT_R_MM = 4.0              # 每个点画成多大的八面体
DEFAULT_BANDS = 6                     # 按深度分几带颜色（0/1 = 单色）
DEFAULT_MESH_PREFIX = "pc_cloud"      # 生成的点云网格文件名前缀（放 assets/meshes/）
MIN_DEPTH_MM = 1.0                    # 深度小于它就当无效像素

# 深度分色：近 → 远（红 → 橙 → 黄 → 绿 → 蓝 → 紫）
DEPTH_COLORS = (
    (0.95, 0.32, 0.28, 1.0),
    (0.98, 0.62, 0.24, 1.0),
    (0.97, 0.88, 0.30, 1.0),
    (0.52, 0.92, 0.42, 1.0),
    (0.28, 0.76, 0.96, 1.0),
    (0.68, 0.46, 0.96, 1.0),
    (0.95, 0.45, 0.85, 1.0),
    (0.60, 0.60, 0.62, 1.0),
)
# 单点 = 一个小八面体（6 顶点 / 8 面）：比立方体省 1/3 面，还不会从侧面看消失
OCTA_VERT = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                     dtype=float)
OCTA_FACE = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
                      [0, 1, 5], [1, 2, 5], [2, 3, 5], [3, 0, 5]], dtype=np.int64)

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


class PointCloudError(RuntimeError):
    """点云 / 标定 / 姿态读不出来，或者场景生成失败（消息是中文，可直接显示在界面上）。"""


# =============================================================== 标定 / 姿态
def read_opencv_matrices(path) -> dict[str, np.ndarray]:
    """读 OpenCV ``FileStorage`` 的 XML（``<K>`` / ``<R>`` / ``<t>`` / ``<distortion>``…）。

    返回 ``{名字: ndarray}``；名字就是 XML 里的元素名。
    """
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, np.ndarray] = {}
    for m in re.finditer(r"<(\w+)\s+type_id=\"opencv-matrix\">(.*?)</\1>", txt, re.S):
        name, body = m.group(1), m.group(2)
        r = re.search(r"<rows>\s*(\d+)\s*</rows>", body)
        c = re.search(r"<cols>\s*(\d+)\s*</cols>", body)
        d = re.search(r"<data>(.*?)</data>", body, re.S)
        if not (r and c and d):
            continue
        rows, cols = int(r.group(1)), int(c.group(1))
        vals = [float(v) for v in NUM_RE.findall(d.group(1))]
        if len(vals) != rows * cols:
            raise PointCloudError(f"{Path(path).name} 的 {name}：{rows}×{cols} 需要 "
                                  f"{rows * cols} 个数，读出来 {len(vals)} 个")
        out[name] = np.asarray(vals, dtype=float).reshape(rows, cols)
    if not out:
        raise PointCloudError(f"{path} 里没读到任何 opencv-matrix")
    return out


def load_extrinsic(path=DEFAULT_EXTRINSIC) -> tuple[np.ndarray, np.ndarray]:
    """读手眼标定：``(R, t)``，把**相机光学系**的点变换到**末端系**，单位 mm。"""
    mats = read_opencv_matrices(path)
    if "R" not in mats or "t" not in mats:
        raise PointCloudError(f"{Path(path).name} 里没有 R/t（手眼标定矩阵该有这两个）")
    R = np.asarray(mats["R"], dtype=float)
    t = np.asarray(mats["t"], dtype=float).ravel()
    if R.shape != (3, 3) or t.size != 3:
        raise PointCloudError(f"标定形状不对：R{R.shape} t{t.shape}")
    return R, t


def load_intrinsic(path=DEFAULT_INTRINSIC) -> tuple[np.ndarray, np.ndarray | None]:
    """读相机内参：``(K 3×3, distortion)``（第二个可能没有）。"""
    mats = read_opencv_matrices(path)
    K = mats.get("K")
    if K is None or K.shape != (3, 3):
        raise PointCloudError(f"{Path(path).name} 里没有 3×3 的 K")
    return np.asarray(K, dtype=float), mats.get("distortion")


def euler_xyz_matrix(rx_rad: float, ry_rad: float, rz_rad: float) -> np.ndarray:
    """外旋 XYZ 欧拉角 → 旋转矩阵 ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``。

    真机 ``pose[3:6]`` 就是这个约定；实现只有一份，在 ``revA1_spec.euler_xyz_matrix``。
    """
    return spec.euler_xyz_matrix(rx_rad, ry_rad, rz_rad)


def pose_matrix(pose) -> tuple[np.ndarray, np.ndarray]:
    """拍摄姿态 ``[x,y,z(mm), rx,ry,rz(deg，外旋 XYZ)]`` → ``(R 3×3, p 3)[m]``（末端→基座）。"""
    try:
        return spec.pose_matrix(pose)
    except ValueError as exc:                       # spec 的报错换成本模块的异常类型
        raise PointCloudError(str(exc)) from exc


def parse_pose6(text) -> tuple[float, ...]:
    """把界面输入框里的 ``"110.9 -5.5 396.4 196.6 34.9 111.8"`` 解析成 6 个数。"""
    if isinstance(text, (list, tuple, np.ndarray)):
        vals = [float(v) for v in np.asarray(text, dtype=float).ravel()]
    else:
        vals = [float(v) for v in NUM_RE.findall(str(text))]
    if len(vals) != 6:
        raise PointCloudError(f"拍摄姿态要 6 个数，现在是 {len(vals)} 个：{text!r}")
    return tuple(vals)


def pose_text(pose) -> str:
    """反过来：6 个数 → 一行文本（界面默认值用）。"""
    vals = [float(v) for v in np.asarray(pose, dtype=float).ravel()]
    return " ".join(f"{v:g}" for v in vals)


# =============================================================== 深度图 → 点云
def decode_depth_png(b64: str) -> np.ndarray:
    """解 16 位单通道 PNG（base64 字符串）→ ``(H, W) uint16`` 深度图。

    手写最小 PNG 解析（签名 + chunk + ``zlib`` 解压 + 每行反滤波），所以**不依赖 OpenCV / Pillow**。
    """
    raw = base64.b64decode(str(b64))
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise PointCloudError("点云里的 png_base64 不是 PNG 数据")
    pos, idat = 8, bytearray()
    width = height = bitdepth = colortype = interlace = None
    while pos + 8 <= len(raw):
        (length,) = struct.unpack(">I", raw[pos:pos + 4])
        ctype = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            (width, height, bitdepth, colortype,
             _comp, _filt, interlace) = struct.unpack(">IIBBBBB", data[:13])
        elif ctype == b"IDAT":
            idat += data
        elif ctype == b"IEND":
            break
    if width is None or height is None:
        raise PointCloudError("PNG 里没有 IHDR")
    if bitdepth != 16 or colortype != 0:
        raise PointCloudError(f"只认 16 位灰度深度图（现在 bitdepth={bitdepth} colortype={colortype}）")
    if interlace:
        raise PointCloudError("隔行（interlaced）PNG 暂不支持")

    buf = np.frombuffer(zlib.decompress(bytes(idat)), dtype=np.uint8)
    bpp, stride = 2, int(width) * 2
    need = (stride + 1) * int(height)
    if buf.size < need:
        raise PointCloudError(f"PNG 数据不完整：{buf.size} < {need}")
    out = np.empty((int(height), int(width)), dtype=np.uint16)
    prev = np.zeros(stride, dtype=np.int32)
    off = 0
    for y in range(int(height)):
        ftype = int(buf[off])
        off += 1
        line = buf[off:off + stride].astype(np.int32)
        off += stride
        if ftype == 1:            # Sub：加左邻
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:          # Up：加上邻
            line += prev
        elif ftype == 3:          # Average：(左 + 上) / 2
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:          # Paeth
            for i in range(stride):
                a = int(line[i - bpp]) if i >= bpp else 0
                b = int(prev[i])
                c = int(prev[i - bpp]) if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif ftype != 0:
            raise PointCloudError(f"不认识的 PNG 滤波类型 {ftype}（第 {y} 行）")
        line &= 0xFF
        prev = line
        out[y] = np.frombuffer(line.astype(np.uint8).tobytes(), dtype=">u2")
    return out


def undistort_normalized(x, y, dist, *, iters: int = 8):
    """归一化像面坐标：(u,v) 畸变后 → 畸变前（迭代反解，不用 OpenCV）。

    ``dist`` 是 OpenCV 的 ``[k1 k2 p1 p2 k3 …]``；``None`` 或全 0 就原样返回。
    """
    if dist is None:
        return x, y
    d = [float(v) for v in np.asarray(dist, dtype=float).ravel()]
    if not d or all(abs(v) < 1e-12 for v in d[:5]):
        return x, y
    k1, k2, p1, p2 = (d + [0.0] * 4)[:4]
    k3 = d[4] if len(d) > 4 else 0.0
    xd, yd = x.copy(), y.copy()
    xu, yu = xd.copy(), yd.copy()
    for _ in range(int(iters)):
        r2 = xu * xu + yu * yu
        k = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        k = np.where(np.abs(k) < 1e-6, 1e-6, k)
        dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
        dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
        xu = (xd - dx) / k
        yu = (yd - dy) / k
    return xu, yu


def deproject_depth(depth_mm, K, dist=None, *, roi=None, offset_xy=(0.0, 0.0), stride: int = 1,
                    depth_min_mm: float = MIN_DEPTH_MM, depth_max_mm=None,
                    scale: float = 1.0) -> np.ndarray:
    """深度图 → 相机光学系 3D 点（mm）：``x=(u-cx)z/fx``、``y=(v-cy)z/fy``、``z=深度``。

    * ``roi=(x,y,w,h)``：只用深度图里的这一块（默认整张）；
    * ``offset_xy``：深度图是**裁剪图**时，把裁剪坐标加回整图坐标（畸变中心 ``cx/cy`` 是整图坐标）；
    * ``stride``：像素步长（抽稀）；
    * ``depth_min/max_mm``：深度有效范围（太近/太远/0 都当无效）；
    * ``scale``：``depth_scale``（OpenCV 常见 1 / 0.001），深度值乘它才是 mm。
    """
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    depth_mm = np.asarray(depth_mm)
    if roi is None:
        bx, by, bw, bh = 0, 0, int(depth_mm.shape[1]), int(depth_mm.shape[0])
    else:
        bx, by, bw, bh = (int(v) for v in roi)
        bw = min(bw, int(depth_mm.shape[1]) - bx)
        bh = min(bh, int(depth_mm.shape[0]) - by)
    if bw <= 0 or bh <= 0:
        raise PointCloudError(f"ROI {roi} 落在深度图 {depth_mm.shape} 外面了")
    step = max(int(stride), 1)
    d = depth_mm[by:by + bh:step, bx:bx + bw:step].astype(float) * float(scale)
    uu, vv = np.meshgrid(np.arange(bx, bx + bw, step)[:d.shape[1]],
                         np.arange(by, by + bh, step)[:d.shape[0]])
    keep = np.isfinite(d) & (d >= float(depth_min_mm))
    if depth_max_mm is not None:
        keep &= d <= float(depth_max_mm)
    if not np.any(keep):
        raise PointCloudError(f"深度图里 {depth_min_mm}~{depth_max_mm} mm 之间没有有效像素")
    u = uu[keep].astype(float) + float(offset_xy[0])
    v = vv[keep].astype(float) + float(offset_xy[1])
    z = d[keep]
    xn, yn = undistort_normalized((u - cx) / fx, (v - cy) / fy, dist)
    return np.column_stack([xn * z, yn * z, z])


# =============================================================== 点云对象 / 读取
@dataclass
class Cloud:
    """一帧点云（单位 mm）。

    ``frame`` 是这份点云**自带的坐标系**：

    * ``"cam"`` —— 相机光学系（x 右 / y 下 / z 深度）：要乘手眼标定 ``R,t`` 才能到末端系；
    * ``"end"`` —— **已经是末端系**（数据里写 ``frame: "o2e"``，即光学→末端的转换已经做过）：
      不能再乘一次手眼，只需用拍摄姿态把它搬到基座系；
    * ``""`` —— 文件里没写，交给 :func:`detect_cloud_frame` 判。
    """

    name: str
    points_mm: np.ndarray
    source: str = ""
    box: tuple[int, int, int, int] | None = None      # 原始 ROI（整图坐标 x,y,w,h）
    frame: str = ""                                   # "cam" / "end" / ""
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.points_mm.shape[0])

    @property
    def depth_mm(self) -> np.ndarray:
        return self.points_mm[:, 2]

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.points_mm
        return p.min(axis=0), p.max(axis=0)

    def frame_label(self) -> str:
        return {"cam": "相机光学系", "end": "末端系（o2e）"}.get(self.frame, "未知（待判定）")

    def summary(self) -> str:
        lo, hi = self.bbox()
        return (f"{self.name}：{self.n} 点（{self.source} / {self.frame_label()}）"
                f"｜x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] "
                f"z[{lo[2]:.0f},{hi[2]:.0f}] mm")


CLOUD_KEYS = ("depth_o2e", "points", "xyz", "point_cloud", "cloud", "vertices", "pcd")
# 数据里声明的坐标系名字 → 本模块的两个内部名字
FRAME_ALIASES = {
    "cam": "cam", "camera": "cam", "optical": "cam", "depth": "cam", "cam_optical": "cam",
    "end": "end", "ee": "end", "tcp": "end", "o2e": "end", "end_effector": "end",
    "flange": "end", "link6": "end",
}


def normalize_frame(name) -> str:
    """把数据里五花八门的坐标系名字（``o2e`` / ``end`` / ``camera``…）归一成 ``"cam"``/``"end"``。

    ``o2e`` = optical→end（相机光学系 → 末端系）**已经做过**，所以它属于 ``"end"``。
    """
    if not name:
        return ""
    return FRAME_ALIASES.get(str(name).strip().lower(), "")


def _points_from_json(obj) -> np.ndarray | None:
    """从 JSON 里挖 3D 点（``[[x,y,z],...]`` 扁平数组形式，或 ``{x:[],y:[],z:[]}`` 形式）。"""
    if not isinstance(obj, dict):
        return None
    for key in CLOUD_KEYS:
        if key in obj:
            arr = np.asarray(obj[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3]
            if arr.ndim == 1 and arr.size >= 3 and arr.size % 3 == 0:
                return arr.reshape(-1, 3)
    if all(k in obj for k in ("x", "y", "z")):
        xs, ys, zs = (np.asarray(obj[k], dtype=float).ravel() for k in ("x", "y", "z"))
        if xs.size and xs.size == ys.size == zs.size:
            return np.column_stack([xs, ys, zs])
    return None


def _box_tuple(box_d) -> tuple[int, int, int, int] | None:
    if isinstance(box_d, dict) and all(k in box_d for k in ("x", "y", "w", "h")):
        return (int(box_d["x"]), int(box_d["y"]), int(box_d["w"]), int(box_d["h"]))
    return None


def load_cloud(path, *, intrinsic_path=DEFAULT_INTRINSIC, stride: int = 1,
               depth_min_mm: float = MIN_DEPTH_MM, depth_max_mm=None,
               undistort: bool = True) -> Cloud:
    """读一帧点云（相机系，mm）。支持 ``*.json``（3D 点 或 16 位深度图）与 ``*.npy``。"""
    path = Path(path)
    if not path.is_file():
        raise PointCloudError(f"点云文件不存在：{path}")
    if path.suffix.lower() == ".npy":
        pts = np.asarray(np.load(str(path)), dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise PointCloudError(f"{path.name} 的形状 {pts.shape} 不是 (N,3) 点云")
        pts = pts[:, :3]
        if float(np.nanmax(pts)) < 10.0:          # 看着像"米"就换成毫米
            pts = pts * 1000.0
        return Cloud(name=path.name, points_mm=pts, source="npy")
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        raise PointCloudError(f"{path.name} 不是合法 JSON：{exc}") from exc

    pts = _points_from_json(obj)
    if pts is not None:
        if pts.shape[0] < 10:
            raise PointCloudError(f"{path.name} 只有 {pts.shape[0]} 个点，太少")
        keep = np.isfinite(pts).all(axis=1) & (pts[:, 2] >= float(depth_min_mm))
        if depth_max_mm is not None:
            keep &= pts[:, 2] <= float(depth_max_mm)
        meta = ({k: v for k, v in obj.items() if k not in CLOUD_KEYS and k != "box"}
                if isinstance(obj, dict) else {})
        declared = obj.get("frame") or obj.get("frame_id") if isinstance(obj, dict) else None
        if declared:
            meta["declared_frame"] = declared
        return Cloud(name=path.name, points_mm=pts[keep], source="json 3D 点",
                     box=_box_tuple(obj.get("box") if isinstance(obj, dict) else None),
                     frame=normalize_frame(declared), meta=meta)

    block = obj.get("depth") if isinstance(obj, dict) and isinstance(obj.get("depth"), dict) else obj
    b64 = None
    if isinstance(block, dict):
        b64 = block.get("png_base64") or block.get("png") or block.get("base64")
    if not b64:
        raise PointCloudError(f"{path.name} 里既没有 3D 点，也没有深度图（png_base64）")
    depth = decode_depth_png(b64)
    K, dist = load_intrinsic(intrinsic_path)
    if not undistort:
        dist = None
    scale = float(block.get("depth_scale") or 1.0) if isinstance(block, dict) else 1.0
    box_d = block.get("box") if isinstance(block, dict) else None
    roi, offset = _box_tuple(box_d), (0.0, 0.0)
    if roi is not None and depth.shape[:2] == (roi[3], roi[2]):
        offset, roi = (float(roi[0]), float(roi[1])), None     # PNG 是裁剪图 → 坐标加回左上角
    pts = deproject_depth(depth, K, dist, roi=roi, offset_xy=offset, stride=stride,
                          depth_min_mm=depth_min_mm, depth_max_mm=depth_max_mm, scale=scale)
    # 深度图是我们自己按内参反投影出来的 → 一定在**相机光学系**（深度 z 就是相机 z）
    return Cloud(name=path.name, points_mm=pts, source="json 深度图 → 反投影",
                 box=_box_tuple(box_d), frame="cam",
                 meta=dict(depth_shape=tuple(int(v) for v in depth.shape), depth_scale=scale,
                           undistort=bool(undistort)))


def subsample(points, max_points: int = DEFAULT_MAX_POINTS, depth=None):
    """点太多就等间隔抽稀（``max_points <= 0`` = 不抽）。返回 ``(点, 深度, 步长)``。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    dep = None if depth is None else np.asarray(depth, dtype=float)
    n = pts.shape[0]
    if max_points is None or int(max_points) <= 0 or n <= int(max_points):
        return pts, dep, 1
    step = int(math.ceil(n / float(max_points)))
    idx = np.arange(0, n, step)
    return pts[idx], (None if dep is None else dep[idx]), step


# =============================================================== 变换 / 落盘
def camera_to_end(points_mm, R, t) -> np.ndarray:
    """相机光学系 → 末端系（手眼标定 ``R,t`` 的正向，mm）。"""
    return (np.asarray(points_mm, dtype=float).reshape(-1, 3)
            @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float).ravel())


def end_to_camera(points_mm, R, t) -> np.ndarray:
    """末端系 → 相机光学系（手眼标定的反向）。判定点云归属哪个坐标系时用它。"""
    return ((np.asarray(points_mm, dtype=float).reshape(-1, 3)
             - np.asarray(t, dtype=float).ravel()) @ np.asarray(R, dtype=float))


def detect_cloud_frame(cloud: Cloud, *, R=None, t=None, intrinsic_path=DEFAULT_INTRINSIC) -> dict:
    """判定这份点云自带的坐标系是 ``"cam"``（相机光学系）还是 ``"end"``（末端系）。

    不靠猜：把点**当成**某个坐标系、反解回相机系投影成像素，看命中这份数据**自己记的 ROI**
    （``cloud.box``）的比例。本机那份 ``o2e`` 数据实测：当成末端系命中 99.8%，当成相机系只有 60.3%
    —— 一眼可辨（道理也直白：ROI 就是当初裁剪的像素框）。

    返回 ``dict(frame, declared, scores, evidence, sure)``；没有 ROI 元数据时只能采信文件声明。
    """
    declared = normalize_frame(cloud.meta.get("declared_frame") or cloud.frame)
    if cloud.box is None:
        frame = declared or "cam"
        return dict(frame=frame, declared=declared, scores={}, sure=bool(declared),
                    evidence=[f"没有 ROI 元数据，按{'文件声明' if declared else '默认相机系'}"
                              f"（{cloud.frame_label()}）"])
    if R is None or t is None:
        R, t = load_extrinsic()
    try:
        K, _dist = load_intrinsic(intrinsic_path)
    except (PointCloudError, OSError):
        frame = declared or "cam"
        return dict(frame=frame, declared=declared, scores={}, sure=bool(declared),
                    evidence=[f"读不到内参 K，按{'文件声明' if declared else '默认相机系'}"
                              f"（{cloud.frame_label()}）"])
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    x0, y0, w, h = (float(v) for v in cloud.box)
    scores: dict[str, float] = {}
    evidence: list[str] = []
    for frame in ("cam", "end"):
        q = cloud.points_mm if frame == "cam" else end_to_camera(cloud.points_mm, R, t)
        z = q[:, 2]
        ok = np.isfinite(z) & (z > 1e-6)
        u = np.zeros(z.shape)
        v = np.zeros(z.shape)
        u[ok] = fx * q[ok, 0] / z[ok] + cx
        v[ok] = fy * q[ok, 1] / z[ok] + cy
        inside = ok & (u >= x0 - 2) & (u <= x0 + w + 2) & (v >= y0 - 2) & (v <= y0 + h + 2)
        scores[frame] = float(inside.mean())
        evidence.append(f"当成{'相机光学系' if frame == 'cam' else '末端系（o2e）'}："
                        f"{scores[frame] * 100:.1f}% 的点反投影落在 ROI "
                        f"x[{x0:.0f},{x0 + w:.0f}] y[{y0:.0f},{y0 + h:.0f}]")
    best = max(scores, key=lambda k: scores[k])
    sure = scores[best] > 0.9 and (scores[best] - scores[min(scores, key=lambda k: scores[k])]) > 0.1
    if not sure and declared:                    # 判不出来就采信文件自己写的
        best = declared
    return dict(frame=best, declared=declared, scores=scores, evidence=evidence, sure=sure)


def transform_to_base(points_mm, R_cam2tcp, t_cam2tcp, pose, *, frame: str = "cam",
                      ref: str = "tcp", tool_offset_mm: float = 42.7,
                      offset_mm=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> np.ndarray:
    """点云（mm）→ **机械臂基座系**（m）。

    ``p_基座 = T_基座←末端(拍摄姿态) · p_末端``；``p_末端`` 从哪来，看 ``frame``：

    * ``frame="cam"``：点云在**相机光学系**里 → 先用手眼标定 ``p_末端 = R·p + t``；
    * ``frame="end"``：点云**已经在末端系**里（``o2e`` 数据就是这样，光学→末端的转换已经做过）
      → 直接用它，**不能再乘一次手眼**。

    之后按 ``ref`` 对齐参考点（标定的"末端"若是法兰盘，就沿工具轴退回一个工具长度），
    再叠加手动微调（``offset_mm`` 沿基座 xyz 平移、``yaw_deg`` 绕基座 z 旋转）。
    """
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    R_bt, p_bt = pose_matrix(pose)
    if normalize_frame(frame) == "end":
        pts_end = pts.copy()
    else:
        pts_end = camera_to_end(pts, R_cam2tcp, t_cam2tcp)
    if str(ref).lower() in ("flange", "ee", "ee_site", "link6"):
        pts_end = pts_end - np.array([0.0, 0.0, float(tool_offset_mm)])
    base = (pts_end / 1000.0) @ R_bt.T + p_bt
    if abs(float(yaw_deg)) > 1e-12:                 # 绕基座 z 转（左右摆一点）
        a = math.radians(float(yaw_deg))
        c, s = math.cos(a), math.sin(a)
        base = base @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    off = np.asarray(offset_mm, dtype=float).ravel()
    if off.size == 3 and np.any(off):
        base = base + off / 1000.0
    return base


def camera_in_base(R_cam2tcp, t_cam2tcp, pose, *, ref: str = "tcp",
                   tool_offset_mm: float = 42.7, offset_mm=(0.0, 0.0, 0.0),
                   yaw_deg: float = 0.0) -> np.ndarray:
    """相机光心在基座系的位置[m]。

    相机光心 = 标定末端系的原点 + ``t``（``t`` 就是相机→末端的平移），所以这一步与
    "点云自带哪个坐标系"无关。
    """
    t = np.asarray(t_cam2tcp, dtype=float).reshape(1, 3)
    return transform_to_base(t, np.eye(3), np.zeros(3), pose, frame="cam", ref=ref,
                             tool_offset_mm=tool_offset_mm, offset_mm=offset_mm,
                             yaw_deg=yaw_deg)[0]


def write_points_obj(path, points_m, radius_m: float) -> Path:
    """把点云写成 OBJ：**每个点一个小八面体**（6 顶点 / 8 面）。

    MuJoCo 没有"点"这个图元，所以用八面体：比立方体少 1/3 的面，而且从任何角度看都不会消失
    （朝向敏感的面片/正方形在侧视时会看不见）。
    """
    pts = np.asarray(points_m, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise PointCloudError("点云是空的，没什么可写")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = (pts[:, None, :] + float(radius_m) * OCTA_VERT[None, :, :]).reshape(-1, 3)
    faces = ((np.arange(pts.shape[0], dtype=np.int64)[:, None] * 6 + 1)
             + OCTA_FACE.reshape(1, -1)).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as fp:
        np.savetxt(fp, verts, fmt="v %.5f %.5f %.5f")
        np.savetxt(fp, faces, fmt="f %d %d %d")
    return path


def depth_band_masks(depth_mm, bands: int) -> list[tuple[tuple[float, float], np.ndarray]]:
    """把深度等分成 ``bands`` 带：返回 ``[((近mm, 远mm), 掩码), ...]``（空带会被丢掉）。"""
    d = np.asarray(depth_mm, dtype=float).ravel()
    if bands is None or int(bands) <= 1:
        lo, hi = float(d.min()), float(d.max())
        return [((lo, hi), np.ones(d.shape[0], dtype=bool))]
    edges = np.linspace(float(d.min()), float(d.max()), int(bands) + 1)
    out: list[tuple[tuple[float, float], np.ndarray]] = []
    for i in range(int(bands)):
        if i < len(edges) - 2:
            mask = (d >= edges[i]) & (d < edges[i + 1])
        else:
            mask = (d >= edges[i]) & (d <= edges[i + 1])
        if np.any(mask):
            out.append(((float(edges[i]), float(edges[i + 1])), mask))
    return out


# =============================================================== 生成场景
@dataclass
class CloudScene:
    """生成好的"带点云的世界场景"（界面 / 命令行共用的返回值）。"""

    scene: Path
    meshes: list[Path]
    points_total: int
    points_used: int
    bands: list[dict] = field(default_factory=list)   # [{n, rgba, mm}]
    cart_height: float = 0.0
    ms: float = 0.0

    @property
    def mesh_mb(self) -> float:
        return sum(p.stat().st_size for p in self.meshes if p.is_file()) / 1e6

    def summary(self) -> str:
        b = "、".join(f"{x['n']}点" for x in self.bands) or "—"
        return (f"场景 {self.scene.name}：用 {self.points_used}/{self.points_total} 点，"
                f"分 {len(self.meshes)} 组（{b}），网格 {self.mesh_mb:.1f} MB，"
                f"生成 {self.ms:.0f} ms")


def build_scene(points_base_m, depth_mm=None, *, scene_out=OUT_SCENE, mesh_dir=MESH_DIR,
                base_scene=BASE_SCENE, cart_height: float = DEFAULT_CART_HEIGHT_M,
                show_cart: bool = True, point_mm: float = DEFAULT_POINT_R_MM,
                bands: int = DEFAULT_BANDS, max_points: int = DEFAULT_MAX_POINTS,
                prefix: str = DEFAULT_MESH_PREFIX, color=None) -> CloudScene:
    """把（基座系）点云写成网格，并生成一份**带点云的世界场景 XML**。

    生成物（都放 ``assets/`` 下，和基场景同目录 —— 这样 ``<include>`` 和 ``meshdir`` 才解析得对）：

    * ``assets/meshes/pc_cloud_*.obj`` —— 每个深度带一个网格（场景的 ``meshdir`` 指向这里）；
    * ``assets/revA1_pc_scene.xml`` —— 基场景 + 点云几何 + 小车 + 地面下移到小车脚下。
    """
    t0 = time.perf_counter()
    pts = np.asarray(points_base_m, dtype=float).reshape(-1, 3)
    dep = None if depth_mm is None else np.asarray(depth_mm, dtype=float).ravel()
    if dep is not None and dep.size != pts.shape[0]:
        dep = None
    total = int(pts.shape[0])
    pts, dep, _step = subsample(pts, max_points, dep)
    if pts.shape[0] == 0:
        raise PointCloudError("点云是空的（都被深度过滤掉了？）")

    specs: list[tuple[np.ndarray, tuple, tuple | None]] = []
    if dep is None:
        specs.append((np.ones(pts.shape[0], dtype=bool), tuple(color or DEPTH_COLORS[0]), None))
    else:
        for i, (rng, mask) in enumerate(depth_band_masks(dep, bands)):
            specs.append((mask, tuple(color or DEPTH_COLORS[i % len(DEPTH_COLORS)]), rng))

    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for old in mesh_dir.glob(f"{prefix}_*.obj"):          # 清掉上一轮的网格
        try:
            old.unlink()
        except OSError:
            pass

    meshes: list[Path] = []
    band_info: list[dict] = []
    for i, (mask, rgba, rng) in enumerate(specs):
        sub = pts[mask]
        if sub.shape[0] == 0:
            continue
        meshes.append(write_points_obj(mesh_dir / f"{prefix}_{i}.obj", sub,
                                       float(point_mm) / 1000.0))
        band_info.append(dict(n=int(sub.shape[0]), rgba=rgba, mm=rng))

    txt = Path(base_scene).read_text(encoding="utf-8")
    if "</asset>" not in txt or "<worldbody>" not in txt:
        raise PointCloudError(f"基场景 {Path(base_scene).name} 里找不到 </asset> / <worldbody>")
    txt = txt.replace("</asset>", "\n".join(
        f'    <mesh name="{prefix}_{i}" file="{p.name}"/>' for i, p in enumerate(meshes))
        + "\n  </asset>", 1)

    body: list[str] = []
    if show_cart and float(cart_height) > 1e-6:
        h = float(cart_height)
        body.append(f'    <geom name="pc_cart" type="box" size="0.30 0.26 {h / 2:.4f}" '
                    f'pos="0 0 {-h / 2:.4f}" rgba="0.30 0.32 0.36 1" '
                    f'contype="0" conaffinity="0"/>')
        body.append('    <geom name="pc_cart_top" type="box" size="0.31 0.27 0.008" '
                    'pos="0 0 -0.008" rgba="0.46 0.48 0.53 1" contype="0" conaffinity="0"/>')
    for i, band in enumerate(band_info):
        r, g, b, a = band["rgba"]
        body.append(f'    <geom name="pc_pts_{i}" type="mesh" mesh="{prefix}_{i}" '
                    f'rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" contype="0" conaffinity="0"/>')
    txt = txt.replace("<worldbody>", "<worldbody>\n" + "\n".join(body), 1)

    if float(cart_height) > 1e-6:                    # 地面挪到小车脚下；名义作业点标记贴到车顶
        txt = re.sub(r'(<geom name="floor"[^>]*?)pos="[^"]*"',
                     lambda m: f'{m.group(1)}pos="0 0 {-float(cart_height):.4f}"', txt, count=1)
        txt = re.sub(r'(<geom name="target_pad"[^>]*?)pos="[^"]*"',
                     r'\1pos="0.4 0 0.002"', txt, count=1)
    txt = re.sub(r'<statistic[^>]*/>',
                 '<statistic center="0 -0.45 -0.15" extent="2.8"/>', txt, count=1)

    scene_out = Path(scene_out)
    scene_out.parent.mkdir(parents=True, exist_ok=True)
    scene_out.write_text(txt, encoding="utf-8")
    return CloudScene(scene=scene_out, meshes=meshes, points_total=total,
                      points_used=int(pts.shape[0]), bands=band_info,
                      cart_height=float(cart_height),
                      ms=(time.perf_counter() - t0) * 1000.0)


# =============================================================== 一步到位 + 报告
def list_clouds() -> list[Path]:
    """``point_cloud/`` 下的点云文件（``*.json`` / ``*.npy``），按名字排序。"""
    if not CLOUD_DIR.is_dir():
        return []
    return sorted((p for p in CLOUD_DIR.iterdir() if p.suffix.lower() in (".json", ".npy")),
                  key=lambda p: p.name)


def default_cloud() -> Path:
    """没指定文件时挑一个：优先带 ``o2e`` 的（那就是现成的 3D 点），否则第一个。"""
    files = list_clouds()
    if not files:
        raise PointCloudError(f"{CLOUD_DIR} 里没有点云文件（*.json / *.npy）")
    for p in files:
        if "o2e" in p.name.lower():
            return p
    return files[0]


def geometry_report(cloud: Cloud, points_base, cam_base, *, cart_height: float = DEFAULT_CART_HEIGHT_M,
                    R=None, t=None, ref: str = "tcp", tool_offset_mm: float = 0.0,
                    frame: str = "cam", frame_probe: dict | None = None) -> list[str]:
    """把"这帧点云摆对了没有"要看的数都列出来（界面显示 / 命令行打印同一份）。"""
    lo, hi = cloud.bbox()
    pb = np.asarray(points_base, dtype=float).reshape(-1, 3)
    blo, bhi = pb.min(axis=0), pb.max(axis=0)
    d = cloud.depth_mm
    declared = cloud.meta.get("declared_frame")
    if frame_probe and frame_probe.get("scores"):
        sc = frame_probe["scores"]
        how = (f"反投影命中 ROI：末端系 {sc.get('end', 0.0) * 100:.1f}% "
               f"vs 相机系 {sc.get('cam', 0.0) * 100:.1f}%")
    else:
        how = "按文件声明（没有 ROI 元数据）" if declared else "默认按相机光学系"
    lines = [
        f"点云    : {cloud.name}（{cloud.source}）{cloud.n} 点"
        + (f"，ROI {cloud.box}" if cloud.box else ""),
        f"坐标系  : {cloud.frame_label()}"
        + (f"· 文件声明 {declared}" if declared else "· 文件没声明")
        + f" · {how} → "
        + ("直接用（不再重复乘手眼标定）" if frame == "end" else "先乘手眼标定到末端系，再搬"),
    ]
    if cloud.meta.get("depth_shape"):
        shape = cloud.meta["depth_shape"]
        lines.append(f"深度图  : {shape[1]}×{shape[0]}（scale {cloud.meta.get('depth_scale')}，"
                     f"畸变校正{'开' if cloud.meta.get('undistort') else '关'}）")
    lines.append(f"相机系  : x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] "
                 f"z[{d.min():.0f},{d.max():.0f}] mm（相机 z 就是深度：越小 = 离相机越近）")
    lines.append(f"相机位置: 基座系 ({cam_base[0]:.3f}, {cam_base[1]:.3f}, {cam_base[2]:.3f}) m"
                 f" · 离地 {cam_base[2] + float(cart_height):.3f} m")
    lines.append(f"基座系  : x[{blo[0]:.3f},{bhi[0]:.3f}] y[{blo[1]:.3f},{bhi[1]:.3f}] "
                 f"z[{blo[2]:.3f},{bhi[2]:.3f}] m")
    lines.append(f"离地高度: {blo[2] + float(cart_height):.2f} ~ {bhi[2] + float(cart_height):.2f} m"
                 f"（地面 = 小车高度 {float(cart_height):.2f} m）")
    if R is not None and t is not None:
        orth = float(np.abs(np.asarray(R).T @ np.asarray(R) - np.eye(3)).max())
        lines.append(f"标定    : |RᵀR−I|max = {orth:.1e}"
                     f"（{'正交 OK' if orth < 1e-6 else '注意：R 不太正交'}）"
                     f"· |t| = {float(np.linalg.norm(t)):.1f} mm · 参考点 {ref}"
                     + (f"（工具长 {tool_offset_mm:.1f} mm 已扣）" if ref == "flange" else ""))
    return lines


def build_cloud_scene(cloud_path=None, *, cloud: Cloud | None = None, pose=DEFAULT_POSE,
                      extrinsic=DEFAULT_EXTRINSIC, intrinsic=DEFAULT_INTRINSIC,
                      frame: str = "auto", ref: str = "tcp", tool_offset_mm: float | None = None,
                      cart_height: float = DEFAULT_CART_HEIGHT_M, show_cart: bool = True,
                      max_points: int = DEFAULT_MAX_POINTS, point_mm: float = DEFAULT_POINT_R_MM,
                      bands: int = DEFAULT_BANDS, stride: int = 1,
                      depth_min_mm: float = MIN_DEPTH_MM, depth_max_mm=None,
                      offset_mm=(0.0, 0.0, 0.0), yaw_deg: float = 0.0, undistort: bool = True,
                      scene_out=OUT_SCENE, mesh_dir=MESH_DIR, base_scene=BASE_SCENE) -> dict:
    """读点云 → 变换到基座系 → 生成"带点云的场景" + 体检报告（界面/命令行都调这一个）。

    ``frame``：``"auto"``（默认，用文件 ROI 反投影判定）/ ``"cam"``（相机光学系）/ ``"end"``（末端系）。
    返回 ``dict(cloud, points_base, cam_base, scene, report, R, t, pose, frame, frame_probe)``。
    """
    if tool_offset_mm is None:                       # 默认用机型表里的工具长度
        try:
            import revA1_spec as _spec                            # noqa: PLC0415
            tool_offset_mm = float(_spec.TOOL_TIP_LOCAL_Z) * 1000.0
        except Exception:  # noqa: BLE001
            tool_offset_mm = 42.7
    if cloud is None:
        cloud = load_cloud(cloud_path or default_cloud(), intrinsic_path=intrinsic, stride=stride,
                           depth_min_mm=depth_min_mm, depth_max_mm=depth_max_mm,
                           undistort=undistort)
    R, t = load_extrinsic(extrinsic)
    pose6 = parse_pose6(pose)
    probe = None
    if not frame or str(frame).lower() in ("auto", "detect"):
        probe = detect_cloud_frame(cloud, R=R, t=t, intrinsic_path=intrinsic)
        use_frame = probe["frame"]
    else:
        use_frame = normalize_frame(frame) or "cam"
    cloud.frame = use_frame                           # 让 cloud.summary() 也反映真实坐标系
    pts_base = transform_to_base(cloud.points_mm, R, t, pose6, frame=use_frame, ref=ref,
                                 tool_offset_mm=float(tool_offset_mm),
                                 offset_mm=offset_mm, yaw_deg=yaw_deg)
    cam_base = camera_in_base(R, t, pose6, ref=ref, tool_offset_mm=float(tool_offset_mm),
                              offset_mm=offset_mm, yaw_deg=yaw_deg)
    scene = build_scene(pts_base, cloud.depth_mm, scene_out=scene_out, mesh_dir=mesh_dir,
                        base_scene=base_scene, cart_height=cart_height, show_cart=show_cart,
                        point_mm=point_mm, bands=bands, max_points=max_points)
    report = geometry_report(cloud, pts_base, cam_base, cart_height=cart_height, R=R, t=t,
                             ref=ref, tool_offset_mm=float(tool_offset_mm),
                             frame=use_frame, frame_probe=probe)
    report.append(scene.summary())
    return dict(cloud=cloud, points_base=pts_base, cam_base=cam_base, scene=scene,
                report=report, R=R, t=t, pose=pose6, frame=use_frame, frame_probe=probe)


def render_preview(scene_path, out_png, *, lookat=None, distance: float = 3.4,
                   azimuth: float = 150.0, elevation: float = -18.0,
                   width: int = 1000, height: int = 700) -> Path:
    """离屏渲染一张预览图（机械臂 + 点云同框）。需要 mujoco / imageio。"""
    try:
        import mujoco                                              # noqa: PLC0415
        import imageio.v2 as imageio                               # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise PointCloudError(f"没有 mujoco / imageio，出不了图：{exc}") from exc
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation = float(azimuth), float(elevation)
    cam.distance = float(distance)
    if lookat is not None:
        cam.lookat[:] = np.asarray(lookat, dtype=float).ravel()[:3]
    r = mujoco.Renderer(model, int(height), int(width))
    try:
        r.update_scene(data, camera=cam)
        img = r.render()
    finally:
        r.close()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_png), img)
    return out_png


# =============================================================== 自检
def run_selftest(args) -> int:
    """无界面自检：标定 / 姿态 / 点云读取（含深度图）/ 变换 / 建模 / 渲染。"""
    print("=" * 78)
    print("point_cloud.py 自检（用 point_cloud/ 里的真实数据；不需要真机）")
    print("=" * 78)
    fails: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    # 1) 手眼标定
    R, t = load_extrinsic()
    orth = float(np.abs(R.T @ R - np.eye(3)).max())
    det = float(np.linalg.det(R))
    check("标定 R 正交、det = 1", orth < 1e-6 and abs(det - 1.0) < 1e-6,
          f"|RᵀR−I|max = {orth:.1e}，det = {det:.9f}")
    check("标定 t 大小 ≈ 245.6 mm（相机沿工具轴前移 24.6 cm）",
          abs(float(np.linalg.norm(t)) - 245.56) < 0.5, f"|t| = {float(np.linalg.norm(t)):.2f} mm")
    check("相机光轴 ≈ 工具轴（eye-in-hand，朝前看）", float(R[2, 2]) > 0.99,
          f"R·ẑ = {np.round(R[:, 2], 4).tolist()}")

    # 2) 拍摄姿态（与真机 pose / robot_link 同一套欧拉约定）
    pose6 = parse_pose6(DEFAULT_POSE)
    R_bt, p_bt = pose_matrix(pose6)
    check("拍摄姿态位置 = 真机 pose（0.1109, -0.0055, 0.3964）m",
          bool(np.allclose(p_bt, (0.110862, -0.005474, 0.396436), atol=1e-6)),
          f"p = {np.round(p_bt, 4).tolist()}")
    try:
        import robot_link as rl                                    # noqa: PLC0415
        axis = rl.euler_xyz_to_axis(*np.radians(pose6[3:6]))
        check("欧拉角约定与 robot_link 一致（外旋 XYZ）",
              bool(np.allclose(R_bt[:, 2], axis)), f"工具轴 {np.round(axis, 4).tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] 与 robot_link 对比（{type(exc).__name__}: {exc}）")

    # 3) 点云读取（三个文件都读一遍）
    files = list_clouds()
    check("point_cloud/ 里有数据", len(files) >= 1, f"{[p.name for p in files]}")
    for p in files[:3]:
        try:
            c = load_cloud(p, stride=2)
            med = float(np.median(c.depth_mm))
            ok = (c.n > 100 and bool(np.isfinite(c.points_mm).all()) and 100 < med < 6000)
            check(f"读点云 {p.name}", ok,
                  f"{c.n} 点｜深度 {c.depth_mm.min():.0f}~{c.depth_mm.max():.0f} mm"
                  f"（中位 {med:.0f}）｜{c.source}")
        except PointCloudError as exc:
            check(f"读点云 {p.name}", False, str(exc))

    # 4) 深度图（urinal-1.json）自己解 PNG + 内参反投影
    png_files = [p for p in files if "urinal-1" in p.name]
    if png_files:
        try:
            c1 = load_cloud(png_files[0], stride=3)
            check("深度图：PNG 解码 + 内参反投影出点云",
                  c1.n > 1000 and c1.source.startswith("json 深度图"),
                  f"{c1.n} 点｜深度 {c1.depth_mm.min():.0f}~{c1.depth_mm.max():.0f} mm｜"
                  f"图 {c1.meta.get('depth_shape')}")
        except PointCloudError as exc:
            check("深度图：PNG 解码 + 内参反投影出点云", False, str(exc))

    # 5) 坐标系判定（o2e = 光学→末端已经做过，不能重复乘手眼）
    probe_o2e = probe_png = None
    try:
        c_o2e = load_cloud(default_cloud(), stride=2)
        probe_o2e = detect_cloud_frame(c_o2e)
        check("坐标系：o2e 数据判为**末端系**（文件声明 + ROI 反投影一致）",
              probe_o2e["frame"] == "end" and probe_o2e["scores"].get("end", 0.0) > 0.9,
              f"声明 {c_o2e.meta.get('declared_frame')}｜" + "；".join(probe_o2e["evidence"]))
        check("坐标系：判错的代价（当成相机系会再乘一次手眼，命中率明显掉下来）",
              probe_o2e["scores"].get("end", 0.0) - probe_o2e["scores"].get("cam", 0.0) > 0.2,
              f"末端 {probe_o2e['scores'].get('end', 0.0) * 100:.1f}% "
              f"vs 相机 {probe_o2e['scores'].get('cam', 0.0) * 100:.1f}%")
        c_png = load_cloud(CLOUD_DIR / "urinal-1.json", stride=8)
        probe_png = detect_cloud_frame(c_png)
        check("坐标系：深度图自己反投影出来的点判为相机光学系",
              probe_png["frame"] == "cam" and probe_png["scores"].get("cam", 0.0) > 0.9,
              "；".join(probe_png["evidence"]))
    except PointCloudError as exc:
        check("坐标系判定", False, str(exc))

    # 6) 变换 + 建模 + 渲染（拿现成 3D 点那个文件走全链路）
    res = None
    try:
        res = build_cloud_scene(cloud_path=default_cloud(), pose=DEFAULT_POSE, stride=2,
                                max_points=20000, point_mm=4.0, bands=6, scene_out=OUT_SCENE)
    except PointCloudError as exc:
        check("全链路：点云 → 基座系 → 场景", False, str(exc))
    if res is not None:
        cloud, base, cam = res["cloud"], res["points_base"], res["cam_base"]
        use_frame, R, t = res["frame"], res["R"], res["t"]
        blo, bhi = base.min(axis=0), base.max(axis=0)
        # 相机光学系下的点（"深度"要按它说话）
        cam_pts = end_to_camera(cloud.points_mm, R, t) if use_frame == "end" else cloud.points_mm
        depth = cam_pts[:, 2]
        check("相机在基座系 ≈ (0.120, -0.201, 0.249) m",
              bool(np.allclose(cam, (0.120, -0.201, 0.249), atol=5e-3)),
              f"{np.round(cam, 4).tolist()}｜离地 {cam[2] + 1.1:.3f} m")
        check("点云整片在机械臂前方（基座系 y < 0）", bhi[1] < 0.0,
              f"y ∈ [{blo[1]:.3f}, {bhi[1]:.3f}] m")
        check("点云在基座下方（z < 0，朝地面）", bhi[2] < 0.1,
              f"z ∈ [{blo[2]:.3f}, {bhi[2]:.3f}] m")
        check("末端系点云都在工具轴正前方（文件 z > 0）" if use_frame == "end" else "相机深度 > 0",
              float(cloud.points_mm[:, 2].min()) > 0.0,
              f"z ∈ [{cloud.points_mm[:, 2].min():.0f}, {cloud.points_mm[:, 2].max():.0f}] mm")
        check("相机深度 0.5 ~ 2.5 m（越小越靠近相机）",
              500.0 < float(depth.min()) and float(depth.max()) < 2500.0,
              f"{depth.min():.0f} ~ {depth.max():.0f} mm")
        zw = base[:, 2] + 1.1
        hist, edges = np.histogram(zw, bins=10)
        k = int(np.argmax(hist))
        peak = 0.5 * (edges[k] + edges[k + 1])
        check("离地高度峰值落在 0.3 ~ 1.2 m（小便池的安装高度）", 0.3 < peak < 1.2,
              f"峰值 {peak:.2f} m｜离地 {zw.min():+.2f}~{zw.max():+.2f} m｜"
              f"埋地下 {100 * (zw < 0).mean():.1f}%")
        tcp = np.asarray(pose_matrix(pose6)[1], dtype=float)      # 拍摄时工具尖（基座系）
        q10, q90 = np.quantile(depth, (0.1, 0.9))
        near, far = base[depth <= q10], base[depth >= q90]
        dn = float(np.linalg.norm(near - tcp, axis=1).mean())
        df = float(np.linalg.norm(far - tcp, axis=1).mean())
        check("深度关系：深度小的那批点离机械臂更近（z 越小越近）", dn < df,
              f"最近 10% 平均 {dn:.2f} m ＜ 最远 10% 平均 {df:.2f} m"
              f"（工具尖 {np.round(tcp, 3).tolist()}）")
        scene = res["scene"]
        check("生成了带点云的场景 + 网格", scene.scene.is_file() and len(scene.meshes) == 6,
              scene.summary())
        try:
            import mujoco                                         # noqa: PLC0415
            model = mujoco.MjModel.from_xml_path(str(scene.scene))
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                     for i in range(model.ngeom)]
            n_pts = sum(1 for n in names if n and n.startswith("pc_pts"))
            check("MuJoCo 能加载这个场景（6 轴机械臂 + 点云几何）",
                  model.nq == 6 and n_pts >= 6, f"nq={model.nq} ngeom={model.ngeom} 点云组={n_pts}")
            fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            fz = float(model.geom_pos[fid][2]) if fid >= 0 else float("nan")
            check("地面跟着小车高度走（z ≈ -1.1）", abs(fz + 1.1) < 1e-6, f"地面 z = {fz:.4f} m")
            look = 0.5 * np.asarray(base.mean(axis=0))
            img = render_preview(scene.scene, RUNS / "pc_selftest.png", lookat=look,
                                 distance=3.0, azimuth=165.0, elevation=-14.0,
                                 width=900, height=620)
            check("离屏渲染出图（机械臂 + 点云同框）",
                  img.is_file() and img.stat().st_size > 5000, f"{img.name} {img.stat().st_size} B")
            try:                                  # 客观一点：数一数画面里有多少"点云颜色"的像素
                import imageio.v2 as imageio2                   # noqa: PLC0415
                arr = np.asarray(imageio2.imread(str(img)), dtype=float).reshape(-1, 3)
                chroma = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1.0)
                hit = 0
                for band in scene.bands:
                    c = np.asarray(band["rgba"][:3], dtype=float)
                    c = c / c.sum()
                    hit += int((np.linalg.norm(chroma - c, axis=1) < 0.08).sum())
                check("画面里真的看得见点云（按分带色相数像素）", hit > 2000, f"{hit} 个像素命中")
            except Exception as exc:  # noqa: BLE001
                print(f"  [SKIP] 点云可见性统计（{type(exc).__name__}: {exc}）")
        except ImportError:
            print("  [SKIP] 没有装 mujoco，跳过「建模 + 渲染」检查")

    # 7) 抽稀 / 分带 / 微调
    if res is not None:
        pts, dep, step = subsample(base, 5000, cloud.depth_mm)
        check("抽稀：数量受控、深度跟着一起抽",
              0 < pts.shape[0] <= 5000 and dep.shape[0] == pts.shape[0],
              f"{base.shape[0]} → {pts.shape[0]} 点（步长 {step}）")
        masks = depth_band_masks(cloud.depth_mm, 6)
        total = sum(int(m.sum()) for _, m in masks)
        check("深度分 6 带：正好覆盖所有点", total == cloud.n and len(masks) == 6,
              f"每带 {[int(m.sum()) for _, m in masks]}")
        b_flange = transform_to_base(cloud.points_mm, R, t, pose6, frame=use_frame,
                                     ref="flange", tool_offset_mm=42.7)
        dmm = float(np.linalg.norm(b_flange - base, axis=1).max())
        check("参考点切到法兰：整片云沿工具轴退一个工具长（42.7 mm）",
              abs(dmm - 0.0427) < 1e-6, f"{dmm * 1000:.2f} mm")
        b_off = transform_to_base(cloud.points_mm, R, t, pose6, frame=use_frame,
                                  offset_mm=(100.0, -50.0, 20.0))
        check("微调 offset 生效（基座系平移）",
              bool(np.allclose(b_off - base, (0.1, -0.05, 0.02), atol=1e-9)))
        b_yaw = transform_to_base(cloud.points_mm, R, t, pose6, frame=use_frame, yaw_deg=90.0)
        check("微调 yaw 生效（绕基座 z 转 90°）",
              bool(np.allclose(b_yaw[:, 0], -base[:, 1], atol=1e-9)))

    # 8) 真机**当前**姿态也能摆（可选）
    if res is not None and not getattr(args, "no_probe", False):
        try:
            import robot_link as rl                                # noqa: PLC0415
            st = rl.HttpStateSource(rl.DEFAULT_HTTP_URL, timeout=2.0).poll_once()
            real_pose = tuple(float(v) for v in np.r_[st.pose[:3] * 1000.0,
                                                      np.degrees(st.pose[3:6])])
            res2 = build_cloud_scene(cloud=res["cloud"], pose=real_pose, stride=2, max_points=5000,
                                     scene_out=ASSETS / "revA1_pc_scene_live.xml")
            b2 = res2["points_base"]
            check("用真机当前姿态也能摆出来", bool(np.isfinite(b2).all()),
                  f"真机姿态 {np.round(real_pose, 2).tolist()} → "
                  f"基座系 z ∈ [{b2[:, 2].min():.2f}, {b2[:, 2].max():.2f}] m")
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] 真机当前姿态（{type(exc).__name__}: {exc}）")

    print("-" * 78)
    if fails:
        print(f"结论：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结论：全部通过 [OK]")
    return 0


# =============================================================== 命令行
def _print_report(lines) -> None:
    for line in lines:
        print("  " + line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="点云 ↔ 机械臂：按手眼标定 + 拍摄姿态把点云放进 MuJoCo 场景",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cloud", default="", help="点云文件（默认 point_cloud/ 里带 o2e 的那个）")
    ap.add_argument("--list", action="store_true", help="列出 point_cloud/ 里的点云文件")
    ap.add_argument("--pose", default=pose_text(DEFAULT_POSE),
                    help="拍摄姿态：x y z(mm) + rx ry rz(deg)")
    ap.add_argument("--extrinsic", default=str(DEFAULT_EXTRINSIC), help="手眼标定（OpenCV XML）")
    ap.add_argument("--intrinsic", default=str(DEFAULT_INTRINSIC), help="相机内参（读深度图时才用）")
    ap.add_argument("--frame", default="auto", choices=("auto", "cam", "end"),
                    help="点云自带哪个坐标系：auto = 用文件 ROI 反投影判定（o2e 数据判为 end）")
    ap.add_argument("--ref", default="tcp", choices=("tcp", "flange"),
                    help="标定的\"末端\"是哪个点：tcp = 工具尖（真机 pose），flange = 法兰盘")
    ap.add_argument("--cart-height", type=float, default=DEFAULT_CART_HEIGHT_M, help="小车高度[m]")
    ap.add_argument("--no-cart", action="store_true", help="不画小车、地面也不下移")
    ap.add_argument("--points", type=int, default=DEFAULT_MAX_POINTS, help="点数上限（0 = 不抽稀）")
    ap.add_argument("--point-mm", type=float, default=DEFAULT_POINT_R_MM, help="每个点的尺寸[mm]")
    ap.add_argument("--bands", type=int, default=DEFAULT_BANDS, help="按深度分几带颜色（1 = 单色）")
    ap.add_argument("--stride", type=int, default=1, help="深度图像素步长（读深度图时抽稀）")
    ap.add_argument("--depth-min", type=float, default=MIN_DEPTH_MM, help="深度下限[mm]")
    ap.add_argument("--depth-max", type=float, default=0.0, help="深度上限[mm]（0 = 不限）")
    ap.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("DX", "DY", "DZ"), help="手动微调：基座系平移[mm]")
    ap.add_argument("--yaw", type=float, default=0.0, help="手动微调：绕基座 z 旋转[deg]")
    ap.add_argument("--no-undistort", action="store_true", help="深度图反投影不做畸变校正")
    ap.add_argument("--out", default=str(OUT_SCENE), help="生成的场景 XML 路径")
    ap.add_argument("--png", default="", help="顺便离屏渲一张预览图（如 runs/pc.png）")
    ap.add_argument("--selftest", action="store_true", help="全链路自检（标定/点云/变换/建模）")
    ap.add_argument("--no-probe", action="store_true", help="自检时不去读真机的当前姿态")
    args = ap.parse_args(argv)

    if args.list:
        files = list_clouds()
        print(f"{CLOUD_DIR} 下 {len(files)} 个点云文件：")
        for p in files:
            print(f"  {p.name}  {p.stat().st_size / 1e6:.2f} MB")
        return 0
    if args.selftest:
        return run_selftest(args)

    res = build_cloud_scene(cloud_path=args.cloud or None, pose=args.pose,
                            extrinsic=args.extrinsic, intrinsic=args.intrinsic,
                            frame=args.frame, ref=args.ref, cart_height=args.cart_height,
                            show_cart=not args.no_cart, max_points=args.points,
                            point_mm=args.point_mm, bands=args.bands, stride=args.stride,
                            depth_min_mm=args.depth_min,
                            depth_max_mm=(args.depth_max or None),
                            offset_mm=tuple(args.offset), yaw_deg=args.yaw,
                            undistort=not args.no_undistort, scene_out=args.out)
    print("=" * 78)
    print("点云 → 机械臂基座系（体检报告）")
    print("=" * 78)
    _print_report(res["report"])
    print(f"\n场景已写好：{res['scene'].scene}")
    print(f"看效果    ：python revA1_gui.py --scene \"{res['scene'].scene}\"")
    if args.png:
        base_pts = np.asarray(res["points_base"], dtype=float)
        look = 0.5 * base_pts.mean(axis=0)
        size = float(np.linalg.norm(base_pts.max(axis=0) - base_pts.min(axis=0)))
        out = render_preview(res["scene"].scene, args.png, lookat=look,
                             distance=max(size * 1.6, 2.0), azimuth=165.0, elevation=-14.0,
                             width=1000, height=700)
        print(f"预览图已存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

