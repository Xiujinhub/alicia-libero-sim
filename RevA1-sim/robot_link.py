#!/usr/bin/env python3
"""真机 ↔ 仿真 的姿态链路：**状态源（HTTP 轮询 / UDP 广播）→ 解析 → 映射 → 每帧跟随**。

它在整个系统里的位置
--------------------
::

    真机控制器 ┬--(HTTP: GET /api/state, JSON)─┐
               └--(UDP 广播/组播: 六轴关节角)──┴─→ StateSource（接收线程 + 解析）
                                                        │  每帧一问（线程安全快照）
                                                        ▼
                                   JointLink.update() → 映射/平滑/限速/看门狗
                                                        ▼
                                   ArmSim.follow()   →  MuJoCo 模型实时动
                                   （界面里的「真机跟随」卡片）

界面只关心"6 个关节角 + 这包是什么时候来的 + 跟得准不准"，不该管 socket / 线程 / 报文格式，
所以这些都收在这个模块里（分工和 ``alicia_libero/leader_arm.py`` 一致）。

本机实测的真机（TB6-R5-RevA1）接口
----------------------------------
**① UDP 广播（推荐，最实时）**：真机往局域网广播状态，本机绑 :6001 就能收到
（实测源地址 ``192.168.66.169:44558``，约 **96 Hz**，一包约 1.1 kB JSON）：

.. code-block:: json

    {
      "data": {
        "joints": [1.7285, -2.3903, 2.1147, 1.1962, 1.7299, -1.7549],   # 弧度
        "pose":   [0.1109, -0.0055, 0.3964, 3.4320, 0.6090, 1.9507],    # 工具尖 x/y/z[m] + 外旋 XYZ 欧拉角[rad]
        "vel": [...], "torque": [...], "enabled": [...], "robottarget": [...],
        "running": true, "speed": "v25", "ts": 1789891582032
      },
      "type": "state", "ver": 1
    }

另外约 1 Hz 会来一包**心跳**（``{"data":{"stream_alive":true,"system_is_init":true,"ts":..}}``，
没有 joints）——它不算错误，本模块单独计数（``heartbeats``）。

**② HTTP 状态接口（同内容，方便命令行/远程查）**：``http://192.168.66.169/`` 只是个 H5 前端
（nginx），真正的状态在 **8080 端口**：

.. code-block:: json

    GET http://192.168.66.169:8080/api/state      # 返回 {"state": {…与上面 data 同字段…}}

两种方式内容一样（都带 joints / pose / vel / torque），界面里可以任选或用「自动」
（HTTP 与 UDP 同时开，哪路新用哪路）。

把 ``joints`` 直接灌进模型 qpos 后 FK 出来的 ``tool_site`` 与真机 ``pose[:3]`` 实测差 0.5 mm，
用 ``pose[3:6]``（外旋 XYZ 欧拉角）算出的工具轴与模型工具轴完全一致
——**所以真机与仿真的关节约定同号，关节角可以直接镜像**（``--probe`` 会现场复核这件事）。

为什么格式要"猜"
----------------
UDP 广播各家写法都不一样，本模块**先把常见写法都认下来**，认不了就把原始字节预览打出来
（界面上直接显示），照着它改两个选项即可：

=====  =====================================================================
格式   认什么
=====  =====================================================================
json   ``{"joints":[1,2,3,4,5,6]}`` / ``{"joint1":..,"joint6":..}`` /
       ``{"q":[..],"unit":"deg"}`` / ``{"state":{"joints":[..]}}``（键名大小写、下划线都无所谓）
csv    ``1,2,3,4,5,6``、``[1 2 3 4 5 6]``、``q=1,2,3,4,5,6;``（抓文本里的数字）
f64    6 个 float64（小端，48 字节）
f32    6 个 float32（小端，24 字节）
i16    6 个 int16（小端；默认按 0.01°/LSB 折算，可用 ``--i16-scale`` 改）
auto   按上面顺序自动试（先文本、再二进制）
=====  =====================================================================

单位也是自动判断：报文里写了 ``unit`` 就听它的；否则 ``|角| > 7`` 当**度**，不然当**弧度**。

映射约定（和 ``leader_arm.py`` 同一套）
--------------------------------------
* ``target = q_raw * signs + offsets``；**绝对模式**（默认）= 完全镜像真机姿态；
  **相对模式** = 使能瞬间记下"真机基准 / 仿真基准"，之后只镜像增量（两边本来就不同姿时用）；
* ``max_speed_deg``：每秒最多允许变化多少度，防真机跳变/毛刺把仿真"甩"出去（0 = 不限）；
* ``smooth``：0 = 最跟手；> 0 是时间常数[s]，越大越平滑但滞后（建议 0.05 ~ 0.15）；
* 看门狗 ``timeout``：超过这么久没有新包 → ``update()`` 返回 ``stale=True``，
  界面按策略"保持不动"或"回 home"（本模块不自己动，避免和界面的急停抢）。

安全约定
--------
* **只读**：本模块只接收/轮询，不往真机发任何指令（下指令要走真机自己的 API，见 README）；
* 接收线程只做"收 + 解析 + 存最近一包"，映射/平滑/限速都在调用线程算（无锁竞争）；
* 报文认不出来不会崩：记一条 ``errors`` 并把原始字节预览留给界面显示。

命令行
------
::

    python robot_link.py --probe                          # 读一次真机状态并复核关节/姿态约定
    python robot_link.py --listen --port 6001             # 只听 UDP 广播并打印（真机就在这里广播）
    python robot_link.py --emit-demo --target 127.0.0.1:6001 --format json --unit deg
    python robot_link.py --selftest                        # 全链路自检（解析 + 假真机 HTTP + UDP 回环）

界面里用它的入口：``revA1_gui.py`` 的「真机跟随」卡片（``python revA1_gui.py --follow``）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ 默认参数
# 本机实测真机（TB6-R5-RevA1）的状态接口：nginx 的 H5 在 80，状态在 8080。
DEFAULT_HOST = "192.168.66.169"
DEFAULT_HTTP_URL = f"http://{DEFAULT_HOST}:8080/api/state"
DEFAULT_UDP_PORT = 6001           # 真机实测：往 255.255.255.255:6001 广播状态（~96 Hz）
DEFAULT_HTTP_HZ = 30.0            # HTTP 轮询频率（真机的状态流 ~30 Hz，再快没意义）
HTTP_TIMEOUT = 1.0
RECV_BUFFER = 65535
SOCKET_TIMEOUT = 0.2              # 接收线程的阻塞上限，用来定期看退出标志
RATE_SAMPLES = 24                 # 算数据率时回看的最近多少个包
JOINTS_N = 6

SOURCES = ("auto", "http", "udp")
SOURCE_LABELS = ("自动（HTTP + UDP，谁新用谁）", "HTTP 状态接口", "UDP 广播")
FORMATS = ("auto", "json", "csv", "f64", "f32", "i16")
FORMAT_LABELS = ("自动", "JSON", "CSV/文本", "float64×6", "float32×6", "int16×6")
UNITS = ("auto", "deg", "rad")
UNIT_LABELS = ("自动", "度", "弧度")
MODES = ("absolute", "relative")
MODE_LABELS = ("绝对（完全镜像真机）", "相对（只镜像增量）")
TIMEOUT_POLICIES = ("hold", "home")
TIMEOUT_LABELS = ("保持不动（推荐）", "回 home")

PREVIEW_BYTES = 72                # 界面上显示多少字节的原始包
# 带 "state" / "data" 外壳的 JSON（真机 HTTP 用 state，UDP 广播用 data）：往里再挖一层
NEST_KEYS = ("state", "data", "nrt_state", "payload")
# 心跳包的标志（只有这些字段、没有 joints —— 别当成解析错误）
HEARTBEAT_KEYS = ("stream_alive", "system_is_init", "heartbeat", "keepalive")
# JSON 里可能放关节角的键名（比较时统一去掉下划线/连字符并转小写）
KEY_CANDIDATES = (
    "jointangles", "jointangle", "jointposition", "jointpositions", "jointpos",
    "joints", "joint", "jpos", "qpos", "q", "positions", "position", "pos",
    "angles", "angle", "values", "data",
)
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
DEG_HINTS = ("deg", "degree", "degrees", "d", "°", "角度")
RAD_HINTS = ("rad", "radian", "radians", "弧度")


class RobotLinkError(RuntimeError):
    """链路配置/启动/解析出错（消息是中文，可以直接显示在界面上）。"""


# =============================================================== 小工具
def parse_six(text, name: str = "六个数字") -> np.ndarray:
    """把界面输入框里的 ``1,1,1,1,1,1`` 解析成 6 个浮点数（和 leader_arm.parse_six 同义）。"""
    if isinstance(text, (list, tuple, np.ndarray, np.generic)):
        vals = [float(v) for v in np.asarray(text, dtype=float).ravel()]
    else:
        vals = [float(v) for v in NUM_RE.findall(str(text))]
    if len(vals) != 6:
        raise RobotLinkError(f"{name}要 6 个数字，现在是 {len(vals)} 个：{text!r}")
    return np.asarray(vals, dtype=float)


def parse_order(text, name: str = "列顺序") -> tuple[int, ...]:
    """把 ``0,1,2,3,4,5`` 解析成 6 个列下标（真机报文里关节角不一定在最前面）。"""
    vals = NUM_RE.findall(str(text))
    if len(vals) != 6:
        raise RobotLinkError(f"{name}要 6 个下标（从 0 开始），现在是 {len(vals)} 个：{text!r}")
    out = tuple(int(v) for v in vals)
    if any(v < 0 for v in out):
        raise RobotLinkError(f"{name}不能是负数：{text!r}")
    return out


def preview_bytes(data: bytes, limit: int = PREVIEW_BYTES) -> str:
    """原始包的预览：能当文本读就当文本，否则给前若干字节的十六进制。"""
    head = bytes(data[:limit])
    try:
        text = head.decode("utf-8")
        if all((c.isprintable() or c in "\r\n\t") for c in text):
            return text.replace("\n", "\\n")
    except UnicodeDecodeError:
        pass
    return head.hex(" ")


def looks_text(data: bytes) -> bool:
    """这一包像不像文本？（可打印字符占比 > 95%）"""
    if not data:
        return False
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError:
        return False
    good = sum(1 for c in text if c.isprintable() or c in "\r\n\t ")
    return good >= max(int(len(text) * 0.95), 1)


def unit_is_deg(vals) -> bool:
    """``auto`` 单位判定：``|角| > 7`` 当度，否则当弧度。"""
    v = np.asarray(vals, dtype=float).ravel()
    return bool(v.size and float(np.max(np.abs(v))) > 7.0)


def guess_unit(vals) -> str:
    """猜出来的单位标签（``"deg"`` / ``"rad"``），只用来显示。"""
    return "deg" if unit_is_deg(vals) else "rad"


def to_rad(vals, unit: str = "auto") -> np.ndarray:
    """把 6 个关节角换算成弧度（``auto`` 时按 :func:`unit_is_deg` 猜）。"""
    v = np.asarray(vals, dtype=float).ravel()
    hint = str(unit or "auto").strip().lower()
    if hint in RAD_HINTS or hint.startswith("rad"):
        return v
    if hint in DEG_HINTS or hint.startswith("deg"):
        return np.radians(v)
    return np.radians(v) if unit_is_deg(v) else v


def pick_order(nums, order=None) -> np.ndarray:
    """按 ``order`` 从数字列表里挑 6 个（真机报文里关节角可能不在最前面）。"""
    idx = tuple(range(JOINTS_N)) if order is None else tuple(int(i) for i in order)
    if len(idx) != JOINTS_N:
        raise RobotLinkError(f"列顺序要 6 个下标：{order!r}")
    vals = list(nums)
    if max(idx) >= len(vals):
        raise RobotLinkError(f"列顺序 {idx} 超出报文的 {len(vals)} 个数字：{vals[:12]}")
    return np.asarray([vals[i] for i in idx], dtype=float)


def _numbers_from_obj(obj) -> tuple[list[float] | None, str | None]:
    """从 JSON 对象里挖关节角 → ``(数字列表, unit 提示)``。

    认四种写法：带 ``state`` 外壳的嵌套对象、常见键（值里是数组/一串数字）、
    ``j1..j6`` / ``joint1..joint6`` 一组键、以及裸数组。
    """
    if isinstance(obj, (list, tuple)):
        nums = [float(v) for v in obj if isinstance(v, (int, float))]
        return (nums or None), None
    if not isinstance(obj, dict):
        return None, None
    unit_hint = None
    for key, val in obj.items():
        if str(key).lower().strip() in ("unit", "units", "angle_unit", "unit_type"):
            unit_hint = str(val).lower()
    # ① 带外壳的（真机 /api/state 的 {"state": {...}}）
    for key in NEST_KEYS:
        if key in obj and isinstance(obj[key], dict):
            nums, hint = _numbers_from_obj(obj[key])
            if nums:
                return nums, hint or unit_hint
    # ② 常见键：值本身是数组，或字符串里是一串数字
    for key, val in obj.items():
        norm = re.sub(r"[^a-z0-9]", "", str(key).lower())
        if norm not in KEY_CANDIDATES:
            continue
        if isinstance(val, (list, tuple)):
            nums = [float(v) for v in val if isinstance(v, (int, float))]
            if nums:
                return nums, unit_hint
        elif isinstance(val, (int, float)) and norm not in ("q", "data"):
            continue
        elif isinstance(val, str):
            nums = [float(v) for v in NUM_RE.findall(val)]
            if nums:
                return nums, unit_hint
    # ③ j1..j6 / joint1..joint6（顺序按编号）
    found: dict[int, float] = {}
    for key, val in obj.items():
        m = re.match(r"^(?:joint|j|axis|axis_?)\s*([1-9])$", str(key).strip().lower())
        if m and isinstance(val, (int, float)):
            found[int(m.group(1))] = float(val)
    if len(found) >= JOINTS_N:
        return [found[i] for i in sorted(found)[:JOINTS_N]], unit_hint
    return None, unit_hint


# =============================================================== 报文 → 关节角
def plausible(q_rad) -> bool:
    """这 6 个数像不像关节角？(|角| ≤ 7 rad ≈ 400°；解码错位通常会给出离谱的数值)"""
    v = np.asarray(q_rad, dtype=float).ravel()
    return bool(v.size == JOINTS_N and np.all(np.isfinite(v)) and np.max(np.abs(v)) <= 7.0)


def _unit_tag(vals, unit: str) -> str:
    """显示用的单位标签。"""
    return unit if unit in ("deg", "rad") else guess_unit(vals)


def _parse_json(text: str, unit: str, order, i16_scale: float):
    obj = json.loads(text)
    nums, hint = _numbers_from_obj(obj)
    if not nums:
        raise RobotLinkError("JSON 里找不到 6 个关节角")
    vals = pick_order(nums, order)
    tag = unit if unit in ("deg", "rad") else (hint if hint in ("deg", "rad") else guess_unit(vals))
    return to_rad(vals, tag), tag


def _parse_csv(text: str, unit: str, order, i16_scale: float):
    nums = [float(v) for v in NUM_RE.findall(text)]
    vals = pick_order(nums, order)
    tag = _unit_tag(vals, unit)
    return to_rad(vals, tag), tag


def _parse_bin(data: bytes, kind: str, unit: str, order, i16_scale: float):
    size, code = {"f64": (8 * JOINTS_N, "d"), "f32": (4 * JOINTS_N, "f"),
                  "i16": (2 * JOINTS_N, "h")}[kind]
    if len(data) < size:
        raise RobotLinkError(f"{kind} 至少要 {size} 字节，这包只有 {len(data)} 字节")
    vals = np.asarray(struct.unpack("<" + code * JOINTS_N, data[:size]), dtype=float)
    if kind == "i16":                      # 默认 0.01°/LSB（真机编码器常见写法）
        vals = vals * float(i16_scale)
        tag = unit if unit in ("deg", "rad") else "deg"
        return to_rad(pick_order(vals, order), tag), tag
    vals = pick_order(vals, order)
    tag = _unit_tag(vals, unit)
    return to_rad(vals, tag), tag


def parse_joint_payload(data: bytes, fmt: str = "auto", unit: str = "auto", *,
                        order=None, i16_scale: float = 0.01):
    """一包原始字节 → ``(6 个关节角[rad], 单位标签, 说明)``。

    ``fmt="auto"`` 时：文本包先当 JSON 再当 CSV；二进制包按 float32 / float64 / int16 依次试，
    并且**只认数值落在关节范围内的解释**（防 48 字节的 float64 被当成 12 个 float32 用）。
    """
    if not data:
        raise RobotLinkError("空包")
    kind = str(fmt or "auto").lower()
    if kind not in FORMATS:
        raise RobotLinkError(f"不认识的格式 {fmt!r}（可选 {'/'.join(FORMATS)}）")
    if kind == "auto":
        kinds = ("json", "csv") if looks_text(data) else ("f32", "f64", "i16")
    else:
        kinds = (kind,)

    notes: list[str] = []
    fallback = None
    for k in kinds:
        try:
            if k in ("json", "csv"):
                text = bytes(data).decode("utf-8", "replace")
                q, tag = (_parse_json if k == "json" else _parse_csv)(text, unit, order, i16_scale)
            else:
                q, tag = _parse_bin(data, k, unit, order, i16_scale)
        except RobotLinkError as exc:
            notes.append(f"{k}: {exc}")
            continue
        except (ValueError, struct.error, UnicodeDecodeError) as exc:
            notes.append(f"{k}: {type(exc).__name__}")
            continue
        if plausible(q):
            return q, tag, f"{k} → {np.round(np.degrees(q), 2).tolist()}°"
        if fallback is None:
            fallback = (q, tag, f"{k}（数值超出关节范围，存疑）")
        notes.append(f"{k}: 解出来 {np.round(q, 2).tolist()} 不像关节角")
    if fallback is not None:
        return fallback
    raise RobotLinkError("认不出这包：" + "；".join(notes[:4])
                         + f"｜原始字节：{preview_bytes(data)}")


# =============================================================== 状态对象
def _vec(obj, key) -> np.ndarray | None:
    """从 JSON 字典里取一个数值数组（取不到就 None）。"""
    if not isinstance(obj, dict):
        return None
    val = obj.get(key)
    if isinstance(val, (list, tuple)) and val:
        try:
            return np.asarray([float(v) for v in val], dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def euler_xyz_to_axis(rx: float, ry: float, rz: float) -> np.ndarray:
    """外旋 XYZ 欧拉角 → 旋转矩阵第 3 列（工具轴方向）。

    真机 ``pose[3:6]`` 就是这个约定：``R = Rz(rz) @ Ry(ry) @ Rx(rx)``（实测与模型工具轴一致）。
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return (Rz @ Ry @ Rx)[:, 2]


@dataclass
class RobotState:
    """真机的一帧状态（接收线程只负责造它，映射/显示都交给调用线程）。"""

    q_rad: np.ndarray                      # 6 个关节角（弧度，joint1..joint6）
    stamp: float = 0.0                     # 收到这一包的时刻（time.perf_counter，算延迟用）
    wall: float = 0.0                      # 收到这一包的时刻（time.time）
    src: str = ""                          # 从哪来的（URL / ip:port）
    seq: int = 0                           # 第几包
    raw: str = ""                          # 原始报文预览（界面显示）
    unit: str = "rad"                      # 原始单位（显示用）
    note: str = ""                         # 解析说明
    pose: np.ndarray | None = None          # 真机 TCP：[x,y,z, euler_xyz]（m / rad）
    vel: np.ndarray | None = None           # 关节速度[rad/s]
    torque: np.ndarray | None = None        # 关节力矩[N·m]
    enabled: np.ndarray | None = None       # 各轴使能
    info: dict = field(default_factory=dict)  # running / speed / ts / err_code ...

    @property
    def deg(self) -> np.ndarray:
        return np.degrees(self.q_rad)

    def tcp(self) -> np.ndarray | None:
        """真机工具尖位置[m]（没有 pose 时返回 None）。"""
        return None if self.pose is None else np.asarray(self.pose[:3], dtype=float)

    def axis(self) -> np.ndarray | None:
        """真机工具轴方向（由 ``pose[3:6]`` 的外旋 XYZ 欧拉角算出，与模型同一套约定）。"""
        if self.pose is None or len(self.pose) < JOINTS_N:
            return None
        return euler_xyz_to_axis(*[float(v) for v in self.pose[3:6]])

    def summary(self) -> str:
        """一行摘要（打印 / 界面显示）。"""
        q = np.round(self.deg, 2)
        txt = f"src={self.src} seq={self.seq} unit={self.unit} q(deg)={q.tolist()}"
        if self.pose is not None:
            txt += f" tcp={np.round(self.pose[:3], 4).tolist()}"
        return txt


def _inner_state(obj) -> dict:
    """剥掉 ``{"state": {...}}`` / ``{"data": {...}}`` 这层壳，返回真正的状态字典。"""
    if isinstance(obj, dict):
        for key in NEST_KEYS:
            val = obj.get(key)
            if isinstance(val, dict) and (_vec(val, "pose") is not None
                                          or _vec(val, "joints") is not None
                                          or _numbers_from_obj(val)[0] is not None):
                return val
    return obj if isinstance(obj, dict) else {}


def parse_http_state(payload: bytes, *, unit: str = "auto", order=None) -> RobotState:
    """真机状态的 JSON → :class:`RobotState`（含 pose / vel / torque）。

    真机两条路（HTTP ``/api/state`` 与 UDP :data: 广播）的 JSON 形状只在**外壳键名**上不同
    （``state`` / ``data``），这里统一处理。
    """
    text = bytes(payload).decode("utf-8", "replace")
    try:
        obj = json.loads(text)
    except ValueError as exc:
        raise RobotLinkError(f"不是合法 JSON（{exc}）：{preview_bytes(payload)}") from exc
    nums, hint = _numbers_from_obj(obj)
    if not nums:
        raise RobotLinkError(f"JSON 里找不到 joints：{preview_bytes(payload)}")
    vals = pick_order(nums, order)
    tag = unit if unit in ("deg", "rad") else (hint if hint in ("deg", "rad") else guess_unit(vals))
    inner = _inner_state(obj)
    info: dict = {}
    for k in ("running", "speed", "ts", "rt_ts", "err_code", "model_state",
              "stream_alive", "rpc_connected", "type", "ver"):
        if k in inner:
            info[k] = inner[k]
        elif isinstance(obj, dict) and k in obj:
            info[k] = obj[k]
    n_q = _numbers_from_obj(obj)[0]
    return RobotState(q_rad=to_rad(vals, tag), unit=tag, raw=preview_bytes(payload),
                      pose=_vec(inner, "pose"), vel=_vec(inner, "vel"),
                      torque=_vec(inner, "torque"), enabled=_vec(inner, "enabled"),
                      info=info, note=f"json → joints（{len(n_q)} 个数字，外壳 {sorted(set(obj) & set(NEST_KEYS)) or '无'}）")


def is_heartbeat(data: bytes) -> bool:
    """这一包是不是"我还活着"的心跳（没有关节角）？是的话不该算解析错误。"""
    if not looks_text(data):
        return False
    try:
        obj = json.loads(bytes(data).decode("utf-8", "replace"))
    except ValueError:
        return False
    if _numbers_from_obj(obj)[0] is not None:
        return False
    txt = json.dumps(obj)
    return any(k in txt for k in HEARTBEAT_KEYS)


# =============================================================== 状态源
class StateSource:
    """状态源公共部分：后台线程 + 线程安全"最近一帧"。

    子类只管在 :meth:`_run` 里"收 + 解析"，解析结果交给 :meth:`publish`；
    调用线程用 :meth:`latest` 取快照（拿的是同一个对象，别再改它）。
    """

    kind = "?"

    def __init__(self, *, unit: str = "auto", order=None, i16_scale: float = 0.01):
        self.unit = unit
        self.order = order
        self.i16_scale = float(i16_scale)
        self.packets = 0
        self.errors = 0
        self.heartbeats = 0
        self.last_error = ""
        self.last_note = ""
        self._seq = 0
        self._stamps: deque = deque(maxlen=RATE_SAMPLES)
        self._latest: RobotState | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------- 生命周期
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._guarded_run,
                                        name=f"revA1-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.5) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t.is_alive():
            t.join(timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def _guarded_run(self) -> None:
        """线程入口：任何异常都收成一条错误记录，绝不抛给界面。"""
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self.fail(f"线程异常：{type(exc).__name__}: {exc}")

    def _run(self) -> None:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    # ---------------------------------------------------------- 数据
    def publish(self, st: RobotState, *, raw_note: str = "") -> None:
        """接收线程：记下最新一帧 + 统计。"""
        self._seq += 1
        st.seq = self._seq
        st.stamp = time.perf_counter()
        st.wall = time.time()
        st.src = st.src or self.endpoint()
        with self._lock:
            self.packets += 1
            self._latest = st
            self._stamps.append(st.stamp)
            self.last_note = raw_note or st.note

    def fail(self, msg: str) -> None:
        """接收线程：记一条解析/通信错误（不中断）。"""
        with self._lock:
            self.errors += 1
            self.last_error = msg

    def heartbeat(self) -> None:
        """接收线程：收到心跳包（没有关节角，只说明真机活着）。"""
        with self._lock:
            self.heartbeats += 1

    def latest(self) -> RobotState | None:
        with self._lock:
            return self._latest

    def rate_hz(self) -> float:
        """这一路实测数据率[Hz]（看最近 :data:`RATE_SAMPLES` 个包的到达时间）。"""
        with self._lock:
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            return 0.0 if span <= 1e-6 else (len(self._stamps) - 1) / float(span)

    def stats(self) -> dict:
        with self._lock:
            return dict(kind=self.kind, packets=self.packets, errors=self.errors,
                        heartbeats=self.heartbeats, last_error=self.last_error,
                        last_note=self.last_note, running=self.running)

    def endpoint(self) -> str:
        """这一路数据的来源描述（显示用）。"""
        return self.kind

    def describe(self) -> str:
        return f"{self.kind} {self.endpoint()}"


class HttpStateSource(StateSource):
    """HTTP 状态接口源：按 ``hz`` 轮询真机的 JSON 状态（真机实测 30 Hz 很轻松）。"""

    kind = "http"

    def __init__(self, url: str = DEFAULT_HTTP_URL, *, hz: float = DEFAULT_HTTP_HZ,
                 timeout: float = HTTP_TIMEOUT, unit: str = "auto", order=None):
        super().__init__(unit=unit, order=order)
        if not str(url).strip():
            raise RobotLinkError("HTTP 地址不能为空")
        self.url = str(url).strip()
        self.hz = float(max(hz, 1.0))
        self.timeout = float(max(timeout, 0.05))
        self.period = 1.0 / self.hz

    def endpoint(self) -> str:
        return self.url

    def describe(self) -> str:
        return f"HTTP 轮询 {self.url} @ {self.hz:.0f} Hz"

    def poll_once(self) -> RobotState:
        """读一次（也给自检/命令行用）：直接返回解析好的状态，异常照样抛。"""
        with urlopen(self.url, timeout=self.timeout) as fp:      # noqa: S310 (局域网 http)
            payload = fp.read()
        return parse_http_state(payload, unit=self.unit, order=self.order)

    def _run(self) -> None:
        quiet = 0.0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.publish(self.poll_once())
                quiet = 0.0
            except RobotLinkError as exc:
                self.fail(str(exc))
                quiet = 1.0
            except Exception as exc:  # noqa: BLE001  (网络/HTTP 各种异常都算"这次没读到")
                self.fail(f"{type(exc).__name__}: {exc}")
                quiet = 1.0
            if quiet and self.latest() is None:
                self._stop.wait(min(self.period, 0.25))       # 连不上时别把 CPU 打满
                continue
            self._stop.wait(max(self.period - (time.perf_counter() - t0), 0.0))


class UdpStateSource(StateSource):
    """UDP 广播源：绑一个端口听真机广播（也支持组播），报文格式自动识别。"""

    kind = "udp"

    def __init__(self, port: int = DEFAULT_UDP_PORT, *, host: str = "", fmt: str = "auto",
                 unit: str = "auto", order=None, i16_scale: float = 0.01, mcast: str = ""):
        super().__init__(unit=unit, order=order, i16_scale=i16_scale)
        self.port = int(port)
        if not (0 < self.port < 65536):
            raise RobotLinkError(f"UDP 端口要在 1..65535：{port}")
        self.host = str(host or "")
        self.fmt = fmt if fmt in FORMATS else "auto"
        self.mcast = str(mcast or "")
        self.sock: socket.socket | None = None
        self.last_addr = ""

    def endpoint(self) -> str:
        who = f"{self.host or '0.0.0.0'}:{self.port}"
        if self.mcast:
            return f"{who} + 组播 {self.mcast}"
        return f"{who} ← {self.last_addr or '（还没收到包）'}"

    def describe(self) -> str:
        return f"UDP 监听 :{self.port}（格式 {self.fmt}）"

    def _open(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass
        s.bind((self.host, self.port))
        if self.mcast:
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                         socket.inet_aton(self.mcast) + socket.inet_aton("0.0.0.0"))
        s.settimeout(SOCKET_TIMEOUT)
        return s

    def _recv_one(self, data: bytes, addr) -> None:
        """收线程：把一包字节变成 :class:`RobotState`（认不出来只记错误）。"""
        if is_heartbeat(data):
            self.heartbeat()                      # 真机约 1 Hz 的心跳，不算错误
            return
        self.last_addr = f"{addr[0]}:{addr[1]}"
        if looks_text(data) and self.fmt in ("auto", "json"):
            try:                                  # 真机广播就是这种 JSON：一次拿到 joints/pose/...
                st = parse_http_state(data, unit=self.unit, order=self.order)
                self.publish(st, raw_note=st.note)
                return
            except RobotLinkError:
                pass                              # 不是那形状 → 交给通用解析
        try:
            q, tag, note = parse_joint_payload(data, self.fmt, self.unit,
                                               order=self.order, i16_scale=self.i16_scale)
        except RobotLinkError as exc:
            self.fail(f"{addr[0]}:{addr[1]} {exc}")
            return
        self.publish(RobotState(q_rad=q, unit=tag, raw=preview_bytes(data), note=note),
                     raw_note=note)

    def _run(self) -> None:
        self.sock = self._open()
        try:
            while not self._stop.is_set():
                try:
                    data, addr = self.sock.recvfrom(RECV_BUFFER)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._stop.is_set():
                        break
                    self.fail(f"socket: {exc}")
                    continue
                self._recv_one(data, addr)
        finally:
            if self.sock is not None:
                self.sock.close()
                self.sock = None

    def stop(self, timeout: float = 1.5) -> None:
        super().stop(timeout)
        if self.sock is not None:            # 兜底：超时窗口内还没退就硬关（recvfrom 会立刻报错退出）
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


# =============================================================== 链路总入口
@dataclass
class FollowOut:
    """一帧跟随结果（:meth:`JointLink.update` 的返回值）。"""

    q_rad: np.ndarray | None          # 本帧要下发的 6 个关节目标（None = 本帧别动）
    state: RobotState | None          # 这一帧依据的真机状态
    stale: bool = False               # 看门狗判定"掉线"
    age: float = 0.0                  # 这一帧数据有多久了[s]
    fresh: bool = True                # 是不是刚收到的新包
    note: str = ""                    # 给界面/日志用的一句话

    @property
    def ok(self) -> bool:
        return self.q_rad is not None and not self.stale

    @property
    def deg(self) -> np.ndarray | None:
        return None if self.q_rad is None else np.degrees(self.q_rad)


class JointLink:
    """真机 → 仿真的姿态链路总入口（界面只用这一个类）。

    ``source``：

    =========  ==========================================================
    ``http``   HTTP 轮询 ``http_url``（真机实测接口，最稳）
    ``udp``    监听 ``port`` 上的 UDP 广播/组播（格式自动识别）
    ``auto``   两个一起开，每帧用**最新的那一路**（真机在哪条路上发都能跟）
    =========  ==========================================================

    用法（界面里就是这么写的）::

        link = JointLink(source="http", http_url=URL)
        link.start()
        ...
        out = link.update(sim.cmd)      # 每帧一次
        if out.ok:
            sim.follow(out.q_rad)
    """

    def __init__(self, *, source: str = "auto", http_url: str = DEFAULT_HTTP_URL,
                 http_hz: float = DEFAULT_HTTP_HZ, port: int = DEFAULT_UDP_PORT,
                 host: str = "", fmt: str = "auto", unit: str = "auto",
                 i16_scale: float = 0.01, order=None, signs=None, offsets=None,
                 mode: str = "absolute", max_speed_deg: float = 0.0, smooth: float = 0.0,
                 timeout: float = 1.0, mcast: str = "", timeout_policy: str = "hold"):
        if source not in SOURCES:
            raise RobotLinkError(f"数据源只能是 {SOURCES}：{source!r}")
        if mode not in MODES:
            raise RobotLinkError(f"映射模式只能是 {MODES}：{mode!r}")
        if timeout_policy not in TIMEOUT_POLICIES:
            raise RobotLinkError(f"断线策略只能是 {TIMEOUT_POLICIES}：{timeout_policy!r}")
        self.source = source
        self.http_url = str(http_url).strip()
        self.http_hz = float(max(http_hz, 1.0))
        self.port = int(port)
        self.host = str(host or "")
        self.mcast = str(mcast or "")
        self.fmt = fmt if fmt in FORMATS else "auto"
        self.unit = unit if unit in UNITS else "auto"
        self.i16_scale = float(i16_scale)
        self.order = None if order is None else tuple(int(i) for i in order)
        self.signs = (np.ones(JOINTS_N) if signs is None
                      else np.asarray(signs, dtype=float).ravel())
        self.offsets = (np.zeros(JOINTS_N) if offsets is None
                        else np.asarray(offsets, dtype=float).ravel())
        if self.signs.size != JOINTS_N or self.offsets.size != JOINTS_N:
            raise RobotLinkError("signs / offsets 都要 6 个数")
        self.mode = mode
        self.max_speed_deg = float(max(max_speed_deg, 0.0))
        self.smooth = float(max(smooth, 0.0))
        self.timeout = float(max(timeout, 0.05))
        self.timeout_policy = timeout_policy

        self.sources: list[StateSource] = []
        self._q_out: np.ndarray | None = None
        self._t_prev = 0.0
        self._raw0: np.ndarray | None = None      # 相对模式的"真机基准"
        self._cmd0: np.ndarray | None = None      # 相对模式的"仿真基准"
        self._last_seen: tuple[str, int] = ("", -1)
        self.notes: list[str] = []                # 最近几条事件（界面/日志用）
        self._note_once: dict[str, float] = {}

    # ---------------------------------------------------------- 生命周期
    def build_sources(self) -> list[StateSource]:
        """按 ``source`` 造状态源（start 之前也可以单独调，方便自检）。"""
        common = dict(unit=self.unit, order=self.order)
        out: list[StateSource] = []
        if self.source in ("http", "auto"):
            out.append(HttpStateSource(self.http_url, hz=self.http_hz, **common))
        if self.source in ("udp", "auto"):
            out.append(UdpStateSource(self.port, host=self.host, fmt=self.fmt,
                                      i16_scale=self.i16_scale, mcast=self.mcast, **common))
        if not out:
            raise RobotLinkError(f"没有可用的数据源：{self.source!r}")
        return out

    def start(self) -> None:
        """起接收线程（端口被占 / URL 非法等问题在这里就直接报错）。"""
        if self.running:
            return
        self.reset()
        self.notes = []
        self.sources = self.build_sources()
        for src in self.sources:
            src.start()
        self.note(f"已启动：{self.describe_sources()}")

    def stop(self) -> None:
        for src in self.sources:
            try:
                src.stop()
            except Exception:  # noqa: BLE001  (退出路径别抛)
                pass
        self.sources = []
        self.reset()
        self.note("已停止")

    def reset(self) -> None:
        """清掉基准/平滑状态（下次 update 重新取基准）。"""
        self._q_out = None
        self._t_prev = 0.0
        self._raw0 = None
        self._cmd0 = None
        self._last_seen = ("", -1)

    @property
    def running(self) -> bool:
        return any(s.running for s in self.sources)

    def note(self, msg: str, *, throttle: float = 0.0) -> None:
        """记一条事件（``throttle`` > 0 时同一条消息在该秒数内只记一次）。"""
        if throttle > 0:
            now = time.perf_counter()
            if now - self._note_once.get(msg, -1e9) < throttle:
                return
            self._note_once[msg] = now
        self.notes.append(msg)
        if len(self.notes) > 40:
            del self.notes[:-40]

    def drain_notes(self) -> list[str]:
        """取走累积的事件（界面每帧刷一次日志用）。"""
        out, self.notes = self.notes, []
        return out

    # ---------------------------------------------------------- 取数
    def latest(self) -> RobotState | None:
        """取所有源里**最新**的那一帧（``auto`` 模式下 HTTP / UDP 谁新用谁）。"""
        best: RobotState | None = None
        for src in self.sources:
            st = src.latest()
            if st is not None and (best is None or st.stamp > best.stamp):
                best = st
        return best

    def packets(self) -> tuple[int, int]:
        """``(收到的包数, 错误数)``（所有源合计；心跳包不计入）。"""
        return (sum(s.packets for s in self.sources), sum(s.errors for s in self.sources))

    def heartbeats(self) -> int:
        """收到的心跳包数（"真机还活着"但没有关节角的那种）。"""
        return sum(s.heartbeats for s in self.sources)

    def errors_text(self) -> str:
        """最近一条错误（合起来显示）。"""
        for src in reversed(self.sources):
            if src.last_error:
                return src.last_error
        return ""

    def rate_hz(self) -> float:
        """实测数据率[Hz]：多路取**最快**的那一路（``auto`` 下就是真正在带仿真的那条）。"""
        return max((s.rate_hz() for s in self.sources), default=0.0)

    def rate_text(self) -> str:
        """数据率文本：两路都活着时写成 ``30.0 + 96.0 Hz``。"""
        rates = [s.rate_hz() for s in self.sources]
        alive = [r for r in rates if r > 0.01]
        return (" + ".join(f"{r:.1f}" for r in alive) + " Hz") if alive else "0.0 Hz"

    def age(self) -> float:
        """最近一帧有多旧[s]（没数据时给 ``inf``）。"""
        st = self.latest()
        return float("inf") if st is None else time.perf_counter() - st.stamp

    # ---------------------------------------------------------- 每帧一问
    def update(self, current_cmd=None) -> FollowOut:
        """取最近一帧 → 映射（符号/偏置）→ 平滑 → 限速 → 给出本帧目标。

        :param current_cmd: 仿真当前 6 个关节目标（弧度）；**相对模式**用它当基准。
        """
        now = time.perf_counter()
        st = self.latest()
        if st is None:
            msg = "还没收到数据" if self.running else "链路未启动"
            self.note(msg, throttle=2.0)
            return FollowOut(None, None, stale=True, age=float("inf"), note=msg)
        age = now - st.stamp
        fresh = (st.src, st.seq) != self._last_seen
        self._last_seen = (st.src, st.seq)
        if age > self.timeout:
            msg = (f"看门狗：{age:.2f} s 没有新包（限 {self.timeout:.2f} s），"
                   f"按「{'保持不动' if self.timeout_policy == 'hold' else '回 home'}」处理")
            self.note(msg, throttle=2.0)
            return FollowOut(None, st, stale=True, age=age, fresh=fresh, note=msg)
        if not fresh and self._q_out is not None:
            # 没有新包：沿用上一帧（平滑/限速状态别乱跳）
            return FollowOut(self._q_out.copy(), st, stale=False, age=age, fresh=False)

        # ① 映射
        if self.mode == "relative":
            if self._raw0 is None:
                self._raw0 = st.q_rad.copy()
                base = self._q_out if current_cmd is None else np.asarray(current_cmd, float)
                self._cmd0 = (np.zeros(JOINTS_N) if base is None
                              else np.asarray(base, dtype=float).ravel().copy())
                self._q_out = self._cmd0.copy()
            target = self._cmd0 + (st.q_rad - self._raw0) * self.signs
        else:
            target = st.q_rad * self.signs + self.offsets

        # ② 平滑（一阶低通，时间常数 self.smooth）+ ③ 限速（度/秒 → 本帧最大变化量）
        q = target.copy()
        dt = now - self._t_prev if self._t_prev else 0.0
        if self._q_out is not None and dt > 1e-4:
            if self.smooth > 0:
                q = self._q_out + (target - self._q_out) * (1.0 - math.exp(-dt / self.smooth))
            if self.max_speed_deg > 0:
                step = math.radians(self.max_speed_deg) * dt
                d = q - self._q_out
                over = np.abs(d) > step
                if np.any(over):
                    q = q.copy()
                    q[over] = self._q_out[over] + np.sign(d[over]) * step
        self._q_out = q.copy()
        self._t_prev = now
        return FollowOut(q, st, stale=False, age=age, fresh=True)

    # ---------------------------------------------------------- 显示
    def describe_sources(self) -> str:
        return " + ".join(s.describe() for s in self.sources) if self.sources else "（未启动）"

    def describe(self) -> str:
        """多行状态（界面卡片 / 命令行打印同一份）。"""
        st = self.latest()
        n_pkt, n_err = self.packets()
        lines = [f"源  : {self.describe_sources()}",
                 f"数据: {self.rate_text()} · 延迟 {self.age() * 1000:.0f} ms · "
                 f"包 {n_pkt} / 错 {n_err} / 心跳 {self.heartbeats()}",
                 f"映射: {self.mode} 模式 · signs {self.signs.astype(int).tolist()} · "
                 f"smooth {self.smooth:.2f} s · 限速 "
                 f"{'关' if self.max_speed_deg <= 0 else f'{self.max_speed_deg:.0f}°/s'} · "
                 f"看门狗 {self.timeout:.2f} s"]
        if st is not None:
            lines.append(f"真机: q(deg) {np.round(st.deg, 2).tolist()}（原单位 {st.unit}）")
            if st.pose is not None:
                ax = st.axis()
                extra = "" if ax is None else f" · 工具轴 {np.round(ax, 3).tolist()}"
                lines.append(f"      TCP {np.round(st.pose[:3], 4).tolist()} m{extra}")
            lines.append(f"包  : {st.raw[:PREVIEW_BYTES]}")
        err = self.errors_text()
        if err:
            lines.append(f"错误: {err[:160]}")
        return "\n".join(lines)


# =============================================================== 造包 / 假真机
def pack_joint_payload(q_rad, fmt: str = "json", unit: str = "rad",
                       *, i16_scale: float = 0.01) -> bytes:
    """把 6 个关节角打成某种格式的字节（自检 / ``--emit-demo`` 用；与解析互为逆运算）。"""
    q = np.asarray(q_rad, dtype=float).ravel()
    if q.size != JOINTS_N:
        raise RobotLinkError(f"要 6 个关节角：{q_rad!r}")
    vals = np.degrees(q) if unit == "deg" else q
    kind = str(fmt).lower()
    if kind == "json":
        obj = {"joints": [round(float(v), 6) for v in vals], "unit": unit,
               "ts": int(time.time() * 1000)}
        return json.dumps(obj).encode("utf-8")
    if kind == "csv":
        return (",".join(f"{float(v):.6f}" for v in vals) + "\n").encode("utf-8")
    if kind == "f64":
        return vals.astype("<f8").tobytes()
    if kind == "f32":
        return vals.astype("<f4").tobytes()
    if kind == "i16":
        return np.round(vals / float(i16_scale)).astype("<i2").tobytes()
    raise RobotLinkError(f"不认识的格式 {fmt!r}（可选 {FORMATS}）")


def split_target(target: str, default_port: int = DEFAULT_UDP_PORT) -> tuple[str, int]:
    """``"192.168.66.255:9000"`` → ``("192.168.66.255", 9000)``。"""
    txt = str(target).strip()
    if not txt:
        return ("127.0.0.1", int(default_port))
    if txt.count(":") == 1:
        host, port = txt.split(":")
        return (host or "127.0.0.1", int(port))
    return (txt, int(default_port))


def emit_demo(target: str = f"127.0.0.1:{DEFAULT_UDP_PORT}", *, fmt: str = "json",
              unit: str = "deg", hz: float = 30.0, seconds: float = 1.0, amp_deg: float = 25.0,
              stop_evt: threading.Event | None = None) -> int:
    """按 ``hz`` 往 ``host:port`` 发关节角报文（模拟真机广播，联调/自检用）。

    关节角是 6 个正弦（幅值 ``amp_deg``，相位错开），方便一眼看出"跟没跟上"。
    """
    host, port = split_target(target)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    period = 1.0 / float(max(hz, 1.0))
    t0 = time.perf_counter()
    sent = 0
    try:
        while True:
            t = time.perf_counter() - t0
            if t > float(seconds) or (stop_evt is not None and stop_evt.is_set()):
                break
            q = np.radians([amp_deg * math.sin(2 * math.pi * (t + 0.12 * i)) for i in range(JOINTS_N)])
            try:
                sock.sendto(pack_joint_payload(q, fmt, unit), (host, port))
            except OSError:
                break
            sent += 1
            time.sleep(max(period - ((time.perf_counter() - t0) - t), 0.0))
    finally:
        sock.close()
    return sent


class FakeStateServer:
    """假真机：在本地起一个 HTTP 服务，模仿真机的 ``/api/state``（自检 / 界面演示用）。

    ``q_fn(t)`` 给任意时刻的 6 个关节角（弧度）；``pose_fn(q)`` 可选，用来补 ``pose``
    （界面自检里用 MuJoCo 正运动学造一个，就能把"真机 TCP ↔ 仿真工具尖"这条误差链也测到）。
    """

    def __init__(self, q_fn=None, *, pose_fn=None, host: str = "127.0.0.1", port: int = 0):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: PLC0415

        self.q_fn = q_fn or (lambda t: np.radians([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        self.pose_fn = pose_fn
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):                                  # noqa: N802 (http 接口名)
                outer.hits += 1
                body = outer.payload()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):                      # 静音
                return

        self._srv = ThreadingHTTPServer((host, int(port)), Handler)
        self.port = int(self._srv.server_address[1])
        self.url = f"http://{host}:{self.port}/api/state"
        self._thread: threading.Thread | None = None

    def payload(self) -> bytes:
        """造一帧和真机同形状的 JSON。"""
        q = np.asarray(self.q_fn(time.time()), dtype=float).ravel()
        state = {
            "joints": [float(v) for v in q],
            "vel": [0.0] * JOINTS_N,
            "torque": [0.5] * JOINTS_N,
            "enabled": [1.0] * JOINTS_N,
            "running": True, "speed": "v25", "err_code": 0,
            "ts": int(time.time() * 1000),
        }
        if self.pose_fn is not None:
            state["pose"] = [float(v) for v in np.asarray(self.pose_fn(q), dtype=float).ravel()]
        obj = {"state": state, "nrt_state": {"stream_alive": True, "system_is_init": True},
               "event": None, "stats": {"state": self.hits}}
        return json.dumps(obj).encode("utf-8")

    def start(self) -> "FakeStateServer":
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


# =============================================================== 真机探针
def free_udp_port() -> int:
    """向系统要一个当前没人用的 UDP 端口（自检用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def fk_diff(q_rad, pose, *, scene=None):
    """用 MuJoCo 正运动学复核"真机关节角 ↔ 真机 pose"是不是同一套约定。

    返回 ``(工具尖位置误差[m], 工具轴夹角[°])``；没装 mujoco / 没模型时返回 ``None``。
    """
    try:
        import mujoco                                        # noqa: PLC0415
        import revA1_spec as spec                            # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    try:
        model = mujoco.MjModel.from_xml_path(str(scene or spec.SCENE_XML))
        data = mujoco.MjData(model)
        data.qpos[:JOINTS_N] = np.asarray(q_rad, dtype=float).ravel()
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
        dpos = float(np.linalg.norm(data.site_xpos[sid] - np.asarray(pose[:3], dtype=float)))
        dax = math.degrees(math.acos(float(np.clip(
            data.site_xmat[sid].reshape(3, 3)[:, 2] @ euler_xyz_to_axis(*pose[3:6]), -1.0, 1.0))))
        return dpos, dax
    except Exception:  # noqa: BLE001
        return None


def probe_once(url: str = DEFAULT_HTTP_URL, *, timeout: float = 2.0, with_fk: bool = True,
               unit: str = "auto") -> RobotState:
    """读一次真机状态（命令行 ``--probe`` 用）；顺带复核它与模型的约定是否一致。"""
    st = HttpStateSource(url, timeout=timeout, unit=unit).poll_once()
    print("-" * 78)
    print(f"真机状态 {url}")
    print(f"  关节角[rad] : {np.round(st.q_rad, 5).tolist()}")
    print(f"  关节角[deg] : {np.round(st.deg, 3).tolist()}")
    print(f"  单位判定    : {st.unit}")
    if st.pose is not None:
        print(f"  工具尖[m]   : {np.round(st.pose[:3], 5).tolist()}")
        print(f"  欧拉角[deg] : {np.round(np.degrees(st.pose[3:6]), 3).tolist()}"
              f"（外旋 XYZ）")
    if st.vel is not None:
        print(f"  关节速度    : {np.round(st.vel, 4).tolist()}")
    if st.torque is not None:
        print(f"  关节力矩    : {np.round(st.torque, 3).tolist()}")
    if st.info:
        print(f"  其它        : {st.info}")
    if with_fk and st.pose is not None:
        diff = fk_diff(st.q_rad, st.pose)
        if diff is None:
            print("  模型复核    : 跳过（没装 mujoco / 找不到模型）")
        else:
            print(f"  模型复核    : 把 joints 灌进模型 qpos 后，工具尖差 "
                  f"{diff[0] * 1000:.2f} mm、工具轴差 {diff[1]:.2f}°"
                  f"（<5 mm / <2° 说明关节约定同号，可以直接镜像）")
    print("-" * 78)
    return st


# =============================================================== 自检
def run_selftest(args) -> int:
    """无界面自检：解析 / 单位 / 列顺序 → 假真机 HTTP → UDP 回环 → 映射 / 限速 / 看门狗。"""
    print("=" * 78)
    print("robot_link.py 自检（不需要真机；会起一个假真机的本地 HTTP 服务）")
    print("=" * 78)
    fails: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    def hand_fed(*, source: str = "udp", **kw):
        """造一个"不起线程、状态手塞"的 JointLink，专门用来测映射/限速/看门狗。"""
        lk = JointLink(source=source, port=free_udp_port(),
                       http_url="http://127.0.0.1:1/api/state", **kw)
        lk.sources = lk.build_sources()
        return lk, lk.sources[0]

    # 0) 三种数据源都能建起来（HTTP 源不吃 UDP 专有参数这种坑，就靠这一条兜）
    for src in SOURCES:
        try:
            n_src = len(JointLink(source=src, port=free_udp_port()).build_sources())
            detail = f"{n_src} 路"
        except Exception as exc:  # noqa: BLE001
            n_src, detail = 0, f"{type(exc).__name__}: {exc}"
        check(f"数据源 {src} 能建起来", (n_src >= (1 if src != "auto" else 2)), detail)

    # 1) 单位 / 数字 / 列顺序
    check("unit=auto：小角按弧度", np.allclose(to_rad([1, 2, 3, 4, 5, 6]), [1, 2, 3, 4, 5, 6]))
    check("unit=auto：大角按度", np.allclose(to_rad([90, 0, 0, 0, 0, 0])[0], np.pi / 2))
    check("unit=deg 强制按度", np.allclose(to_rad([180] * 6, "deg"), np.pi))
    check("parse_order", parse_order("0,1,2,3,4,5") == (0, 1, 2, 3, 4, 5))
    check("pick_order 挑列", np.allclose(pick_order([9, 1, 2, 3, 4, 5, 6], (1, 2, 3, 4, 5, 6)),
                                        [1, 2, 3, 4, 5, 6]))
    check("parse_six 抓文本里的数字", np.allclose(parse_six("1, 2;3 4|5 6"), [1, 2, 3, 4, 5, 6]))

    # 2) 打包 → 解包 回环（各种格式 / 单位）
    q = np.radians([10.0, -20.0, 30.0, -40.0, 50.0, -60.0])
    for fmt, unit in (("json", "rad"), ("json", "deg"), ("csv", "rad"), ("csv", "deg"),
                      ("f64", "rad"), ("f32", "rad"), ("f64", "deg"), ("i16", "deg")):
        data = pack_joint_payload(q, fmt, unit)
        try:
            got, tag, _note = parse_joint_payload(data, "auto")
            err = float(np.max(np.abs(got - q)))
        except RobotLinkError as exc:
            err, tag = float("inf"), str(exc)[:40]
        tol = 3e-4 if fmt == "i16" else 1e-5
        check(f"回环 {fmt}/{unit}", err < tol, f"最大误差 {err:.2e} rad（单位判定 {tag}）")

    # 3) JSON 的几种写法 + 真机 /api/state 的形状
    st = parse_http_state(json.dumps({"joints": list(q)}).encode())
    check("JSON {joints:[...]}", np.allclose(st.q_rad, q))
    named = {f"joint{i + 1}": float(np.degrees(q[i])) for i in range(JOINTS_N)}
    named["unit"] = "deg"
    st = parse_http_state(json.dumps(named).encode())
    check("JSON joint1..joint6 + unit=deg", np.allclose(st.q_rad, q, atol=1e-6))
    st = parse_http_state(json.dumps({"q": list(np.degrees(q)), "unit": "deg"}).encode())
    check("JSON {q:[...], unit:deg}", np.allclose(st.q_rad, q, atol=1e-6))
    real = {"state": {"joints": list(q), "pose": [0.11, -0.02, 0.39, 0.4, 0.5, 0.6],
                      "vel": [0.0] * JOINTS_N, "torque": [1.0] * JOINTS_N,
                      "ts": 12345, "running": True, "speed": "v25"},
            "nrt_state": {"stream_alive": True}, "stats": {"state": 7}}
    st = parse_http_state(json.dumps(real).encode())
    check("真机形状 {state:{joints, pose}}", np.allclose(st.q_rad, q) and st.tcp() is not None
          and st.info.get("running") is True, f"tcp={np.round(st.tcp(), 3).tolist()}")
    check("pose[3:6]（外旋 XYZ）→ 单位工具轴",
          abs(float(np.linalg.norm(st.axis())) - 1.0) < 1e-9,
          f"axis={np.round(st.axis(), 3).tolist()}")
    try:
        parse_joint_payload(b"\x01\x02\x03\x04", "auto")
        ok_bad = False
    except RobotLinkError as exc:
        ok_bad = "认不出" in str(exc)
    check("乱包：只报错不崩，并带上原始字节预览", ok_bad)

    # 4) 假真机（HTTP）端到端
    home = np.radians([18.0, -137.0, 121.0, 68.0, 99.0, -101.0])

    def q_fn(t):
        return home + 0.05 * np.sin(2.0 * math.pi * 0.5 * t)

    srv = FakeStateServer(q_fn=q_fn, pose_fn=lambda _q: [0.11, -0.02, 0.39, 0.4, 0.5, 0.6]).start()
    src = HttpStateSource(srv.url, hz=60.0)
    src.start()
    time.sleep(0.6)
    st = src.latest()
    src.stop()
    check("HTTP 源：收到数据", st is not None and src.packets >= 5,
          f"{src.packets} 包 / 错 {src.errors} / 假真机被读 {srv.hits} 次")
    check("HTTP 源：关节角解析正确",
          st is not None and float(np.max(np.abs(st.q_rad - q_fn(time.time())))) < 0.02)
    check("HTTP 源：pose / vel / torque 一起带过来",
          st is not None and st.pose is not None and st.vel is not None and st.torque is not None)
    srv.stop()
    dead = HttpStateSource(f"http://127.0.0.1:{free_udp_port()}/api/state", hz=20.0, timeout=0.3)
    dead.start()
    time.sleep(0.5)
    dead.stop()
    check("HTTP 源：连不上只记错误、不崩",
          dead.packets == 0 and dead.errors > 0 and dead.latest() is None,
          f"错 {dead.errors} 次：{dead.last_error[:60]}")

    # 5) UDP 回环（真机就是用这种方式往外广播的）
    port = free_udp_port()
    usrc = UdpStateSource(port, fmt="auto")
    usrc.start()
    sent = emit_demo(f"127.0.0.1:{port}", fmt="json", unit="deg", hz=60.0,
                     seconds=0.5, amp_deg=25.0)
    time.sleep(0.25)
    ust = usrc.latest()
    usrc.stop()
    check("UDP 源：收到广播", sent > 10 and usrc.packets > 5,
          f"发了 {sent} 包 / 收到 {usrc.packets} 包 / 错 {usrc.errors}")
    check("UDP 源：解析出关节角",
          ust is not None and plausible(ust.q_rad)
          and float(np.max(np.abs(ust.deg))) <= 26.0,
          f"{None if ust is None else np.round(ust.deg, 2).tolist()}")
    port2 = free_udp_port()
    usrc2 = UdpStateSource(port2)
    usrc2.start()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for _ in range(3):
            s.sendto(b"#!not-a-joint-packet!?", ("127.0.0.1", port2))
    time.sleep(0.4)
    usrc2.stop()
    check("UDP 源：乱包只记错误、不崩", usrc2.packets == 0 and usrc2.errors >= 1,
          f"错 {usrc2.errors} 次：{usrc2.last_error[:60]}")

    # 5b) 真机广播的真实形状：{"data":{...},"type":"state"} + 心跳包
    hb = json.dumps({"data": {"stream_alive": True, "system_is_init": True, "ts": 123},
                     "type": "nrt_state"}).encode()
    check("心跳包能认出来（不当解析错误）",
          is_heartbeat(hb) and not is_heartbeat(json.dumps({"data": {"joints": list(q)}}).encode()))
    port3 = free_udp_port()
    usrc3 = UdpStateSource(port3)
    usrc3.start()
    broadcast = json.dumps({"data": {"joints": list(q), "pose": [0.11, -0.02, 0.39, 0.4, 0.5, 0.6],
                                     "vel": [0.0] * JOINTS_N, "torque": [1.0] * JOINTS_N,
                                     "enabled": [1.0] * JOINTS_N, "running": True,
                                     "speed": "v25", "ts": 1789891582032},
                            "type": "state", "ver": 1}).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(hb, ("127.0.0.1", port3))
        s.sendto(broadcast, ("127.0.0.1", port3))
    time.sleep(0.4)
    st_bc = usrc3.latest()
    usrc3.stop()
    check("UDP：真机广播形状（data 外壳）解出 joints + pose",
          st_bc is not None and np.allclose(st_bc.q_rad, q) and st_bc.pose is not None,
          f"{None if st_bc is None else np.round(st_bc.deg, 2).tolist()}°")
    check("UDP：心跳单独计数、不算错误",
          usrc3.packets == 1 and usrc3.heartbeats == 1 and usrc3.errors == 0,
          f"包 {usrc3.packets} / 心跳 {usrc3.heartbeats} / 错 {usrc3.errors}")

    # 6) 映射：绝对 / 相对 / 符号 / 偏置
    qa = np.radians([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lk, s = hand_fed()
    s.publish(RobotState(q_rad=qa))
    out = lk.update()
    check("绝对模式：原样镜像真机", out.ok and np.allclose(out.q_rad, qa),
          f"{np.round(out.deg, 2).tolist()}°")
    lk2, s2 = hand_fed(signs=(-1, 1, 1, 1, 1, 1),
                       offsets=(0.0, 0.0, 0.0, 0.0, 0.0, np.pi))
    s2.publish(RobotState(q_rad=qa))
    out2 = lk2.update()
    want = qa * np.array([-1.0, 1, 1, 1, 1, 1]) + np.array([0, 0, 0, 0, 0, np.pi])
    check("符号 / 偏置生效", np.allclose(out2.q_rad, want),
          f"{np.round(out2.deg, 1).tolist()}°")
    base = np.radians([5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
    lk3, s3 = hand_fed(mode="relative")
    s3.publish(RobotState(q_rad=base))
    out3 = lk3.update(np.zeros(6))
    check("相对模式：第一帧不跳变（以仿真当前姿态为基准）",
          np.allclose(out3.q_rad, np.zeros(6)), f"{np.round(out3.deg, 3).tolist()}°")
    delta = np.radians([3.0, -2.0, 1.0, 0.5, -0.5, 2.0])
    s3.publish(RobotState(q_rad=base + delta))
    out3 = lk3.update(np.zeros(6))
    check("相对模式：只镜像增量", np.allclose(out3.q_rad, delta, atol=1e-9),
          f"{np.round(out3.deg, 3).tolist()}°")

    # 7) 限速 / 平滑
    lk4, s4 = hand_fed(max_speed_deg=60.0)
    s4.publish(RobotState(q_rad=np.zeros(6)))
    lk4.update()
    t0 = time.perf_counter()
    time.sleep(0.1)
    s4.publish(RobotState(q_rad=np.radians([90.0] * 6)))
    out4 = lk4.update()
    dt = time.perf_counter() - t0
    cap = 60.0 * dt * 1.5
    check("限速：一步不超过 max_speed × dt",
          0.0 < float(np.max(out4.deg)) <= cap,
          f"一步走了 {float(np.max(out4.deg)):.2f}°（dt={dt * 1000:.0f} ms，上限 {cap:.2f}°）")
    check("限速：没有新包就沿用上一帧（不乱跳）",
          np.allclose(lk4.update().q_rad, out4.q_rad))
    lk5, s5 = hand_fed(smooth=0.2)
    s5.publish(RobotState(q_rad=np.zeros(6)))
    lk5.update()
    time.sleep(0.05)
    s5.publish(RobotState(q_rad=np.radians([90.0] * 6)))
    out5 = lk5.update()
    check("平滑：一阶低通，第一帧只走一部分",
          0.0 < float(np.max(out5.deg)) < 90.0, f"第一帧 {float(np.max(out5.deg)):.2f}°")

    # 8) 看门狗
    lk6, s6 = hand_fed(timeout=0.2)
    s6.publish(RobotState(q_rad=qa))
    check("看门狗：有新包时正常跟随", lk6.update().ok)
    time.sleep(0.3)
    out6 = lk6.update()
    check("看门狗：超时变 stale 且不给目标",
          out6.stale and out6.q_rad is None and "看门狗" in out6.note, out6.note)

    # 9) 真机（可选：读不到就 SKIP，不算失败）
    if not getattr(args, "no_probe", False):
        url = getattr(args, "url", DEFAULT_HTTP_URL)
        try:
            st = HttpStateSource(url, timeout=2.0).poll_once()
            check(f"真机 {url} 可达", plausible(st.q_rad),
                  f"{np.round(st.deg, 1).tolist()}°")
            if st.pose is not None:
                diff = fk_diff(st.q_rad, st.pose)
                if diff is None:
                    print("  [SKIP] 模型复核（没装 mujoco / 找不到模型）")
                else:
                    check("模型复核：真机 joints 与 pose 是同一套约定（<5 mm / <2°）",
                          diff[0] < 5e-3 and diff[1] < 2.0,
                          f"工具尖差 {diff[0] * 1000:.2f} mm / 工具轴差 {diff[1]:.2f}°")
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] 真机 {url} 现在读不到（{type(exc).__name__}: {exc}）")
        # 真机也在 UDP 上广播（本机实测 :6001，~96 Hz）
        try:
            lv = UdpStateSource(int(getattr(args, "port", DEFAULT_UDP_PORT)))
            lv.start()
            time.sleep(1.2)
            live, n_pkt, n_hb = lv.latest(), lv.packets, lv.heartbeats
            lv.stop()
            if live is None:
                print(f"  [SKIP] 真机 UDP :{lv.port} 这 1.2 s 没收到广播"
                      f"（真机没在广播？端口被占？）错 {lv.errors}：{lv.last_error[:60]}")
            else:
                check(f"真机 UDP :{lv.port} 广播可达（{n_pkt} 包 / 心跳 {n_hb}）",
                      plausible(live.q_rad), f"{np.round(live.deg, 1).tolist()}°")
                if live.pose is not None:
                    diff = fk_diff(live.q_rad, live.pose)
                    if diff is not None:
                        check("模型复核（UDP 包）：joints 与 pose 是同一套约定",
                              diff[0] < 5e-3 and diff[1] < 2.0,
                              f"工具尖差 {diff[0] * 1000:.2f} mm / 工具轴差 {diff[1]:.2f}°")
        except OSError as exc:
            print(f"  [SKIP] 真机 UDP 收不了（端口被占？）：{exc}")

    print("-" * 78)
    if fails:
        print(f"结论：{len(fails)} 项失败 -> {fails}")
        return 1
    print("结论：全部通过 [OK]")
    return 0


# =============================================================== 命令行
def run_listen(args) -> int:
    """没有界面时的联调：把收到的包按行打印（Ctrl+C 退出）。"""
    lk = JointLink(source=args.source, http_url=args.url, http_hz=args.hz, port=args.port,
                   host=args.host, fmt=args.format, unit=args.unit,
                   i16_scale=args.i16_scale, mcast=args.mcast)
    lk.start()
    print(f"监听中：{lk.describe_sources()}（Ctrl+C 退出）")
    seen: tuple[str, int] = ("", -1)
    try:
        while True:
            time.sleep(0.2)
            for line in lk.drain_notes():
                print(f"[link] {line}")
            st = lk.latest()
            if st is not None and (st.src, st.seq) != seen:
                seen = (st.src, st.seq)
                print(f"[{time.strftime('%H:%M:%S')}] {st.summary()}")
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        lk.stop()
    return 0


def run_emit_demo(args) -> int:
    """模拟真机往某地址发报文（配合 ``--listen`` 或界面联调）。"""
    print(f"往 {args.target} 发 {args.format}/{args.unit} 报文："
          f"{args.hz:.0f} Hz × {args.seconds:g} s（幅度 {args.amp_deg:g}°）")
    sent = emit_demo(args.target, fmt=args.format, unit=args.unit, hz=args.hz,
                     seconds=args.seconds, amp_deg=args.amp_deg)
    print(f"发完：{sent} 包")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="真机 ↔ 仿真 姿态链路（HTTP 状态接口 / UDP 广播 → 6 个关节角）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="无界面自检（含假真机 / UDP 回环）")
    ap.add_argument("--probe", action="store_true", help="读一次真机状态，并复核它与模型的约定")
    ap.add_argument("--listen", action="store_true", help="持续接收并打印（联调）")
    ap.add_argument("--emit-demo", action="store_true", help="模拟真机发报文（联调）")
    ap.add_argument("--source", default="auto", choices=SOURCES, help="数据源（--listen 用）")
    ap.add_argument("--url", default=DEFAULT_HTTP_URL, help="真机 HTTP 状态接口")
    ap.add_argument("--port", type=int, default=DEFAULT_UDP_PORT, help="UDP 监听端口")
    ap.add_argument("--host", default="", help="UDP 绑定地址（默认全部网卡）")
    ap.add_argument("--mcast", default="", help="组播地址（可选，如 239.0.0.1）")
    ap.add_argument("--target", default=f"127.0.0.1:{DEFAULT_UDP_PORT}",
                    help="--emit-demo 的发送目标 host:port")
    ap.add_argument("--format", default="json", choices=FORMATS, help="报文格式")
    ap.add_argument("--unit", default="auto", choices=UNITS, help="报文单位")
    ap.add_argument("--i16-scale", type=float, default=0.01, help="int16 格式的 度/LSB")
    ap.add_argument("--hz", type=float, default=30.0, help="轮询频率 / 发送频率")
    ap.add_argument("--seconds", type=float, default=1.0, help="--emit-demo 发多久")
    ap.add_argument("--amp-deg", type=float, default=25.0, help="--emit-demo 正弦幅度[度]")
    ap.add_argument("--no-probe", action="store_true", help="自检时不去碰真机")
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest(args)
    if args.emit_demo:
        return run_emit_demo(args)
    if args.listen:
        return run_listen(args)
    if args.probe:
        try:
            probe_once(args.url)
        except Exception as exc:  # noqa: BLE001
            print(f"读不到真机 {args.url}：{type(exc).__name__}: {exc}")
            return 1
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
