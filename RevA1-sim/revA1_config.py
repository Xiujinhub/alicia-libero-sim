#!/usr/bin/env python3
"""RevA1-sim 的**用户配置**：按关节的限位覆盖（``config/joint_limits.json``）。

为什么要有这份配置
------------------
模型里的关节范围（``jnt_range`` / ``actuator_ctrlrange``）是从 URDF 生成的，而**真机实际可用
范围**可能与 URDF 不一致（本机实测：URDF 里 ``joint1/2/4/5/6`` 都是 ±180°，**只有 ``joint3``
给到 ±164°**）。真机跟随会按模型范围**截断**目标角（``arm_core.ArmSim.follow``），范围对不上
就会出现"真机已经转过去了、仿真却卡在限位上"（表现：末端姿态与真机不一致，甚至看着更竖直）。

于是把"每个关节的限位"做成可编辑 + 可持久化的用户配置：

* 文件：``RevA1-sim/config/joint_limits.json``（JSON、**单位度**、手改也行）；
* 界面：「关节限位」卡片 —— 改完点「应用」立即生效（只改内存），点「保存为默认」写进该文件
  （下次启动自动加载），点「恢复 URDF 默认」= 删掉该文件；
* 换目录：环境变量 ``REVA1_CONFIG_DIR`` 或 ``--config-dir``（自检用临时目录，互不干扰）。

文件长这样::

    {
      "_note": "关节限位（度）。界面「关节限位」卡片写入；删掉本文件即恢复 URDF 默认。",
      "unit": "deg",
      "limits": {
        "joint1": [-180.0, 180.0],
        "joint3": [-180.0, 180.0]
      }
    }

本模块**只依赖标准库 + revA1_spec**（不 import mujoco / Qt），谁都能用；
"把值写进模型"那一步在 ``arm_core.apply_joint_limits``。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import revA1_spec as spec

HERE = Path(__file__).resolve().parent
CONFIG_DIR_ENV = "REVA1_CONFIG_DIR"      # 换个配置目录（自检 / 多套参数用）
DEFAULT_CONFIG_DIR = HERE / "config"
LIMITS_NAME = "joint_limits.json"
MAX_ABS_DEG = 360.0                      # 限位的合理上限（超过它多半是填错了）
DIFF_TOL_DEG = 0.01                      # 比 URDF 差多少才算"改过"（界面输入是 0.1° 步进）


class ConfigError(ValueError):
    """配置文件读不了 / 写不进去 / 内容不合法。"""


# ---------------------------------------------------------------- 路径
def config_dir() -> Path:
    """配置目录：``REVA1_CONFIG_DIR`` 优先，否则 ``RevA1-sim/config/``。"""
    env = os.environ.get(CONFIG_DIR_ENV, "").strip()
    return Path(env).expanduser() if env else DEFAULT_CONFIG_DIR


def limits_path() -> Path:
    """关节限位配置文件（``config/joint_limits.json``）。"""
    return config_dir() / LIMITS_NAME


# ---------------------------------------------------------------- 默认值 / 校验
def urdf_limits() -> dict[str, tuple[float, float]]:
    """URDF（``revA1_spec.LIMITS``）里的关节限位，**度**。"""
    return {jn: (math.degrees(spec.LIMITS[jn][0]), math.degrees(spec.LIMITS[jn][1]))
            for jn in spec.JOINTS}


def sanitize(limits) -> dict[str, tuple[float, float]]:
    """校验并规整一份限位（``{关节名: (最小°, 最大°)}``），不认识/不合法的直接报错。"""
    if limits is None:
        return {}
    if not isinstance(limits, dict):
        raise ConfigError(f"限位应该是一个字典/对象，收到 {type(limits).__name__}")
    out: dict[str, tuple[float, float]] = {}
    for name, pair in limits.items():
        key = str(name).strip()
        if key not in spec.LIMITS:
            raise ConfigError(f"不认识的关节名 {key!r}（只认 {spec.JOINTS}）")
        try:
            lo, hi = (float(v) for v in tuple(pair))
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"{key} 的限位要两个数 [最小, 最大]：{pair!r}") from exc
        if not (lo < hi):
            raise ConfigError(f"{key} 的限位要满足 最小 < 最大（收到 {lo} / {hi}）")
        if max(abs(lo), abs(hi)) > MAX_ABS_DEG:
            raise ConfigError(f"{key} 的限位超出 ±{MAX_ABS_DEG:.0f}°（收到 {lo} / {hi}）")
        out[key] = (lo, hi)
    return out


def effective(saved: dict | None = None) -> dict[str, tuple[float, float]]:
    """最终生效的限位 = URDF 默认 + 覆盖（缺的关节用 URDF 值）。"""
    out = urdf_limits()
    out.update(dict(saved or {}))
    return out


def describe(limits: dict[str, tuple[float, float]] | None = None) -> str:
    """一句话描述（日志/界面用）：哪些关节与 URDF 不同。"""
    eff = effective(limits)
    base = urdf_limits()
    diff = [f"{jn} ±{eff[jn][1]:.1f}°（URDF ±{base[jn][1]:.1f}°）"
            for jn in spec.JOINTS if _differs(eff[jn], base[jn])]
    return "与 URDF 相同" if not diff else "；".join(diff)


def _differs(pair, base) -> bool:
    """这一对限位算不算"和 URDF 不一样"（界面输入是 0.1° 步进，所以留 0.01° 容差）。"""
    return (abs(float(pair[0]) - float(base[0])) > DIFF_TOL_DEG
            or abs(float(pair[1]) - float(base[1])) > DIFF_TOL_DEG)


# ---------------------------------------------------------------- 读写
def load_limits(path: str | Path | None = None) -> dict[str, tuple[float, float]] | None:
    """读配置里的限位（**度**）。文件不存在返回 ``None``；内容不合法抛 :class:`ConfigError`。"""
    p = Path(path) if path is not None else limits_path()
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"{p} 不是合法 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} 顶层应该是一个对象")
    body = raw.get("limits", raw)           # 允许直接写成 {"joint3": [...]}（手改省事）
    return sanitize(body)


def save_limits(limits, path: str | Path | None = None) -> Path:
    """把限位写进配置文件（只写**与 URDF 不同**的项，读起来干净）。返回文件路径。"""
    p = Path(path) if path is not None else limits_path()
    clean = sanitize(limits)
    base = urdf_limits()
    keep = {jn: [round(float(v[0]), 6), round(float(v[1]), 6)]
            for jn, v in clean.items() if _differs(v, base[jn])}
    body = {
        "_note": "关节限位（度）。界面「关节限位」卡片写入；删掉本文件即恢复 URDF 默认。",
        "_urdf": "revA1_spec.LIMITS（joint1/2/4/5/6 = ±180°，joint3 = ±164°）",
        "unit": "deg",
        "limits": {jn: keep[jn] for jn in spec.JOINTS if jn in keep},
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{p} 写不进去：{exc}") from exc
    return p


def clear_limits(path: str | Path | None = None) -> bool:
    """删掉配置文件（= 恢复 URDF 默认）。文件本来就不在返回 ``False``。"""
    p = Path(path) if path is not None else limits_path()
    if not p.exists():
        return False
    try:
        p.unlink()
    except OSError as exc:
        raise ConfigError(f"{p} 删不掉：{exc}") from exc
    return True
