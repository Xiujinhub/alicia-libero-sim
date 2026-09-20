"""MuJoCo 的 OpenGL 后端自动选择（跨平台 / WSL 免手配）。

和 `lerobot_codeit/sim/gl_backend.py` 是同一份逻辑，这里自带一份，保证 RevA1-sim
目录可以单独拷走使用。

为什么需要它
------------
MuJoCo 是在 **import 的那一刻** 读环境变量 ``MUJOCO_GL`` 决定用哪个 GL 后端的：

| 平台 | 可用后端 | 说明 |
|---|---|---|
| Windows | ``wgl``（默认） | 不用管 |
| macOS | ``cgl``（默认） | 不用管 |
| Linux 有 X/Wayland（含 WSLg） | ``glfw`` | 离屏渲染 + ``mujoco.viewer`` 都能用 |
| Linux 纯无头（无 ``DISPLAY``） | ``egl`` / ``osmesa`` | 只能离屏渲染，开不了窗口 |

所以：

    ⚠️ 本模块必须在 ``import mujoco`` **之前** 被 import。

用法::

    from gl_backend import configure_gl
    configure_gl()          # 幂等；环境里已有 MUJOCO_GL 就完全不动
    import mujoco

想手动指定后端，直接设环境变量即可（本模块不会覆盖用户设置）::

    MUJOCO_GL=egl python viewer.py --headless --seconds 3
"""

from __future__ import annotations

import os
import sys

_CONFIGURED = False
_CHOSEN: str | None = None


def configure_gl(verbose: bool = False) -> str | None:
    """按平台挑一个能用的 MUJOCO_GL 后端（只在用户没设置时生效），返回最终取值。

    Windows / macOS 下返回 None，表示用 MuJoCo 自己的默认后端。
    """
    global _CONFIGURED, _CHOSEN
    if _CONFIGURED:
        return _CHOSEN
    _CONFIGURED = True

    env_value = os.environ.get("MUJOCO_GL")
    if env_value:
        _CHOSEN = env_value                      # 尊重用户设置
    elif sys.platform.startswith("linux"):
        # WSLg / 本机桌面都会给 DISPLAY 或 WAYLAND_DISPLAY；glfw 既能离屏渲染也能开窗口。
        # 真·无头服务器（docker/CI）没有 DISPLAY，只能走 egl。
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        _CHOSEN = "glfw" if has_display else "egl"
        os.environ["MUJOCO_GL"] = _CHOSEN

    if verbose and _CHOSEN:
        print(f"[gl] 自动选择 MUJOCO_GL={_CHOSEN}")
    return _CHOSEN
