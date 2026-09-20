#!/usr/bin/env python3
"""交互控制台入口（**已改为 PySide6 界面**；旧 tkinter 版本另存为 ``interactive_control_tk.py``）。

这个文件现在只是一层转发：参数原样交给 ``revA1_gui.main``。之所以保留它，是为了让原有的
命令行（README / 脚本 / 肌肉记忆里的 ``python interactive_control.py``）继续可用。

    python interactive_control.py                  # = python revA1_gui.py
    python interactive_control.py --selftest       # 无窗口自检（控制逻辑 + 渲染通路）
    python interactive_control.py --ui-test        # 建窗口 → 脚本化点一遍控件 → 存图

直接跑 ``revA1_gui.py`` 完全等价。新功能（关节 −/+ 微调、笛卡尔直线点到点、点位示教、
多初值 IK）都在那边，这里不再放任何界面代码。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import revA1_gui  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return revA1_gui.main(argv)


if __name__ == "__main__":
    sys.exit(main())
