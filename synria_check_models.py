"""校验 Synria-Robot-Descriptions 里所有模型能否被 MuJoCo 加载。

用法（在 conda lerobot 环境里执行）：
    conda run --no-capture-output -n lerobot python synria_check_models.py
"""

import glob
import os

import mujoco

REPO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Synria-Robot-Descriptions-main",
    "synriard",
)

# 已知问题：这些 URDF 的 mesh 路径多写了一层 "meshes/"（如 ../../meshes/meshes/...），
# MuJoCo 直接加载会报 Error opening file；对应的 MJCF 版本是正常的，请优先用 MJCF。
KNOWN_BAD_URDF = (
    "Alicia_M_v1_0_follower.urdf",
    "Alicia_M_v1_1_vertical.urdf",
    "Bessica_D_v1_1_covered.urdf",
)


EXT = {"mjcf": "xml", "urdf": "urdf"}


def check(kind):
    print(f"\n===== {kind.upper()} ===== (mujoco {mujoco.__version__})")
    ext = EXT[kind]
    files = sorted(glob.glob(os.path.join(REPO, kind, "**", f"*.{ext}"), recursive=True))
    ok = fail = 0
    for path in files:
        rel = os.path.relpath(path, os.path.join(REPO, kind))
        try:
            model = mujoco.MjModel.from_xml_path(path)
            ok += 1
            print(f"OK   nq={model.nq:<3} nu={model.nu:<3} ngeom={model.ngeom:<4} {rel}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            tag = "已知问题" if os.path.basename(path) in KNOWN_BAD_URDF else "FAIL"
            print(f"{tag} {rel}\n     {str(exc).strip().splitlines()[0]}")
    print(f"---- {ok} ok / {fail} failed / {len(files)} total ----")


if __name__ == "__main__":
    check("mjcf")
    check("urdf")
