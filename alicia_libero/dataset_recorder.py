#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集采集：把连跑 / 手动示教的每一"局"录成 **LeRobot 数据集**（每个相机一个 mp4）。

为什么单独一个模块
------------------
1. **格式**：LeRobot（``lerobot.datasets.LeRobotDataset``）——``meta/info.json`` 元数据 +
   每帧 parquet + （``use_videos=True`` 时）**每个相机键一个 mp4**；
2. **不卡主循环**：建库 / 写 JPEG / 编视频全在**后台线程**里做，主循环只往有界队列投帧
   （队列满了就丢帧、并把这一局标记成"损坏"，宁可不落盘也不拖慢仿真）；
3. **断点续采**：载入同名数据集继续 ``add_frame`` / ``save_episode`` 即可 —— 本仓库的
   lerobot 已打 "ALICIA-D PATCH"：续采时 parquet / episodes / **每相机 mp4** 都续写同一文件，
   所以"一个相机一个视频"在多次采集之间也成立。

一"局"（episode）的边界 = **两次复位之间**（连跑每局都会复位 → 一局一个 episode；手动示教按
``R`` 复位即收局）。成功与否由 ``add_frame(success=...)`` 逐帧累计，收局时按「仅成功局」决定
``save_episode()`` 落盘还是 ``clear_episode_buffer()`` 丢弃。
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

RECORD_WIDTH = 320          # 录像宽度（像素）；高度按渲染比例折成**偶数**（900×640 → 320×228）
REPO_PREFIX = "alicia"      # repo_id 前缀（只用本地 root，这半截只为信息完整）
QUEUE_LIMIT = 400           # 队列最多攒多少帧（≈0.9MB/帧 → 顶多 ~360MB）
DEFAULT_FPS = 10            # 采集帧率（Hz）：仿真跑 60Hz，这里按**仿真时钟**抽帧
DEFAULT_NAME = "alicia_libero"
ASPECT = 900.0 / 640.0      # 渲染分辨率 VIEW_W / VIEW_H，用来算录像高度


def datasets_dir() -> Path:
    """数据集根目录：本项目 ``Datasets/``（可用环境变量 ``ALICIA_DATASETS_DIR`` 覆盖）。"""
    env = os.environ.get("ALICIA_DATASETS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "Datasets"


def stamped_name(base: str) -> str:
    """非续采时的**新名字**：``<base>_<yyyyMMdd_HHMMSS>``（绝不覆盖已有数据集）。"""
    return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def record_shape(width: int = RECORD_WIDTH) -> tuple[int, int]:
    """录像尺寸 (H, W)：高度取偶数 —— h264 的 yuv420p 不接受奇数边。"""
    height = int(round(width / ASPECT))
    return height - height % 2, int(width)


def resized(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """等比缩到录像尺寸，**总是返回新数组**（MuJoCo 的渲染缓冲下一帧就被覆盖）。"""
    height, width = shape
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.shape[0] == height and frame.shape[1] == width:
        return np.array(frame, copy=True)
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR),
                      dtype=np.uint8)


def video_key(camera: str) -> str:
    """相机 → LeRobot 特征名（``observation.images.<cam>``，一路一个 mp4）。"""
    return f"observation.images.{camera}"


def sweep_images(root: Path) -> bool:
    """删掉 LeRobot 编视频用的**临时帧目录** ``images/``（编完视频它就没用了）。

    ⚠ 为什么需要这一步：``use_videos=True`` 时 lerobot 先把每帧落成临时图（
    ``images/<相机>/episode-XXXXXX/frame-XXXXXX.png``）再编码成 mp4；编码完**文件**会被删，
    但**空目录树留在数据集里**（实测：一局录完根目录仍是 ``data/ images/ meta/ videos/``）。
    既然是"一个相机一个视频"，这些空目录纯属噪声 —— 收局/关闭/打开时统一扫掉。

    Windows 上目录可能被写入线程短暂占用，失败重试几次；实在删不掉也不影响数据
    （视频已经在 ``videos/`` 里了）。
    """
    target = Path(root) / "images"
    if not target.is_dir():
        return False
    for attempt in range(3):
        try:
            shutil.rmtree(target)
            return True
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    return False


def features_for(cameras: list[str], shape: tuple[int, int], state_dim: int = 7) -> dict:
    """LeRobot features 声明：4 路相机都是 ``video``，外加状态 / 动作 / 奖励 / 成功标记。"""
    names = [f"J{i + 1}" for i in range(state_dim - 1)] + ["gripper"]
    feats = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": names},
        "action": {"dtype": "float32", "shape": (state_dim,), "names": names},
        # ⚠ 成功标记用 float32（0/1）而不是 bool：float32 在 LeRobot 里是验证最充分的那条路径
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.success": {"dtype": "float32", "shape": (1,), "names": None},
    }
    for camera in cameras:
        feats[video_key(camera)] = {"dtype": "video", "shape": (*shape, 3),
                                    "names": ["height", "width", "channels"]}
    return feats



class DatasetRecorder:
    """采集器：主循环投帧 → 后台线程建库 / 续采 / 写盘；任何异常都记进 ``error``。

    线程约定（重要）：

    * **主线程**只做两件事：``add_frame()``（把帧拷一份塞进队列）和读计数（整数，读脏了也无害）；
    * **后台线程**独占 LeRobot 数据集对象（``add_frame`` / ``save_episode`` / ``finalize``）；
    * 队列有上限；满了**丢帧并把这一局标记损坏**（该局不落盘），绝不阻塞主循环。
    """

    def __init__(self) -> None:
        self.wanted = False                # 用户的采集开关（界面上的勾选框）——只表示"要不要采"
        self.armed = False                 # 数据集已经打开、正在记录（真正的采集状态）
        self.name = ""
        self.path: Path | None = None
        self.resumed = False               # 本次是"续采同一个数据集"吗
        self.fps = float(DEFAULT_FPS)
        self.cameras: list[str] = []
        self.shape = record_shape()
        self.only_success = True
        self.episodes = 0                  # 本次会话落盘的局数（后台线程写）
        self.frames = 0                    # 本次会话落盘的帧数（丢弃的局不算，后台线程写）
        self.buffered = 0                  # 当前这一局已攒的帧数（还没落盘，后台线程写）
        self.dropped = 0                   # 本次会话丢掉的帧数
        self.error = ""                    # 最近一次错误（界面直接显示）
        self._dataset = None
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._task_text = ""
        self._last_sim_time = -1.0         # 按仿真时钟抽帧用
        self._pending = 0                  # 主线程：距上次收局已投了多少帧
        self._w_frames = 0                 # 后台线程：当前这一局攒了多少帧
        self._w_success = False
        self._w_dropped = False

    # ────────────────── 主线程接口 ──────────────────
    @property
    def active(self) -> bool:
        return self.armed

    @property
    def queued(self) -> int:
        """还有多少帧在队列里等后台写（界面状态行显示）。"""
        return 0 if self._queue is None else self._queue.qsize()

    def due(self, sim_time: float) -> bool:
        """按**仿真时钟**抽帧：距上一帧够 ``1/fps`` 就返回 True（与界面帧率、墙钟都无关）。

        ⚠ 这里只看"数据集是否已打开"；"什么时候算在采集"由调用方决定（本工程 = 连续任务进行中）。
        """
        if not self.armed:
            return False
        if sim_time < self._last_sim_time + 1.0 / self.fps - 1e-9:
            return False
        self._last_sim_time = sim_time
        return True

    def start(self, base_name: str, task_text: str, fps: float, cameras: list[str],
              resume: bool = False, only_success: bool = True) -> Path:
        """开始采集：后台建库 / 续采。返回数据集目录（此刻可能还在创建中）。"""
        self.stop(save=True)
        base = (base_name or "").strip() or DEFAULT_NAME
        self.name = base if resume else stamped_name(base)
        self.path = datasets_dir() / self.name
        self.resumed = bool(resume) and (self.path / "meta" / "info.json").exists()
        self.fps = max(1.0, float(fps))
        self.cameras = list(cameras)
        self.only_success = bool(only_success)
        self.error = ""
        self.episodes = self.frames = self.dropped = 0
        self.buffered = 0
        self._pending = 0
        self._w_frames = 0
        self._w_success = self._w_dropped = False
        self._last_sim_time = -1.0
        self.armed = True
        self.wanted = True
        self._queue = queue.Queue(maxsize=QUEUE_LIMIT)
        self._thread = threading.Thread(target=self._run, args=(self.path, task_text),
                                        name="alicia-dataset", daemon=True)
        self._thread.start()
        return self.path

    def add_frame(self, images: dict, state: np.ndarray, action: np.ndarray,
                  task_text: str, success: bool) -> None:
        """投一帧（不阻塞）。⚡ 图像在这里就**拷一份**：MuJoCo 渲染缓冲下一帧会被覆盖。"""
        if not self.armed or self._queue is None or not images:
            return
        payload = ({camera: np.array(frame, copy=True) for camera, frame in images.items()},
                   np.asarray(state, dtype=np.float32), np.asarray(action, dtype=np.float32),
                   task_text, bool(success))
        if self._push(("frame", payload)):
            self._pending += 1

    def end_episode(self) -> None:
        """收掉当前这一局（复位 / 停止采集时调用）；这一局没有帧就什么也不做。"""
        if self._queue is None or self._pending == 0:
            self._pending = 0
            return
        self._pending = 0
        self._push(("end", None), keep=True)

    def stop(self, save: bool = True, timeout: float = 30.0) -> None:
        """收尾：结束当前局（``save=True`` 时按"仅成功局"落盘）→ 等后台写完 → 关数据集。"""
        if self._queue is None or self._thread is None:
            self.armed = False
            return
        if save:
            self.end_episode()
        self.armed = False
        self._push(("close", None), keep=True)
        self._thread.join(timeout=timeout)
        self._thread = None
        self._queue = None

    def stats_text(self) -> str:
        """一行摘要（界面状态行用）。"""
        bits = [f"{self.episodes} 局", f"{self.frames} 帧"]
        if self.buffered:
            bits.append(f"本局 {self.buffered}")
        if self.queued:
            bits.append(f"待写 {self.queued}")
        if self.dropped:
            bits.append(f"丢帧 {self.dropped}")
        return " / ".join(bits)

    # ────────────────── 内部：队列 ──────────────────
    def _push(self, item, keep: bool = False) -> bool:
        """入队。``keep=True``（收局 / 关闭消息）时若队列满了就挤掉一帧，保证控制消息不丢。"""
        queue_ = self._queue
        if queue_ is None:
            return False
        try:
            queue_.put_nowait(item)
            return True
        except queue.Full:
            pass
        if not keep:
            self.dropped += 1
            self._w_dropped = True
            return False
        try:                                   # 控制消息不能丢：腾一格（等同于丢一帧）
            queue_.get_nowait()
            self.dropped += 1
            self._w_dropped = True
            queue_.put_nowait(item)
            return True
        except (queue.Empty, queue.Full):
            return False

    # ────────────────── 内部：后台线程 ──────────────────
    def _run(self, path: Path, task_text: str) -> None:
        queue_ = self._queue                      # 用局部引用：stop() 超时后会把 self._queue 置空
        try:
            self._open(path, task_text)
        except Exception as exc:               # noqa: BLE001（错误只上报，不抛穿线程）
            self.error = f"打开数据集失败：{exc}"
            self.armed = False
            self._dataset = None
            self._drain(queue_)
            return
        while True:
            kind, payload = queue_.get()
            try:
                if kind == "frame":
                    self._write_frame(*payload)
                elif kind == "end":
                    self._write_end()
                else:                          # "close"
                    break
            except Exception as exc:           # noqa: BLE001
                self.error = f"写盘失败：{exc}"
                self.armed = False
                break
        self._close()

    def _drain(self, queue_) -> None:
        """丢弃队列里剩下的消息（打不开数据集时用）。"""
        while not queue_.empty():
            try:
                queue_.get_nowait()
            except queue.Empty:
                break

    def _open(self, path: Path, task_text: str) -> None:
        """建库或续采。lerobot 的导入也放这儿 —— 不开采集就不背这个启动包袱。"""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset   # noqa: PLC0415

        path.parent.mkdir(parents=True, exist_ok=True)
        repo_id = f"{REPO_PREFIX}/{self.name}"
        if (path / "meta" / "info.json").exists():                     # 断点续采
            dataset = LeRobotDataset(repo_id=repo_id, root=path)
            self._check_resume_compatible(dataset)
            self.resumed = True
        else:                                                          # 新建数据集
            dataset = LeRobotDataset.create(
                repo_id=repo_id, fps=int(round(self.fps)),
                features=features_for(self.cameras, self.shape), root=path,
                robot_type="alicia_d", use_videos=True,
                image_writer_processes=0, image_writer_threads=4,
                video_backend="pyav", batch_encoding_size=1)
            self.resumed = False
        self._dataset = dataset
        self._task_text = task_text
        sweep_images(path)                     # 上次异常退出可能留下临时帧目录，开录前扫干净

    def _check_resume_compatible(self, dataset) -> None:
        """续采前核对**相机集合** / 帧率 / 分辨率：不一致就报错，而不是把数据写坏。

        ⚠ 相机集合要**完全一致**：LeRobot 的 features 在建库时就定死了，帧里少一路（或
        多一路）都会在 ``add_frame`` 的校验里直接报错，所以宁可开库时就说清楚。
        """
        problems = []
        meta = dataset.meta
        if int(meta.fps) != int(round(self.fps)):
            problems.append(f"帧率 {meta.fps} ≠ {int(round(self.fps))}")
        recorded = {k.split("observation.images.")[-1] for k in meta.video_keys}
        wanted = set(self.cameras)
        if recorded != wanted:
            missing = sorted(wanted - recorded)
            extra = sorted(recorded - wanted)
            detail = []
            if missing:
                detail.append(f"少录 {missing}")
            if extra:
                detail.append(f"多录 {extra}")
            problems.append(f"相机集合不一致（{'；'.join(detail)}）")
        for camera in self.cameras:
            key = video_key(camera)
            feature = meta.features.get(key)
            if feature is None:
                problems.append(f"缺少相机 {camera}")
            elif tuple(feature["shape"]) != (*self.shape, 3):
                problems.append(f"{camera} 分辨率 {tuple(feature['shape'][:2])} ≠ {self.shape}")
        if problems:
            raise ValueError("同名数据集与本次设置不一致（" + "；".join(problems)
                             + "）—— 改个名字或关掉「断点续采」")

    def _write_frame(self, images: dict, state: np.ndarray, action: np.ndarray,
                     task_text: str, success: bool) -> None:
        """后台线程：真正写一帧（缩图 + JPEG + 缓冲）。"""
        self._w_frames += 1
        self._w_success = self._w_success or bool(success)
        flag = np.array([1.0 if success else 0.0], dtype=np.float32)
        frame = {
            "observation.state": state,
            "action": action,
            "next.reward": flag,
            "next.success": flag,
            "task": task_text or self._task_text,
        }
        for camera in self.cameras:
            frame[video_key(camera)] = resized(images[camera], self.shape)
        self._dataset.add_frame(frame)          # timestamp / frame_index 由 lerobot 补
        self.buffered = self._w_frames          # 当前这一局已攒多少帧（界面显示用）

    def _write_end(self) -> None:
        """后台线程：一局结束 —— 按「仅成功局」决定落盘还是丢弃。"""
        dataset, frames = self._dataset, self._w_frames
        success, dropped = self._w_success, self._w_dropped
        self._w_frames = 0
        self._w_success = self._w_dropped = False
        self.buffered = 0
        self._last_sim_time = -1.0             # 下一局第一帧一定录得上
        if dataset is None or frames == 0:
            return
        if dropped or (self.only_success and not success):
            dataset.clear_episode_buffer(delete_images=True)
            sweep_images(self.path)                # 顺手扫掉临时帧目录（视频之外不该留 images/）
            return
        # ⚠ parallel_encoding=False：Windows + Qt 里进程池不稳，顺序编码放在后台线程做
        # （4 路 × 320×228 的一局 ≈ 1s），本来也不占主循环。
        dataset.save_episode(parallel_encoding=False)
        self.episodes += 1
        self.frames += frames                  # 只有**落盘的**局才计入帧数（丢弃的不算）
        sweep_images(self.path)                # 视频编完就把临时帧目录删掉（只留 data/meta/videos）

    def _close(self) -> None:
        """后台线程：收尾（关图像写入线程 + parquet writer，否则数据集读不出来）。"""
        dataset, self._dataset = self._dataset, None
        if dataset is None:
            return
        try:
            dataset.stop_image_writer()
        except Exception:                       # noqa: BLE001（没开线程 writer 时无所谓）
            pass
        try:
            dataset.finalize()
        except Exception as exc:                # noqa: BLE001
            self.error = f"收尾失败：{exc}"
        if self.path is not None:               # 关库时再扫一遍：数据集里只留 data/meta/videos
            sweep_images(self.path)
