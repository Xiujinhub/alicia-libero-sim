#!/usr/bin/env python3
"""RevA1-sim 网页画面复刻：把 MuJoCo 模拟画面推成网页 **MJPEG 流**。

画面里包含机械臂、点云、三个工具 TCP 向量箭头、工具尖轨迹（任务信号驱动），
但**不含**任何控制机械臂的控件（关节微调 / IK / 真机跟随按钮等）——网页端只看画面，
可以左键旋转 / 中键平移 / 滚轮缩放 / 双击复位视角。

运行（在 ``RevA1-sim`` 目录下，或在任意位置直接跑本脚本）：

.. code-block:: bash

    python webserver/server.py                      # 基场景（机械臂 + 地面 + 工具箭头）
    python webserver/server.py --point-cloud        # 机械臂 + 点云 + 小车
    python webserver/server.py --follow             # 再接真机跟随（真机动 → 画面跟着动）
    python webserver/server.py --task-listen        # 监听 6501 任务信号：start 记轨迹 / over 清空
    python webserver/server.py --port 8080 --width 960 --height 600 --fps 30

然后浏览器打开 ``http://127.0.0.1:8080/``。

只依赖标准库 + 已经装好的 ``mujoco`` / ``numpy`` / ``Pillow``，复用本工程现成的
``arm_core`` / ``point_cloud`` / ``robot_link`` / ``revA1_spec``（**不修改**这些模块）。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from gl_backend import configure_gl  # noqa: E402  (必须在 import mujoco 之前)

configure_gl()
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

import arm_core as core  # noqa: E402
import point_cloud as pc  # noqa: E402
import revA1_spec as spec  # noqa: E402
import robot_link as link  # noqa: E402

INDEX_HTML = (HERE / "index.html").read_text(encoding="utf-8")


class Viewer:
    """加载场景 + 真机跟随/任务信号 + 离屏渲染 → JPEG 帧，全部在**后台线程**里跑。

    MuJoCo 的 ``Renderer`` 不是线程安全的，所以这里让它在渲染线程里创建和使用，
    HTTP 线程只通过 :meth:`jpeg` 读最新一帧字节（线程安全）。
    """

    def __init__(self, args):
        self.width = int(args.width)
        self.height = int(args.height)
        self.fps = float(max(args.fps, 1.0))

        # 场景：基场景 或 点云场景（build_cloud_scene 生成 / 复用已有 revA1_pc_scene.xml）
        scene = self._resolve_scene(args)
        self.sim = core.ArmSim(scene, log_scene=False)

        # 相机：默认看向工具尖，让画面始终聚焦末端
        self.cam = core.Camera(distance=float(args.distance),
                               azimuth=float(args.azimuth),
                               elevation=float(args.elevation))
        if args.lookat:
            self.cam.look_at([float(x) for x in args.lookat])
        else:
            self.cam.look_at(self.sim.tip())

        # 真机跟随（可选）：真机姿态实时映到模型
        self.link = None
        if args.follow:
            self.link = link.JointLink(source=args.follow_source,
                                       http_url=args.follow_url,
                                       port=args.follow_port)
            self.link.start()

        # 任务信号（可选）：start 记轨迹 / over 清空
        self.task_listener = None
        self.task_active = False
        # 要记录轨迹的工具（默认只记喷嘴1；喷嘴2、夹爪默认关闭）
        self.trail_names = [t.strip() for t in str(args.trail_tools).split(",")
                            if t.strip() in spec.TOOLS]
        if args.task_listen:
            self.task_listener = link.TaskListener(args.task_port)
            self.task_listener.start()

        # 帧缓冲（线程安全）
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._fps_now = 0.0
        self._render_ms = 0.0
        self._running = True
        self.renderer = None

        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="revA1-web-render")
        self._thread.start()

    # ------------------------------------------------------------------ 场景
    def _resolve_scene(self, args) -> str:
        if not args.point_cloud:
            return str(spec.SCENE_XML)
        try:
            res = pc.build_cloud_scene(args.cloud_file or None,
                                       max_points=args.max_points,
                                       point_mm=args.point_mm,
                                       bands=args.bands)
        except pc.PointCloudError as exc:
            print(f"[web] 点云场景生成失败（退回基场景）：{exc}")
            return str(spec.SCENE_XML)
        scene = str(res["scene"].scene)
        print(f"[web] 点云场景：{scene}")
        for line in res["report"][:6]:
            print(f"[web]   {line}")
        return scene

    # ------------------------------------------------------------------ 渲染循环
    def _loop(self):
        self.renderer = mujoco.Renderer(self.sim.model, self.height, self.width)
        acc = 0.0
        last = time.perf_counter()
        n_frames = 0
        t_fps0 = last
        try:
            while self._running:
                now = time.perf_counter()
                dt_wall = min(now - last, 0.25)
                last = now

                # 真机跟随：先更新目标，再推进物理
                if self.link is not None:
                    out = self.link.update(self.sim.cmd)
                    if out.ok:
                        self.sim.follow(out.q_rad)

                # 任务信号 → 工具尖轨迹
                if self.task_listener is not None:
                    for sig in self.task_listener.drain_signals():
                        if sig.motion == link.MOTION_START:
                            self.task_active = True
                        elif sig.motion == link.MOTION_OVER:
                            self.task_active = False
                            self.sim.clear_trails()
                    if self.task_active:
                        self.sim.record_trails(self.trail_names)

                # 推进物理（按真实时间补步，最多 80 步防雪崩）
                acc += dt_wall
                n = 0
                while acc >= self.sim.dt and n < 80:
                    self.sim.step()
                    acc -= self.sim.dt
                    n += 1
                if n >= 80:
                    acc = 0.0

                # 渲染 + 画工具向量箭头 + 画轨迹 → JPEG
                t0 = time.perf_counter()
                self.renderer.update_scene(self.sim.data, camera=self.cam.mjv)
                self.sim.add_tool_arrows(self.renderer.scene)
                self.sim.draw_trails(self.renderer.scene)
                arr = self.renderer.render()
                buf = io.BytesIO()
                Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
                self._render_ms = (0.9 * self._render_ms
                                   + 0.1 * ((time.perf_counter() - t0) * 1000.0))
                with self._lock:
                    self._jpeg = buf.getvalue()

                # 帧率统计 + 限帧（省 CPU）
                n_frames += 1
                if now - t_fps0 >= 1.0:
                    self._fps_now = n_frames / (now - t_fps0)
                    n_frames = 0
                    t_fps0 = now
                sleep = 1.0 / self.fps - (time.perf_counter() - now)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            if self.renderer is not None:
                try:
                    self.renderer.close()
                except Exception:  # noqa: BLE001
                    pass
                self.renderer = None

    # ------------------------------------------------------------------ 供 HTTP 用
    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        return dict(fps=self._fps_now, render_ms=self._render_ms,
                    mode=self.sim.mode, tip=[round(float(x), 3) for x in self.sim.tip()],
                    follow=bool(self.link is not None and self.link.running),
                    task=self.task_active,
                    scene=self.sim.scene.name)

    def orbit(self, dx, dy) -> None:
        self.cam.orbit(float(dx), float(dy))

    def zoom(self, notches) -> None:
        self.cam.zoom(float(notches))

    def reset_view(self) -> None:
        self.cam.reset()

    def pan(self, dx, dy) -> None:
        self.cam.pan(float(dx), float(dy))

    def shutdown(self) -> None:
        self._running = False
        if self.link is not None:
            self.link.stop()
            self.link = None
        if self.task_listener is not None:
            self.task_listener.stop()
            self.task_listener = None


class Handler(BaseHTTPRequestHandler):
    """HTTP 路由：页面 / MJPEG 流 / 单帧快照 / 状态 / 相机控制。"""

    server_version = "RevA1SimWeb/1.0"
    viewer: "Viewer" = None  # 由 main() 赋值

    # ------------------------------------------------------------ 请求
    def do_GET(self) -> None:  # noqa: N802 (http.server 的接口名)
        url = urlparse(self.path)
        path, qs = url.path, parse_qs(url.query)
        v = self.viewer
        try:
            if path in ("/", "/index.html"):
                self._send_html(INDEX_HTML)
            elif path == "/stream":
                self._send_stream()
            elif path == "/snapshot.jpg":
                jpeg = v.jpeg()
                if jpeg is None:
                    self.send_error(503, "首帧还没渲染出来")
                else:
                    self._send_bytes(jpeg, "image/jpeg")
            elif path == "/status":
                self._send_json(v.status())
            elif path == "/cam/orbit":
                v.orbit(qs.get("dx", ["0"])[0], qs.get("dy", ["0"])[0])
                self._send_empty()
            elif path == "/cam/pan":
                v.pan(qs.get("dx", ["0"])[0], qs.get("dy", ["0"])[0])
                self._send_empty()
            elif path == "/cam/zoom":
                v.zoom(qs.get("n", ["0"])[0])
                self._send_empty()
            elif path == "/cam/reset":
                v.reset_view()
                self._send_empty()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):  # noqa: A003
        # 关掉默认的访问日志（渲染线程已经够吵了）
        pass

    # ------------------------------------------------------------ 响应
    def _send_html(self, html: str) -> None:
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

    def _send_json(self, obj) -> None:
        self._send_bytes(json.dumps(obj).encode("utf-8"),
                         "application/json; charset=utf-8")

    def _send_empty(self) -> None:
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_bytes(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_stream(self) -> None:
        """MJPEG 流：multipart/x-mixed-replace，浏览器 ``<img src="/stream">`` 直接显示。"""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        boundary = b"--frame\r\n"
        while True:
            jpeg = self.viewer.jpeg()
            if jpeg is None:
                time.sleep(0.01)
                continue
            try:
                self.wfile.write(boundary)
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            time.sleep(0.01)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="RevA1-sim 网页画面复刻（MJPEG 流，只看画面、无控制控件）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="监听地址")
    ap.add_argument("--port", type=int, default=8080, help="监听端口")
    ap.add_argument("--width", type=int, default=960, help="渲染宽度[px]")
    ap.add_argument("--height", type=int, default=600, help="渲染高度[px]")
    ap.add_argument("--fps", type=float, default=30.0, help="渲染/推流帧率上限")
    ap.add_argument("--distance", type=float, default=2.5, help="相机距离[m]")
    ap.add_argument("--azimuth", type=float, default=135.0, help="相机方位角[deg]")
    ap.add_argument("--elevation", type=float, default=-25.0, help="相机仰角[deg]")
    ap.add_argument("--lookat", nargs=3, type=float, default=None,
                    help="相机看向点（x y z，米；默认 = 工具尖）")
    ap.add_argument("--point-cloud", action="store_true",
                    help="加载点云场景（机械臂 + 点云 + 小车）")
    ap.add_argument("--cloud-file", default="", help="指定点云文件（默认自动挑 point_cloud/ 里的）")
    ap.add_argument("--max-points", type=int, default=60000, help="点云点数上限")
    ap.add_argument("--point-mm", type=float, default=4.0, help="点云每个点的尺寸[mm]")
    ap.add_argument("--bands", type=int, default=6, help="点云按深度分几带颜色")
    ap.add_argument("--follow", action="store_true", help="启动真机跟随（真机姿态实时映到模型）")
    ap.add_argument("--follow-source", default="auto", choices=("auto", "http", "udp"))
    ap.add_argument("--follow-url", default=link.DEFAULT_HTTP_URL, help="真机 HTTP 状态接口")
    ap.add_argument("--follow-port", type=int, default=link.DEFAULT_UDP_PORT,
                    help="真机 UDP 广播端口")
    ap.add_argument("--task-listen", action="store_true", help="监听 6501 任务信号驱动轨迹")
    ap.add_argument("--task-port", type=int, default=link.DEFAULT_TASK_PORT, help="任务信号 UDP 端口")
    ap.add_argument("--trail-tools", default="喷嘴1",
                    help="要记录轨迹的工具（逗号分隔，可选 夹爪/喷嘴1/喷嘴2；默认只记喷嘴1）")
    args = ap.parse_args(argv)

    viewer = Viewer(args)
    Handler.viewer = viewer
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"RevA1-sim 网页画面：http://127.0.0.1:{args.port}/")
    print(f"场景：{viewer.sim.scene}")
    print(f"渲染：{args.width}x{args.height} @ {args.fps:.0f} fps"
          + (" · 真机跟随" if viewer.link else "")
          + (" · 任务信号轨迹" if viewer.task_listener else ""))
    if viewer.task_listener:
        print(f"轨迹工具：{'、'.join(viewer.trail_names)}")
    print("Ctrl+C 退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 退出")
    finally:
        viewer.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



