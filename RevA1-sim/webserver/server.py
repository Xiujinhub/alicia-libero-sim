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
    python webserver/server.py --point-cloud --follow --task-listen

然后浏览器打开 ``http://127.0.0.1:8080/``。

只依赖标准库 + 已经装好的 ``mujoco`` / ``numpy`` / ``Pillow``，复用本工程现成的
``arm_core`` / ``point_cloud`` / ``robot_link`` / ``revA1_spec``（**不修改**这些模块）。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import subprocess
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

def _load_config() -> dict:
    """读 ``config/config.json``（缺文件/损坏时返回空 dict）。"""
    try:
        return json.loads((ROOT / "config" / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


INDEX_HTML = (HERE / "index.html").read_text(encoding="utf-8").replace(
    "__JETSON_IP__", str(_load_config().get("jetson_ip") or link.DEFAULT_HOST))


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
                                       port=args.follow_port,
                                       tcp_host=args.follow_tcp_host,
                                       tcp_port=args.follow_tcp_port)
            self.link.start()

        # 任务信号（可选）：start 记轨迹 / over 清空
        self.task_listener = None
        self.task_active = False
        # 要记录轨迹的工具（默认喷嘴1 + 刷子；喷嘴2、夹爪默认关闭）
        self.trail_names = [t.strip() for t in str(args.trail_tools).split(",")
                            if t.strip() in spec.TOOLS]
        if args.task_listen:
            # 触发源：HTTP /api/task/motion（web_server 已把 UDP 广播按 seq 去重后缓存）；
            # 需要时 --task-udp 再把 6501 广播也收上（双通道，谁先到用谁、自动去重）
            self.task_listener = link.TaskLink(args.task_port, base=args.task_base,
                                              udp=bool(args.task_udp))
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
        groups = pc.load_cloud_groups()          # 配置里的多组点云（拍照位姿 → 多份点云）
        try:
            if groups and not args.cloud_file:
                # 配了多组 → 每组按自己的拍照位姿变换到基座系，最后合并进同一个场景
                print(f"[web] 点云组：{len(groups)} 组"
                      f"（{'、'.join(str(g['label']) for g in groups)}）")
                res = pc.build_multi_cloud_scene(groups,
                                                 max_points=args.max_points,
                                                 point_mm=args.point_mm,
                                                 bands=args.bands)
            else:
                res = pc.build_cloud_scene(args.cloud_file or None,
                                           max_points=args.max_points,
                                           point_mm=args.point_mm,
                                           bands=args.bands)
        except pc.PointCloudError as exc:
            print(f"[web] 点云场景生成失败（退回基场景）：{exc}")
            return str(spec.SCENE_XML)
        scene = str(res["scene"].scene)
        print(f"[web] 点云场景：{scene}")
        for line in res["report"]:
            print(f"[web]   {line}")
        return scene

    # ------------------------------------------------------------------ 渲染循环
    def _grow_framebuffer(self) -> None:
        """按需放大离屏 framebuffer。

        MuJoCo 默认 ``vis.global_.offwidth/offheight`` 只有 1280×960，渲染尺寸一旦超过它，
        ``mujoco.Renderer`` 会直接抛 ``Image width ... > framebuffer width``。这里**只放大、
        不缩小**（场景 XML 里特意开大的值不动），够这次渲染就行。
        """
        g = self.sim.model.vis.global_
        w, h = int(self.width), int(self.height)
        if w > int(g.offwidth) or h > int(g.offheight):
            g.offwidth = max(int(g.offwidth), w)
            g.offheight = max(int(g.offheight), h)
            print(f"[web] 离屏 framebuffer 放大到 {int(g.offwidth)}x{int(g.offheight)}"
                  f"（渲染 {w}x{h}；MuJoCo 默认 1280×960）")

    def _loop(self):
        self._grow_framebuffer()
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


def _is_private_ip(ip: str) -> bool:
    """RFC1918 私网段判断（10/8、172.16/12、192.168/16），用于过滤虚拟网卡的假 IP。"""
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)


def _lan_ips() -> list[str]:
    """探测本机局域网 IPv4 地址（跳过 127.x 回环和虚拟网卡假 IP）。"""
    ips: list[str] = []
    # 1) UDP connect 到公网地址，拿到系统实际出网网卡的 IP（不真正发包）
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        probe.close()
        if ip and not ip.startswith("127."):
            ips.append(ip)
    except OSError:
        pass
    # 2) 兜底：枚举 hostname 解析出的所有私网 IPv4（多网卡 / 上面失败时）
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and _is_private_ip(ip):
                ips.append(ip)
    except OSError:
        pass
    return ips


def _allow_firewall(port: int) -> bool:
    """放行 Windows 防火墙入站 TCP 端口（需管理员权限；非 Windows 视为无需处理）。"""
    if os.name != "nt":
        return True
    rule = f"RevA1-sim web {port}"
    add = ["netsh", "advfirewall", "firewall", "add", "rule",
           f"name={rule}", "dir=in", "action=allow", "protocol=TCP",
           f"localport={port}"]
    show = ["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule}"]
    try:
        subprocess.run(add, capture_output=True, timeout=15, check=False)
        out = subprocess.run(show, capture_output=True, timeout=15, check=False)
        return out.returncode == 0 and b"Allow" in out.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="RevA1-sim 网页画面复刻（MJPEG 流，只看画面、无控制控件）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="监听地址")
    ap.add_argument("--port", type=int, default=8080, help="监听端口")
    ap.add_argument("--width", type=int, default=4320, help="渲染宽度[px]")
    ap.add_argument("--height", type=int, default=1920, help="渲染高度[px]")
    ap.add_argument("--fps", type=float, default=30.0, help="渲染/推流帧率上限")
    ap.add_argument("--distance", type=float, default=2.5, help="相机距离[m]")
    ap.add_argument("--azimuth", type=float, default=175.0, help="相机方位角[deg]")
    ap.add_argument("--elevation", type=float, default=-45.0, help="相机仰角[deg]")
    ap.add_argument("--lookat", nargs=3, type=float, default=None,
                    help="相机看向点（x y z，米；默认 = 工具尖）")
    ap.add_argument("--point-cloud", action="store_true",
                    help="加载点云场景（机械臂 + 点云 + 小车）")
    ap.add_argument("--cloud-file", default="", help="指定点云文件（默认自动挑 point_cloud/ 里的）")
    ap.add_argument("--max-points", type=int, default=60000, help="点云点数上限")
    ap.add_argument("--point-mm", type=float, default=4.0, help="点云每个点的尺寸[mm]")
    ap.add_argument("--bands", type=int, default=6, help="点云按深度分几带颜色")
    ap.add_argument("--follow", action="store_true", help="启动真机跟随（真机姿态实时映到模型）")
    ap.add_argument("--follow-source", default="http", choices=link.SOURCES,
                    help="跟随的数据源（默认 http：轮询真机 /api/state，和 clean-robot 的 vue 端一样）")
    ap.add_argument("--follow-url", default=link.DEFAULT_HTTP_URL,
                    help="真机 HTTP 状态接口（GET /api/state）")
    ap.add_argument("--follow-tcp-host", default=link.DEFAULT_HOST,
                    help="真机 TCP 状态流地址（默认取 config.json 的 jetson_ip）")
    ap.add_argument("--follow-tcp-port", type=int, default=link.DEFAULT_TCP_PORT,
                    help="真机 TCP 状态流端口")
    ap.add_argument("--follow-port", type=int, default=link.DEFAULT_UDP_PORT,
                    help="真机 UDP 广播端口（--follow-source udp/auto 用）")
    ap.add_argument("--task-listen", action="store_true",
                    help="监听任务信号驱动轨迹（HTTP /api/task/motion；默认不碰 UDP）")
    ap.add_argument("--task-port", type=int, default=link.DEFAULT_TASK_PORT,
                    help="任务信号 UDP 端口（只在 --task-udp 时用）")
    ap.add_argument("--task-base", default=link.DEFAULT_TASK_BASE,
                    help="任务信号 HTTP 基址（…/api/task/motion + …/api/task/status）")
    ap.add_argument("--task-udp", action="store_true",
                    help="再把 UDP 广播也收上（双通道，重复信号自动去重）")
    ap.add_argument("--trail-tools", default="刷子",
                    help="要记录轨迹的工具（逗号分隔，可选 夹爪/喷嘴1/喷嘴2/刷子；默认刷子）")
    args = ap.parse_args(argv)

    viewer = Viewer(args)
    Handler.viewer = viewer
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"RevA1-sim 网页画面（本机）：http://127.0.0.1:{args.port}/")
    for ip in _lan_ips():
        print(f"局域网访问：http://{ip}:{args.port}/")
    if not _allow_firewall(args.port):
        print("（提示）未自动放行防火墙，其它电脑若访问不了，请以管理员身份执行：")
        print(f"  netsh advfirewall firewall add rule name=\"RevA1-sim web {args.port}\" "
              f"dir=in action=allow protocol=TCP localport={args.port}")
    print(f"场景：{viewer.sim.scene}")
    print(f"渲染：{args.width}x{args.height} @ {args.fps:.0f} fps"
          + (" · 真机跟随" if viewer.link else "")
          + (" · 任务信号轨迹" if viewer.task_listener else ""))
    if viewer.task_listener:
        print(viewer.task_listener.describe())
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



