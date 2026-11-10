"""连续运行：模型只加载一次，不断取图 → 决策 → 下发灯板。

三种取图方式（任选其一）：
  # 1) 摄像头：每隔 N 秒抓一帧（默认 1 秒），决策变了才下发；带预览窗口，按 q 退出
  .venv/Scripts/python scripts/live.py --camera 0 --interval 1 --show

  # 2) 盯一个文件夹：往里面丢图片 / 手机拍照同步进来，就自动处理新图
  .venv/Scripts/python scripts/live.py --watch inbox

  # 3) 交互：在命令行里一张张输入（拖入）图片路径，回车即处理；输入 q 退出
  .venv/Scripts/python scripts/live.py

通用参数：--distance 60  --mode scene_priority  --fake（发假板端）  --no-send（只推理）
          --stable 2（摄像头：同一决策连续出现 2 次才切换，防闪）  --off-on-exit（退出时关灯）
Ctrl+C 随时退出。台架上限约 13 帧/秒，摄像头间隔不要小于 0.1 秒。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fill_light.pipeline import FillLightPipeline, imread_rgb  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def key_of(res: dict) -> tuple:
    """决策指纹：颜色 / 灯型 / 亮度（取整到 5%）/ 侧光 / 顶光 一样就视为同一决策，不重复下发。"""
    lp = res["light_params"]
    hs = lambda c: None if not c else (round(c["hue"] / 10) % 36, round(c["sat"] / 5))   # 色相 10°、饱和度 5% 以内算没变
    return (lp["light_type"], hs(lp["main"]), hs(lp["contour"]),
            round(lp["brightness_pct"] / 5), tuple(sorted(lp["side_boost_pct"].items())),
            lp["top_gradient_pct"], hs(lp["eye_patch"]))


def brief(res: dict) -> str:
    lp, env = res["light_params"], res["env_light"]
    color = lp["t1_format"] or (lp["main"]["name"] + lp["main"]["hex"]
                                + (("/" + lp["contour"]["name"]) if lp["light_type"] == "sandwich" else ""))
    face = "无人脸" if not res["perception"]["faces"] else f"脸{res['perception']['faces']}"
    return (f"{face} | {env['env_light_category']} {env['env_light_name']} S{env['saturation']:.0f}% "
            f"{env['pollution_level']} | {lp['light_type']} {color} 亮{lp['brightness_pct']:.0f}%")


class Runner:
    def __init__(self, a):
        self.a = a
        self.pipe = FillLightPipeline(a.cfg)
        self.link = None
        if not a.no_send:
            from fill_light.bench import BenchLink
            self.link = BenchLink(self.pipe.cfg, *(("127.0.0.1", 8080) if a.fake else (a.host, a.port)))
        self.last_key = None
        self.pending_key, self.pending_n = None, 0

    def handle(self, rgb, tag: str, force: bool = False) -> None:
        t0 = time.perf_counter()
        res = self.pipe.run(rgb, mode=self.a.mode, object_type=self.a.object, distance_cm=self.a.distance)
        res.pop("_internal", None)
        ms = (time.perf_counter() - t0) * 1000
        k = key_of(res)
        stamp = time.strftime("%H:%M:%S")
        if not force:                                     # 摄像头防闪：同一新决策连续 N 次才切
            if k == self.last_key:
                self.pending_key, self.pending_n = None, 0
                print(f"[{stamp}] {tag} | 决策未变，不重发 | {ms:.0f}ms", end="\r")
                return
            if k != self.pending_key:
                self.pending_key, self.pending_n = k, 0
            self.pending_n += 1
            if self.pending_n < self.a.stable:
                return
        print(f"\n[{stamp}] {tag} | {brief(res)} | 推理 {ms:.0f}ms")
        if self.link is None:
            self.last_key = k
            return
        try:
            sent = self.link.send_result(res)
            self.last_key = k
            print(f"           → 已下发：{sent['rtt_ms']}ms，估算 {sent['current_a_sent']}A"
                  + (f"（台架限流 ×{sent['bench_scale']}）" if sent["bench_scale"] < 1 else ""))
        except Exception as e:  # noqa: BLE001  —— 单次失败不退出，下一帧再试
            print(f"           → 下发失败：{e}")

    # ---------- 三种取图方式 ----------
    def run_camera(self) -> None:
        cap = cv2.VideoCapture(self.a.camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(self.a.camera)
        if not cap.isOpened():
            raise SystemExit(f"打不开摄像头 {self.a.camera}（换个编号试试：--camera 1）")
        print(f"摄像头 {self.a.camera} 已打开，每 {self.a.interval}s 决策一次；"
              f"{'预览窗口按 q 退出' if self.a.show else 'Ctrl+C 退出'}")
        next_t = 0.0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    print("读帧失败，重试…"); time.sleep(0.5); continue
                if self.a.show:
                    cv2.imshow("smart_fill_light (q 退出)", bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if time.monotonic() >= next_t:
                    next_t = time.monotonic() + self.a.interval
                    h, w = bgr.shape[:2]
                    s = 1280 / max(h, w)
                    if s < 1:
                        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                    self.handle(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "camera")
                elif not self.a.show:
                    time.sleep(0.02)
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_watch(self) -> None:
        d = Path(self.a.watch)
        d.mkdir(parents=True, exist_ok=True)
        seen = {p: p.stat().st_mtime for p in d.iterdir() if p.suffix.lower() in EXTS}
        print(f"盯文件夹 {d.resolve()}：放进新图片就处理（已有 {len(seen)} 张忽略）；Ctrl+C 退出")
        while True:
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in EXTS:
                    continue
                m = p.stat().st_mtime
                if seen.get(p) == m:
                    continue
                time.sleep(0.3)                           # 等文件写完
                seen[p] = p.stat().st_mtime
                try:
                    self.handle(imread_rgb(p), p.name, force=True)
                except Exception as e:  # noqa: BLE001
                    print(f"\n{p.name} 处理失败：{e}")
            time.sleep(0.5)

    def run_interactive(self) -> None:
        print("输入（或把图片拖进来）图片路径回车处理；q 退出")
        while True:
            try:
                s = input("\n图片> ").strip().strip('"').strip("'")
            except EOFError:
                break
            if s.lower() in ("q", "quit", "exit"):
                break
            if not s:
                continue
            p = Path(s)
            if not p.exists():
                print(f"找不到：{p}")
                continue
            try:
                self.handle(imread_rgb(p), p.name, force=True)
            except Exception as e:  # noqa: BLE001
                print(f"处理失败：{e}")

    def close(self) -> None:
        if self.link is not None and self.a.off_on_exit:
            try:
                self.link.off(); print("\n已关灯")
            except Exception:  # noqa: BLE001
                pass
        self.pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="连续运行：取图 → 决策 → 下发灯板")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--camera", type=int, help="摄像头编号（0 = 默认摄像头）")
    g.add_argument("--watch", help="盯一个文件夹，处理新放进来的图片")
    ap.add_argument("--interval", type=float, default=1.0, help="摄像头决策间隔（秒）")
    ap.add_argument("--stable", type=int, default=2, help="摄像头：新决策连续出现几次才切换")
    ap.add_argument("--show", action="store_true", help="摄像头：显示预览窗口")
    ap.add_argument("--mode", default="skin_priority", choices=["skin_priority", "scene_priority"])
    ap.add_argument("--object", default=None)
    ap.add_argument("--distance", type=float, default=None)
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--off-on-exit", action="store_true")
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    r = Runner(a)
    try:
        if a.camera is not None:
            r.run_camera()
        elif a.watch:
            r.run_watch()
        else:
            r.run_interactive()
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        r.close()


if __name__ == "__main__":
    main()
