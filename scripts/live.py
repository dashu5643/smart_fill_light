"""连续运行：模型只加载一次，不断取图 → 决策 → 下发灯板。Ctrl+C 退出。

  python scripts/live.py                                   # 交互：输入（拖入）图片路径，回车即处理；q 退出
  python scripts/live.py --watch inbox                     # 盯一个文件夹：放进新图片就处理
  python scripts/live.py --camera 0 --interval 1 --show    # 摄像头：每秒决策一次，决策变了才下发；预览窗口按 q 退出

通用参数：--distance 60  --mode scene_priority  --fake（发假板端）  --no-send（只推理）  --off-on-exit（退出时关灯）
摄像头防闪：新决策连续出现 --stable 次（默认 2）才切换；色相差 <10°、饱和度差 <5%、亮度差 <5% 视为没变。
台架上限约 13 帧/秒，--interval 不要小于 0.1。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fill_light import FillLightPipeline, imread_rgb  # noqa: E402
from fill_light import cli  # noqa: E402
from fill_light.images import bgr_to_rgb, list_images  # noqa: E402
from fill_light.result import describe  # noqa: E402

Frames = Iterator[tuple[np.ndarray, str]]      # (RGB 图, 来源名)


# ---------------- 三种取图方式 ----------------
def camera_frames(index: int, interval: float, show: bool) -> Frames:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"打不开摄像头 {index}（换个编号试试：--camera 1）")
    print(f"摄像头 {index} 已打开，每 {interval}s 决策一次；{'预览窗口按 q 退出' if show else 'Ctrl+C 退出'}")
    next_t = 0.0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("读帧失败，重试…")
                time.sleep(0.5)
                continue
            if show:
                cv2.imshow("smart_fill_light (q 退出)", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
            if time.monotonic() >= next_t:
                next_t = time.monotonic() + interval
                yield bgr_to_rgb(bgr), "camera"
            elif not show:
                time.sleep(0.02)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def watched_frames(folder: str) -> Frames:
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    seen = {p: p.stat().st_mtime for p in list_images(d)}
    print(f"盯文件夹 {d.resolve()}：放进新图片就处理（已有 {len(seen)} 张忽略）；Ctrl+C 退出")
    while True:
        for p in list_images(d):
            if seen.get(p) == p.stat().st_mtime:
                continue
            time.sleep(0.3)                               # 等文件写完
            seen[p] = p.stat().st_mtime
            try:
                rgb = imread_rgb(p)
            except Exception as e:  # noqa: BLE001
                print(f"\n{p.name} 读取失败：{e}")
                continue
            yield rgb, p.name
        time.sleep(0.5)


def typed_frames() -> Frames:
    print("输入（或把图片拖进来）图片路径，回车处理；q 退出")
    while True:
        try:
            s = input("\n图片> ").strip().strip('"').strip("'")
        except EOFError:
            return
        if s.lower() in ("q", "quit", "exit"):
            return
        if not s:
            continue
        p = Path(s)
        if not p.exists():
            print(f"找不到：{p}")
            continue
        try:
            rgb = imread_rgb(p)
        except Exception as e:  # noqa: BLE001
            print(f"读取失败：{e}")
            continue
        yield rgb, p.name


# ---------------- 决策 → 下发 ----------------
def fingerprint(lp: dict) -> tuple:
    """决策指纹：灯型、颜色（色相 10°、饱和度 5% 一档）、亮度（5% 一档）、侧光、顶光都一样，就算同一个决策。"""
    def hs(c):
        return None if not c else (round(c["hue"] / 10) % 36, round(c["sat"] / 5))
    return (lp["light_type"], hs(lp["main"]), hs(lp["contour"]), round(lp["brightness_pct"] / 5),
            tuple(sorted(lp["side_boost_pct"].items())), lp["top_gradient_pct"], hs(lp["eye_patch"]))


class Runner:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.pipe = FillLightPipeline(a.cfg)
        self.link = None if a.no_send else cli.open_link(a)
        self.sent_key = None                              # 灯板上当前的决策
        self.pending_key, self.pending_n = None, 0        # 候选的新决策，以及它连续出现的次数

    def handle(self, rgb: np.ndarray, tag: str, debounce: bool) -> None:
        t0 = time.perf_counter()
        res = self.pipe.run(rgb, **cli.run_kwargs(self.a))
        ms = (time.perf_counter() - t0) * 1000
        key = fingerprint(res.light_params)
        stamp = time.strftime("%H:%M:%S")
        if debounce:
            if key == self.sent_key:
                self.pending_key, self.pending_n = None, 0
                print(f"[{stamp}] {tag} | 决策未变，不重发 | {ms:.0f}ms", end="\r")
                return
            if key != self.pending_key:
                self.pending_key, self.pending_n = key, 0
            self.pending_n += 1
            if self.pending_n < self.a.stable:
                return
        print(f"\n[{stamp}] {tag} | {describe(res.to_dict(with_leds=False))} | 推理 {ms:.0f}ms")
        if self.link is None:
            self.sent_key = key
            return
        try:
            sent = self.link.send_result(res)
            self.sent_key = key
            print(f"           → 已下发：{sent.describe()}")
        except Exception as e:  # noqa: BLE001  —— 单次失败不退出，下一帧再试
            print(f"           → 下发失败：{e}")

    def close(self) -> None:
        if self.link is not None and self.a.off_on_exit:
            try:
                self.link.off()
                print("\n已关灯")
            except Exception:  # noqa: BLE001
                pass
        self.pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="连续运行：取图 → 决策 → 下发灯板")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--camera", type=int, help="摄像头编号（0 = 默认摄像头）")
    g.add_argument("--watch", help="盯一个文件夹，处理新放进来的图片")
    ap.add_argument("--interval", type=float, default=1.0, help="摄像头：决策间隔（秒）")
    ap.add_argument("--stable", type=int, default=2, help="摄像头：新决策连续出现几次才切换")
    ap.add_argument("--show", action="store_true", help="摄像头：显示预览窗口")
    cli.add_run_args(ap)
    cli.add_board_args(ap)
    ap.add_argument("--no-send", action="store_true", help="只推理，不下发")
    ap.add_argument("--off-on-exit", action="store_true", help="退出时关灯")
    a = ap.parse_args()
    cli.utf8_stdout(line_buffering=True)

    runner = Runner(a)
    if a.camera is not None:
        frames, debounce = camera_frames(a.camera, a.interval, a.show), True
    elif a.watch:
        frames, debounce = watched_frames(a.watch), False
    else:
        frames, debounce = typed_frames(), False
    try:
        for rgb, tag in frames:
            try:
                runner.handle(rgb, tag, debounce)
            except Exception as e:  # noqa: BLE001  —— 单帧出错不退出
                print(f"\n{tag} 处理失败：{e}")
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        frames.close()
        runner.close()


if __name__ == "__main__":
    main()
