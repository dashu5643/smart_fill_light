"""算法决策 → 下发到灯板（经 ss2-light-bench 台架通道）。

用法（在项目根目录）：
  # 0) 先确认连通 & 方位
  .venv/Scripts/python scripts/to_bench.py --caps
  .venv/Scripts/python scripts/to_bench.py --identify          # 左红 / 右蓝 / 内侧一列白 / 顶行绿

  # 1) 一张图：推理 → 下发 → 存验收记录
  .venv/Scripts/python scripts/to_bench.py --image 取景.jpg --distance 60

  # 2) 已有推理结果 JSON（infer.py 输出，需带 frame.leds）直接下发
  .venv/Scripts/python scripts/to_bench.py --result outputs/test/xxx.json

  # 3) 一个目录逐张轮播，每张停 3 秒（看对比）
  .venv/Scripts/python scripts/to_bench.py --dir testset/t1 --hold 3

  # 关灯
  .venv/Scripts/python scripts/to_bench.py --off

不接板子：另开一个窗口在 ss2-light-bench 目录跑 `py ss2light.py fake`，这边加 `--fake`
（= --host 127.0.0.1 --port 8080），每收一帧会画到 ss2-light-bench 目录下的 fake-frame.ppm。
只想看下发内容不发：加 `--dry-run`（离线读 lightboard-v1.uci）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fill_light.bench import BenchLink, identify_leds, save_record  # noqa: E402
from fill_light.pipeline import load_cfg  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def describe(res: dict) -> str:
    lp, env = res["light_params"], res["env_light"]
    color = lp["t1_format"] or (lp["main"]["name"] + lp["main"]["hex"]
                                + (("/" + lp["contour"]["name"]) if lp["light_type"] == "sandwich" else ""))
    return (f"{env['env_light_category']} {env['env_light_name']} {env['pollution_level']} | "
            f"{lp['light_type']} {color} 亮{lp['brightness_pct']:.0f}%")


def report_sent(sent: dict) -> str:
    s = (f"估算电流 {sent['current_a_algo']}A" +
         (f" → 超台架上限 {sent['limit_a']}A，整体×{sent['bench_scale']} = {sent['current_a_sent']}A"
          if sent["bench_scale"] < 1 else ""))
    if sent.get("sent"):
        s += f" | 板子回复 {sent['reply']} | {sent['rtt_ms']}ms"
    else:
        s += " | 未发送（dry-run）"
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="算法决策下发到灯板（ss2-light-bench）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="取景图片：推理后下发")
    g.add_argument("--result", help="infer.py 输出的 JSON（含 frame.leds）")
    g.add_argument("--dir", help="目录逐张推理下发")
    g.add_argument("--identify", action="store_true", help="下发方位测试帧")
    g.add_argument("--caps", action="store_true", help="读灯板配置")
    g.add_argument("--off", action="store_true", help="关灯")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--fake", action="store_true", help="连本机假板端（127.0.0.1:8080）")
    ap.add_argument("--dry-run", action="store_true", help="只生成下发内容不发")
    ap.add_argument("--mode", default="skin_priority", choices=["skin_priority", "scene_priority"])
    ap.add_argument("--object", default=None)
    ap.add_argument("--distance", type=float, default=None, help="灯板到人脸距离 cm")
    ap.add_argument("--transition-ms", type=int, default=None)
    ap.add_argument("--hold", type=float, default=3.0, help="--dir 时每张停留秒数")
    ap.add_argument("--no-record", action="store_true", help="不存验收记录")
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_cfg(a.cfg)
    host, port = (("127.0.0.1", 8080) if a.fake else (a.host, a.port))
    link = BenchLink(cfg, host, port, offline=a.dry_run)
    rec_dir = ROOT / cfg.get("bench", {}).get("record_dir", "outputs/bench/records")

    try:
        if a.caps:
            print(json.dumps(link.caps(), ensure_ascii=False, indent=1))
            return 0
        if a.off:
            print(link.off())
            return 0
        if a.identify:
            sent = link.send_leds(identify_leds(), transition_ms=0)
            print("方位测试帧：" + report_sent(sent))
            print("预期（站在灯前面向灯板）：左板红、右板蓝；靠屏幕的一列白；最上一行绿。\n"
                  "不一致时改 configs/strategy.yaml → bench.swap_boards / mirror_cols / flip_rows")
            return 0

        if a.result:
            res = json.loads(Path(a.result).read_text(encoding="utf-8"))
            sent = link.send_result(res, a.transition_ms)
            print(f"{Path(a.result).name} | {describe(res)}\n  {report_sent(sent)}")
            if not a.no_record:
                print(f"  记录：{save_record(rec_dir, Path(a.result).stem, None, res, sent)}")
            return 0

        from fill_light.pipeline import FillLightPipeline, imread_rgb
        imgs = [Path(a.image)] if a.image else sorted(p for p in Path(a.dir).iterdir() if p.suffix.lower() in EXTS)
        pipe = FillLightPipeline(a.cfg)
        try:
            for i, p in enumerate(imgs):
                res = pipe.run(imread_rgb(p), mode=a.mode, object_type=a.object, distance_cm=a.distance)
                res.pop("_internal", None)
                sent = link.send_result(res, a.transition_ms)
                print(f"{p.name} | {describe(res)}\n  {report_sent(sent)}")
                if not a.no_record:
                    print(f"  记录：{save_record(rec_dir, p.stem, p, res, sent)}")
                if a.dir and i < len(imgs) - 1:
                    time.sleep(a.hold)
        finally:
            pipe.close()
    except link.ss.ModelError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"连不上板子 {link.light.base}：{e}\n"
              f"——插好 USB-C、等电脑拿到 192.168.7.x 地址再试；不接板子可用 --fake（先起假板端）或 --dry-run",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
