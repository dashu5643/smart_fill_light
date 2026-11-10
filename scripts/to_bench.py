"""灯板工具：连通检查、方位校正、把算法结果下发到灯板（经 ss2-light-bench 台架）。

  python scripts/to_bench.py --caps                        # 能不能连上板子
  python scripts/to_bench.py --identify                    # 方位测试帧：左红 / 右蓝 / 靠屏幕一列白 / 顶行绿
  python scripts/to_bench.py --image 取景.jpg --distance 60 # 推理 → 下发 → 存验收记录
  python scripts/to_bench.py --result outputs/xxx.json     # 下发已有的结果 JSON（需带 frame.leds）
  python scripts/to_bench.py --dir testset/t1 --hold 3     # 逐张轮播，每张停 3 秒
  python scripts/to_bench.py --off                         # 关灯

不接板子：另开一个窗口在 ss2-light-bench 目录运行 `py ss2light.py fake`，这边加 --fake。
只想看下发内容、不发送：加 --dry-run。连接参数和方位开关在 configs/bench.yaml。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fill_light import cli  # noqa: E402
from fill_light.bench import IDENTIFY_EXPECTED, identify_leds, save_record  # noqa: E402
from fill_light.images import list_images  # noqa: E402
from fill_light.result import describe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="灯板工具：连通检查、方位校正、下发算法结果")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="取景图片：推理后下发")
    g.add_argument("--dir", help="目录里的图片逐张推理下发")
    g.add_argument("--result", help="infer.py 输出的 JSON（含 frame.leds）")
    g.add_argument("--identify", action="store_true", help="下发方位测试帧")
    g.add_argument("--caps", action="store_true", help="读灯板配置")
    g.add_argument("--off", action="store_true", help="关灯")
    cli.add_run_args(ap)
    cli.add_board_args(ap)
    ap.add_argument("--dry-run", action="store_true", help="只生成下发内容，不发送")
    ap.add_argument("--transition-ms", type=int, default=None, help="渐变时长（默认取 configs/bench.yaml）")
    ap.add_argument("--hold", type=float, default=3.0, help="--dir 时每张停留秒数")
    ap.add_argument("--no-record", action="store_true", help="不存验收记录")
    a = ap.parse_args()
    cli.utf8_stdout()

    try:
        link = cli.open_link(a, offline=a.dry_run)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    def send(name: str, result, image: Path | None = None) -> None:
        """result：pipe.run() 的返回值，或读进来的结果 JSON。"""
        d = result if isinstance(result, dict) else result.to_dict()
        sent = link.send_result(result, a.transition_ms)
        print(f"{name} | {describe(d)}\n  {sent.describe()}")
        if not a.no_record:
            print(f"  记录：{save_record(link.record_dir, Path(name).stem, image, d, sent)}")

    try:
        if a.caps:
            print(json.dumps(link.caps(), ensure_ascii=False, indent=1))
        elif a.off:
            print(link.off())
        elif a.identify:
            print("方位测试帧：" + link.send_leds(identify_leds(), transition_ms=0).describe())
            print(f"预期（站在灯前、面向灯板）：{IDENTIFY_EXPECTED}。\n"
                  "不一致时改 configs/bench.yaml 的 swap_boards / mirror_cols / flip_rows")
        elif a.result:
            send(Path(a.result).name, json.loads(Path(a.result).read_text(encoding="utf-8")))
        else:
            from fill_light import FillLightPipeline, imread_rgb
            images = [Path(a.image)] if a.image else list_images(a.dir)
            with FillLightPipeline(a.cfg) as pipe:
                for i, p in enumerate(images):
                    send(p.name, pipe.run(imread_rgb(p), **cli.run_kwargs(a)), p)
                    if i < len(images) - 1:
                        time.sleep(a.hold)
    except (link.ss.ModelError, ValueError, FileNotFoundError) as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"连不上板子 {link.light.base}：{e}\n"
              "——插好 USB-C、等电脑拿到 192.168.7.x 地址再试；不接板子可用 --fake（先起假板端）或 --dry-run",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
