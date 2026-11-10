"""一张图片 → 打光策略 + 灯板参数。

  python scripts/infer.py --image photo.jpg                     # 肤色优先（默认）
  python scripts/infer.py --image photo.jpg --distance 30       # 指定灯板到人脸的距离（cm）
  python scripts/infer.py --image photo.jpg --mode scene_priority
  python scripts/infer.py --image food.jpg --object food_cn     # 无人脸：指定物品类别
  python scripts/infer.py --dir samples --out outputs/batch     # 一批图
  python scripts/infer.py --image photo.jpg --send              # 算完顺便下发到灯板

每张图输出 <名>.json（策略 + 灯板参数）和 <名>_report.jpg（报告图）到 --out 目录。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fill_light import FillLightPipeline, imread_rgb  # noqa: E402
from fill_light import cli  # noqa: E402
from fill_light.images import list_images  # noqa: E402
from fill_light.report import render_report  # noqa: E402
from fill_light.result import describe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="一张图片 → 打光策略 + 灯板参数")
    ap.add_argument("--image", help="一张图片")
    ap.add_argument("--dir", help="一个目录里的全部图片")
    ap.add_argument("--out", default="outputs", help="输出目录（默认 outputs）")
    cli.add_run_args(ap)
    ap.add_argument("--no-frame", action="store_true", help="JSON 里不写 216 颗灯珠的逐珠数据")
    ap.add_argument("--no-report", action="store_true", help="不出报告图")
    ap.add_argument("--send", action="store_true", help="算完下发到灯板（连接参数见 configs/bench.yaml）")
    cli.add_board_args(ap)
    a = ap.parse_args()
    cli.utf8_stdout()

    images = ([Path(a.image)] if a.image else []) + (list_images(a.dir) if a.dir else [])
    if not images:
        ap.error("需要 --image 或 --dir")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    link = cli.open_link(a) if a.send else None
    summary = []
    with FillLightPipeline(a.cfg) as pipe:
        for p in images:
            rgb = imread_rgb(p)
            t0 = time.perf_counter()
            res = pipe.run(rgb, **cli.run_kwargs(a))
            ms = (time.perf_counter() - t0) * 1000

            d = res.to_dict(with_leds=not a.no_frame)
            d["latency_ms"] = round(ms, 1)
            (out / f"{p.stem}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
            if not a.no_report:
                render_report(rgb, res, out / f"{p.stem}_report.jpg")
            print(f"{p.name} | {describe(d)} | {ms:.0f}ms")

            if link is not None:
                try:
                    print(f"  → 已下发灯板：{link.send_result(res).describe()}")
                except Exception as e:  # noqa: BLE001
                    print(f"  → 下发失败：{e}（检查 USB-C 连线、电脑是否拿到 192.168.7.x 地址）")

            env, mf = d["perception"]["environment"], d["perception"]["main_face"]
            summary.append({"image": p.name, "scenario": d["strategy"]["scenario"], **d["light_params"],
                            "skin": mf and mf["skin_level"], "cast": env["cast_family"],
                            "s_pct": env["pollution_s_pct"]})

    if a.dir:
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"输出目录：{out.resolve()}")


if __name__ == "__main__":
    main()
