"""单帧推理：一张（或一批）图片 → 补光策略 JSON + 可视化报告。

用法：
  python scripts/infer.py --image samples/zone_image28.png
  python scripts/infer.py --image photo.jpg --mode scene_priority
  python scripts/infer.py --image food.jpg --object food_cn
  python scripts/infer.py --dir samples --out outputs/batch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fill_light.pipeline import FillLightPipeline, imread_rgb  # noqa: E402
from fill_light.visualize import render_report  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--dir")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--mode", default="skin_priority", choices=["skin_priority", "scene_priority"])
    ap.add_argument("--object", default=None, help="物品类别：food_cn/food_west/dessert/salad/beauty/jewelry/plant")
    ap.add_argument("--distance", type=float, default=None,
                    help="灯板到人脸距离 cm（决定补光饱和度衰减；不填则按脸宽估计）")
    ap.add_argument("--no-frame", action="store_true", help="JSON 中省略 216 颗逐珠数据")
    ap.add_argument("--send", action="store_true", help="推理后下发到灯板（ss2-light-bench 台架，单张图时用）")
    ap.add_argument("--fake", action="store_true", help="配合 --send：发到本机假板端 127.0.0.1:8080")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    imgs = []
    if args.image:
        imgs.append(Path(args.image))
    if args.dir:
        imgs += sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in EXTS)
    if not imgs:
        ap.error("需要 --image 或 --dir")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = FillLightPipeline()
    link = None
    if args.send:
        from fill_light.bench import BenchLink
        link = BenchLink(pipe.cfg, *(("127.0.0.1", 8080) if args.fake else (None, None)))
    summary = []
    try:
        for p in imgs:
            rgb = imread_rgb(p)
            t0 = time.perf_counter()
            res = pipe.run(rgb, mode=args.mode, object_type=args.object, distance_cm=args.distance)
            ms = (time.perf_counter() - t0) * 1000
            render_report(rgb, res, str(out / f"{p.stem}_report.jpg"))
            res.pop("_internal")
            res["latency_ms"] = round(ms, 1)
            if link is not None:
                try:
                    sent = link.send_result(res)
                    print(f"  → 已下发灯板：板子回复 {sent['reply']}，估算电流 {sent['current_a_sent']}A"
                          + (f"（台架限流 ×{sent['bench_scale']}）" if sent["bench_scale"] < 1 else ""))
                except Exception as e:  # noqa: BLE001
                    print(f"  → 下发失败：{e}（检查 USB-C 连线、电脑是否拿到 192.168.7.x 地址）")
            if args.no_frame:
                res["frame"].pop("leds")
            (out / f"{p.stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            st, lp = res["strategy"], res["light_params"]
            mf = res["perception"]["main_face"]
            env = res["perception"]["environment"]
            line = (f"{p.name:24s} | 脸{res['perception']['faces']} | "
                    f"{res['env_light']['env_light_category']} {env['cast_family_zh']} S{env['pollution_s_pct']:.0f}%"
                    f"({res['env_light']['pollution_level']}) | "
                    f"{((mf['skin_level'] + ' ' + mf['skin_group_zh']) if mf['skin_reliable'] else '肤色不可判') if mf else '-':14s} | "
                    f"{lp['light_type']:8s} "
                    f"{lp['t1_format'] or (lp['main']['name'] + lp['main']['hex'] + ('/' + lp['contour']['name'] if lp['light_type'] == 'sandwich' else ''))} "
                    f"亮{lp['brightness_pct']:.0f}% | {ms:.0f}ms")
            print(line)
            summary.append({"image": p.name, "scenario": st["scenario"], **lp,
                            "skin": mf and mf["skin_level"], "cast": env["cast_family"],
                            "s_pct": env["pollution_s_pct"]})
    finally:
        pipe.close()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
