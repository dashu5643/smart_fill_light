"""用 T1《色彩矫正测试》实拍照片评估环境光识别 / 污染等级 / 距离估计 / 输出策略。

testset/t1/labels.csv 由 色彩矫正测试.xlsx 单元格图片导出：
  ambient   = purple / pink / blue / green（环境光模拟色）
  condition = none（无补光）/ flash（闪光灯）/ fill_high / fill_low（互补色补光后）
  skin_group= medium（真人黄种人）/ deep（黑人模特头）/ fair（白人模特头）
  distance_cm = 60 / 30

用法：python scripts/eval_t1.py [--csv testset/t1/labels.csv] [--out outputs/eval_t1]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fill_light.pipeline import FillLightPipeline, imread_rgb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "testset/t1/labels.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs/eval_t1"))
    ap.add_argument("--report", action="store_true", help="同时输出每张图的可视化报告")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = FillLightPipeline()
    recs = []
    try:
        for r in rows:
            p = Path(args.csv).parent / r["file"]
            rgb = imread_rgb(p)
            res = pipe.run(rgb)
            if args.report:
                from fill_light.visualize import render_report
                render_report(rgb, res, str(out / f"{p.stem}_report.jpg"))
            res.pop("_internal")
            env = res["perception"]["environment"]
            mf = res["perception"]["main_face"]
            recs.append({
                **r,
                "pred_family": env["cast_family"], "s_pct": env["pollution_s_pct"],
                "level": env["pollution_level"], "hue": env["cast_hue"],
                "face_hue": env["face_estimate"]["hue"], "face_s": env["face_estimate"]["s_pct"],
                "bg_hue": env["background_estimate"]["hue"], "bg_s": env["background_estimate"]["s_pct"],
                "pred_skin": mf and (mf["skin_group"] if mf.get("skin_reliable", True) else "unknown"), "pred_level": mf and mf["skin_level"],
                "dist_est": mf and mf.get("distance_cm"),
                "multi": env.get("multi_color_mode"),
                "light": res["light_params"],
            })
    finally:
        pipe.close()
    (out / "eval.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    def show(title, sel):
        print(f"\n== {title}（{len(sel)} 张）")
        acc = sum(x["pred_family"] == x["ambient"] for x in sel)
        print(f"环境光色系正确率：{acc}/{len(sel)} = {acc / max(len(sel), 1):.0%}")
        conf = collections.Counter((x["ambient"], x["pred_family"]) for x in sel)
        for amb in ("purple", "pink", "blue", "green"):
            row = {k[1]: v for k, v in conf.items() if k[0] == amb}
            ss = [x["s_pct"] for x in sel if x["ambient"] == amb]
            lv = collections.Counter(x["level"] for x in sel if x["ambient"] == amb)
            hs = sorted(round(x["hue"]) for x in sel if x["ambient"] == amb)
            if ss:
                print(f"  {amb:6s} → {dict(row)}  S% 中位 {sorted(ss)[len(ss) // 2]:.0f}  等级 {dict(lv)}  色相 {hs[0]}~{hs[-1]}")
        sk = [x for x in sel if x["pred_skin"]]
        if sk:
            ok = sum(x["pred_skin"] == x["skin_group"] for x in sk)
            unk = sum(x["pred_skin"] == "unknown" for x in sk)
            wrong = len(sk) - ok - unk
            print(f"肤色组：正确 {ok} / 判为不可判 {unk} / 判错 {wrong}（共 {len(sk)}）  "
                  f"{dict(collections.Counter((x['skin_group'], x['pred_skin']) for x in sk))}")
        de = [x for x in sel if x["dist_est"]]
        if de:
            for d in ("60", "30"):
                v = sorted(x["dist_est"] for x in de if x["distance_cm"] == d)
                if v:
                    print(f"距离 {d}cm 估计：中位 {v[len(v) // 2]:.0f}cm  范围 {v[0]:.0f}~{v[-1]:.0f}")

    show("无补光（只有环境色光）", [x for x in recs if x["condition"] == "none"])
    show("闪光灯（环境色光 + 白光）", [x for x in recs if x["condition"] == "flash"])
    fills = [x for x in recs if x["condition"].startswith("fill")]
    print(f"\n== 互补色补光后（{len(fills)} 张）：残余污染等级分布 "
          f"{dict(collections.Counter(x['level'] for x in fills))}")
    # 无补光 vs 补光后的残余 S%（同肤色/设备/距离/环境配对）
    base = {(x["ambient"], x["skin_group"], x["device"], x["distance_cm"]): x["s_pct"]
            for x in recs if x["condition"] == "none"}
    for cond in ("fill_low", "fill_high"):
        d = [base[k] - x["s_pct"] for x in recs if x["condition"] == cond
             for k in [(x["ambient"], x["skin_group"], x["device"], x["distance_cm"])] if k in base]
        if d:
            print(f"  {cond}: 污染 S% 平均下降 {sum(d) / len(d):.1f}（{len(d)} 对）")


if __name__ == "__main__":
    main()
