"""在 T1《色彩矫正测试》实拍集上评估：环境光色系、污染等级、肤色组、补光后的残余污染。

  python scripts/eval_t1.py                # 打印统计，明细写到 outputs/eval_t1/eval.json
  python scripts/eval_t1.py --report       # 同时给每张图出报告图

标注文件 testset/t1/labels.csv（由 色彩矫正测试.xlsx 导出，数据不入库）：
  ambient     环境光颜色：purple / pink / blue / green
  condition   none 无补光 / flash 闪光灯 / fill_high、fill_low 互补色补光后
  skin_group  medium 真人黄种人 / deep 黑人模特头 / fair 白人模特头
  distance_cm 60 / 30
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fill_light import FillLightPipeline, imread_rgb  # noqa: E402
from fill_light import cli  # noqa: E402

AMBIENTS = ("purple", "pink", "blue", "green")


def predict(rows: list[dict], folder: Path, out: Path, with_report: bool) -> list[dict]:
    """逐张推理，返回 标注 + 预测 的记录。"""
    recs = []
    with FillLightPipeline() as pipe:
        for r in rows:
            p = folder / r["file"]
            rgb = imread_rgb(p)
            res = pipe.run(rgb)
            if with_report:
                from fill_light.report import render_report
                render_report(rgb, res, out / f"{p.stem}_report.jpg")
            env, mf = res.perception["environment"], res.perception["main_face"]
            recs.append({
                **r,
                "pred_family": env["cast_family"], "s_pct": env["pollution_s_pct"],
                "level": env["pollution_level"], "hue": env["cast_hue"],
                "face_hue": env["face_estimate"]["hue"], "face_s": env["face_estimate"]["s_pct"],
                "bg_hue": env["background_estimate"]["hue"], "bg_s": env["background_estimate"]["s_pct"],
                "pred_skin": mf and (mf["skin_group"] if mf["skin_reliable"] else "unknown"),
                "pred_level": mf and mf["skin_level"],
                "dist_est": mf and mf["distance_cm"],
                "multi": env["multi_color_mode"],
                "light": res.light_params,
            })
    return recs


def show(title: str, sel: list[dict]) -> None:
    print(f"\n== {title}（{len(sel)} 张）")
    acc = sum(x["pred_family"] == x["ambient"] for x in sel)
    print(f"环境光色系正确率：{acc}/{len(sel)} = {acc / max(len(sel), 1):.0%}")
    conf = Counter((x["ambient"], x["pred_family"]) for x in sel)
    for amb in AMBIENTS:
        group = [x for x in sel if x["ambient"] == amb]
        if not group:
            continue
        row = {k[1]: v for k, v in conf.items() if k[0] == amb}
        ss = sorted(x["s_pct"] for x in group)
        hs = sorted(round(x["hue"]) for x in group)
        print(f"  {amb:6s} → {row}  S% 中位 {ss[len(ss) // 2]:.0f}  "
              f"等级 {dict(Counter(x['level'] for x in group))}  色相 {hs[0]}~{hs[-1]}")
    sk = [x for x in sel if x["pred_skin"]]
    if sk:
        ok = sum(x["pred_skin"] == x["skin_group"] for x in sk)
        unk = sum(x["pred_skin"] == "unknown" for x in sk)
        print(f"肤色组：正确 {ok} / 判为不可判 {unk} / 判错 {len(sk) - ok - unk}（共 {len(sk)}）  "
              f"{dict(Counter((x['skin_group'], x['pred_skin']) for x in sk))}")
    for d in ("60", "30"):
        v = sorted(x["dist_est"] for x in sel if x["dist_est"] and x["distance_cm"] == d)
        if v:
            print(f"距离 {d}cm 估计：中位 {v[len(v) // 2]:.0f}cm  范围 {v[0]:.0f}~{v[-1]:.0f}")


def show_fill_effect(recs: list[dict]) -> None:
    """补光前后配对（同环境光 / 肤色 / 设备 / 距离），看污染 S% 降了多少。"""
    fills = [x for x in recs if x["condition"].startswith("fill")]
    print(f"\n== 互补色补光后（{len(fills)} 张）：残余污染等级分布 {dict(Counter(x['level'] for x in fills))}")
    pair = lambda x: (x["ambient"], x["skin_group"], x["device"], x["distance_cm"])
    base = {pair(x): x["s_pct"] for x in recs if x["condition"] == "none"}
    for cond in ("fill_low", "fill_high"):
        d = [base[pair(x)] - x["s_pct"] for x in recs if x["condition"] == cond and pair(x) in base]
        if d:
            print(f"  {cond}: 污染 S% 平均下降 {sum(d) / len(d):.1f}（{len(d)} 对）")


def main() -> None:
    ap = argparse.ArgumentParser(description="在 T1 实拍集上评估")
    ap.add_argument("--csv", default=str(ROOT / "testset/t1/labels.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs/eval_t1"))
    ap.add_argument("--report", action="store_true", help="同时给每张图出报告图")
    a = ap.parse_args()
    cli.utf8_stdout()

    csv_path, out = Path(a.csv), Path(a.out)
    if not csv_path.exists():
        raise SystemExit(f"找不到标注文件 {csv_path}（T1 实拍数据不入库，需要单独拷贝到 testset/t1/）")
    out.mkdir(parents=True, exist_ok=True)
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    recs = predict(rows, csv_path.parent, out, a.report)
    (out / "eval.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    show("无补光（只有环境色光）", [x for x in recs if x["condition"] == "none"])
    show("闪光灯（环境色光 + 白光）", [x for x in recs if x["condition"] == "flash"])
    show_fill_effect(recs)


if __name__ == "__main__":
    main()
