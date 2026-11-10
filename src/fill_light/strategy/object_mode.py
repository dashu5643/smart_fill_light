"""无人脸 → 物品模式：指定类别 / 暗光候选 / 色光互补 / 模仿环境光。"""
from __future__ import annotations

from ..analysis.types import Analysis
from ..labels import FAMILY_ZH
from .complement import complement_colors
from .palette import cap, named
from .types import LightColor, Strategy

OBJECT_BRIGHTNESS = 0.7
DARK_MEAN_L = 22                 # 背景平均亮度低于此值按暗光处理


def object_strategy(cfg: dict, an: Analysis, object_type: str | None) -> Strategy:
    oc = cfg["object_mode"]
    cast = an.cast
    reasons = ["未检测到人脸 → 物品拍摄模式"]

    if object_type and object_type in oc:                       # 1) 调用方指定了物品类别
        rule = oc[object_type]
        if rule["type"] == "sandwich":
            inner, outer, light_type = named(cfg, rule["inner"]), named(cfg, rule["outer"]), "sandwich"
        else:
            inner = outer = named(cfg, rule["color"])
            light_type = "mono"
        reasons.append(f"用户指定物品类别 {object_type}")
        return Strategy(mode="object", scenario=f"物品 · {object_type}", light_type=light_type, inner=inner,
                        outer=outer, brightness=OBJECT_BRIGHTNESS, reasons=reasons)

    if an.exposure == "dark" or an.env.mean_L < DARK_MEAN_L:    # 2) 太暗，认不出环境色
        a, b, pick = (named(cfg, n) for n in oc["dark_fallback"])
        reasons.append("暗光环境无法识别环境色，给出 浅蓝/青/中性偏黄 三个候选")
        return Strategy(mode="object", scenario="物品 · 暗光", light_type="mono", inner=pick, outer=pick,
                        brightness=OBJECT_BRIGHTNESS, alternatives=[a, b], reasons=reasons)

    if cast.level != "none" and cast.family in oc["complement_families"]:   # 3) 实测过的色光 → 互补
        outer, inner, alts, factors = complement_colors(cfg, an, "medium", None, reasons)
        reasons.insert(1, f"背景{FAMILY_ZH[cast.family]}污染 S={cast.s_pct}% → 背景主色互补色")
        return Strategy(mode="object", scenario=f"物品 · {FAMILY_ZH[cast.family]}互补矫正", light_type="dual",
                        inner=inner, outer=outer, brightness=1.0, reasons=reasons, alternatives=alts,
                        sat_factors=factors)

    hue = cast.bg_hue if cast.bg_hue is not None else 40                    # 4) 其余：模仿环境光
    sat = min(max((cast.bg_s or 0) / 100 * 0.6, 0.08), 0.3)
    pick = cap(cfg, LightColor("模仿环境色", hue, sat))
    reasons.append(f"模仿环境光色彩打光（色相 {hue:.0f}°，降饱和）；食物/静物类别可通过参数指定")
    return Strategy(mode="object", scenario="物品 · 模仿环境光", light_type="mono", inner=pick, outer=pick,
                    brightness=OBJECT_BRIGHTNESS, reasons=reasons,
                    alternatives=[named(cfg, "暖橙"), named(cfg, "中性偏黄")])
