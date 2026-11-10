"""互补色矫正：T1 内外圈参数表 × 污染系数 × 距离系数 × 多色系数。"""
from __future__ import annotations

import numpy as np

from ..analysis.types import Analysis
from .palette import clip, hue_name
from .types import LightColor


def pollution_scale(cfg: dict, s_pct: float) -> float:
    """污染越轻，补光饱和度越低。"""
    for thr, k in cfg["pollution_sat_scale"]:
        if s_pct >= thr:
            return float(k)
    return 1.0


def distance_scale(cfg: dict, dist_cm: float | None) -> float:
    """灯离脸越近，补光饱和度越低（表内各点之间线性插值，两端截断）。"""
    d = dist_cm if dist_cm is not None else cfg["distance"]["default_cm"]
    pts = cfg["distance"]["attenuation"]
    return float(np.interp(d, [p[0] for p in pts], [p[1] for p in pts]))


def complement_colors(cfg: dict, an: Analysis, group: str, dist_cm: float | None, reasons: list[str],
                      sat_cap_group: str | None = None) -> tuple[LightColor, LightColor, list[LightColor], dict]:
    """返回 (外圈, 内圈, 备选, 饱和度计算过程)。group = 查表用的肤色组。"""
    cast = an.cast
    e = cfg["complement_table"][cast.family][group]
    k_p = pollution_scale(cfg, cast.s_pct)
    k_d = distance_scale(cfg, dist_cm)
    k_m = cfg["cast"]["multi_color"]["sat_boost"] if cast.multi_color else 1.0
    k = k_p * k_d * k_m
    (oh, os_), (ih, is_) = e["outer"], e["inner"]
    outer = LightColor(f"外圈·{hue_name(oh)}", float(oh), os_ / 100 * k)
    inner = LightColor(f"内圈·{hue_name(ih)}", float(ih), is_ / 100 * k)

    caps = cfg["skin_sat_caps"]                     # 《环境光分类标准》§5.3，与 T1 实测冲突，默认不启用
    if caps.get("enforce") and sat_cap_group:
        cap = caps[sat_cap_group][cast.level] / 100
        if inner.sat > cap or outer.sat > cap:
            reasons.append(f"肤色×污染饱和度上限 {cap:.0%}（分类标准 §5.3）")
        outer.sat, inner.sat = min(outer.sat, cap), min(inner.sat, cap)
    outer, inner = clip(outer), clip(inner)

    factors = {"base_outer_S": os_, "base_inner_S": is_, "pollution_k": k_p, "distance_k": round(k_d, 3),
               "distance_cm": dist_cm, "multi_color_k": k_m, "total_k": round(k, 3)}
    reasons.append(f"T1 实测参数（60cm）【{oh}-{os_}***{ih}-{is_}】"
                   + (f"，实测效果 {e['score']}/5" if e.get("score") else "，理论值未实测"))
    reasons.append(f"饱和度 = 基准 × 污染系数 {k_p} × 距离系数 {k_d:.2f}（≈{dist_cm or cfg['distance']['default_cm']:.0f}cm）"
                   + (f" × 多色增强 {k_m}" if k_m != 1 else "")
                   + f" → 【{oh}-{outer.sat * 100:.0f}***{ih}-{inner.sat * 100:.0f}】")
    if e.get("note"):
        reasons.append(f"参数备注：{e['note']}")

    # 备选：当前是低饱和版就给高饱和版，反之亦然（绿/黄/青光没有高饱和版）
    alts = []
    v = cfg["complement_variants"]
    if os_ > v["low"][0]:
        alts.append(clip(LightColor(f"低饱和版 外圈【{oh}-{v['low'][0]}***{ih}-{v['low'][1]}】×{k:.2f}",
                                    float(oh), v["low"][0] / 100 * k)))
    elif cast.family not in v["no_high"]:
        alts.append(clip(LightColor(f"高饱和版 外圈【{oh}-{v['high'][0]}***{ih}-{v['high'][1]}】×{k:.2f}",
                                    float(oh), v["high"][0] / 100 * k)))
    return outer, inner, alts, factors
