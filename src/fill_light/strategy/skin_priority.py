"""单人 · 肤色优先：有色光污染 → T1 互补色（内外圈）；无污染 → 肤色 × 环境基调的常规肤色光。"""
from __future__ import annotations

from .. import color as C
from ..analysis.types import Analysis, FaceAnalysis
from ..labels import FAMILY_ZH, GROUP_ZH, LEVEL_ZH, TONE_ZH
from .complement import complement_colors
from .palette import apply_taboo, cap, named
from .types import LightColor, Strategy


def skin_priority_strategy(cfg: dict, an: Analysis, fa: FaceAnalysis) -> Strategy:
    cast, skin = an.cast, fa.skin
    reasons = [f"环境光：{FAMILY_ZH.get(cast.family)}（色相 {cast.hue:.0f}°），污染 S={cast.s_pct}%（{LEVEL_ZH[cast.level]}）"]
    warnings: list[str] = []
    if skin.reliable:
        group = skin.group
        reasons.insert(0, f"肤色 {skin.level}（ITA {skin.ita}°，{GROUP_ZH[skin.group]}，底调 {skin.undertone}）")
    else:
        group = cfg["skin"]["unknown_fallback_group"]
        reasons.insert(0, f"肤色不可判（{skin.unreliable_reason}）→ 按{GROUP_ZH[group]}参数")
        warnings.append(f"强色光下肤色无法可靠识别，已按{GROUP_ZH[group]}参数补光；"
                        f"如为白人/深肤色，可在备选中切换")
    if cast.level != "none":
        return _polluted(cfg, an, fa, group, reasons, warnings)
    return _clean(cfg, an, fa, group, reasons, warnings)


def _polluted(cfg, an: Analysis, fa: FaceAnalysis, group: str, reasons, warnings) -> Strategy:
    cast = an.cast
    outer, inner, alts, factors = complement_colors(cfg, an, group, fa.distance_cm, reasons, sat_cap_group=group)
    if fa.skin.reliable and group == "deep" and C.hue_in(outer.hue, cfg["taboo"]["deep"]["forbid_hues"]):
        warnings.append("深肤色禁冷色，但实测表给出青绿互补；当前以实测为准（效果 3/5，待验证）")
    scenario = f"{FAMILY_ZH[cast.family]}污染 · 互补色矫正（内外圈）"
    if cast.multi_color:
        reasons.append(f"多色光源（霓虹）模式：主环境光 {FAMILY_ZH[cast.family]}"
                       + (f"，次环境光 {FAMILY_ZH.get(cast.secondary)}（记录，不单独补光）" if cast.secondary else ""))
        scenario = "多色霓虹 · " + scenario
    return Strategy(mode="skin_priority", scenario=scenario, light_type="dual", inner=inner, outer=outer,
                    brightness=0.0, reasons=reasons, warnings=warnings, alternatives=alts, sat_factors=factors)


def _clean(cfg, an: Analysis, fa: FaceAnalysis, group: str, reasons, warnings) -> Strategy:
    skin = fa.skin
    base = cfg["base_light"][group][an.env_tone]
    sat_tag = base.get("sat")
    if base["type"] == "sandwich":
        inner, outer, light_type = named(cfg, base["inner"], sat_tag), named(cfg, base["outer"], sat_tag), "sandwich"
    else:
        inner = outer = named(cfg, base["color"], sat_tag)
        light_type = "mono"
    override = cfg["undertone_override"].get(f"{group}_{skin.undertone}")
    if override and skin.reliable:
        inner = named(cfg, override)
        if light_type == "mono":
            outer = inner
        reasons.append(f"底调修正：{skin.undertone} → {override}")
    inner = apply_taboo(cfg, inner, fa, reasons)
    outer = apply_taboo(cfg, outer, fa, reasons) if light_type != "mono" else inner
    if an.exposure == "over":
        reasons.append("画面过曝 → 归 L0，不做彩色矫正")
    reasons.append(f"无明显光污染（L0），环境基调 {TONE_ZH[an.env_tone]} → "
                   f"{'夹心 ' + inner.name + '/' + outer.name if light_type == 'sandwich' else '单色 ' + inner.name}")
    low = inner.sat < 0.4                           # 当前偏淡 → 备选给高饱和版，反之给低饱和版
    presets = cfg["saturation_presets"]
    alt = cap(cfg, LightColor(inner.name + ("·高饱和" if low else "·低饱和"), inner.hue,
                              presets["high"] if low else presets["low"]))
    return Strategy(mode="skin_priority", scenario=f"{TONE_ZH[an.env_tone]}环境 · 肤色光", light_type=light_type,
                    inner=inner, outer=outer, brightness=0.0, reasons=reasons, warnings=warnings,
                    alternatives=[alt])
