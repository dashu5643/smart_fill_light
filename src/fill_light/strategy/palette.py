"""色板：按名字取色、饱和度上限、肤色禁忌色替换。"""
from __future__ import annotations

from .. import color as C
from ..analysis.types import FaceAnalysis
from .types import LightColor

_HUE_NAMES = [(15, "红"), (35, "橙"), (55, "黄"), (90, "黄青"), (135, "绿"), (170, "青绿"), (200, "青"),
              (250, "蓝"), (285, "紫"), (320, "紫粉"), (345, "粉"), (361, "红")]


def hue_name(h: float) -> str:
    h = h % 360
    return next(n for t, n in _HUE_NAMES if h < t)


def clip(c: LightColor) -> LightColor:
    c.sat = round(min(max(c.sat, 0.0), 1.0), 3)
    return c


def cap(cfg: dict, c: LightColor) -> LightColor:
    """无污染肤色光的色系饱和度上限（如青绿过浓脸发脏）。"""
    for rule in cfg.get("hue_family_sat_cap", []):
        if C.hue_in(c.hue, [rule["range"]]):
            c.sat = min(c.sat, rule["cap"])
    return clip(c)


def named(cfg: dict, name: str, sat_tag: str | None = None) -> LightColor:
    """configs 的 palette 里按名字取色；sat_tag = low / high 时改用预设饱和度。"""
    p = cfg["palette"][name]
    sat = cfg["saturation_presets"][sat_tag] if sat_tag else p["sat"]
    return cap(cfg, LightColor(name, float(p["hue"]), float(sat)))


def apply_taboo(cfg: dict, c: LightColor, fa: FaceAnalysis | None, reasons: list[str]) -> LightColor:
    """深肤色禁绿/青/纯白，橄榄调禁青/蓝：命中则换成替代色。肤色不可靠时不处理。"""
    if fa is None or fa.skin is None or not fa.skin.reliable:
        return c
    t = cfg["taboo"]
    if fa.skin.group == "deep":
        rule = t["deep"]
        if C.hue_in(c.hue, rule["forbid_hues"]) or (rule.get("forbid_white") and c.sat < 0.06):
            reasons.append(f"深肤色禁绿/青/纯白：{c.name} → {rule['replace']}")
            return named(cfg, rule["replace"])
    if fa.skin.undertone == "olive":
        rule = t["olive"]
        if C.hue_in(c.hue, rule["forbid_hues"]):
            reasons.append(f"橄榄调禁青/蓝：{c.name} → {rule['replace']}")
            return named(cfg, rule["replace"])
    return c
