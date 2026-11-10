"""单人 · 场景光优先：保留现场氛围。外侧轮廓光贴近场景色，面部主光降饱和并向肤色色相靠拢。"""
from __future__ import annotations

from ..analysis.types import Analysis, FaceAnalysis
from .palette import apply_taboo, cap
from .types import LightColor, Strategy

SCENE_MIN_COLORED_SHARE = 0.08   # 彩色像素占比超过此值才算"有场景主色调"


def scene_priority_strategy(cfg: dict, an: Analysis, fa: FaceAnalysis | None) -> Strategy:
    sc = cfg["scene_priority"]
    env = an.env
    reasons = []
    if env.dominant_hue is not None and env.colored_share > SCENE_MIN_COLORED_SHARE:
        scene_hue = env.dominant_hue
        scene_sat = min(0.9, 0.35 + env.colored_share)
        reasons.append(f"场景主色调 {scene_hue:.0f}°（彩色占比 {env.colored_share:.0%}）")
    else:
        scene_hue, scene_sat = an.cast.bg_hue or 40.0, max((an.cast.bg_s or 0) / 100, 0.15)
        reasons.append(f"场景无强彩色光源，跟随背景光源色相 {scene_hue:.0f}°")
    to_skin = ((sc["skin_hue"] - scene_hue + 180) % 360) - 180
    inner_hue = (scene_hue + to_skin * sc["pull_to_skin"]) % 360
    inner = cap(cfg, LightColor("场景色·降饱和", inner_hue, min(scene_sat * sc["inner_sat_ratio"], sc["inner_sat_cap"])))
    outer = cap(cfg, LightColor("场景色·轮廓", scene_hue, min(scene_sat * sc["outer_sat_ratio"], sc["outer_sat_cap"])))
    inner = apply_taboo(cfg, inner, fa, reasons)
    reasons.append("面部区降饱和并向肤色色相靠拢，外侧轮廓光贴近场景色")
    if env.dark_share > sc["shadow_contour_min_ratio"]:
        reasons.append(f"背景暗部占比 {env.dark_share:.0%}，外侧加同色调轮廓光")
    return Strategy(mode="scene_priority", scenario="场景光优先 · 氛围保留", light_type="sandwich",
                    inner=inner, outer=outer, brightness=0.0, reasons=reasons)
