"""暗光环境：画面太暗、认不出环境光颜色 → 只提亮（暖白满亮）。"""
from __future__ import annotations

from ..analysis.types import Analysis
from .palette import named
from .types import Strategy


def dark_scene_strategy(cfg: dict, an: Analysis) -> Strategy:
    pick = named(cfg, "暖白")
    return Strategy(mode="dark", scenario="暗光环境 · 仅提亮", light_type="mono", inner=pick, outer=pick,
                    brightness=float(cfg["brightness"]["dark_scene"]),
                    reasons=[f"画面明度 V={an.env.frame_V:.0%}，环境过暗无法判断环境光色彩 → 暖白满亮提亮"],
                    warnings=["环境过暗，建议靠近光源或开启补光后重新识别"],
                    alternatives=[named(cfg, "中性偏黄"), named(cfg, "淡粉")])
