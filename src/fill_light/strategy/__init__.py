"""③ 决策层：语义特征 → 补光策略（与硬件无关）。规则和色值都来自 configs/strategy.yaml。"""
from .engine import MODES, decide
from .types import LightColor, Strategy

__all__ = ["decide", "MODES", "Strategy", "LightColor"]
