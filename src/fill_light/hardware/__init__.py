"""④ 执行层：策略 → SureShot 双灯板逐珠帧（2 × 18 行 × 6 列 × RGBW，0-255）。"""
from .frame import LedFrame
from .layout import BoardLayout
from .renderer import render_frame

__all__ = ["render_frame", "LedFrame", "BoardLayout"]
