"""灯板帧：每颗灯珠的 RGBW 值。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LedFrame:
    arrays: dict[str, np.ndarray]      # 板名 → rows × cols × 4（RGBW，0-255 整数）
    layout: dict                       # 行列数、列方向、内侧列数、通道顺序
    current_clamp_scale: float         # 电流钳位系数（1.0 = 没有被缩小）
    duty: float                        # 全部通道的平均占空比（0-1）

    @property
    def leds(self) -> dict[str, list]:
        """leds[板名][行][列] = [R, G, B, W]。"""
        return {k: v.tolist() for k, v in self.arrays.items()}

    def to_dict(self, with_leds: bool = True) -> dict:
        d = {"format": "per_led_rgbw", "layout": self.layout,
             "current_clamp_scale": self.current_clamp_scale, "duty": self.duty}
        if with_leds:
            d["leds"] = self.leds
        return d
