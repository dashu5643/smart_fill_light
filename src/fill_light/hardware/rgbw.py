"""颜色 → RGBW 通道值，以及整机电流钳位。"""
from __future__ import annotations

import numpy as np

from .. import color as C


def hsv_to_rgbw(hue: float, sat: float, val: float) -> np.ndarray:
    """RGB 三色的公共部分交给 W 通道（显色更好、更省电）。返回 [R, G, B, W]，0-1。"""
    rgb = C.hsv_rgb(hue, sat, 1.0) * val
    w = float(rgb.min())
    return np.array([rgb[0] - w, rgb[1] - w, rgb[2] - w, w])


def clamp_total_duty(boards: dict[str, np.ndarray], max_total: float) -> float:
    """所有通道之和超过 max_total 时整体等比缩小（原地修改），颜色比例不变。返回缩放系数。"""
    total = sum(a.sum() for a in boards.values())
    if total <= max_total:
        return 1.0
    scale = max_total / total
    for k in boards:
        boards[k] *= scale
    return scale
