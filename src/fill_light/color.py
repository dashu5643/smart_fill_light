"""色彩工具：sRGB / 线性 RGB / XYZ / Lab / HSV 互转，色温估计，色相区间工具。"""
from __future__ import annotations

import colorsys
import math

import numpy as np

_M_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def linear_to_lab(lin: np.ndarray) -> np.ndarray:
    """lin: (..., 3) 线性 RGB 0-1 → Lab"""
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    eps = 216 / 24389
    k = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (k * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_linear(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    eps = 216 / 24389
    k = 24389 / 27

    def finv(t):
        t3 = t ** 3
        return np.where(t3 > eps, t3, (116 * t - 16) / k)

    xyz = np.stack([finv(fx), np.where(L > k * eps, fy ** 3, L / k), finv(fz)], axis=-1) * _WHITE
    return np.clip(xyz @ _M_XYZ2RGB.T, 0.0, None)


def rgb8_to_lab(rgb: np.ndarray) -> np.ndarray:
    return linear_to_lab(srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255.0))


def ita_deg(lab: np.ndarray) -> float:
    L, _, b = lab
    return float(math.degrees(math.atan2(L - 50.0, b if abs(b) > 1e-6 else 1e-6)))


def lab_hue_deg(lab: np.ndarray) -> float:
    return float(math.degrees(math.atan2(lab[2], lab[1])) % 360)


def rgb_hsv(rgb01) -> tuple[float, float, float]:
    """sRGB 0-1 → (hue 0-360, sat 0-1, val 0-1)"""
    r, g, b = [float(np.clip(x, 0, 1)) for x in rgb01]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s, v


def hsv_rgb(hue: float, sat: float, val: float) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb((hue % 360) / 360.0, float(np.clip(sat, 0, 1)), float(np.clip(val, 0, 1)))
    return np.array([r, g, b])


def hsv_hex(hue: float, sat: float, val: float = 1.0) -> str:
    r, g, b = (hsv_rgb(hue, sat, val) * 255 + 0.5).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def cct_from_linear_rgb(lin_rgb) -> float | None:
    """McCamy 近似；仅对接近黑体轨迹的光源有意义。"""
    xyz = _M_RGB2XYZ @ np.asarray(lin_rgb, dtype=np.float64)
    s = xyz.sum()
    if s <= 1e-9:
        return None
    x, y = xyz[0] / s, xyz[1] / s
    n = (x - 0.3320) / (0.1858 - y)
    cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
    if not (1000 <= cct <= 25000):
        return None
    return float(cct)


def hue_in(hue: float, ranges) -> bool:
    h = hue % 360
    for lo, hi in ranges:
        if lo <= h < hi:
            return True
    return False


def hue_dist(a: float, b: float) -> float:
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def classify_hue(hue: float, classes: dict) -> str:
    for name, ranges in classes.items():
        if hue_in(hue, ranges):
            return name
    return "unknown"
