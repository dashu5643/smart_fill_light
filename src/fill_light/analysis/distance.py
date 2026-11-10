"""灯到人脸的距离（决定补光饱和度的衰减）。

优先用调用方给的真实灯距；按脸宽估出来的是镜头距离，T1 实拍中与灯距不同步，默认不启用。
"""
from __future__ import annotations

import math

import numpy as np


def estimate_distance_cm(width_ratio: float, cfg: dict) -> float:
    """针孔模型：距离 = 真实脸宽 / (2·tan(视场角/2)·脸宽占比)。"""
    dc = cfg["distance"]
    half = math.tan(math.radians(float(dc["hfov_deg"]) / 2))
    d = float(dc["face_width_cm"]) / (2 * half * max(width_ratio, 1e-3))
    return float(np.clip(d, 15.0, 300.0))


def resolve_distance_cm(width_ratio: float, cfg: dict, given_cm: float | None = None) -> float:
    if given_cm is not None:
        return float(given_cm)
    dc = cfg["distance"]
    if dc.get("estimate_from_face", True) is False:
        return float(dc["default_cm"])
    return round(estimate_distance_cm(width_ratio, cfg))
