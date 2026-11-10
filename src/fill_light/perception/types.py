"""感知层输出：只有观测值，不含任何判断。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceObs:
    box: tuple[int, int, int, int]                 # x, y, w, h
    center_x_ratio: float
    width_ratio: float                              # 脸宽 / 图宽
    landmarks: np.ndarray                           # 478 × (x, y, z)，像素坐标
    regions: dict[str, dict] = field(default_factory=dict)   # 区域名 → {lab, lin, n}
    skin_lab: np.ndarray | None = None              # 整脸皮肤 Lab 中位数（原始，未矫正）
    skin_lin: np.ndarray | None = None
    yaw_proxy: float = 0.0                          # 偏转近似：鼻尖相对两颊外缘的位置


@dataclass
class EnvObs:
    illuminant_lin: np.ndarray                      # 背景估计光源（线性 RGB，max=1）
    illum_hue: float
    illum_sat: float
    cct: float | None
    mean_L: float
    p90_L: float
    colored_share: float                            # 高饱和像素占比
    dominant_hue: float | None
    hue_peaks: list[tuple[float, float]]            # [(hue, 权重占比)]
    dark_share: float                               # 大面积暗部占比
    frame_V: float = 0.5                            # 全画面平均明度 V（0-1）
    frame_S: float = 0.0                            # 全画面平均饱和度 S（0-1）


@dataclass
class Observation:
    image_size: tuple[int, int]
    faces: list[FaceObs]                            # 按脸宽从大到小
    env: EnvObs
    roi_radius_ratio: float = 0.07                  # 采样圆半径 / 脸宽（报告图画采样圈用）
