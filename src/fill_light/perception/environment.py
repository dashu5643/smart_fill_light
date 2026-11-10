"""环境：背景光源估计、彩色光源色相峰、明暗统计。"""
from __future__ import annotations

import cv2
import numpy as np

from .. import color as C
from .types import EnvObs

SOG_P = 6.0                      # Shades-of-Gray 的范数
MAX_BG_SAMPLES = 60000           # 背景像素下采样到这个量级
COLORED_MIN_S, COLORED_MIN_V = 0.45, 0.25     # "彩色像素"的定义
PEAK_MIN_GAP = 40                # 两个色相峰至少相隔的度数


def observe_environment(rgb: np.ndarray, lin: np.ndarray, lab: np.ndarray, bg_mask: np.ndarray,
                        min_peak_share: float) -> EnvObs:
    if bg_mask.sum() < 500:                         # 背景太少（脸占满画面）→ 用全图
        bg_mask = np.ones(bg_mask.shape, bool)
    step = max(1, int(np.sqrt(bg_mask.sum() / MAX_BG_SAMPLES)))
    m = bg_mask[::step, ::step]
    px_lin = lin[::step, ::step][m]
    px_lab = lab[::step, ::step][m]
    px_rgb = rgb[::step, ::step][m].astype(np.float64) / 255.0

    # 1) 光源估计：Shades-of-Gray，排除过曝和过暗像素
    ok = (px_lin.max(1) < 0.97) & (px_lin.max(1) > 0.01)
    src = px_lin[ok] if ok.sum() > 200 else px_lin
    e = np.power(np.mean(np.power(src, SOG_P), 0), 1 / SOG_P)
    e = e / max(e.max(), 1e-9)
    hue, sat, _ = C.rgb_hsv(C.linear_to_srgb(e))
    cct = C.cct_from_linear_rgb(e) if sat < 0.35 else None

    # 2) 彩色光源（霓虹）：高饱和像素的色相直方图找峰
    hsv = cv2.cvtColor((px_rgb[None] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV_FULL)[0].astype(np.float64)
    hh = hsv[:, 0] / 255.0 * 360.0
    ss = hsv[:, 1] / 255.0
    vv = hsv[:, 2] / 255.0
    colored = (ss > COLORED_MIN_S) & (vv > COLORED_MIN_V)
    peaks: list[tuple[float, float]] = []
    if colored.sum() > 50:
        hist, edges = np.histogram(hh[colored], bins=36, range=(0, 360), weights=ss[colored] * vv[colored])
        hist = np.convolve(np.r_[hist[-1], hist, hist[0]], [0.25, 0.5, 0.25], "valid")
        total = hist.sum() + 1e-9
        for i in np.argsort(-hist):
            share = hist[i] / total
            if share < min_peak_share:
                break
            hc = (edges[i] + 5.0) % 360
            if all(C.hue_dist(hc, ph) > PEAK_MIN_GAP for ph, _ in peaks):
                peaks.append((float(hc), float(share)))

    # 3) 明暗统计：背景亮度 + 全画面平均 V / S（曝光判定用）
    L = px_lab[:, 0]
    small = rgb[::4, ::4].astype(np.float64) / 255.0
    mx, mn = small.max(-1), small.min(-1)
    return EnvObs(
        illuminant_lin=e, illum_hue=hue, illum_sat=sat, cct=cct,
        mean_L=float(L.mean()), p90_L=float(np.percentile(L, 90)),
        colored_share=float(colored.mean()), dominant_hue=peaks[0][0] if peaks else None, hue_peaks=peaks,
        dark_share=float(((L < 25) & (ss < 0.35)).mean()),
        frame_V=float(mx.mean()),
        frame_S=float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()),
    )
