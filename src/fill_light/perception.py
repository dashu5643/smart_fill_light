"""感知层：人脸检测 + 面部区域采样 + 环境光估计。

输入一帧 RGB 图片，输出结构化观测（不做任何灯光决策）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from . import color as C

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# 采样点（MediaPipe 478 点索引）。"img_l"/"img_r" 指图像左/右侧
ROI_POINTS = {
    "forehead": [151],
    "cheek_img_l": [205, 50],
    "cheek_img_r": [425, 280],
    "chin": [199],
    "undereye_img_l": [(145, 118)],
    "undereye_img_r": [(374, 347)],
}


@dataclass
class FaceObs:
    box: tuple[int, int, int, int]                 # x, y, w, h
    center_x_ratio: float
    width_ratio: float
    landmarks: np.ndarray
    regions: dict[str, dict] = field(default_factory=dict)   # 区域名 → {lab, lin, n}
    skin_lab: np.ndarray | None = None              # 整脸皮肤 Lab 中位数（原始，未矫正）
    skin_lin: np.ndarray | None = None
    yaw_proxy: float = 0.0


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
    faces: list[FaceObs]
    env: EnvObs


class Perceiver:
    def __init__(self, cfg: dict, root: Path) -> None:
        self.cfg = cfg
        fcfg = cfg["face"]
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(root / cfg["paths"]["face_landmarker"])),
            running_mode=RunningMode.IMAGE,
            num_faces=int(fcfg["max_faces"]),
            min_face_detection_confidence=float(fcfg["min_detection_confidence"]),
        )
        self._lm = FaceLandmarker.create_from_options(opts)

    def close(self) -> None:
        self._lm.close()

    # ------------------------------------------------------------------
    def observe(self, rgb: np.ndarray) -> Observation:
        h, w = rgb.shape[:2]
        lin = C.srgb_to_linear(rgb.astype(np.float64) / 255.0)
        lab = C.linear_to_lab(lin)

        faces = self._detect(rgb)
        face_mask = np.zeros((h, w), np.uint8)
        obs_faces = []
        for pts in faces:
            fo = self._face_obs(pts, lin, lab, w, h)
            if fo is None:
                continue
            obs_faces.append(fo)
            hull = cv2.convexHull(pts[:, :2].astype(np.int32))
            cv2.fillConvexPoly(face_mask, hull, 1)
        if obs_faces:
            # 背景排除人脸及其周边（头发/脖子），按脸宽膨胀
            k = max(3, int(max(f.box[2] for f in obs_faces) * 0.35))
            face_mask = cv2.dilate(face_mask, np.ones((k, k), np.uint8))
        env = self._env_obs(rgb, lin, lab, face_mask == 0)
        small = rgb[::4, ::4].astype(np.float64) / 255.0
        mx, mn = small.max(-1), small.min(-1)
        env.frame_V = float(mx.mean())
        env.frame_S = float(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
        obs_faces.sort(key=lambda f: -f.box[2])
        return Observation(image_size=(w, h), faces=obs_faces, env=env)

    # ------------------------------------------------------------------
    def _detect(self, rgb: np.ndarray) -> list[np.ndarray]:
        h, w = rgb.shape[:2]
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        out = []
        for face in res.face_landmarks or []:
            out.append(np.array([[p.x * w, p.y * h, p.z * w] for p in face], dtype=np.float32))
        return out

    def _face_obs(self, pts, lin, lab, w, h) -> FaceObs | None:
        oval = pts[FACE_OVAL, :2]
        x0, y0 = oval.min(0)
        x1, y1 = oval.max(0)
        fw, fh = float(x1 - x0), float(y1 - y0)
        if fw < 8:
            return None
        fo = FaceObs(
            box=(int(x0), int(y0), int(fw), int(fh)),
            center_x_ratio=float((x0 + x1) / 2 / w),
            width_ratio=fw / w,
            landmarks=pts,
        )
        r = max(2, int(fw * float(self.cfg["face"]["roi_radius_ratio"])))
        all_px = []
        for name, idxs in ROI_POINTS.items():
            mask = np.zeros((h, w), np.uint8)
            rr = r if "undereye" not in name else max(2, int(r * 0.6))
            for idx in idxs:
                if isinstance(idx, tuple):
                    c = (pts[idx[0], :2] * 0.45 + pts[idx[1], :2] * 0.55)
                else:
                    c = pts[idx, :2]
                cv2.circle(mask, (int(c[0]), int(c[1])), rr, 1, -1)
            sel = mask.astype(bool)
            px_lab, px_lin = _trim(lab[sel], lin[sel])
            if len(px_lab) < 6:
                continue
            fo.regions[name] = {
                "lab": np.median(px_lab, 0),
                "lin": np.median(px_lin, 0),
                "n": int(len(px_lab)),
            }
            if "undereye" not in name:
                all_px.append((px_lab, px_lin))
        if all_px:
            la = np.concatenate([a for a, _ in all_px])
            li = np.concatenate([b for _, b in all_px])
            fo.skin_lab = np.median(la, 0)
            fo.skin_lin = np.median(li, 0)
        # 偏转近似：鼻尖相对两颊外缘的位置
        nose = pts[1, 0]
        fo.yaw_proxy = float(((nose - x0) / max(fw, 1)) - 0.5)
        return fo

    def _env_obs(self, rgb, lin, lab, bg_mask) -> EnvObs:
        if bg_mask.sum() < 500:
            bg_mask = np.ones(bg_mask.shape, bool)
        # 下采样提速
        step = max(1, int(np.sqrt(bg_mask.sum() / 60000)))
        m = bg_mask[::step, ::step]
        px_lin = lin[::step, ::step][m]
        px_lab = lab[::step, ::step][m]
        px_rgb = rgb[::step, ::step][m].astype(np.float64) / 255.0

        # 1) 光源估计：Shades-of-Gray (p=6)，排除过曝和过暗像素
        ok = (px_lin.max(1) < 0.97) & (px_lin.max(1) > 0.01)
        src = px_lin[ok] if ok.sum() > 200 else px_lin
        p = 6.0
        e = np.power(np.mean(np.power(src, p), 0), 1 / p)
        e = e / max(e.max(), 1e-9)
        hue, sat, _ = C.rgb_hsv(C.linear_to_srgb(e))
        cct = C.cct_from_linear_rgb(e) if sat < 0.35 else None

        # 2) 彩色光源：高饱和像素的色相直方图
        hsv = cv2.cvtColor((px_rgb[None] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV_FULL)[0].astype(np.float64)
        hh = hsv[:, 0] / 255.0 * 360.0
        ss = hsv[:, 1] / 255.0
        vv = hsv[:, 2] / 255.0
        colored = (ss > 0.45) & (vv > 0.25)
        colored_share = float(colored.mean())
        peaks: list[tuple[float, float]] = []
        dominant = None
        if colored.sum() > 50:
            wts = ss[colored] * vv[colored]
            hist, edges = np.histogram(hh[colored], bins=36, range=(0, 360), weights=wts)
            hist = np.convolve(np.r_[hist[-1], hist, hist[0]], [0.25, 0.5, 0.25], "valid")
            total = hist.sum() + 1e-9
            for i in np.argsort(-hist):
                share = hist[i] / total
                if share < float(self.cfg["cast"]["multi_color"]["min_peak_share"]):
                    break
                hc = (edges[i] + 5.0) % 360
                if all(C.hue_dist(hc, ph) > 40 for ph, _ in peaks):
                    peaks.append((float(hc), float(share)))
            if peaks:
                dominant = peaks[0][0]

        L = px_lab[:, 0]
        dark_share = float(((L < 25) & (ss < 0.35)).mean())
        return EnvObs(
            illuminant_lin=e, illum_hue=hue, illum_sat=sat, cct=cct,
            mean_L=float(L.mean()), p90_L=float(np.percentile(L, 90)),
            colored_share=colored_share, dominant_hue=dominant, hue_peaks=peaks,
            dark_share=dark_share,
        )


def _trim(lab_px: np.ndarray, lin_px: np.ndarray, lo=10, hi=90):
    """去掉高光和阴影（按 L 的分位数）。"""
    if len(lab_px) < 10:
        return lab_px, lin_px
    L = lab_px[:, 0]
    a, b = np.percentile(L, [lo, hi])
    keep = (L >= a) & (L <= b)
    return lab_px[keep], lin_px[keep]
