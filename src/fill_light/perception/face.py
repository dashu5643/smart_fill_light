"""人脸：MediaPipe 478 点检测 + 面部 6 个圆形区域采样。"""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from .types import FaceObs

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# 采样点（478 点索引）。"img_l"/"img_r" 指图像左/右侧；二元组 = 两点之间按 0.45 : 0.55 取点
ROI_POINTS = {
    "forehead": [151],
    "cheek_img_l": [205, 50],
    "cheek_img_r": [425, 280],
    "chin": [199],
    "undereye_img_l": [(145, 118)],
    "undereye_img_r": [(374, 347)],
}
MIN_FACE_PX = 8          # 脸宽小于此像素数直接丢弃
MIN_ROI_PX = 6           # 区域内有效像素少于此数不采样


class FaceDetector:
    def __init__(self, model_path: str | Path, max_faces: int, min_confidence: float) -> None:
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=int(max_faces),
            min_face_detection_confidence=float(min_confidence),
        )
        self._lm = FaceLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> list[np.ndarray]:
        """每张脸一个 478×3 数组（像素坐标）。"""
        h, w = rgb.shape[:2]
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return [np.array([[p.x * w, p.y * h, p.z * w] for p in face], dtype=np.float32)
                for face in res.face_landmarks or []]

    def close(self) -> None:
        self._lm.close()


def roi_centers(pts: np.ndarray) -> dict[str, list[np.ndarray]]:
    """区域名 → 圆心列表（像素坐标）。采样和报告图共用。"""
    return {name: [pts[i[0], :2] * 0.45 + pts[i[1], :2] * 0.55 if isinstance(i, tuple) else pts[i, :2]
                   for i in idxs]
            for name, idxs in ROI_POINTS.items()}


def roi_radius(name: str, base: int) -> int:
    """眼下区域用小一号的圆。"""
    return base if "undereye" not in name else max(2, int(base * 0.6))


def sample_face(pts: np.ndarray, lin: np.ndarray, lab: np.ndarray, radius_ratio: float) -> FaceObs | None:
    h, w = lab.shape[:2]
    oval = pts[FACE_OVAL, :2]
    x0, y0 = oval.min(0)
    x1, y1 = oval.max(0)
    fw, fh = float(x1 - x0), float(y1 - y0)
    if fw < MIN_FACE_PX:
        return None
    fo = FaceObs(
        box=(int(x0), int(y0), int(fw), int(fh)),
        center_x_ratio=float((x0 + x1) / 2 / w),
        width_ratio=fw / w,
        landmarks=pts,
    )
    r = max(2, int(fw * radius_ratio))
    all_px = []
    for name, centers in roi_centers(pts).items():
        mask = np.zeros((h, w), np.uint8)
        for c in centers:
            cv2.circle(mask, (int(c[0]), int(c[1])), roi_radius(name, r), 1, -1)
        sel = mask.astype(bool)
        px_lab, px_lin = _trim(lab[sel], lin[sel])
        if len(px_lab) < MIN_ROI_PX:
            continue
        fo.regions[name] = {"lab": np.median(px_lab, 0), "lin": np.median(px_lin, 0), "n": int(len(px_lab))}
        if "undereye" not in name:                  # 眼下不参与整脸肤色
            all_px.append((px_lab, px_lin))
    if all_px:
        fo.skin_lab = np.median(np.concatenate([a for a, _ in all_px]), 0)
        fo.skin_lin = np.median(np.concatenate([b for _, b in all_px]), 0)
    fo.yaw_proxy = float(((pts[1, 0] - x0) / max(fw, 1)) - 0.5)
    return fo


def _trim(lab_px: np.ndarray, lin_px: np.ndarray, lo: float = 10, hi: float = 90):
    """去掉高光和阴影（按 L 的分位数）。"""
    if len(lab_px) < 10:
        return lab_px, lin_px
    L = lab_px[:, 0]
    a, b = np.percentile(L, [lo, hi])
    keep = (L >= a) & (L <= b)
    return lab_px[keep], lin_px[keep]
