"""感知层入口：一帧 RGB → Observation。"""
from __future__ import annotations

import cv2
import numpy as np

from .. import color as C
from ..config import resolve
from .environment import observe_environment
from .face import FaceDetector, sample_face
from .types import Observation

FACE_DILATE_RATIO = 0.35         # 背景要避开人脸周边（头发/脖子）：按脸宽的这个比例膨胀


class Perceiver:
    def __init__(self, cfg: dict) -> None:
        f = cfg["face"]
        self._roi_radius_ratio = float(f["roi_radius_ratio"])
        self._min_peak_share = float(cfg["cast"]["multi_color"]["min_peak_share"])
        self._detector = FaceDetector(resolve(cfg["paths"]["face_landmarker"]), f["max_faces"],
                                      f["min_detection_confidence"])

    def close(self) -> None:
        self._detector.close()

    def observe(self, rgb: np.ndarray) -> Observation:
        h, w = rgb.shape[:2]
        lin = C.srgb_to_linear(rgb.astype(np.float64) / 255.0)
        lab = C.linear_to_lab(lin)

        faces = []
        face_mask = np.zeros((h, w), np.uint8)
        for pts in self._detector.detect(rgb):
            fo = sample_face(pts, lin, lab, self._roi_radius_ratio)
            if fo is None:
                continue
            faces.append(fo)
            cv2.fillConvexPoly(face_mask, cv2.convexHull(pts[:, :2].astype(np.int32)), 1)
        if faces:
            k = max(3, int(max(f.box[2] for f in faces) * FACE_DILATE_RATIO))
            face_mask = cv2.dilate(face_mask, np.ones((k, k), np.uint8))
        env = observe_environment(rgb, lin, lab, face_mask == 0, self._min_peak_share)
        faces.sort(key=lambda f: -f.box[2])
        return Observation(image_size=(w, h), faces=faces, env=env, roi_radius_ratio=self._roi_radius_ratio)
