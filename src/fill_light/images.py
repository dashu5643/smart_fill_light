"""图片读取：统一成 RGB、长边不超过 max_side。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def bgr_to_rgb(bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imread_rgb(path: str | Path, max_side: int = 1280) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)          # 支持中文路径
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr_to_rgb(bgr, max_side)


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
