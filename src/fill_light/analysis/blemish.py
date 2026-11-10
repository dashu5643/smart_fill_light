"""瑕疵：眼周暗沉（眼下比脸颊暗）、面中泛红（脸颊比额头红）。肤色不可靠时只记数值、不下结论。"""
from __future__ import annotations

import numpy as np

from ..perception.types import FaceObs

CHEEKS = ("cheek_img_l", "cheek_img_r")
UNDEREYES = ("undereye_img_l", "undereye_img_r")


def detect_blemishes(f: FaceObs, skin_reliable: bool, cfg: dict) -> tuple[list[str], dict]:
    bcfg = cfg["blemish"]
    R = f.regions
    found, detail = [], {}
    eye_L = [R[k]["lab"][0] for k in UNDEREYES if k in R]
    cheek_L = [R[k]["lab"][0] for k in CHEEKS if k in R]
    if eye_L and cheek_L:
        d = float(np.mean(cheek_L) - np.mean(eye_L))
        detail["undereye_delta_L"] = round(d, 1)
        if d > bcfg["dark_circle_delta_L"] and skin_reliable:
            found.append("dark_circle")
    cheek_a = [R[k]["lab"][1] for k in CHEEKS if k in R]
    if cheek_a and "forehead" in R:
        d = float(np.mean(cheek_a) - R["forehead"]["lab"][1])
        detail["cheek_minus_forehead_a"] = round(d, 1)
        if d > bcfg["redness_delta_a"] and skin_reliable:
            found.append("redness")
    return found, detail
