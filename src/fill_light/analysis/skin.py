"""肤色：去偏色 → ITA → L1-L6 → 浅/中/深 + 底调；强色光下标记"不可靠"。"""
from __future__ import annotations

import numpy as np

from .. import color as C
from ..perception.types import FaceObs
from .types import SkinInfo


def analyze_skin(f: FaceObs, illum_lin: np.ndarray | None, cast_s_pct: float, cfg: dict) -> SkinInfo:
    scfg = cfg["skin"]
    # 去偏色：von Kries 白点矫正（以 G 通道为基准）后再算 ITA
    lin = f.skin_lin.copy()
    if illum_lin is not None:
        gain = illum_lin / max(illum_lin[1], 1e-6)
        lin = lin / np.maximum(gain, 1e-3)
    lab_c = C.linear_to_lab(lin)
    ita = C.ita_deg(lab_c)
    level = next(x["level"] for x in scfg["ita_levels"] if ita >= x["min"])

    # 可靠性：强色光下某个颜色通道几乎没有信号，去偏色也救不回来
    rc = scfg["reliability"]
    face_L = float(f.skin_lab[0])
    ch_ratio = float(f.skin_lin.min() / max(f.skin_lin.max(), 1e-6))
    reasons = []
    if cast_s_pct >= rc["max_cast_s"]:
        reasons.append(f"重度色光 S={cast_s_pct:.0f}%")
    if ch_ratio < rc["min_channel_ratio"]:
        reasons.append(f"皮肤颜色通道缺失（最小/最大={ch_ratio:.2f}）")
    if face_L < rc["min_face_L"]:
        reasons.append(f"脸部过暗 L={face_L:.0f}")

    return SkinInfo(lab_raw=[round(float(x), 1) for x in f.skin_lab],
                    lab_corrected=[round(float(x), 1) for x in lab_c],
                    ita=round(ita, 1), level=level, group=scfg["groups"][level],
                    undertone=_undertone(lab_c, scfg["undertone"]), face_L=round(face_L, 1),
                    reliable=not reasons, unreliable_reason="；".join(reasons))


def _undertone(lab: np.ndarray, ucfg: dict) -> str:
    _, a, b = lab
    h = C.lab_hue_deg(lab)
    if a < ucfg["olive_max_a"] and b > ucfg["olive_min_b"]:
        return "olive"
    if h >= ucfg["warm_min_hue"]:
        return "warm"
    if h <= ucfg["cool_max_hue"]:
        return "cool"
    return "neutral"
