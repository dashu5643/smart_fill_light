"""环境光：曝光异常、色光污染分类（《环境光分类标准》）、环境基调。

色光污染 = 人脸当色卡反推的光源 + 背景估计的光源，在色轮上按权重做向量融合。
"""
from __future__ import annotations

import numpy as np

from .. import color as C
from ..perception.types import EnvObs, FaceObs
from .types import CastInfo

WARM_MAX_CCT, COOL_MIN_CCT = 4300, 5800          # 色温 → 暖 / 冷的分界（K）
WARM_HUES, COOL_HUES = [[0, 70], [300, 360]], [[160, 300]]
DISAGREE_HUE = 60                                # 人脸与背景色相差超过此值 → 更信人脸
CORRECTION_MAX_SAT = 0.85                        # 肤色去偏色时光源饱和度上限


def judge_exposure(env: EnvObs, cfg: dict) -> tuple[str, str | None]:
    """返回 (normal / dark / over, 说明)。暗光和过曝都不判色光污染。"""
    ex = cfg["exposure"]
    if env.frame_V < ex["dark_V"]:
        return "dark", f"暗光环境（画面明度 V={env.frame_V:.0%} < {ex['dark_V']:.0%}），不做环境光分类，只提亮"
    if env.frame_V > ex["over_V"] and env.frame_S < ex["over_S"]:
        return "over", "画面过曝且无明显色彩，归 L0"
    return "normal", None


def background_light(env: EnvObs, cfg: dict) -> tuple[float, float]:
    """背景光源的 (色相, 饱和度 0-1)。背景有大面积彩色光源（霓虹）时，用主色相代替灰世界估计。"""
    hue, sat = env.illum_hue, env.illum_sat
    if env.dominant_hue is not None and env.colored_share > cfg["cast"]["multi_color"]["min_colored_share"]:
        hue = env.dominant_hue
        sat = max(sat, min(0.9, env.colored_share * 1.5))
    return hue, sat


def estimate_cast(env: EnvObs, bg: tuple[float, float], main_face: FaceObs | None, exposure: str,
                  cfg: dict) -> tuple[CastInfo, np.ndarray | None]:
    """返回 (环境光分类, 肤色去偏色用的光源；无污染时为 None)。"""
    ccfg = cfg["cast"]
    bg_h, bg_s = bg
    face_e = _illuminant_from_face(main_face, ccfg["ref_skin_ab"]) if main_face else None
    if face_e is not None:
        f_h, f_s, _ = C.rgb_hsv(C.linear_to_srgb(face_e))
        hd = C.hue_dist(f_h, bg_h)
        w = ccfg["face_weight_disagree"] if hd > DISAGREE_HUE else ccfg["face_weight"]
        v = w * _polar(f_h, f_s) + (1 - w) * _polar(bg_h, bg_s)
        hue = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
        sat = float(np.linalg.norm(v))
        source, agree = "fused", max(0.0, 1.0 - hd / 90.0)
    else:
        f_h = f_s = None
        hue, sat, source, agree = bg_h, bg_s, "background", None

    s_pct = round(sat * 100, 1)
    raw_family = C.classify_hue(hue, ccfg["hue_classes"])
    level = _level(s_pct, raw_family, ccfg) if exposure == "normal" else "none"
    family = raw_family if level != "none" else "neutral"

    multi = len(env.hue_peaks) >= 2 and env.colored_share > ccfg["multi_color"]["min_colored_share"]
    secondary = None
    if multi:
        others = [C.classify_hue(h, ccfg["hue_classes"]) for h, _ in env.hue_peaks]
        secondary = next((x for x in others if x != family), None)

    # 置信度（启发式）：人脸/背景两路色相的一致程度 + S 距离等级边界的余量
    edges = [ccfg["levels"]["none"], ccfg["levels"]["light"], ccfg["levels"]["medium"]]
    margin = min(1.0, min(abs(s_pct - e) for e in edges) / 10.0)
    conf = 0.4 + 0.3 * (agree if agree is not None else 0.5) + 0.3 * margin

    cast = CastInfo(hue=round(hue, 1), s_pct=s_pct, level=level, family=family, raw_family=raw_family,
                    source=source, face_hue=None if f_h is None else round(f_h, 1),
                    face_s=None if f_s is None else round(f_s * 100, 1),
                    bg_hue=round(bg_h, 1), bg_s=round(bg_s * 100, 1),
                    multi_color=bool(multi and level != "none"), secondary=secondary,
                    confidence=round(conf, 2))
    illum = None
    if level != "none":
        illum = C.srgb_to_linear(C.hsv_rgb(cast.hue, min(sat, CORRECTION_MAX_SAT), 1.0))
    return cast, illum


def classify_env_tone(env: EnvObs, bg: tuple[float, float]) -> str:
    """环境基调 warm / cool / neutral：接近白光时看色温，否则看背景色相。"""
    bg_h, bg_s = bg
    if env.cct is not None and bg_s < 0.3:
        return "warm" if env.cct < WARM_MAX_CCT else ("cool" if env.cct > COOL_MIN_CCT else "neutral")
    if bg_s < 0.08:
        return "neutral"
    if C.hue_in(bg_h, WARM_HUES):
        return "warm"
    if C.hue_in(bg_h, COOL_HUES):
        return "cool"
    return "neutral"


def _illuminant_from_face(face: FaceObs, ref_ab) -> np.ndarray | None:
    """人脸当色卡：实测肤色 ÷ 同亮度的参考肤色 = 光源颜色。"""
    if face.skin_lab is None:
        return None
    ref_lin = C.lab_to_linear(np.array([float(face.skin_lab[0]), ref_ab[0], ref_ab[1]]))
    e = face.skin_lin / np.maximum(ref_lin, 1e-4)
    return e / max(e.max(), 1e-9)


def _polar(hue: float, sat: float) -> np.ndarray:
    return sat * np.array([np.cos(np.radians(hue)), np.sin(np.radians(hue))])


def _level(s_pct: float, family: str, ccfg: dict) -> str:
    lv = ccfg["levels"]
    floor = ccfg.get("class_min_s", {}).get(family)
    if floor is not None and s_pct < floor:
        return "none"
    if s_pct < lv["none"]:
        return "none"
    if s_pct < lv["light"]:
        return "light"
    if s_pct < lv["medium"]:
        return "medium"
    return "heavy"
