"""分析层入口：Observation → Analysis。"""
from __future__ import annotations

from ..perception.types import FaceObs, Observation
from .ambient import background_light, classify_env_tone, estimate_cast, judge_exposure
from .blemish import detect_blemishes
from .direction import light_direction
from .distance import resolve_distance_cm
from .skin import analyze_skin
from .types import Analysis, FaceAnalysis


def analyze(obs: Observation, cfg: dict, distance_cm: float | None = None) -> Analysis:
    """distance_cm：调用方给的灯到人脸距离；不给则按 configs 的 distance 一节处理。"""
    env = obs.env
    min_w = float(cfg["face"]["min_face_width_ratio"])
    usable = lambda f: f.skin_lab is not None and f.width_ratio >= min_w
    main = next((f for f in obs.faces if usable(f)), None)        # 最大的可用人脸

    exposure, note = judge_exposure(env, cfg)
    notes = [note] if note else []
    bg = background_light(env, cfg)
    cast, illum = estimate_cast(env, bg, main, exposure, cfg)
    env_tone = classify_env_tone(env, bg)

    faces = []
    for f in obs.faces:
        dist = resolve_distance_cm(f.width_ratio, cfg, distance_cm)
        faces.append(_analyze_face(f, illum, cast.s_pct, env, cfg, dist) if usable(f) else
                     FaceAnalysis(obs=f, skin=None, direction="unknown", distance_cm=dist, usable=False))
    if not faces:
        notes.append("未检测到人脸 → 物品模式")
    elif main is None:
        notes.append("人脸过小或采样不足，肤色判断不可靠")
    if distance_cm is not None:
        notes.append(f"距离由调用方指定：{distance_cm}cm")
    return Analysis(faces=faces, cast=cast, env_tone=env_tone, env=env, exposure=exposure, notes=notes)


def _analyze_face(f: FaceObs, illum, cast_s_pct: float, env, cfg: dict, dist: float) -> FaceAnalysis:
    skin = analyze_skin(f, illum, cast_s_pct, cfg)
    direction, d_detail = light_direction(f, skin, env, cfg)
    blemishes, b_detail = detect_blemishes(f, skin.reliable, cfg)
    return FaceAnalysis(obs=f, skin=skin, direction=direction, direction_detail=d_detail,
                        blemishes=blemishes, blemish_detail=b_detail, distance_cm=dist)
