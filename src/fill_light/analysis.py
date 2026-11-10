"""分析层：把感知观测转成决策用的语义特征。

- 环境光分类（《环境光分类标准》）：L0 中性 / L1 蓝 / L2 绿 / L3 紫 / L4 暖粉 / L5 黄 / L6 青
  + 污染程度 S% + 等级（L0 / P-Light / P-Mid / P-Heavy）+ 多色光源模式 + 次环境光
- 画面异常：暗光 / 过曝
- 肤色：去偏色后的 Lab → ITA → L1-L6 → fair/medium/deep，底调；强色光下判"不可靠"
- 拍摄距离：由脸宽占比估计（决定补光饱和度衰减）
- 光线方向：顺 / 逆 / 侧 / 顶
- 瑕疵：眼周暗沉、面中泛红
- 环境基调：暖 / 冷 / 中性
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import color as C
from .perception import EnvObs, FaceObs, Observation

P_LEVEL = {"none": "L0", "light": "P-Light", "medium": "P-Mid", "heavy": "P-Heavy"}


@dataclass
class CastInfo:
    hue: float                 # 污染光源色相（HSV 度）
    s_pct: float               # 污染程度 0-100
    level: str                 # none / light / medium / heavy
    family: str                # neutral / blue / green / purple / pink / yellow / cyan
    raw_family: str            # 不考虑等级时色相所属类别（L0 时也保留，便于排查）
    source: str                # face / background / fused
    face_hue: float | None = None
    face_s: float | None = None
    bg_hue: float | None = None
    bg_s: float | None = None
    multi_color: bool = False
    secondary: str | None = None
    confidence: float = 0.0


@dataclass
class SkinInfo:
    lab_raw: list[float]
    lab_corrected: list[float]
    ita: float
    level: str
    group: str                 # fair / medium / deep（不可靠时 = 推测值）
    undertone: str             # warm / cool / neutral / olive
    face_L: float
    reliable: bool = True
    unreliable_reason: str = ""


@dataclass
class FaceAnalysis:
    obs: FaceObs
    skin: SkinInfo | None
    direction: str             # front / back / side_img_l_dark / side_img_r_dark / top
    direction_detail: dict = field(default_factory=dict)
    blemishes: list[str] = field(default_factory=list)
    blemish_detail: dict = field(default_factory=dict)
    distance_cm: float | None = None
    usable: bool = True


@dataclass
class Analysis:
    faces: list[FaceAnalysis]
    cast: CastInfo
    env_tone: str              # warm / cool / neutral
    env: EnvObs
    exposure: str = "normal"   # normal / dark / over
    notes: list[str] = field(default_factory=list)


def _illum_from_face(face: FaceObs, ref_ab) -> np.ndarray | None:
    if face.skin_lab is None:
        return None
    L = float(face.skin_lab[0])
    ref_lin = C.lab_to_linear(np.array([L, ref_ab[0], ref_ab[1]]))
    e = face.skin_lin / np.maximum(ref_lin, 1e-4)
    return e / max(e.max(), 1e-9)


def _hs(e_lin: np.ndarray) -> tuple[float, float]:
    h, s, _ = C.rgb_hsv(C.linear_to_srgb(e_lin))
    return h, s


def _level(s_pct: float, family: str, cfg: dict) -> str:
    lv = cfg["cast"]["levels"]
    floor = cfg["cast"].get("class_min_s", {}).get(family)
    if floor is not None and s_pct < floor:
        return "none"
    if s_pct < lv["none"]:
        return "none"
    if s_pct < lv["light"]:
        return "light"
    if s_pct < lv["medium"]:
        return "medium"
    return "heavy"


def estimate_distance_cm(width_ratio: float, cfg: dict) -> float:
    dc = cfg["distance"]
    half = math.tan(math.radians(float(dc["hfov_deg"]) / 2))
    d = float(dc["face_width_cm"]) / (2 * half * max(width_ratio, 1e-3))
    return float(np.clip(d, 15.0, 300.0))


def analyze(obs: Observation, cfg: dict) -> Analysis:
    env = obs.env
    notes: list[str] = []
    ccfg = cfg["cast"]
    min_w = float(cfg["face"]["min_face_width_ratio"])

    main = next((f for f in obs.faces if f.width_ratio >= min_w and f.skin_lab is not None), None)

    # ---------- 0. 画面曝光异常 ----------
    ex = cfg["exposure"]
    exposure = "normal"
    if env.frame_V < ex["dark_V"]:
        exposure = "dark"
        notes.append(f"暗光环境（画面明度 V={env.frame_V:.0%} < {ex['dark_V']:.0%}），不做环境光分类，只提亮")
    elif env.frame_V > ex["over_V"] and env.frame_S < ex["over_S"]:
        exposure = "over"
        notes.append("画面过曝且无明显色彩，归 L0")

    # ---------- 1. 环境光（人脸 ROI 色偏优先 + 背景主色调辅助）----------
    mc = ccfg["multi_color"]
    bg_e = env.illuminant_lin
    bg_h, bg_s = _hs(bg_e)
    multi = len(env.hue_peaks) >= 2 and env.colored_share > mc["min_colored_share"]
    if env.dominant_hue is not None and env.colored_share > mc["min_colored_share"]:
        # 背景有大面积彩色光源（霓虹）：主色调代替灰世界估计
        bg_h = env.dominant_hue
        bg_s = max(bg_s, min(0.9, env.colored_share * 1.5))
    face_e = _illum_from_face(main, ccfg["ref_skin_ab"]) if main else None
    agree = None
    if face_e is not None:
        f_h, f_s = _hs(face_e)
        hd = C.hue_dist(f_h, bg_h)
        w = ccfg["face_weight_disagree"] if hd > 60 else ccfg["face_weight"]
        vf = f_s * np.array([np.cos(np.radians(f_h)), np.sin(np.radians(f_h))])
        vb = bg_s * np.array([np.cos(np.radians(bg_h)), np.sin(np.radians(bg_h))])
        v = w * vf + (1 - w) * vb
        hue = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
        sat = float(np.linalg.norm(v))
        source = "fused"
        agree = max(0.0, 1.0 - hd / 90.0)
    else:
        f_h = f_s = None
        hue, sat, source = bg_h, bg_s, "background"
    s_pct = round(sat * 100, 1)
    raw_family = C.classify_hue(hue, ccfg["hue_classes"])
    level = _level(s_pct, raw_family, cfg)
    if exposure != "normal":
        level = "none"
    family = raw_family if level != "none" else "neutral"
    secondary = None
    if multi:
        sec = [C.classify_hue(h, ccfg["hue_classes"]) for h, _ in env.hue_peaks]
        sec = [x for x in sec if x != family]
        secondary = sec[0] if sec else None
    # 置信度（启发式）：人脸/背景两路估计的色相一致程度 + S 距离等级边界的余量
    edges = [ccfg["levels"]["none"], ccfg["levels"]["light"], ccfg["levels"]["medium"]]
    margin = min(1.0, min(abs(s_pct - e) for e in edges) / 10.0)
    conf = 0.4 + 0.3 * (agree if agree is not None else 0.5) + 0.3 * margin
    cast = CastInfo(hue=round(hue, 1), s_pct=s_pct, level=level, family=family, raw_family=raw_family,
                    source=source, face_hue=None if f_h is None else round(f_h, 1),
                    face_s=None if f_s is None else round(f_s * 100, 1),
                    bg_hue=round(bg_h, 1), bg_s=round(bg_s * 100, 1),
                    multi_color=bool(multi and level != "none"), secondary=secondary if multi else None,
                    confidence=round(conf, 2))

    # ---------- 2. 环境基调（暖 / 冷 / 中性） ----------
    if env.cct is not None and bg_s < 0.3:
        env_tone = "warm" if env.cct < 4300 else ("cool" if env.cct > 5800 else "neutral")
    elif bg_s < 0.08:
        env_tone = "neutral"
    elif C.hue_in(bg_h, [[0, 70], [300, 360]]):
        env_tone = "warm"
    elif C.hue_in(bg_h, [[160, 300]]):
        env_tone = "cool"
    else:
        env_tone = "neutral"

    # ---------- 3. 每张脸 ----------
    illum_for_corr = None
    if cast.level != "none":
        illum_for_corr = C.srgb_to_linear(C.hsv_rgb(cast.hue, min(sat, 0.85), 1.0))
    faces = [_analyze_face(f, illum_for_corr, cfg, env, min_w, cast) for f in obs.faces]
    if not faces:
        notes.append("未检测到人脸 → 物品模式")
    elif main is None:
        notes.append("人脸过小或采样不足，肤色判断不可靠")
    return Analysis(faces=faces, cast=cast, env_tone=env_tone, env=env, exposure=exposure, notes=notes)


def _analyze_face(f: FaceObs, illum_lin, cfg: dict, env: EnvObs, min_w: float, cast: CastInfo) -> FaceAnalysis:
    dist = estimate_distance_cm(f.width_ratio, cfg)
    if f.skin_lab is None or f.width_ratio < min_w:
        return FaceAnalysis(obs=f, skin=None, direction="unknown", distance_cm=round(dist), usable=False)

    # 去偏色：von Kries 白点矫正后再算 ITA
    lin = f.skin_lin.copy()
    if illum_lin is not None:
        g = illum_lin / max(illum_lin[1], 1e-6)
        lin = lin / np.maximum(g, 1e-3)
    lab_c = C.linear_to_lab(lin)
    ita = C.ita_deg(lab_c)
    level = next(x["level"] for x in cfg["skin"]["ita_levels"] if ita >= x["min"])
    group = cfg["skin"]["groups"][level]
    ucfg = cfg["skin"]["undertone"]
    _, a, b = lab_c
    h = C.lab_hue_deg(lab_c)
    if a < ucfg["olive_max_a"] and b > ucfg["olive_min_b"]:
        undertone = "olive"
    elif h >= ucfg["warm_min_hue"]:
        undertone = "warm"
    elif h <= ucfg["cool_max_hue"]:
        undertone = "cool"
    else:
        undertone = "neutral"

    # 肤色可靠性：强色光下某个颜色通道几乎没有信号，去偏色也救不回来
    rc = cfg["skin"]["reliability"]
    raw_lin = f.skin_lin
    ch_ratio = float(raw_lin.min() / max(raw_lin.max(), 1e-6))
    reasons = []
    if cast.s_pct >= rc["max_cast_s"]:
        reasons.append(f"重度色光 S={cast.s_pct:.0f}%")
    if ch_ratio < rc["min_channel_ratio"]:
        reasons.append(f"皮肤颜色通道缺失（最小/最大={ch_ratio:.2f}）")
    if float(f.skin_lab[0]) < rc["min_face_L"]:
        reasons.append(f"脸部过暗 L={float(f.skin_lab[0]):.0f}")
    skin = SkinInfo(lab_raw=[round(float(x), 1) for x in f.skin_lab],
                    lab_corrected=[round(float(x), 1) for x in lab_c],
                    ita=round(ita, 1), level=level, group=group, undertone=undertone,
                    face_L=round(float(f.skin_lab[0]), 1),
                    reliable=not reasons, unreliable_reason="；".join(reasons))

    # 光线方向
    dcfg = cfg["direction"]
    R = f.regions
    get_L = lambda k: float(R[k]["lab"][0]) if k in R else None
    lL, rL, fL, cL = get_L("cheek_img_l"), get_L("cheek_img_r"), get_L("forehead"), get_L("chin")
    detail = {"cheek_img_l_L": lL, "cheek_img_r_L": rL, "forehead_L": fL, "chin_L": cL,
              "bg_p90_L": round(env.p90_L, 1)}
    direction = "front"
    back_delta = dcfg["back_delta_L"] + (30.0 if group == "deep" else 0.0)
    if (env.mean_L - skin.face_L > back_delta * 0.8 and env.p90_L - skin.face_L > back_delta
            and skin.face_L < 50):
        direction = "back"
    elif dcfg.get("skip_side_top_when_skin_unreliable", True) and not skin.reliable:
        detail["side_top_skipped"] = "肤色不可靠（强色光）"
    elif lL is not None and rL is not None and abs(lL - rL) > dcfg["side_delta_L"] and abs(f.yaw_proxy) < 0.12:
        direction = "side_img_l_dark" if lL < rL else "side_img_r_dark"
    elif fL is not None and cL is not None and fL - cL > dcfg["top_delta_L"]:
        direction = "top"

    # 瑕疵（强色光下肤色不可靠时不判）
    bcfg = cfg["blemish"]
    blem, bdet = [], {}
    ue = [R[k]["lab"][0] for k in ("undereye_img_l", "undereye_img_r") if k in R]
    ck = [R[k]["lab"][0] for k in ("cheek_img_l", "cheek_img_r") if k in R]
    if ue and ck:
        d = float(np.mean(ck) - np.mean(ue))
        bdet["undereye_delta_L"] = round(d, 1)
        if d > bcfg["dark_circle_delta_L"] and skin.reliable:
            blem.append("dark_circle")
    ca = [R[k]["lab"][1] for k in ("cheek_img_l", "cheek_img_r") if k in R]
    if ca and "forehead" in R:
        d = float(np.mean(ca) - R["forehead"]["lab"][1])
        bdet["cheek_minus_forehead_a"] = round(d, 1)
        if d > bcfg["redness_delta_a"] and skin.reliable:
            blem.append("redness")
    return FaceAnalysis(obs=f, skin=skin, direction=direction, direction_detail=detail,
                        blemishes=blem, blemish_detail=bdet, distance_cm=round(dist))
