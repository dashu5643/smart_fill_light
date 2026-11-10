"""一次推理的结果，以及对外输出的 JSON 结构。"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .analysis.types import Analysis
from .hardware.frame import LedFrame
from .labels import FAMILY_ZH, GROUP_ZH, LEVEL_ZH, P_LEVEL, SKIN_TONE_TYPE
from .perception.types import Observation
from .strategy.types import LightColor, Strategy


@dataclass
class Result:
    mode: str
    env_light: dict            # 环境光分类（《环境光分类标准》§7 的字段）
    perception: dict           # 感知与分析摘要（人脸、肤色、光线方向、环境估计）
    strategy: Strategy         # 打光策略（含决策依据、警告、备选）
    light_params: dict         # 灯板参数 · 精简版（灯型 + 两路颜色 + 亮度 + 修饰）
    frame: LedFrame            # 灯板参数 · 逐珠版（216 颗 RGBW）
    observation: Observation   # ↓ 中间结果，供报告图和调试用，不进 JSON
    analysis: Analysis

    def to_dict(self, with_leds: bool = True) -> dict:
        return {"mode": self.mode, "env_light": self.env_light, "perception": self.perception,
                "strategy": self.strategy.to_dict(), "light_params": self.light_params,
                "frame": self.frame.to_dict(with_leds)}


def build_result(mode: str, obs: Observation, an: Analysis, st: Strategy, frame: LedFrame, cfg: dict) -> Result:
    return Result(mode=mode, env_light=env_light_dict(an, cfg), perception=perception_dict(an),
                  strategy=st, light_params=light_params_dict(st), frame=frame, observation=obs, analysis=an)


def _main_face(an: Analysis):
    return next((f for f in an.faces if f.usable), None)


def env_light_dict(an: Analysis, cfg: dict) -> dict:
    cast = an.cast
    cats = cfg["cast"]["categories"]
    cat = cats.get(cast.family, cats["neutral"])
    main = _main_face(an)
    if main is None or main.skin is None:
        skin_type = None
    else:
        skin_type = SKIN_TONE_TYPE[main.skin.group] if main.skin.reliable else "Unknown"
    return {
        "env_light_category": cat["code"],
        "env_light_name": cat["name"],
        "hue": round(cast.hue),
        "saturation": round(cast.s_pct),
        "brightness": round(an.env.frame_V * 100),
        "pollution_level": P_LEVEL[cast.level],
        "multi_color_mode": cast.multi_color,
        "secondary_category": cats[cast.secondary]["code"] if cast.secondary in cats else None,
        "face_detected": bool(an.faces),
        "skin_tone_type": skin_type,
        "exposure": an.exposure,
        "confidence": cast.confidence,
        "timestamp": int(time.time() * 1000),
    }


def perception_dict(an: Analysis) -> dict:
    cast, env = an.cast, an.env
    main = _main_face(an)
    return {
        "faces": len(an.faces),
        "usable_faces": sum(f.usable for f in an.faces),
        "environment": {
            "cast_family": cast.family, "cast_family_zh": FAMILY_ZH.get(cast.family),
            "raw_family": cast.raw_family,
            "cast_hue": cast.hue, "pollution_s_pct": cast.s_pct,
            "pollution_level": cast.level, "pollution_level_zh": LEVEL_ZH[cast.level],
            "face_estimate": {"hue": cast.face_hue, "s_pct": cast.face_s},
            "background_estimate": {"hue": cast.bg_hue, "s_pct": cast.bg_s},
            "env_tone": an.env_tone, "cct_k": None if env.cct is None else round(env.cct),
            "colored_share": round(env.colored_share, 3),
            "hue_peaks": [[round(h), round(s, 2)] for h, s in env.hue_peaks],
            "mean_L": round(env.mean_L, 1),
            "frame_V": round(env.frame_V, 3),
            "multi_color_mode": cast.multi_color,
        },
        "main_face": None if main is None else {
            "box": main.obs.box,
            "skin_level": main.skin.level, "skin_group": main.skin.group,
            "skin_group_zh": GROUP_ZH[main.skin.group], "ita": main.skin.ita,
            "skin_reliable": main.skin.reliable, "skin_unreliable_reason": main.skin.unreliable_reason,
            "distance_cm": main.distance_cm,
            "undertone": main.skin.undertone,
            "lab_raw": main.skin.lab_raw, "lab_corrected": main.skin.lab_corrected,
            "light_direction": main.direction, "direction_detail": main.direction_detail,
            "blemishes": main.blemishes, "blemish_detail": main.blemish_detail,
        },
        "notes": an.notes,
    }


def light_params_dict(st: Strategy) -> dict:
    """精简灯光参数。main = 靠屏幕的列（mono 全板 / sandwich 面部主光 / dual 内圈），
    contour = 外侧的列（sandwich 轮廓光 / dual 外圈；mono 时与 main 相同）。"""
    def color(c: LightColor | None):
        return None if c is None else {"name": c.name, "hue": round(c.hue, 1), "sat": round(c.sat * 100, 1),
                                       "hex": c.hex}
    return {
        "light_type": st.light_type,
        "main": color(st.inner),
        "contour": color(st.outer),
        "t1_format": (f"【{st.outer.hue:.0f}-{st.outer.sat * 100:.0f}***{st.inner.hue:.0f}-{st.inner.sat * 100:.0f}】"
                      if st.light_type == "dual" else None),
        "brightness_pct": round(st.brightness * 100, 1),
        "side_boost_pct": {k: round(v * 100) for k, v in st.side_boost.items()},
        "top_gradient_pct": round(st.top_gradient * 100),
        "eye_patch": color(st.eye_patch),
    }


def describe(d: dict) -> str:
    """结果字典 → 一行中文摘要（命令行各入口共用）。"""
    lp, env, p = d["light_params"], d["env_light"], d["perception"]
    mf = p["main_face"]
    if not p["faces"]:
        who = "无人脸"
    elif mf is None:
        who = f"脸{p['faces']} 过小"
    else:
        who = f"脸{p['faces']} " + (f"{mf['skin_level']} {mf['skin_group_zh']}" if mf["skin_reliable"] else "肤色不可判")
    light = lp["t1_format"] or (lp["main"]["name"] + lp["main"]["hex"]
                                + (("/" + lp["contour"]["name"]) if lp["light_type"] == "sandwich" else ""))
    return (f"{who} | {env['env_light_category']} {env['env_light_name']} S{env['saturation']}% "
            f"{env['pollution_level']} | {lp['light_type']} {light} 亮{lp['brightness_pct']:.0f}%")
