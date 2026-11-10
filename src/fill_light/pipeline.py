"""单帧补光策略流水线：一张图 → 一套策略 + 灯板帧。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

import time

from .analysis import P_LEVEL, analyze
from .hardware import render_frame, summary_params
from .perception import Perceiver
from .strategy import GROUP_ZH, LEVEL_ZH, FAMILY_ZH, StrategyEngine

ROOT = Path(__file__).resolve().parents[2]


def load_cfg(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "strategy.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def imread_rgb(path: str | Path, max_side: int = 1280) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)          # 支持中文路径
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FillLightPipeline:
    def __init__(self, cfg_path: str | Path | None = None) -> None:
        self.cfg = load_cfg(cfg_path)
        self.perceiver = Perceiver(self.cfg, ROOT)
        self.engine = StrategyEngine(self.cfg)

    def close(self) -> None:
        self.perceiver.close()

    def run(self, rgb: np.ndarray, mode: str = "skin_priority", object_type: str | None = None,
            distance_cm: float | None = None) -> dict:
        """distance_cm：灯板到人脸的距离（cm）。不传则按脸宽占比估计（假设灯与镜头同位）。"""
        obs = self.perceiver.observe(rgb)
        an = analyze(obs, self.cfg)
        if distance_cm is not None:
            for f in an.faces:
                f.distance_cm = float(distance_cm)
            an.notes.append(f"距离由调用方指定：{distance_cm}cm")
        elif self.cfg["distance"].get("estimate_from_face", True) is False:
            for f in an.faces:
                f.distance_cm = float(self.cfg["distance"]["default_cm"])
        st = self.engine.decide(an, mode=mode, object_type=object_type)
        frame = render_frame(st, self.cfg)
        main = next((f for f in an.faces if f.usable), None)
        cats = self.cfg["cast"]["categories"]
        cat = cats.get(an.cast.family, cats["neutral"])
        result = {
            "mode": mode,
            # 《环境光分类标准》§7 输出数据结构
            "env_light": {
                "env_light_category": cat["code"],
                "env_light_name": cat["name"],
                "hue": round(an.cast.hue),
                "saturation": round(an.cast.s_pct),
                "brightness": round(an.env.frame_V * 100),
                "pollution_level": P_LEVEL[an.cast.level],
                "multi_color_mode": an.cast.multi_color,
                "secondary_category": cats[an.cast.secondary]["code"] if an.cast.secondary in cats else None,
                "face_detected": bool(an.faces),
                "skin_tone_type": (None if main is None or main.skin is None else
                                   ({"fair": "Fair", "medium": "Medium", "deep": "Deep"}[main.skin.group]
                                    if main.skin.reliable else "Unknown")),
                "exposure": an.exposure,
                "confidence": an.cast.confidence,
                "timestamp": int(time.time() * 1000),
            },
            "perception": {
                "faces": len(an.faces),
                "usable_faces": sum(f.usable for f in an.faces),
                "environment": {
                    "cast_family": an.cast.family, "cast_family_zh": FAMILY_ZH.get(an.cast.family),
                    "raw_family": an.cast.raw_family,
                    "cast_hue": an.cast.hue, "pollution_s_pct": an.cast.s_pct,
                    "pollution_level": an.cast.level, "pollution_level_zh": LEVEL_ZH[an.cast.level],
                    "face_estimate": {"hue": an.cast.face_hue, "s_pct": an.cast.face_s},
                    "background_estimate": {"hue": an.cast.bg_hue, "s_pct": an.cast.bg_s},
                    "env_tone": an.env_tone, "cct_k": None if an.env.cct is None else round(an.env.cct),
                    "colored_share": round(an.env.colored_share, 3),
                    "hue_peaks": [[round(h), round(s, 2)] for h, s in an.env.hue_peaks],
                    "mean_L": round(an.env.mean_L, 1),
                    "frame_V": round(an.env.frame_V, 3),
                    "multi_color_mode": an.cast.multi_color,
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
            },
            "strategy": st.to_dict(),
            "light_params": summary_params(st),
            "frame": {k: v for k, v in frame.items() if not k.startswith("_")},
        }
        result["_internal"] = {"obs": obs, "analysis": an, "strategy": st, "frame_arrays": frame["_arrays"]}
        return result
