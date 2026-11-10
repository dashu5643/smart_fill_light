"""光线方向：逆光 / 侧光 / 顶光 / 顺光，由面部各区域的亮度差判断。"""
from __future__ import annotations

from ..perception.types import EnvObs, FaceObs
from .types import SkinInfo

DEEP_SKIN_BACK_EXTRA = 30.0      # 深肤色本身就暗，逆光阈值放宽
BACK_MAX_FACE_L = 50             # 脸不暗就不算逆光
SIDE_MAX_YAW = 0.12              # 头转得太多时两颊亮度差不可信


def light_direction(f: FaceObs, skin: SkinInfo, env: EnvObs, cfg: dict) -> tuple[str, dict]:
    dcfg = cfg["direction"]
    L = lambda k: float(f.regions[k]["lab"][0]) if k in f.regions else None
    lL, rL, fL, cL = L("cheek_img_l"), L("cheek_img_r"), L("forehead"), L("chin")
    detail = {"cheek_img_l_L": lL, "cheek_img_r_L": rL, "forehead_L": fL, "chin_L": cL,
              "bg_p90_L": round(env.p90_L, 1)}
    back_delta = dcfg["back_delta_L"] + (DEEP_SKIN_BACK_EXTRA if skin.group == "deep" else 0.0)
    if (env.mean_L - skin.face_L > back_delta * 0.8 and env.p90_L - skin.face_L > back_delta
            and skin.face_L < BACK_MAX_FACE_L):
        return "back", detail
    if dcfg.get("skip_side_top_when_skin_unreliable", True) and not skin.reliable:
        # 强色光下两颊亮度差主要来自色光本身，不代表真实明暗
        detail["side_top_skipped"] = "肤色不可靠（强色光）"
        return "front", detail
    if lL is not None and rL is not None and abs(lL - rL) > dcfg["side_delta_L"] and abs(f.yaw_proxy) < SIDE_MAX_YAW:
        return ("side_img_l_dark" if lL < rL else "side_img_r_dark"), detail
    if fL is not None and cL is not None and fL - cL > dcfg["top_delta_L"]:
        return "top", detail
    return "front", detail
