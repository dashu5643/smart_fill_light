"""组装下发内容：坐标重排 + 按台架电流上限等比缩放。"""
from __future__ import annotations

from .mapping import frame_to_pixels


def estimate_current_a(px: list[list[float]], ss) -> float:
    """按台架工具给的每通道满亮电流估算整灯电流（A）。"""
    ma = ss.I_CH_MA
    return sum(p[0] / 100 * ma["r"] + p[1] / 100 * ma["g"] + p[2] / 100 * ma["b"] + p[3] / 100 * ma["w"]
               for p in px) / 1000


def build_payload(leds: dict, caps: dict, ss, bcfg: dict) -> tuple[dict, dict]:
    """返回 (交给 ss2light.Light.apply 的内容, 电流信息)。超过上限时整体等比缩小，颜色比例不变。"""
    px = frame_to_pixels(leds, caps, bool(bcfg.get("swap_boards")), bool(bcfg.get("mirror_cols")),
                         bool(bcfg.get("flip_rows")))
    i0 = estimate_current_a(px, ss)
    limit = min(float(bcfg.get("max_current_a", 0.95)), float(ss.I_BENCH_MAX_A))
    scale = 1.0
    if i0 > limit:
        scale = limit / i0
        px = [[round(x * scale, 2) for x in p] for p in px]
    payload = {
        "schema": ss.SCHEMA,
        "on": True,
        "intensity": 100,                                     # 亮度已经乘进逐颗值里
        "zones": {p["name"]: {"intensity": 100} for p in caps["panels"]},   # 清掉之前残留的面板亮度
        "pixels": px,
    }
    info = {"current_a_algo": round(i0, 3), "current_a_sent": round(estimate_current_a(px, ss), 3),
            "bench_scale": round(scale, 3), "limit_a": limit}
    return payload, info
