"""执行层：策略 → SureShot 双灯板逐珠帧（2 × 18 行 × 6 列 × RGBW，0-255）。

- 夹心灯 sandwich：靠屏幕的 inner_cols 列 = 面部主光；外侧列 = 轮廓光
- 内外圈 dual：靠屏幕的 ring_inner_cols 列 = 内圈（小面积、更浓）；其余列 = 外圈（大面积）
- 侧光补偿：暗侧整块灯板增亮
- 顶光渐变：从上到下线性增亮
- 眼周修饰：eye 行带内侧列换成修饰色并提亮
- RGB → RGBW：公共白分量交给 W 通道（提高显色、降低功耗）
- 电流钳位：总占空比超过上限时整体等比缩放
"""
from __future__ import annotations

import numpy as np

from . import color as C
from .strategy import Strategy


def _rgbw(hue: float, sat: float, val: float) -> np.ndarray:
    rgb = C.hsv_rgb(hue, sat, 1.0) * val
    w = float(rgb.min())
    return np.array([rgb[0] - w, rgb[1] - w, rgb[2] - w, w])


def _is_inner(board: str, col: int, cols: int, inner_cols: int) -> bool:
    # 左板 x0=外 → x5=内；右板 x0=内 → x5=外
    return col >= cols - inner_cols if board == "left" else col < inner_cols


def render_frame(st: Strategy, cfg: dict) -> dict:
    hw = cfg["hardware"]
    R, Cc = int(hw["rows"]), int(hw["cols"])
    inner_cols = int(hw["ring_inner_cols"] if st.light_type == "dual" else hw["inner_cols"])
    bands = hw["row_bands"]
    eye_rows = range(bands["eye"][0], bands["eye"][1] + 1)
    frame = {}
    for board in hw["boards"]:
        arr = np.zeros((R, Cc, 4))
        boost = 1.0 + float(st.side_boost.get(board, 0.0))
        for r in range(R):
            grad = 1.0 + st.top_gradient * (r / (R - 1) - 0.5) * 2 if st.top_gradient else 1.0
            for c in range(Cc):
                inner = _is_inner(board, c, Cc, inner_cols)
                h = st.inner if (inner or st.light_type == "mono") else st.outer
                val = st.brightness * boost * grad
                if st.eye_patch is not None and inner and r in eye_rows:
                    h = st.eye_patch
                    val = val + st.eye_patch_extra_val
                arr[r, c] = _rgbw(h.hue, h.sat, min(val, 1.0))
        frame[board] = arr

    total = sum(a.sum() for a in frame.values())
    cap = float(hw["max_total_duty"]) * 2 * R * Cc * 4
    scale = 1.0
    if total > cap:
        scale = cap / total
        for k in frame:
            frame[k] *= scale
    out = {k: np.round(v * 255).astype(int) for k, v in frame.items()}
    return {
        "format": "per_led_rgbw",
        "layout": {"boards": hw["boards"], "rows": R, "cols": Cc,
                   "col_order": {"left": "x0=外侧→x5=内侧", "right": "x0=内侧→x5=外侧"},
                   "inner_cols": inner_cols,
                   "channel_order": "RGBW"},
        "current_clamp_scale": round(scale, 3),
        "duty": round(float(sum(v.sum() for v in out.values()) / (255 * 2 * R * Cc * 4)), 3),
        "leds": {k: v.tolist() for k, v in out.items()},
        "_arrays": out,
    }


def summary_params(st: Strategy) -> dict:
    """精简协议：HSV 参数（给不支持逐珠的灯板，如导光板 / 夹心灯两路）。"""
    def h(x):
        return None if x is None else {"name": x.name, "hue": round(x.hue, 1), "sat": round(x.sat * 100, 1),
                                       "hex": x.hex}
    return {
        "light_type": st.light_type,
        "main": h(st.inner),            # mono=全板；sandwich=面部主光(内4列)；dual=内圈(内2列)
        "contour": h(st.outer),         # sandwich=轮廓光(外2列)；dual=外圈(外4列)
        "t1_format": (f"【{st.outer.hue:.0f}-{st.outer.sat * 100:.0f}***{st.inner.hue:.0f}-{st.inner.sat * 100:.0f}】"
                      if st.light_type == "dual" else None),
        "brightness_pct": round(st.brightness * 100, 1),
        "side_boost_pct": {k: round(v * 100) for k, v in st.side_boost.items()},
        "top_gradient_pct": round(st.top_gradient * 100),
        "eye_patch": h(st.eye_patch),
    }
