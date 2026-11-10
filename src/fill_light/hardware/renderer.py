"""策略 → 灯板帧。

灯型决定哪些列用哪个颜色：
  mono      全部同色
  sandwich  靠屏幕 inner_cols 列 = 面部主光，其余 = 轮廓光
  dual      靠屏幕 ring_inner_cols 列 = 内圈，其余 = 外圈
再叠加：侧光（暗侧整板增亮）、顶光（从上到下渐亮）、眼周修饰（眼部行带的内侧列换色并提亮）。
"""
from __future__ import annotations

import numpy as np

from ..strategy.types import Strategy
from .frame import LedFrame
from .layout import BoardLayout
from .rgbw import clamp_total_duty, hsv_to_rgbw


def render_frame(st: Strategy, cfg: dict) -> LedFrame:
    hw = cfg["hardware"]
    lay = BoardLayout.from_config(cfg)
    inner_cols = int(hw["ring_inner_cols"] if st.light_type == "dual" else hw["inner_cols"])
    eye_first, eye_last = hw["row_bands"]["eye"]

    boards = {}
    for board in lay.boards:
        arr = np.zeros((lay.rows, lay.cols, 4))
        boost = 1.0 + float(st.side_boost.get(board, 0.0))
        for r in range(lay.rows):
            grad = 1.0 + st.top_gradient * (r / (lay.rows - 1) - 0.5) * 2 if st.top_gradient else 1.0
            for c in range(lay.cols):
                inner = lay.is_inner(board, c, inner_cols)
                color = st.inner if (inner or st.light_type == "mono") else st.outer
                val = st.brightness * boost * grad
                if st.eye_patch is not None and inner and eye_first <= r <= eye_last:
                    color = st.eye_patch
                    val = val + st.eye_patch_extra_val
                arr[r, c] = hsv_to_rgbw(color.hue, color.sat, min(val, 1.0))
        boards[board] = arr

    scale = clamp_total_duty(boards, float(hw["max_total_duty"]) * lay.channel_count)
    arrays = {k: np.round(v * 255).astype(int) for k, v in boards.items()}
    duty = float(sum(v.sum() for v in arrays.values()) / (255 * lay.channel_count))
    return LedFrame(arrays=arrays, layout=lay.describe(inner_cols),
                    current_clamp_scale=round(scale, 3), duty=round(duty, 3))
