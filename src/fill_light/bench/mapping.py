"""坐标重排：算法帧（板 / 行 / 列）→ 台架的灯珠编号顺序。

算法帧：leds[板][行][列] = [R,G,B,W] 0-255，观察者视角，row 0 = 最上，列从左到右。
台架：  pixels = 216 组 [r,g,b,w] 0-100，按灯珠编号 1..216；每块面板的编号走向由灯板配置
        （lightboard-v1.uci）的 first_index / scan / col_dir / row_dir 描述。
"""
from __future__ import annotations

import numpy as np


def panel_index(p: dict, col: int, row: int) -> int:
    """面板内 (col, row)（面向灯板：col 0 = 最左，row 0 = 最上）→ 全局灯珠编号（0 起）。"""
    cols, rows = int(p["cols"]), int(p["rows"])
    c = col if p.get("col_dir", "left_to_right") == "left_to_right" else cols - 1 - col
    r = row if p.get("row_dir", "top_to_bottom") == "top_to_bottom" else rows - 1 - row
    if p.get("scan", "column_major") == "column_major":
        k = c * rows + r
    else:                                   # row_major（V1 灯板没有，留着兼容）
        k = r * cols + c
    return int(p["first_index"]) - 1 + k


def frame_to_pixels(leds: dict, caps: dict, swap_boards: bool = False, mirror_cols: bool = False,
                    flip_rows: bool = False) -> list[list[float]]:
    """算法帧 → 台架 pixels。三个开关是实物方位校正（见 configs/bench.yaml）。"""
    n = int(caps["pixels"])
    px = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    filled = 0
    for p in caps["panels"]:
        name = p["name"]
        src = {"left": "right", "right": "left"}[name] if swap_boards else name
        if src not in leds:
            raise ValueError(f"算法帧里没有 {src} 板")
        arr = np.asarray(leds[src], dtype=float)                 # rows × cols × 4
        rows, cols = int(p["rows"]), int(p["cols"])
        if arr.shape[:2] != (rows, cols):
            raise ValueError(f"{src} 板尺寸 {arr.shape[:2]} 与台架 {name} 面板 {(rows, cols)} 不一致")
        for row in range(rows):
            for col in range(cols):
                v = arr[rows - 1 - row if flip_rows else row, cols - 1 - col if mirror_cols else col]
                px[panel_index(p, col, row)] = [round(float(x) / 255 * 100, 2) for x in v]
                filled += 1
    if filled != n:
        raise ValueError(f"面板合计 {filled} 颗 ≠ 灯珠数 {n}")
    return px
