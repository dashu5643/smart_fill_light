"""测试图案。"""
from __future__ import annotations

IDENTIFY_EXPECTED = "左板红、右板蓝；靠屏幕的一列白；最上一行绿"


def identify_leds(rows: int = 18, cols: int = 6) -> dict:
    """方位测试帧（观察者视角）：见 IDENTIFY_EXPECTED。实物不一致就改 configs/bench.yaml 的方位开关。"""
    out = {}
    for board, base in (("left", [60, 0, 0, 0]), ("right", [0, 0, 60, 0])):
        arr = [[list(base) for _ in range(cols)] for _ in range(rows)]
        inner = cols - 1 if board == "left" else 0
        for r in range(rows):
            arr[r][inner] = [0, 0, 0, 60]
        for c in range(cols):
            arr[0][c] = [0, 60, 0, 0]
        out[board] = arr
    return out
