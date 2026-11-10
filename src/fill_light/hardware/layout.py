"""SureShot 双灯板的几何：2 块板 × 18 行 × 6 列，观察者视角（站在灯前、面向灯板）。

行：row 0 = 最上。列：两块板都是从左到右编号，所以
左板 col 0 = 外侧 → col 5 = 内侧（靠屏幕）；右板 col 0 = 内侧 → col 5 = 外侧。
"""
from __future__ import annotations

from dataclasses import dataclass

CHANNELS = "RGBW"
COL_ORDER = {"left": "x0=外侧→x5=内侧", "right": "x0=内侧→x5=外侧"}


@dataclass(frozen=True)
class BoardLayout:
    boards: tuple[str, ...]
    rows: int
    cols: int

    @classmethod
    def from_config(cls, cfg: dict) -> "BoardLayout":
        hw = cfg["hardware"]
        return cls(tuple(hw["boards"]), int(hw["rows"]), int(hw["cols"]))

    @property
    def channel_count(self) -> int:
        """全部通道数（216 颗 × 4 = 864）。"""
        return len(self.boards) * self.rows * self.cols * len(CHANNELS)

    def is_inner(self, board: str, col: int, inner_cols: int) -> bool:
        """该列是否属于靠屏幕的 inner_cols 列。"""
        return col >= self.cols - inner_cols if board == "left" else col < inner_cols

    def describe(self, inner_cols: int) -> dict:
        return {"boards": list(self.boards), "rows": self.rows, "cols": self.cols,
                "col_order": COL_ORDER, "inner_cols": inner_cols, "channel_order": CHANNELS}
