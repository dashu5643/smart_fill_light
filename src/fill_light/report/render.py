"""把三块面板拼成一张报告图。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..result import Result
from .panels import HEIGHT, board_panel, photo_panel, text_panel


def render_report(rgb: np.ndarray, res: Result, out_path: str | Path) -> None:
    """原图标注 | 灯板预览 | 文字说明，横向拼成一张图。"""
    panels = [photo_panel(rgb, res), board_panel(res), text_panel(res)]
    canvas = Image.new("RGB", (sum(p.width for p in panels), HEIGHT), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width
    canvas.save(str(out_path), quality=90)
