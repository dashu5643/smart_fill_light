"""智能补光 · 单帧策略引擎：一张图 → 打光策略 + 灯板参数。

    from fill_light import FillLightPipeline, imread_rgb
    with FillLightPipeline() as pipe:
        res = pipe.run(imread_rgb("photo.jpg"), distance_cm=60)
    res.light_params      # 灯型、两路颜色、亮度
    res.frame.leds        # 216 颗灯珠的 RGBW
"""
from .config import load_config
from .images import imread_rgb
from .pipeline import FillLightPipeline
from .result import Result

__all__ = ["FillLightPipeline", "Result", "imread_rgb", "load_config"]
__version__ = "0.2.0"
