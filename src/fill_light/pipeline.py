"""单帧补光流水线：一张图 → 打光策略 + 灯板参数。

    图片 → ① 感知 → ② 分析 → ③ 决策 → ④ 执行 → Result
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import analyze
from .config import load_config
from .hardware import render_frame
from .perception import Perceiver
from .result import Result, build_result
from .strategy import decide


class FillLightPipeline:
    """人脸模型只在创建时加载一次，之后可以反复 run()。用完 close()，或用 with。"""

    def __init__(self, cfg_path: str | Path | None = None) -> None:
        self.cfg = load_config(cfg_path)
        self.perceiver = Perceiver(self.cfg)

    def run(self, rgb: np.ndarray, mode: str = "skin_priority", object_type: str | None = None,
            distance_cm: float | None = None) -> Result:
        """
        rgb          H×W×3 的 RGB 图（uint8）
        mode         skin_priority 肤色优先（默认）/ scene_priority 场景光优先
        object_type  无人脸时的物品类别（configs 的 object_mode 一节），可不填
        distance_cm  灯板到人脸的距离；不填按 configs 的 distance 一节（默认 60cm）
        """
        obs = self.perceiver.observe(rgb)
        an = analyze(obs, self.cfg, distance_cm)
        st = decide(an, self.cfg, mode=mode, object_type=object_type)
        frame = render_frame(st, self.cfg)
        return build_result(mode, obs, an, st, frame, self.cfg)

    def close(self) -> None:
        self.perceiver.close()

    def __enter__(self) -> "FillLightPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
