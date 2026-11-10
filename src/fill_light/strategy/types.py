"""决策层输出：一套与硬件无关的补光策略。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .. import color as C


@dataclass
class LightColor:
    name: str
    hue: float                     # 0-360
    sat: float                     # 0-1

    @property
    def hex(self) -> str:
        return C.hsv_hex(self.hue, self.sat, 1.0)


@dataclass
class Strategy:
    """灯型 light_type：
      mono      单色：两块灯板同色（inner = outer）
      sandwich  夹心：靠屏幕的列 = 面部主光 inner，外侧列 = 轮廓光 outer
      dual      内外圈互补：靠屏幕的列 = 内圈 inner（面积小、更浓），其余 = 外圈 outer
    """
    mode: str                      # skin_priority / scene_priority / multi_safe / multi_same / multi_scene / object / dark
    scenario: str                  # 场景结论（中文）
    light_type: str
    inner: LightColor
    outer: LightColor
    brightness: float              # 0-1
    side_boost: dict = field(default_factory=dict)       # 侧光：{暗侧灯板: 增亮比例}
    top_gradient: float = 0.0                            # 顶光：从上到下的增亮幅度
    eye_patch: LightColor | None = None                  # 眼周行带的修饰色
    eye_patch_extra_val: float = 0.0
    alternatives: list[LightColor] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sat_factors: dict = field(default_factory=dict)      # 饱和度计算过程（基准 × 污染 × 距离 × 多色）

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("inner", "outer", "eye_patch"):
            if d.get(k):
                d[k]["hex"] = getattr(self, k).hex
        for i, a in enumerate(self.alternatives):
            d["alternatives"][i]["hex"] = a.hex
        return d
