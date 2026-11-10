"""分析层输出：决策要用的语义特征。"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..perception.types import EnvObs, FaceObs


@dataclass
class CastInfo:
    """环境光（色光污染）。"""
    hue: float                 # 污染光源色相（HSV 度）
    s_pct: float               # 污染程度 0-100
    level: str                 # none / light / medium / heavy
    family: str                # neutral / blue / green / purple / pink / yellow / cyan
    raw_family: str            # 不考虑等级时色相所属类别（level=none 时也保留，便于排查）
    source: str                # fused（人脸 + 背景）/ background
    face_hue: float | None = None
    face_s: float | None = None
    bg_hue: float | None = None
    bg_s: float | None = None
    multi_color: bool = False  # 多色光源（霓虹）模式
    secondary: str | None = None
    confidence: float = 0.0


@dataclass
class SkinInfo:
    lab_raw: list[float]
    lab_corrected: list[float] # 去偏色后
    ita: float
    level: str                 # L1-L6
    group: str                 # fair / medium / deep（不可靠时 = 推测值）
    undertone: str             # warm / cool / neutral / olive
    face_L: float
    reliable: bool = True
    unreliable_reason: str = ""


@dataclass
class FaceAnalysis:
    obs: FaceObs
    skin: SkinInfo | None
    direction: str             # front / back / side_img_l_dark / side_img_r_dark / top / unknown
    direction_detail: dict = field(default_factory=dict)
    blemishes: list[str] = field(default_factory=list)   # dark_circle / redness
    blemish_detail: dict = field(default_factory=dict)
    distance_cm: float | None = None
    usable: bool = True        # False = 脸太小或采样不足，不参与肤色决策


@dataclass
class Analysis:
    faces: list[FaceAnalysis]
    cast: CastInfo
    env_tone: str              # warm / cool / neutral
    env: EnvObs
    exposure: str = "normal"   # normal / dark / over
    notes: list[str] = field(default_factory=list)
