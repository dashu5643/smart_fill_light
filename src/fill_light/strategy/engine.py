"""决策层入口：Analysis → Strategy。只做分流，具体规则在各场景模块里。"""
from __future__ import annotations

from ..analysis.types import Analysis
from .dark_scene import dark_scene_strategy
from .modifiers import apply_modifiers
from .multi_face import multi_face_strategy
from .object_mode import object_strategy
from .scene_priority import scene_priority_strategy
from .skin_priority import skin_priority_strategy
from .types import Strategy

MODES = ("skin_priority", "scene_priority")


def decide(an: Analysis, cfg: dict, mode: str = "skin_priority", object_type: str | None = None) -> Strategy:
    """mode：skin_priority 肤色优先（默认）/ scene_priority 场景光优先。object_type：无人脸时的物品类别。"""
    usable = [f for f in an.faces if f.usable]
    if an.exposure == "dark":
        return dark_scene_strategy(cfg, an)
    if not an.faces:
        return object_strategy(cfg, an, object_type)
    if len(usable) >= 2:
        st = multi_face_strategy(cfg, an, usable, mode)
    elif not usable:
        st = multi_face_strategy(cfg, an, [], mode)
        st.warnings.append("人脸过小，采用全肤色安全色")
    elif mode == "scene_priority":
        st = scene_priority_strategy(cfg, an, usable[0])
    else:
        st = skin_priority_strategy(cfg, an, usable[0])
    if usable:
        apply_modifiers(cfg, st, an, usable[0], multi=len(usable) >= 2 and st.mode != "multi_same")
    return st
