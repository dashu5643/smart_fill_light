"""多人（《多人场景补光方案》）：灯板只能出一种光，不分人。

两人且肤色相同 → 按单人策略；其他情况 → 按环境光选一个对所有肤色都安全的颜色。
"""
from __future__ import annotations

from ..analysis.types import Analysis, FaceAnalysis
from ..labels import FAMILY_ZH, GROUP_ZH
from .complement import distance_scale
from .palette import clip
from .scene_priority import scene_priority_strategy
from .skin_priority import skin_priority_strategy
from .types import LightColor, Strategy


def multi_face_strategy(cfg: dict, an: Analysis, faces: list[FaceAnalysis], mode: str) -> Strategy:
    """faces = 可用的人脸；为空表示检测到的脸都太小。"""
    mc = cfg["multi_face"]
    cast = an.cast
    groups = sorted({f.skin.group for f in faces if f.skin})
    all_reliable = bool(faces) and all(f.skin and f.skin.reliable for f in faces)
    reasons = [f"检测到 {len(an.faces)} 张人脸（有效 {len(faces)}），肤色组："
               f"{', '.join(GROUP_ZH[g] for g in groups) or '未知'}"
               + ("" if all_reliable or not faces else "（强色光下不可靠）"),
               "灯板只能输出一种光：第一代不做分人/分区"]

    if mode == "scene_priority":
        st = scene_priority_strategy(cfg, an, faces[0] if faces else None)
        st.inner.sat = min(st.inner.sat, mc["scene_sat_cap"])
        st.mode, st.scenario = "multi_scene", "多人 · 场景光优先（邻近低饱和色）"
        st.reasons = reasons + st.reasons
        return st

    if mc["same_group_as_single"] and len(faces) == 2 and len(groups) == 1 and all_reliable:
        st = skin_priority_strategy(cfg, an, faces[0])
        st.mode = "multi_same"
        st.scenario = "双人同肤色 · " + st.scenario
        st.reasons = reasons + ["两人肤色相同 → 用该肤色专属最佳色"] + st.reasons
        return st

    # 安全交集色：中度以上污染按环境光选色，否则只轻微提亮
    key = cast.family if cast.level in ("medium", "heavy") else "none"
    if cast.level == "light":
        reasons.append(f"{FAMILY_ZH[cast.family]}污染轻（S={cast.s_pct}% < 30），危害低 → 安全色轻微提亮")
    e = mc["safe_by_env"].get(key, mc["safe_by_env"]["none"])
    dist = min((f.distance_cm for f in faces if f.distance_cm), default=None)
    k_d = distance_scale(cfg, dist)
    pick = clip(LightColor(e["name"], float(e["color"][0]), e["color"][1] / 100 * k_d))
    if key != "none":
        reasons.append(f"环境光校正优先：{FAMILY_ZH[cast.family]}（S={cast.s_pct}%）→ 多人安全色 {e['name']} "
                       f"H{e['color'][0]}° S{e['color'][1]}%（60cm）")
    else:
        reasons.append(f"无明显污染 → 全肤色安全色 {e['name']} 轻微提亮")
    worst = next((g for g in ("deep", "medium") if g in groups), groups[0] if groups else None)
    if worst:
        reasons.append(f"最坏情况肤色：{GROUP_ZH[worst]}；暖色区低饱和对所有肤色安全")
    if k_d != 1:
        reasons.append(f"距离 ≈{dist}cm，饱和度 × {k_d:.2f}")
    alts = [clip(LightColor(a["name"], float(a["color"][0]), a["color"][1] / 100 * k_d))
            for a in mc["safe_alternatives"] if a["name"] != e["name"]]
    return Strategy(mode="multi_safe", scenario="多人 · 全肤色安全交集色", light_type="mono", inner=pick,
                    outer=pick, brightness=0.0, reasons=reasons, alternatives=alts,
                    sat_factors={"distance_cm": dist, "distance_k": round(k_d, 3)})
