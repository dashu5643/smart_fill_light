"""修饰项：在选好颜色的策略上定亮度，并按光线方向和瑕疵做补偿（原地修改 Strategy）。"""
from __future__ import annotations

from ..analysis.types import Analysis, FaceAnalysis
from .palette import named
from .types import Strategy

FAR_FACE_WIDTH_RATIO = 0.15      # 脸宽占比小于此值视为离得远，亮度加一点
DEFAULT_FACE_L = 50.0


def apply_modifiers(cfg: dict, st: Strategy, an: Analysis, fa: FaceAnalysis, multi: bool) -> None:
    """fa = 主脸；multi = 按多人处理（不做只针对单人的侧光补偿和眼周修饰）。"""
    b, d = cfg["brightness"], cfg["direction"]
    skin = fa.skin

    # 1) 基础亮度：有色光污染 → 满亮；否则脸越暗越亮
    if an.cast.level != "none":
        val = b["polluted"]
        st.reasons.append("有色光污染：补光满亮（T1 参数均在 V=100 下实测）")
    else:
        val = b["base"] + (b["target_face_L"] - (skin.face_L if skin else DEFAULT_FACE_L)) * b["per_L"]
        if fa.obs.width_ratio < FAR_FACE_WIDTH_RATIO:
            val += b["far_face_bonus"]
    if skin and skin.reliable and skin.group == "deep" and val < b["deep_min"]:
        val = b["deep_min"]
        st.reasons.append("深肤色：保证亮度足够")

    # 2) 光线方向
    if fa.direction == "back":
        val = b["backlight"]
        st.reasons.append("逆光：补光做主力，亮度拉满")
        st.warnings.append("逆光：建议调整拍摄位置（环境光分类标准 §4.5）")
    elif fa.direction.startswith("side") and not multi:
        # 图像左侧脸暗 → 补右板（灯板按观察者视角）；输入是镜像图时相反
        board = "right" if fa.direction == "side_img_l_dark" else "left"
        if cfg["hardware"].get("image_mirrored"):
            board = "left" if board == "right" else "right"
        st.side_boost = {board: d["side_boost"]}
        st.reasons.append(f"侧光：暗侧（{'左' if board == 'left' else '右'}板）补光 +{int(d['side_boost'] * 100)}%")
        val = min(val, 1.0 / (1.0 + d["side_boost"]))           # 给暗侧增亮留余量，避免被 1.0 截断
    elif fa.direction == "top":
        st.top_gradient = d["top_gradient"]
        st.reasons.append("顶光：下方行补光加强，均匀化下颌阴影")
    st.brightness = round(min(max(val, b["min"]), b["max"]), 3)

    # 3) 瑕疵
    if not multi and "dark_circle" in fa.blemishes and skin and st.light_type != "dual":
        patch = cfg["blemish"]["eye_patch"][skin.group]
        st.eye_patch = named(cfg, patch["color"])
        st.eye_patch_extra_val = patch["extra_val"]
        st.reasons.append(f"检测到眼周暗沉（ΔL={fa.blemish_detail.get('undereye_delta_L')}），眼周行带 {patch['color']} 提亮")
    if "redness" in fa.blemishes:
        st.warnings.append("检测到面中泛红；第一代分区混光效果有限，仅提示（可选青色中和）")
