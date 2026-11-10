"""报告图的三块面板：原图标注 | 灯板预览 | 文字说明。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import color as C
from ..labels import LIGHT_TYPE_ZH
from ..perception.face import FACE_OVAL, roi_centers, roi_radius
from ..result import Result
from ..strategy.types import LightColor

HEIGHT = 720
_FONT_CANDIDATES = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                    "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]


def _font(size: int):
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _swatch(c: LightColor) -> tuple[int, ...]:
    return tuple(int(x * 255) for x in C.hsv_rgb(c.hue, c.sat, 1.0))


def _led_rgb(rgbw: np.ndarray) -> tuple[int, ...]:
    """RGBW → 屏幕显示色（W 叠加到三色上；暗值抬一点，避免看不清）。"""
    r, g, b, w = [x / 255 for x in rgbw]
    c = 0.15 + 0.85 * np.clip(np.array([r + w, g + w, b + w]), 0, 1)
    return tuple(int(x * 255) for x in c)


def _wrap(t: str, n: int) -> list[str]:
    return [t[i:i + n] for i in range(0, len(t), n)] or [""]


def photo_panel(rgb: np.ndarray, res: Result) -> Image.Image:
    """原图 + 人脸轮廓（白）+ 采样区域（绿圈）。"""
    img = Image.fromarray(rgb)
    s = HEIGHT / img.height
    img = img.resize((int(img.width * s), HEIGHT))
    d = ImageDraw.Draw(img)
    ratio = res.observation.roi_radius_ratio
    for f in res.observation.faces:
        pts = f.landmarks[FACE_OVAL, :2] * s
        d.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=(255, 255, 255), width=2)
        base = max(2, int(f.box[2] * ratio * s))
        for name, centers in roi_centers(f.landmarks).items():
            rr = roi_radius(name, base)
            for c in centers:
                c = c * s
                d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], outline=(0, 255, 160), width=2)
    return img


def board_panel(res: Result) -> Image.Image:
    """左板 - 屏幕 - 右板 的逐珠预览 + 灯光参数。"""
    st, arrays = res.strategy, res.frame.arrays
    cell, gap, screen_w, top = 22, 3, 120, 70
    rows, cols = arrays["left"].shape[:2]
    bw = cols * (cell + gap)
    pw = bw * 2 + screen_w + 40
    panel = Image.new("RGB", (pw, HEIGHT), (245, 245, 245))
    d = ImageDraw.Draw(panel)
    f_s, f_m, f_l = _font(14), _font(16), _font(20)
    d.text((20, 20), "灯板预览（观察者视角）", font=f_l, fill=(30, 30, 30))
    for board in ("left", "right"):
        x0 = 20 if board == "left" else 20 + bw + screen_w
        for r in range(rows):
            for c in range(cols):
                x, y = x0 + c * (cell + gap), top + r * (cell + gap)
                d.rounded_rectangle([x, y, x + cell, y + cell], 4, fill=_led_rgb(arrays[board][r, c]),
                                    outline=(200, 200, 200))
        d.text((x0, top + rows * (cell + gap) + 6), "左板" if board == "left" else "右板", font=f_m, fill=(60, 60, 60))
    sx = 20 + bw + 10
    d.rectangle([sx, top + 40, sx + screen_w - 20, top + rows * (cell + gap) - 40], fill=(40, 40, 40))
    d.text((sx + 22, top + 180), "屏幕", font=f_m, fill=(220, 220, 220))

    y0 = top + rows * (cell + gap) + 40
    lines = [f"灯型：{LIGHT_TYPE_ZH[st.light_type]}   亮度 {res.light_params['brightness_pct']}%"]
    if st.light_type == "dual":
        lines.append(f"外圈：H{st.outer.hue:.0f}° S{st.outer.sat * 100:.0f}%  {st.outer.hex}")
        lines.append(f"内圈：H{st.inner.hue:.0f}° S{st.inner.sat * 100:.0f}%  {st.inner.hex}")
    else:
        lines.append(f"主光：{st.inner.name}  H{st.inner.hue:.0f}° S{st.inner.sat * 100:.0f}%  {st.inner.hex}")
    if st.light_type == "sandwich":
        lines.append(f"轮廓：{st.outer.name}  H{st.outer.hue:.0f}° S{st.outer.sat * 100:.0f}%  {st.outer.hex}")
    if st.eye_patch:
        lines.append(f"眼周修饰：{st.eye_patch.name} +{int(st.eye_patch_extra_val * 100)}%")
    lines.append(f"电流钳位系数 {res.frame.current_clamp_scale}")
    for i, t in enumerate(lines):
        d.text((20, y0 + i * 24), t, font=f_s, fill=(40, 40, 40))
    swatches = {"dual": [st.outer, st.inner], "sandwich": [st.inner, st.outer], "mono": [st.inner]}[st.light_type]
    for i, c in enumerate(swatches):
        d.rounded_rectangle([pw - 70, y0 + 24 + i * 26, pw - 30, y0 + 44 + i * 26], 4, fill=_swatch(c),
                            outline=(150, 150, 150))
    return panel


def text_panel(res: Result) -> Image.Image:
    """场景结论、感知结果、决策依据、警告、备选。"""
    st, p, el = res.strategy, res.perception, res.env_light
    env, mf = p["environment"], p["main_face"]
    f_s, f_m, f_l = _font(14), _font(16), _font(20)
    panel = Image.new("RGB", (520, HEIGHT), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    y = 20
    d.text((20, y), f"场景结论：{st.scenario}", font=f_l, fill=(20, 20, 20)); y += 36
    info = [
        f"人脸：{p['faces']} 张（有效 {p['usable_faces']}）",
        f"环境光：{el['env_light_category']} {el['env_light_name']}  H{env['cast_hue']:.0f}°  "
        f"S={env['pollution_s_pct']}%  {el['pollution_level']}" + ("  多色" if el["multi_color_mode"] else ""),
        f"  人脸估计 H{env['face_estimate']['hue']}° S{env['face_estimate']['s_pct']}% · "
        f"背景估计 H{env['background_estimate']['hue']}° S{env['background_estimate']['s_pct']}%",
        f"环境基调：{env['env_tone']}   色温≈{env['cct_k'] or '-'}K   彩色占比 {env['colored_share']:.0%}",
    ]
    if mf:
        skin = (f"肤色：{mf['skin_level']} {mf['skin_group_zh']}  ITA {mf['ita']}°  底调 {mf['undertone']}"
                if mf["skin_reliable"] else "肤色：强色光下不可判")
        info += [skin + f"   距离≈{mf['distance_cm']}cm",
                 f"光线方向：{mf['light_direction']}   瑕疵：{', '.join(mf['blemishes']) or '无'}"]
    for t in info:
        d.text((20, y), t, font=f_s, fill=(60, 60, 60)); y += 22
    y += 10
    d.text((20, y), "决策依据", font=f_m, fill=(20, 20, 20)); y += 26
    for t in st.reasons:
        for chunk in _wrap(t, 34):
            d.text((26, y), chunk, font=f_s, fill=(40, 40, 40)); y += 20
        y += 2
    if st.warnings:
        y += 6
        for t in st.warnings:
            for chunk in _wrap("! " + t, 34):
                d.text((26, y), chunk, font=f_s, fill=(180, 80, 0)); y += 20
    if st.alternatives:
        y += 8
        d.text((20, y), "备选：", font=f_m, fill=(20, 20, 20)); y += 26
        for a in st.alternatives:
            d.rounded_rectangle([26, y, 50, y + 18], 3, fill=_swatch(a), outline=(150, 150, 150))
            d.text((58, y), f"{a.name} H{a.hue:.0f}° S{a.sat * 100:.0f}%", font=f_s, fill=(40, 40, 40)); y += 24
    return panel
