"""结果可视化：原图标注 | 灯板预览（左板-屏幕-右板）| 策略说明。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import color as C
from .perception import FACE_OVAL

_FONT_CANDIDATES = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                    "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]


def _font(size: int):
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _led_rgb(rgbw: np.ndarray) -> tuple[int, int, int]:
    r, g, b, w = [x / 255 for x in rgbw]
    c = np.clip(np.array([r + w, g + w, b + w]), 0, 1)
    # 显示时把暗值抬一点，避免看不清
    c = 0.15 + 0.85 * c
    return tuple(int(x * 255) for x in c)


def render_report(rgb: np.ndarray, result: dict, out_path: str) -> None:
    it = result["_internal"]
    obs, an, st, arrays = it["obs"], it["analysis"], it["strategy"], it["frame_arrays"]
    H = 720
    img = Image.fromarray(rgb)
    s = H / img.height
    img = img.resize((int(img.width * s), H))
    d = ImageDraw.Draw(img)
    for i, f in enumerate(obs.faces):
        pts = f.landmarks[FACE_OVAL, :2] * s
        d.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=(255, 255, 255), width=2)
        for name, reg in f.regions.items():
            pass
        from .perception import ROI_POINTS
        r = max(2, int(f.box[2] * 0.07 * s))
        for name, idxs in ROI_POINTS.items():
            for idx in idxs:
                c = (f.landmarks[idx[0], :2] * 0.45 + f.landmarks[idx[1], :2] * 0.55) if isinstance(idx, tuple) \
                    else f.landmarks[idx, :2]
                c = c * s
                rr = r if "undereye" not in name else max(2, int(r * 0.6))
                d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], outline=(0, 255, 160), width=2)

    # 灯板预览
    cell, gap = 22, 3
    R, Cc = arrays["left"].shape[:2]
    bw = Cc * (cell + gap)
    screen_w = 120
    pw = bw * 2 + screen_w + 40
    panel = Image.new("RGB", (pw, H), (245, 245, 245))
    pd = ImageDraw.Draw(panel)
    top = 70
    f_s, f_m, f_l = _font(14), _font(16), _font(20)
    pd.text((20, 20), "灯板预览（观察者视角）", font=f_l, fill=(30, 30, 30))
    for bi, board in enumerate(["left", "right"]):
        x0 = 20 if board == "left" else 20 + bw + screen_w
        for r in range(R):
            for c in range(Cc):
                col = _led_rgb(arrays[board][r, c])
                x, y = x0 + c * (cell + gap), top + r * (cell + gap)
                pd.rounded_rectangle([x, y, x + cell, y + cell], 4, fill=col, outline=(200, 200, 200))
        pd.text((x0, top + R * (cell + gap) + 6), "左板" if board == "left" else "右板", font=f_m, fill=(60, 60, 60))
    sx = 20 + bw + 10
    pd.rectangle([sx, top + 40, sx + screen_w - 20, top + R * (cell + gap) - 40], fill=(40, 40, 40))
    pd.text((sx + 22, top + 180), "屏幕", font=f_m, fill=(220, 220, 220))
    # 行带标注
    bands = result["frame"]["layout"]
    yb = top + R * (cell + gap) + 40
    sp = result["light_params"]
    lt_zh = {"sandwich": "夹心灯", "dual": "内外圈互补", "mono": "单色灯"}[st.light_type]
    lines = [f"灯型：{lt_zh}   亮度 {sp['brightness_pct']}%"]
    if st.light_type == "dual":
        lines.append(f"外圈：H{st.outer.hue:.0f}° S{st.outer.sat * 100:.0f}%  {st.outer.hex}")
        lines.append(f"内圈：H{st.inner.hue:.0f}° S{st.inner.sat * 100:.0f}%  {st.inner.hex}")
    else:
        lines.append(f"主光：{st.inner.name}  H{st.inner.hue:.0f}° S{st.inner.sat * 100:.0f}%  {st.inner.hex}")
    if st.light_type == "sandwich":
        lines.append(f"轮廓：{st.outer.name}  H{st.outer.hue:.0f}° S{st.outer.sat * 100:.0f}%  {st.outer.hex}")
    if st.eye_patch:
        lines.append(f"眼周修饰：{st.eye_patch.name} +{int(st.eye_patch_extra_val * 100)}%")
    lines.append(f"电流钳位系数 {result['frame']['current_clamp_scale']}")
    for i, t in enumerate(lines):
        pd.text((20, yb + i * 24), t, font=f_s, fill=(40, 40, 40))
    # 色块
    for i, h in enumerate([st.outer, st.inner] if st.light_type == "dual" else
                          ([st.inner, st.outer] if st.light_type == "sandwich" else [st.inner])):
        rgb_ = tuple(int(x * 255) for x in C.hsv_rgb(h.hue, h.sat, 1.0))
        pd.rounded_rectangle([pw - 70, yb + 24 + i * 26, pw - 30, yb + 44 + i * 26], 4, fill=rgb_, outline=(150, 150, 150))

    # 文字说明
    tw = 520
    txt = Image.new("RGB", (tw, H), (255, 255, 255))
    td = ImageDraw.Draw(txt)
    p = result["perception"]
    env = p["environment"]
    y = 20
    td.text((20, y), f"场景结论：{st.scenario}", font=f_l, fill=(20, 20, 20)); y += 36
    mf = p["main_face"]
    el = result["env_light"]
    info = [
        f"人脸：{p['faces']} 张（有效 {p['usable_faces']}）",
        f"环境光：{el['env_light_category']} {el['env_light_name']}  H{env['cast_hue']:.0f}°  "
        f"S={env['pollution_s_pct']}%  {el['pollution_level']}" + ("  多色" if el["multi_color_mode"] else ""),
        f"  人脸估计 H{env['face_estimate']['hue']}° S{env['face_estimate']['s_pct']}% · 背景估计 H{env['background_estimate']['hue']}° S{env['background_estimate']['s_pct']}%",
        f"环境基调：{env['env_tone']}   色温≈{env['cct_k'] or '-'}K   彩色占比 {env['colored_share']:.0%}",
    ]
    if mf:
        sk = (f"肤色：{mf['skin_level']} {mf['skin_group_zh']}  ITA {mf['ita']}°  底调 {mf['undertone']}"
              if mf["skin_reliable"] else "肤色：强色光下不可判")
        info += [sk + f"   距离≈{mf['distance_cm']}cm",
                 f"光线方向：{mf['light_direction']}   瑕疵：{', '.join(mf['blemishes']) or '无'}"]
    for t in info:
        td.text((20, y), t, font=f_s, fill=(60, 60, 60)); y += 22
    y += 10
    td.text((20, y), "决策依据", font=f_m, fill=(20, 20, 20)); y += 26
    for t in st.reasons:
        for chunk in _wrap(t, 34):
            td.text((26, y), chunk, font=f_s, fill=(40, 40, 40)); y += 20
        y += 2
    if st.warnings:
        y += 6
        for t in st.warnings:
            for chunk in _wrap("! " + t, 34):
                td.text((26, y), chunk, font=f_s, fill=(180, 80, 0)); y += 20
    if st.alternatives:
        y += 8
        td.text((20, y), "备选：", font=f_m, fill=(20, 20, 20)); y += 26
        for a in st.alternatives:
            rgb_ = tuple(int(x * 255) for x in C.hsv_rgb(a.hue, a.sat, 1.0))
            td.rounded_rectangle([26, y, 50, y + 18], 3, fill=rgb_, outline=(150, 150, 150))
            td.text((58, y), f"{a.name} H{a.hue:.0f}° S{a.sat * 100:.0f}%", font=f_s, fill=(40, 40, 40)); y += 24

    canvas = Image.new("RGB", (img.width + pw + tw, H), (255, 255, 255))
    canvas.paste(img, (0, 0))
    canvas.paste(panel, (img.width, 0))
    canvas.paste(txt, (img.width + pw, 0))
    canvas.save(out_path, quality=90)


def _wrap(t: str, n: int):
    return [t[i:i + n] for i in range(0, len(t), n)] or [""]
