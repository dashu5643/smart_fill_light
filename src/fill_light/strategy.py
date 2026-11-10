"""决策层：语义特征 → 一套补光策略（与硬件无关）。

策略 = 灯型 + 两路颜色 + 亮度 + 修饰项（侧光补偿、顶光渐变、眼周修饰）。
灯型：
  mono      单色：两块灯板全部同色
  sandwich  夹心：靠屏幕 4 列 = 面部主光(inner)，外侧 2 列 = 轮廓光(outer)    —— 无污染肤色光 / 场景光
  dual      内外圈：靠屏幕 2 列 = 内圈(inner，小面积更浓)，其余 = 外圈(outer)  —— T1 互补色矫正
所有颜色、阈值来自 configs/strategy.yaml。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import color as C
from .analysis import Analysis, FaceAnalysis


@dataclass
class Hue:
    name: str
    hue: float
    sat: float

    @property
    def hex(self) -> str:
        return C.hsv_hex(self.hue, self.sat, 1.0)


@dataclass
class Strategy:
    mode: str                      # skin_priority / scene_priority / multi_safe / multi_same / object / dark
    scenario: str
    light_type: str                # mono / sandwich / dual
    inner: Hue                     # 面部主光（夹心内侧）/ 内圈（dual）/ 全板（mono）
    outer: Hue                     # 轮廓光（夹心外侧）/ 外圈（dual）/ = inner（mono）
    brightness: float              # 0-1
    side_boost: dict = field(default_factory=dict)
    top_gradient: float = 0.0
    eye_patch: Hue | None = None
    eye_patch_extra_val: float = 0.0
    alternatives: list[Hue] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sat_factors: dict = field(default_factory=dict)   # 饱和度计算过程（基准 × 污染 × 距离 × 多色）

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("inner", "outer", "eye_patch"):
            if d.get(k):
                d[k]["hex"] = getattr(self, k).hex
        for i, a in enumerate(self.alternatives):
            d["alternatives"][i]["hex"] = a.hex
        return d


FAMILY_ZH = {"blue": "蓝光", "green": "绿光", "purple": "紫光", "pink": "暖粉光", "yellow": "黄光",
             "cyan": "青光", "neutral": "中性白光", "unknown": "未知"}
GROUP_ZH = {"fair": "浅肤色", "medium": "黄/浅棕肤色", "deep": "深肤色"}
TONE_ZH = {"warm": "暖色系", "cool": "冷色系", "neutral": "中性"}
LEVEL_ZH = {"none": "无", "light": "轻度", "medium": "中度", "heavy": "重度"}


def _interp(x: float, pts) -> float:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(np.interp(x, xs, ys))


class StrategyEngine:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    # ---------------------------------------------------------------
    def _named(self, name: str, sat_tag: str | None = None, sat_mul: float = 1.0) -> Hue:
        p = self.cfg["palette"][name]
        sat = p["sat"]
        if sat_tag:
            sat = self.cfg["saturation_presets"][sat_tag]
        return self._cap(Hue(name, float(p["hue"]), float(sat * sat_mul)))

    def _cap(self, h: Hue) -> Hue:
        for rule in self.cfg.get("hue_family_sat_cap", []):
            if C.hue_in(h.hue, [rule["range"]]):
                h.sat = min(h.sat, rule["cap"])
        h.sat = round(min(max(h.sat, 0.0), 1.0), 3)
        return h

    @staticmethod
    def _clip(h: Hue) -> Hue:
        h.sat = round(min(max(h.sat, 0.0), 1.0), 3)
        return h

    def _taboo(self, h: Hue, fa: FaceAnalysis | None, reasons: list[str], exempt=False) -> Hue:
        if fa is None or fa.skin is None or exempt or not fa.skin.reliable:
            return h
        t = self.cfg["taboo"]
        if fa.skin.group == "deep":
            rule = t["deep"]
            if C.hue_in(h.hue, rule["forbid_hues"]) or (rule.get("forbid_white") and h.sat < 0.06):
                reasons.append(f"深肤色禁绿/青/纯白：{h.name} → {rule['replace']}")
                return self._named(rule["replace"])
        if fa.skin.undertone == "olive":
            rule = t["olive"]
            if C.hue_in(h.hue, rule["forbid_hues"]):
                reasons.append(f"橄榄调禁青/蓝：{h.name} → {rule['replace']}")
                return self._named(rule["replace"])
        return h

    # ---------------- 饱和度系数 ----------------
    def _pollution_scale(self, s_pct: float) -> float:
        for thr, k in self.cfg["pollution_sat_scale"]:
            if s_pct >= thr:
                return float(k)
        return 1.0

    def _distance_scale(self, dist_cm: float | None) -> float:
        d = dist_cm if dist_cm is not None else self.cfg["distance"]["default_cm"]
        return _interp(d, self.cfg["distance"]["attenuation"])

    @staticmethod
    def _hname(h: float) -> str:
        names = [(15, "红"), (35, "橙"), (55, "黄"), (90, "黄青"), (135, "绿"), (170, "青绿"), (200, "青"),
                 (250, "蓝"), (285, "紫"), (320, "紫粉"), (345, "粉"), (361, "红")]
        h = h % 360
        return next(n for t, n in names if h < t)

    # ---------------------------------------------------------------
    def decide(self, an: Analysis, mode: str = "skin_priority", object_type: str | None = None) -> Strategy:
        usable = [f for f in an.faces if f.usable]
        if an.exposure == "dark":
            st = self._dark(an)
        elif not an.faces:
            return self._object(an, object_type)
        elif len(usable) >= 2:
            st = self._multi(an, usable, mode)
        elif not usable:
            st = self._multi(an, [], mode)
            st.warnings.append("人脸过小，采用全肤色安全色")
        elif mode == "scene_priority":
            st = self._scene(an, usable[0])
        else:
            st = self._skin(an, usable[0])
        if usable and st.mode not in ("dark",):
            self._modifiers(st, an, usable[0], multi=len(usable) >= 2 and st.mode != "multi_same")
        return st

    # ---------------- 互补色矫正（T1 内外圈参数表） ----------------
    def _complement(self, an: Analysis, group: str, dist_cm: float | None, reasons: list[str],
                    sat_cap_group: str | None = None) -> tuple[Hue, Hue, list[Hue], dict]:
        cfg = self.cfg
        cast = an.cast
        e = cfg["complement_table"][cast.family][group]
        k_p = self._pollution_scale(cast.s_pct)
        k_d = self._distance_scale(dist_cm)
        k_m = cfg["cast"]["multi_color"]["sat_boost"] if cast.multi_color else 1.0
        k = k_p * k_d * k_m
        oh, os_ = e["outer"]
        ih, is_ = e["inner"]
        outer = Hue(f"外圈·{self._hname(oh)}", float(oh), os_ / 100 * k)
        inner = Hue(f"内圈·{self._hname(ih)}", float(ih), is_ / 100 * k)
        caps = cfg["skin_sat_caps"]
        if caps.get("enforce") and sat_cap_group:
            cap = caps[sat_cap_group][cast.level] / 100
            if inner.sat > cap or outer.sat > cap:
                reasons.append(f"肤色×污染饱和度上限 {cap:.0%}（分类标准 §5.3）")
            outer.sat, inner.sat = min(outer.sat, cap), min(inner.sat, cap)
        outer, inner = self._clip(outer), self._clip(inner)
        fac = {"base_outer_S": os_, "base_inner_S": is_, "pollution_k": k_p, "distance_k": round(k_d, 3),
               "distance_cm": dist_cm, "multi_color_k": k_m, "total_k": round(k, 3)}
        reasons.append(f"T1 实测参数（60cm）【{oh}-{os_}***{ih}-{is_}】"
                       + (f"，实测效果 {e['score']}/5" if e.get("score") else "，理论值未实测"))
        reasons.append(f"饱和度 = 基准 × 污染系数 {k_p} × 距离系数 {k_d:.2f}（≈{dist_cm or cfg['distance']['default_cm']:.0f}cm）"
                       + (f" × 多色增强 {k_m}" if k_m != 1 else "")
                       + f" → 【{oh}-{outer.sat * 100:.0f}***{ih}-{inner.sat * 100:.0f}】")
        if e.get("note"):
            reasons.append(f"参数备注：{e['note']}")
        # 高/低饱和备选
        alts = []
        v = cfg["complement_variants"]
        cur_low = os_ <= v["low"][0]
        if cur_low and cast.family not in v["no_high"]:
            alts.append(self._clip(Hue(f"高饱和版 外圈【{oh}-{v['high'][0]}***{ih}-{v['high'][1]}】×{k:.2f}",
                                       float(oh), v["high"][0] / 100 * k)))
        elif not cur_low:
            alts.append(self._clip(Hue(f"低饱和版 外圈【{oh}-{v['low'][0]}***{ih}-{v['low'][1]}】×{k:.2f}",
                                       float(oh), v["low"][0] / 100 * k)))
        return outer, inner, alts, fac

    # ---------------- 肤色优先（单人） ----------------
    def _skin(self, an: Analysis, fa: FaceAnalysis) -> Strategy:
        cfg = self.cfg
        cast, skin = an.cast, fa.skin
        reasons = [f"环境光：{FAMILY_ZH.get(cast.family)}（色相 {cast.hue:.0f}°），污染 S={cast.s_pct}%（{LEVEL_ZH[cast.level]}）"]
        warnings: list[str] = []
        if skin.reliable:
            group = skin.group
            reasons.insert(0, f"肤色 {skin.level}（ITA {skin.ita}°，{GROUP_ZH[skin.group]}，底调 {skin.undertone}）")
        else:
            group = cfg["skin"]["unknown_fallback_group"]
            reasons.insert(0, f"肤色不可判（{skin.unreliable_reason}）→ 按{GROUP_ZH[group]}参数")
            warnings.append(f"强色光下肤色无法可靠识别，已按{GROUP_ZH[group]}参数补光；"
                            f"如为白人/深肤色，可在备选中切换")
        if cast.level != "none":
            outer, inner, alts, fac = self._complement(an, group, fa.distance_cm, reasons,
                                                       sat_cap_group=group)
            if skin.reliable and group == "deep" and C.hue_in(outer.hue, cfg["taboo"]["deep"]["forbid_hues"]):
                warnings.append("深肤色禁冷色，但实测表给出青绿互补；当前以实测为准（效果 3/5，待验证）")
            scenario = f"{FAMILY_ZH[cast.family]}污染 · 互补色矫正（内外圈）"
            if cast.multi_color:
                reasons.append(f"多色光源（霓虹）模式：主环境光 {FAMILY_ZH[cast.family]}"
                               + (f"，次环境光 {FAMILY_ZH.get(cast.secondary)}（记录，不单独补光）" if cast.secondary else ""))
                scenario = "多色霓虹 · " + scenario
            st = Strategy(mode="skin_priority", scenario=scenario, light_type="dual", inner=inner, outer=outer,
                          brightness=0.0, reasons=reasons, warnings=warnings, alternatives=alts, sat_factors=fac)
            return st

        b = cfg["base_light"][group][an.env_tone]
        ov = cfg["undertone_override"]
        key = f"{group}_{skin.undertone}"
        sat_tag = b.get("sat")
        if b["type"] == "sandwich":
            inner = self._named(b["inner"], sat_tag)
            outer = self._named(b["outer"], sat_tag)
            light_type = "sandwich"
        else:
            inner = self._named(b["color"], sat_tag)
            outer, light_type = inner, "mono"
        if key in ov and skin.reliable:
            inner = self._named(ov[key])
            if light_type == "mono":
                outer = inner
            reasons.append(f"底调修正：{skin.undertone} → {ov[key]}")
        inner = self._taboo(inner, fa, reasons)
        outer = self._taboo(outer, fa, reasons) if light_type != "mono" else inner
        if an.exposure == "over":
            reasons.append("画面过曝 → 归 L0，不做彩色矫正")
        reasons.append(f"无明显光污染（L0），环境基调 {TONE_ZH[an.env_tone]} → "
                       f"{'夹心 ' + inner.name + '/' + outer.name if light_type == 'sandwich' else '单色 ' + inner.name}")
        st = Strategy(mode="skin_priority", scenario=f"{TONE_ZH[an.env_tone]}环境 · 肤色光", light_type=light_type,
                      inner=inner, outer=outer, brightness=0.0, reasons=reasons, warnings=warnings)
        alt_sat = cfg["saturation_presets"]["high"] if inner.sat < 0.4 else cfg["saturation_presets"]["low"]
        st.alternatives.append(self._cap(Hue(inner.name + ("·高饱和" if inner.sat < 0.4 else "·低饱和"),
                                             inner.hue, alt_sat)))
        return st

    # ---------------- 场景光优先（单人） ----------------
    def _scene(self, an: Analysis, fa: FaceAnalysis | None) -> Strategy:
        sc = self.cfg["scene_priority"]
        env = an.env
        reasons = []
        if env.dominant_hue is not None and env.colored_share > 0.08:
            sh = env.dominant_hue
            ssat = min(0.9, 0.35 + env.colored_share)
            reasons.append(f"场景主色调 {sh:.0f}°（彩色占比 {env.colored_share:.0%}）")
        else:
            sh, ssat = an.cast.bg_hue or 40.0, max((an.cast.bg_s or 0) / 100, 0.15)
            reasons.append(f"场景无强彩色光源，跟随背景光源色相 {sh:.0f}°")
        d = ((sc["skin_hue"] - sh + 180) % 360) - 180
        ih = (sh + d * sc["pull_to_skin"]) % 360
        inner = self._cap(Hue("场景色·降饱和", ih, min(ssat * sc["inner_sat_ratio"], sc["inner_sat_cap"])))
        outer = self._cap(Hue("场景色·轮廓", sh, min(ssat * sc["outer_sat_ratio"], sc["outer_sat_cap"])))
        inner = self._taboo(inner, fa, reasons)
        reasons.append("面部区降饱和并向肤色色相靠拢，外侧轮廓光贴近场景色")
        if env.dark_share > sc["shadow_contour_min_ratio"]:
            reasons.append(f"背景暗部占比 {env.dark_share:.0%}，外侧加同色调轮廓光")
        return Strategy(mode="scene_priority", scenario="场景光优先 · 氛围保留", light_type="sandwich",
                        inner=inner, outer=outer, brightness=0.0, reasons=reasons)

    # ---------------- 多人（《多人场景补光方案》） ----------------
    def _multi(self, an: Analysis, faces: list[FaceAnalysis], mode: str) -> Strategy:
        mc = self.cfg["multi_face"]
        cast = an.cast
        groups = [f.skin.group for f in faces if f.skin]
        all_reliable = bool(faces) and all(f.skin and f.skin.reliable for f in faces)
        gset = sorted(set(groups))
        reasons = [f"检测到 {len(an.faces)} 张人脸（有效 {len(faces)}），肤色组："
                   f"{', '.join(GROUP_ZH[g] for g in gset) or '未知'}"
                   + ("" if all_reliable or not faces else "（强色光下不可靠）"),
                   "灯板只能输出一种光：第一代不做分人/分区"]
        if mode == "scene_priority":
            st = self._scene(an, faces[0] if faces else None)
            st.inner.sat = min(st.inner.sat, mc["scene_sat_cap"])
            st.mode, st.scenario = "multi_scene", "多人 · 场景光优先（邻近低饱和色）"
            st.reasons = reasons + st.reasons
            return st
        # 双人且肤色相同 → 用该肤色专属参数（与单人一致）
        if (mc["same_group_as_single"] and len(faces) == 2 and len(gset) == 1 and all_reliable):
            st = self._skin(an, faces[0])
            st.mode = "multi_same"
            st.scenario = "双人同肤色 · " + st.scenario
            st.reasons = reasons + ["两人肤色相同 → 用该肤色专属最佳色"] + st.reasons
            return st
        # 最坏情况肤色（最深优先）用于说明
        worst = "deep" if "deep" in gset else ("medium" if "medium" in gset else (gset[0] if gset else None))
        key = cast.family if cast.level in ("medium", "heavy") else "none"
        if cast.level == "light":
            reasons.append(f"{FAMILY_ZH[cast.family]}污染轻（S={cast.s_pct}% < 30），危害低 → 安全色轻微提亮")
        e = mc["safe_by_env"].get(key, mc["safe_by_env"]["none"])
        dist = min((f.distance_cm for f in faces if f.distance_cm), default=None)
        k_d = self._distance_scale(dist)
        pick = self._clip(Hue(e["name"], float(e["color"][0]), e["color"][1] / 100 * k_d))
        if key != "none":
            reasons.append(f"环境光校正优先：{FAMILY_ZH[cast.family]}（S={cast.s_pct}%）→ 多人安全色 {e['name']} "
                           f"H{e['color'][0]}° S{e['color'][1]}%（60cm）")
        else:
            reasons.append(f"无明显污染 → 全肤色安全色 {e['name']} 轻微提亮")
        if worst:
            reasons.append(f"最坏情况肤色：{GROUP_ZH[worst]}；暖色区低饱和对所有肤色安全")
        if k_d != 1:
            reasons.append(f"距离 ≈{dist}cm，饱和度 × {k_d:.2f}")
        alts = [self._clip(Hue(a["name"], float(a["color"][0]), a["color"][1] / 100 * k_d))
                for a in mc["safe_alternatives"] if a["name"] != e["name"]]
        return Strategy(mode="multi_safe", scenario="多人 · 全肤色安全交集色", light_type="mono", inner=pick,
                        outer=pick, brightness=0.0, reasons=reasons, alternatives=alts,
                        sat_factors={"distance_cm": dist, "distance_k": round(k_d, 3)})

    # ---------------- 暗光 ----------------
    def _dark(self, an: Analysis) -> Strategy:
        h = self._named("暖白")
        st = Strategy(mode="dark", scenario="暗光环境 · 仅提亮", light_type="mono", inner=h, outer=h,
                      brightness=float(self.cfg["brightness"]["dark_scene"]),
                      reasons=[f"画面明度 V={an.env.frame_V:.0%}，环境过暗无法判断环境光色彩 → 暖白满亮提亮"],
                      warnings=["环境过暗，建议靠近光源或开启补光后重新识别"],
                      alternatives=[self._named("中性偏黄"), self._named("淡粉")])
        return st

    # ---------------- 物品 ----------------
    def _object(self, an: Analysis, object_type: str | None) -> Strategy:
        oc = self.cfg["object_mode"]
        env = an.env
        reasons = ["未检测到人脸 → 物品拍摄模式"]
        if object_type and object_type in oc:
            r = oc[object_type]
            if r["type"] == "sandwich":
                inner, outer, lt = self._named(r["inner"]), self._named(r["outer"]), "sandwich"
            else:
                inner = outer = self._named(r["color"])
                lt = "mono"
            reasons.append(f"用户指定物品类别 {object_type}")
            st = Strategy("object", f"物品 · {object_type}", lt, inner, outer, 0.0, reasons=reasons)
        elif an.exposure == "dark" or env.mean_L < 22:
            c = [self._named(n) for n in oc["dark_fallback"]]
            reasons.append("暗光环境无法识别环境色，给出 浅蓝/青/中性偏黄 三个候选")
            st = Strategy("object", "物品 · 暗光", "mono", c[2], c[2], 0.0, alternatives=c[:2], reasons=reasons)
        elif an.cast.level != "none" and an.cast.family in oc["complement_families"]:
            # 《环境光分类标准》§4.5：无人脸 → 背景主色调对应的互补色
            outer, inner, alts, fac = self._complement(an, "medium", None, reasons)
            reasons.insert(1, f"背景{FAMILY_ZH[an.cast.family]}污染 S={an.cast.s_pct}% → 背景主色互补色")
            st = Strategy("object", f"物品 · {FAMILY_ZH[an.cast.family]}互补矫正", "dual", inner, outer, 0.0,
                          reasons=reasons, alternatives=alts, sat_factors=fac)
            st.brightness = 1.0
            return st
        else:
            h = an.cast.bg_hue if an.cast.bg_hue is not None else 40
            s = min(max((an.cast.bg_s or 0) / 100 * 0.6, 0.08), 0.3)
            inner = self._cap(Hue("模仿环境色", h, s))
            reasons.append(f"模仿环境光色彩打光（色相 {h:.0f}°，降饱和）；食物/静物类别可通过参数指定")
            st = Strategy("object", "物品 · 模仿环境光", "mono", inner, inner, 0.0, reasons=reasons,
                          alternatives=[self._named("暖橙"), self._named("中性偏黄")])
        st.brightness = 0.7
        return st

    # ---------------- 亮度 / 方向 / 瑕疵 ----------------
    def _modifiers(self, st: Strategy, an: Analysis, fa: FaceAnalysis, multi: bool) -> None:
        b = self.cfg["brightness"]
        L = fa.skin.face_L if fa.skin else 50.0
        if an.cast.level != "none":
            val = b["polluted"]
            st.reasons.append("有色光污染：补光满亮（T1 参数均在 V=100 下实测）")
        else:
            val = b["base"] + (b["target_face_L"] - L) * b["per_L"]
            if fa.obs.width_ratio < 0.15:
                val += b["far_face_bonus"]
        if fa.skin and fa.skin.reliable and fa.skin.group == "deep" and val < b["deep_min"]:
            val = b["deep_min"]
            st.reasons.append("深肤色：保证亮度足够")
        d = self.cfg["direction"]
        if fa.direction == "back":
            val = b["backlight"]
            st.reasons.append("逆光：补光做主力，亮度拉满")
            st.warnings.append("逆光：建议调整拍摄位置（环境光分类标准 §4.5）")
        elif fa.direction.startswith("side") and not multi:
            dark_img_l = fa.direction == "side_img_l_dark"
            board = "right" if dark_img_l else "left"
            if self.cfg["hardware"].get("image_mirrored"):
                board = "left" if board == "right" else "right"
            st.side_boost = {board: d["side_boost"]}
            st.reasons.append(f"侧光：暗侧（{'左' if board == 'left' else '右'}板）补光 +{int(d['side_boost'] * 100)}%")
            # 留出侧光增亮余量，避免被 1.0 截断
            val = min(val, 1.0 / (1.0 + d["side_boost"]))
        elif fa.direction == "top":
            st.top_gradient = d["top_gradient"]
            st.reasons.append("顶光：下方行补光加强，均匀化下颌阴影")
        st.brightness = round(min(max(val, b["min"]), b["max"]), 3)

        if not multi and fa.blemishes and "dark_circle" in fa.blemishes and fa.skin and st.light_type != "dual":
            ep = self.cfg["blemish"]["eye_patch"][fa.skin.group]
            st.eye_patch = self._named(ep["color"])
            st.eye_patch_extra_val = ep["extra_val"]
            st.reasons.append(f"检测到眼周暗沉（ΔL={fa.blemish_detail.get('undereye_delta_L')}），眼周行带 {ep['color']} 提亮")
        if "redness" in fa.blemishes:
            st.warnings.append("检测到面中泛红；第一代分区混光效果有限，仅提示（可选青色中和）")
