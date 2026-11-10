"""各模块的单元测试：不加载人脸模型，不需要图片和灯板。

    python -m unittest discover tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fill_light import load_config  # noqa: E402
from fill_light.bench.mapping import frame_to_pixels, panel_index  # noqa: E402
from fill_light.bench.patterns import identify_leds  # noqa: E402
from fill_light.hardware import BoardLayout, render_frame  # noqa: E402
from fill_light.hardware.rgbw import clamp_total_duty, hsv_to_rgbw  # noqa: E402
from fill_light.strategy import LightColor, Strategy  # noqa: E402
from fill_light.strategy.complement import distance_scale, pollution_scale  # noqa: E402
from fill_light.strategy.palette import hue_name, named  # noqa: E402

CFG = load_config()
CYAN, LIME = LightColor("青", 190, 1.0), LightColor("黄青", 75, 0.75)

# 与 ss2-light-bench/lightboard-v1.uci 相同的面板描述：两块板 180° 对装
CAPS = {"pixels": 216, "panels": [
    {"name": "right", "first_index": 1, "cols": 6, "rows": 18, "scan": "column_major",
     "col_dir": "left_to_right", "row_dir": "top_to_bottom"},
    {"name": "left", "first_index": 109, "cols": 6, "rows": 18, "scan": "column_major",
     "col_dir": "right_to_left", "row_dir": "bottom_to_top"},
]}


def strategy(light_type: str, **kw) -> Strategy:
    return Strategy(mode="test", scenario="test", light_type=light_type, inner=CYAN, outer=LIME,
                    brightness=1.0, **kw)


class ConfigTest(unittest.TestCase):
    def test_complement_table_covers_every_light_and_skin_group(self):
        families = set(CFG["cast"]["hue_classes"])
        self.assertEqual(set(CFG["complement_table"]), families)
        for family, groups in CFG["complement_table"].items():
            self.assertEqual(set(groups), {"fair", "medium", "deep"}, family)

    def test_named_colors_exist_in_palette(self):
        palette = CFG["palette"]
        names = [v for g in CFG["base_light"].values() for rule in g.values()
                 for k, v in rule.items() if k in ("color", "inner", "outer")]
        names += list(CFG["undertone_override"].values()) + CFG["object_mode"]["dark_fallback"]
        names += [t["replace"] for t in CFG["taboo"].values()]
        for n in names:
            self.assertIn(n, palette)


class StrategyHelpersTest(unittest.TestCase):
    def test_distance_scale_interpolates_and_clamps(self):
        self.assertAlmostEqual(distance_scale(CFG, 60), 1.0)
        self.assertAlmostEqual(distance_scale(CFG, 30), 0.53)
        self.assertAlmostEqual(distance_scale(CFG, 45), (0.53 + 1.0) / 2)
        self.assertAlmostEqual(distance_scale(CFG, 10), 0.40)       # 比表里最近的还近：截断
        self.assertAlmostEqual(distance_scale(CFG, 200), 1.0)
        self.assertAlmostEqual(distance_scale(CFG, None), 1.0)      # 不给距离 = 默认 60cm

    def test_pollution_scale_steps(self):
        self.assertEqual([pollution_scale(CFG, s) for s in (90, 70, 50, 20)], [1.0, 1.0, 0.85, 0.7])

    def test_palette(self):
        c = named(CFG, "暖白")
        self.assertEqual((c.hue, c.sat), (35.0, 0.12))
        self.assertEqual(named(CFG, "暖白", "high").sat, CFG["saturation_presets"]["high"])
        self.assertEqual([hue_name(h) for h in (0, 45, 120, 190, 230, 275, 359)],
                         ["红", "黄", "绿", "青", "蓝", "紫", "红"])


class HardwareTest(unittest.TestCase):
    def test_rgbw_moves_common_white_to_w_channel(self):
        np.testing.assert_allclose(hsv_to_rgbw(0, 0.0, 1.0), [0, 0, 0, 1])          # 纯白只用 W
        np.testing.assert_allclose(hsv_to_rgbw(0, 1.0, 0.5), [0.5, 0, 0, 0])        # 纯红不用 W
        np.testing.assert_allclose(hsv_to_rgbw(0, 0.5, 1.0), [0.5, 0, 0, 0.5])      # 淡红 = 红 + 白

    def test_clamp_scales_everything_proportionally(self):
        boards = {"left": np.full((2, 2, 4), 1.0), "right": np.full((2, 2, 4), 0.5)}
        self.assertEqual(clamp_total_duty(boards, 100), 1.0)                         # 没超：不动
        scale = clamp_total_duty(boards, 12)                                         # 总和 24 → 12
        self.assertAlmostEqual(scale, 0.5)
        self.assertAlmostEqual(boards["left"][0, 0, 0], 0.5)
        self.assertAlmostEqual(boards["right"][0, 0, 0], 0.25)

    def test_inner_columns_are_the_ones_next_to_the_screen(self):
        lay = BoardLayout.from_config(CFG)
        self.assertEqual(lay.channel_count, 864)
        self.assertEqual([lay.is_inner("left", c, 2) for c in range(6)], [False] * 4 + [True] * 2)
        self.assertEqual([lay.is_inner("right", c, 2) for c in range(6)], [True] * 2 + [False] * 4)

    def test_light_types_fill_the_right_columns(self):
        cyan = [0, 212, 255, 0]
        mono = render_frame(strategy("mono"), CFG)
        self.assertEqual({k: v.shape for k, v in mono.arrays.items()}, {"left": (18, 6, 4), "right": (18, 6, 4)})
        self.assertTrue(all((v == cyan).all() for v in mono.arrays.values()))        # 单色：全板主色

        for light_type, n_inner in (("dual", 2), ("sandwich", 4)):
            f = render_frame(strategy(light_type), CFG)
            self.assertEqual(f.layout["inner_cols"], n_inner)
            left, right = f.arrays["left"], f.arrays["right"]
            self.assertTrue((left[:, 6 - n_inner:] == cyan).all() and (right[:, :n_inner] == cyan).all())
            self.assertFalse((left[:, :6 - n_inner] == cyan).all(axis=2).any())
            np.testing.assert_array_equal(left, right[:, ::-1])                      # 两块板左右对称

    def test_modifiers(self):
        base = render_frame(strategy("mono"), CFG).arrays
        half = Strategy(mode="t", scenario="t", light_type="mono", inner=CYAN, outer=CYAN, brightness=0.5,
                        side_boost={"right": 0.25}, top_gradient=0.0)
        f = render_frame(half, CFG).arrays
        self.assertGreater(f["right"].sum(), f["left"].sum())                        # 侧光：暗侧板更亮
        top = render_frame(Strategy(mode="t", scenario="t", light_type="mono", inner=CYAN, outer=CYAN,
                                    brightness=0.5, top_gradient=0.35), CFG).arrays["left"]
        self.assertGreater(top[-1].sum(), top[0].sum())                              # 顶光：越往下越亮
        eye = render_frame(strategy("sandwich", eye_patch=LightColor("橘", 24, 0.35)), CFG).arrays["left"]
        first, last = CFG["hardware"]["row_bands"]["eye"]
        self.assertFalse((eye[first:last + 1, 2:] == base["left"][first:last + 1, 2:]).all())   # 眼部行带换色
        np.testing.assert_array_equal(eye[last + 1:], render_frame(strategy("sandwich"), CFG).arrays["left"][last + 1:])


class BenchMappingTest(unittest.TestCase):
    def test_panel_index_matches_board_wiring(self):
        right, left = CAPS["panels"]
        # 右板：1 号在左上角，沿列向下；左板：109 号在右下角，沿列向上，216 号在左上角
        self.assertEqual(panel_index(right, 0, 0), 0)
        self.assertEqual(panel_index(right, 0, 17), 17)
        self.assertEqual(panel_index(right, 5, 17), 107)
        self.assertEqual(panel_index(left, 5, 17), 108)
        self.assertEqual(panel_index(left, 5, 0), 125)
        self.assertEqual(panel_index(left, 0, 0), 215)

    def test_every_led_is_filled_exactly_once(self):
        seen = sorted(panel_index(p, c, r) for p in CAPS["panels"] for c in range(6) for r in range(18))
        self.assertEqual(seen, list(range(216)))

    def test_frame_to_pixels_scales_to_percent_and_keeps_position(self):
        leds = {b: np.zeros((18, 6, 4), int).tolist() for b in ("left", "right")}
        leds["right"][0][0] = [255, 0, 0, 0]          # 右板左上角
        leds["left"][17][5] = [0, 0, 0, 51]           # 左板右下角
        px = frame_to_pixels(leds, CAPS)
        self.assertEqual(len(px), 216)
        self.assertEqual(px[0], [100.0, 0.0, 0.0, 0.0])
        self.assertEqual(px[108], [0.0, 0.0, 0.0, 20.0])
        self.assertEqual(sum(1 for p in px if any(p)), 2)
        swapped = frame_to_pixels(leds, CAPS, swap_boards=True)
        self.assertEqual(swapped[215], [100.0, 0.0, 0.0, 0.0])      # 换到左板的左上角

    def test_identify_pattern(self):
        leds = identify_leds()
        self.assertEqual(leds["left"][5][0], [60, 0, 0, 0])          # 左板红
        self.assertEqual(leds["right"][5][5], [0, 0, 60, 0])         # 右板蓝
        self.assertEqual(leds["left"][5][5], [0, 0, 0, 60])          # 靠屏幕一列白
        self.assertEqual(leds["right"][5][0], [0, 0, 0, 60])
        self.assertEqual(leds["left"][0][2], [0, 60, 0, 0])          # 顶行绿


if __name__ == "__main__":
    unittest.main()
