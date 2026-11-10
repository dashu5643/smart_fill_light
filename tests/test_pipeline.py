"""整条流水线的测试：用程序生成的纯色图片，不需要照片和灯板（会加载一次人脸模型）。

    python -m unittest discover tests
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fill_light import FillLightPipeline, Result  # noqa: E402
from fill_light.result import describe  # noqa: E402


def solid(r: int, g: int, b: int) -> np.ndarray:
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (r, g, b)
    return img


GRAY, BLACK, PURPLE = solid(128, 128, 128), solid(0, 0, 0), solid(150, 40, 220)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = FillLightPipeline()

    @classmethod
    def tearDownClass(cls):
        cls.pipe.close()

    def test_result_structure(self):
        res = self.pipe.run(GRAY)
        self.assertIsInstance(res, Result)
        d = res.to_dict()
        self.assertEqual(list(d), ["mode", "env_light", "perception", "strategy", "light_params", "frame"])
        json.dumps(d, ensure_ascii=False)                              # 能直接存成 JSON
        leds = d["frame"]["leds"]
        self.assertEqual((len(leds["left"]), len(leds["left"][0]), len(leds["left"][0][0])), (18, 6, 4))
        self.assertEqual(sum(len(row) for b in leds.values() for row in b), 216)
        self.assertNotIn("leds", res.to_dict(with_leds=False)["frame"])
        self.assertIsInstance(describe(d), str)

    def test_neutral_scene_without_face_goes_to_object_mode(self):
        res = self.pipe.run(GRAY)
        self.assertEqual(res.env_light["env_light_category"], "L0")
        self.assertFalse(res.env_light["face_detected"])
        self.assertEqual((res.strategy.mode, res.light_params["light_type"]), ("object", "mono"))

    def test_object_type_picks_the_configured_light(self):
        lp = self.pipe.run(GRAY, object_type="jewelry").light_params
        self.assertEqual((lp["light_type"], lp["main"]["name"], lp["contour"]["name"]),
                         ("sandwich", "中性偏黄", "浅蓝"))

    def test_dark_scene_only_brightens(self):
        res = self.pipe.run(BLACK)
        self.assertEqual(res.env_light["exposure"], "dark")
        self.assertEqual(res.strategy.mode, "dark")
        self.assertEqual((res.light_params["main"]["name"], res.light_params["brightness_pct"]), ("暖白", 100.0))

    def test_purple_light_gets_complementary_dual_ring(self):
        res = self.pipe.run(PURPLE)
        self.assertEqual((res.env_light["env_light_category"], res.env_light["pollution_level"]), ("L3", "P-Heavy"))
        lp = res.light_params
        self.assertEqual((lp["light_type"], lp["t1_format"]), ("dual", "【75-75***190-100】"))
        left, right = res.frame.arrays["left"], res.frame.arrays["right"]
        self.assertEqual(left[0, 5].tolist(), [0, 212, 255, 0])        # 内圈（靠屏幕 2 列）：青
        self.assertEqual(right[0, 0].tolist(), [0, 212, 255, 0])
        self.assertEqual(left[0, 0].tolist(), [143, 191, 0, 64])       # 外圈：黄青

    def test_same_image_gives_same_decision(self):
        a, b = (self.pipe.run(PURPLE).to_dict() for _ in range(2))
        for d in (a, b):
            d["env_light"].pop("timestamp")
        self.assertEqual(a, b)

    def test_frame_respects_current_limit(self):
        limit = self.pipe.cfg["hardware"]["max_total_duty"]
        for img in (GRAY, BLACK, PURPLE):
            f = self.pipe.run(img).frame
            self.assertLessEqual(f.duty, limit + 1e-3)
            self.assertTrue(all(0 <= v.min() and v.max() <= 255 for v in f.arrays.values()))


if __name__ == "__main__":
    unittest.main()
