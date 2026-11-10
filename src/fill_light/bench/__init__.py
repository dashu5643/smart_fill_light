"""灯板下发（可选）：把算法帧经 ss2-light-bench 台架工具发到灯板。算法四层不依赖本包。"""
from .link import FAKE_BOARD, BenchLink, SendReport, load_bench_config
from .patterns import IDENTIFY_EXPECTED, identify_leds
from .records import save_record

__all__ = ["BenchLink", "SendReport", "FAKE_BOARD", "load_bench_config", "identify_leds", "IDENTIFY_EXPECTED",
           "save_record"]
