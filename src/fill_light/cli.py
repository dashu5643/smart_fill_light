"""命令行入口共用的参数定义（scripts/ 下各脚本使用），保证各入口的参数名和含义一致。"""
from __future__ import annotations

import argparse
import sys

from .strategy import MODES


def utf8_stdout(line_buffering: bool = False) -> None:
    """Windows 控制台默认不是 UTF-8，中文和符号会乱码或报错。"""
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=line_buffering)


def add_run_args(ap: argparse.ArgumentParser) -> None:
    """推理参数：与 FillLightPipeline.run() 一一对应。"""
    ap.add_argument("--mode", default=MODES[0], choices=MODES,
                    help="skin_priority 肤色优先（默认）/ scene_priority 场景光优先")
    ap.add_argument("--object", default=None,
                    help="无人脸时的物品类别：food_cn / food_west / dessert / salad / beauty / jewelry / plant")
    ap.add_argument("--distance", type=float, default=None, help="灯板到人脸的距离 cm（不填按 60cm）")
    ap.add_argument("--cfg", default=None, help="算法参数表（默认 configs/strategy.yaml）")


def run_kwargs(a: argparse.Namespace) -> dict:
    return {"mode": a.mode, "object_type": a.object, "distance_cm": a.distance}


def add_board_args(ap: argparse.ArgumentParser) -> None:
    """灯板连接参数：不填用 configs/bench.yaml 里的地址。"""
    ap.add_argument("--host", help="板子地址")
    ap.add_argument("--port", type=int)
    ap.add_argument("--fake", action="store_true", help="连本机假板端（127.0.0.1:8080）")


def open_link(a: argparse.Namespace, offline: bool = False):
    from .bench import FAKE_BOARD, BenchLink          # 用到灯板时才加载台架工具
    host, port = FAKE_BOARD if a.fake else (a.host, a.port)
    return BenchLink(host, port, offline=offline)
