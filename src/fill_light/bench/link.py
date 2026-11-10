"""与灯板的连接。下发 / 渐变 / 序号 / 板端拒绝原因全部复用 ss2light.Light，这里不另写通信。"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from ..config import ROOT, resolve
from ..result import Result
from .payload import build_payload
from .tool import UCI_NAME, load_ss2light

DEFAULT_BENCH_CONFIG = ROOT / "configs" / "bench.yaml"
FAKE_BOARD = ("127.0.0.1", 8080)     # ss2light.py fake 起的本机假板端


def load_bench_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_BENCH_CONFIG).read_text(encoding="utf-8"))


@dataclass
class SendReport:
    sent: bool                 # False = 离线，只生成了下发内容
    payload: dict              # 交给台架的内容（含 216 颗灯珠的值，0-100）
    current_a_algo: float      # 算法帧的估算电流
    current_a_sent: float      # 限流后实际下发的估算电流
    bench_scale: float         # 台架限流的缩放系数（1.0 = 没有缩小）
    limit_a: float
    reply: dict | None = None  # 板子的回复
    rtt_ms: float | None = None

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("payload")
        return d

    def describe(self) -> str:
        s = f"估算电流 {self.current_a_algo}A"
        if self.bench_scale < 1:
            s += f" → 超台架上限 {self.limit_a}A，整体×{self.bench_scale} = {self.current_a_sent}A"
        return s + (f" | 板子回复 {self.reply} | {self.rtt_ms}ms" if self.sent else " | 未发送（dry-run）")


class BenchLink:
    """用法：BenchLink().send_result(pipe.run(...))。offline=True 时只生成下发内容、不连板子。"""

    def __init__(self, host: str | None = None, port: int | None = None, offline: bool = False,
                 cfg_path: str | Path | None = None) -> None:
        self.cfg = load_bench_config(cfg_path)
        self.ss, self.tool_dir = load_ss2light(self.cfg.get("tool_dir", "../ss2-light-bench"))
        state = self.cfg.get("state_file")
        if state:
            state = resolve(state)
            state.parent.mkdir(parents=True, exist_ok=True)
            state = str(state)
        self.offline = offline
        self.light = self.ss.Light(host or self.cfg.get("host", self.ss.DEFAULT_HOST),
                                   int(port or self.cfg.get("port", 80)), state)

    @property
    def record_dir(self) -> Path:
        return resolve(self.cfg.get("record_dir", "outputs/bench/records"))

    def caps(self) -> dict:
        """灯板配置（面板、灯珠数、编号走向）。离线时读工具目录里的配置文件。"""
        if self.offline:
            return self.ss.parse_uci(self.tool_dir / UCI_NAME)
        return self.light.caps()

    def send_leds(self, leds: dict, transition_ms: int | None = None) -> SendReport:
        payload, info = build_payload(leds, self.caps(), self.ss, self.cfg)
        if self.offline:
            return SendReport(sent=False, payload=payload, **info)
        ms = int(self.cfg.get("transition_ms", 0) if transition_ms is None else transition_ms)
        t0 = time.perf_counter()
        reply = self.light.apply(payload, ms)
        return SendReport(sent=True, payload=payload, reply=reply,
                          rtt_ms=round((time.perf_counter() - t0) * 1000, 1), **info)

    def send_result(self, result: Result | dict, transition_ms: int | None = None) -> SendReport:
        """result：pipe.run() 的返回值，或它存成的 JSON（dict）。"""
        leds = result.frame.leds if isinstance(result, Result) else result.get("frame", {}).get("leds")
        if leds is None:
            raise ValueError("结果里没有 frame.leds（生成时用了 --no-frame？）")
        return self.send_leds(leds, transition_ms)

    def off(self) -> dict:
        return self.light.apply({"on": False})
