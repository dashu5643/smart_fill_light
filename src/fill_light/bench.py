"""算法 → 灯光台架（ss2-light-bench / ss2light.py）桥接。

算法侧帧（hardware.render_frame）：
    leds[board][row][col] = [R,G,B,W] 0-255，观察者视角（站在灯前、面向灯板）
    row 0 = 最上；左板 col0=外侧→col5=内侧，右板 col0=内侧→col5=外侧（即两板都是"从左到右"）
台架侧（lightboard-v1.uci + ICD-5）：
    pixels = 216 组 [r,g,b,w] 0-100，按灯珠编号 1..216 顺序；
    每块面板 first_index / cols / rows / scan=column_major / col_dir / row_dir 描述编号走向
    （col 0 = 面向灯板时最左列，row 0 = 最上行）

本模块只做三件事：坐标重排、0-255 → 0-100、按台架电流上限等比缩放。
下发 / 渐变 / 序号 / 板端拒绝原因全部复用 ss2light.Light，不另写通信。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# ---------- 载入台架工具 ----------

def load_ss2light(tool_dir: str | Path):
    p = Path(tool_dir)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    f = p / "ss2light.py"
    if not f.exists():
        raise FileNotFoundError(f"找不到台架工具 {f}；在 configs/strategy.yaml 的 bench.tool_dir 里改路径")
    if "ss2light" in sys.modules:
        return sys.modules["ss2light"], p
    spec = importlib.util.spec_from_file_location("ss2light", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ss2light"] = mod
    spec.loader.exec_module(mod)
    return mod, p


# ---------- 坐标重排 ----------

def panel_index(p: dict, col: int, row: int) -> int:
    """面板内 (col,row)（面向灯板：col0=最左，row0=最上）→ 全局灯珠编号（0 起）。"""
    cols, rows = int(p["cols"]), int(p["rows"])
    c = col if p.get("col_dir", "left_to_right") == "left_to_right" else cols - 1 - col
    r = row if p.get("row_dir", "top_to_bottom") == "top_to_bottom" else rows - 1 - row
    if p.get("scan", "column_major") == "column_major":
        k = c * rows + r
    else:                                   # row_major（V1 没有，留着兼容）
        k = r * cols + c
    return int(p["first_index"]) - 1 + k


def frame_to_pixels(leds: dict, caps: dict, swap_boards: bool = False, mirror_cols: bool = False,
                    flip_rows: bool = False) -> list[list[float]]:
    """算法帧 → 台架 pixels（216 × [r,g,b,w]，0-100）。"""
    n = int(caps["pixels"])
    px = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    filled = 0
    for p in caps["panels"]:
        name = p["name"]
        src = {"left": "right", "right": "left"}[name] if swap_boards else name
        if src not in leds:
            raise ValueError(f"算法帧里没有 {src} 板")
        arr = np.asarray(leds[src], dtype=float)                 # rows × cols × 4
        rows, cols = int(p["rows"]), int(p["cols"])
        if arr.shape[:2] != (rows, cols):
            raise ValueError(f"{src} 板尺寸 {arr.shape[:2]} 与台架 {name} 面板 {(rows, cols)} 不一致")
        for row in range(rows):
            for col in range(cols):
                v = arr[rows - 1 - row if flip_rows else row, cols - 1 - col if mirror_cols else col]
                px[panel_index(p, col, row)] = [round(float(x) / 255 * 100, 2) for x in v]
                filled += 1
    if filled != n:
        raise ValueError(f"面板合计 {filled} 颗 ≠ 灯珠数 {n}")
    return px


def estimate_current_a(px: list[list[float]], ss) -> float:
    ma = ss.I_CH_MA
    return sum(p[0] / 100 * ma["r"] + p[1] / 100 * ma["g"] + p[2] / 100 * ma["b"] + p[3] / 100 * ma["w"]
               for p in px) / 1000


def build_payload(leds: dict, caps: dict, ss, bcfg: dict) -> tuple[dict, dict]:
    """返回 (下发给 Light.apply 的参数, 说明信息)。"""
    px = frame_to_pixels(leds, caps, bool(bcfg.get("swap_boards")), bool(bcfg.get("mirror_cols")),
                         bool(bcfg.get("flip_rows")))
    i0 = estimate_current_a(px, ss)
    limit = float(bcfg.get("max_current_a", 0.95))
    limit = min(limit, float(ss.I_BENCH_MAX_A))
    scale = 1.0
    if i0 > limit:
        scale = limit / i0
        px = [[round(x * scale, 2) for x in p] for p in px]
    payload = {
        "schema": ss.SCHEMA,
        "on": True,
        "intensity": 100,                                     # 亮度已经乘进逐颗值里
        "zones": {p["name"]: {"intensity": 100} for p in caps["panels"]},   # 清掉之前残留的面板亮度
        "pixels": px,
    }
    info = {"current_a_algo": round(i0, 3), "current_a_sent": round(estimate_current_a(px, ss), 3),
            "bench_scale": round(scale, 3), "limit_a": limit}
    return payload, info


# ---------- 对外接口 ----------

class BenchLink:
    """把算法结果下发到灯板：BenchLink(cfg).send_result(pipe.run(...))."""

    def __init__(self, cfg: dict, host: str | None = None, port: int | None = None, offline: bool = False):
        self.bcfg = dict(cfg.get("bench", {}))
        self.ss, self.tool_dir = load_ss2light(self.bcfg.get("tool_dir", "../ss2-light-bench"))
        state = self.bcfg.get("state_file")
        if state:
            sp = Path(state) if Path(state).is_absolute() else ROOT / state
            sp.parent.mkdir(parents=True, exist_ok=True)
            state = str(sp)
        self.offline = offline
        self.light = self.ss.Light(host or self.bcfg.get("host", self.ss.DEFAULT_HOST),
                                   int(port or self.bcfg.get("port", 80)), state)

    def caps(self) -> dict:
        if self.offline:
            return self.ss.parse_uci(self.tool_dir / "lightboard-v1.uci")
        return self.light.caps()

    def prepare(self, leds: dict) -> tuple[dict, dict]:
        return build_payload(leds, self.caps(), self.ss, self.bcfg)

    def send_leds(self, leds: dict, transition_ms: int | None = None) -> dict:
        payload, info = self.prepare(leds)
        if self.offline:
            return {"sent": False, "payload": payload, **info}
        ms = int(self.bcfg.get("transition_ms", 0) if transition_ms is None else transition_ms)
        t0 = time.perf_counter()
        reply = self.light.apply(payload, ms)
        return {"sent": True, "reply": reply, "rtt_ms": round((time.perf_counter() - t0) * 1000, 1),
                "payload": payload, **info}

    def send_result(self, result: dict, transition_ms: int | None = None) -> dict:
        leds = result.get("frame", {}).get("leds")
        if leds is None:
            raise ValueError("结果里没有 frame.leds（生成时用了 --no-frame？）")
        return self.send_leds(leds, transition_ms)

    def off(self) -> dict:
        return self.light.apply({"on": False})


def identify_leds(rows: int = 18, cols: int = 6) -> dict:
    """方位测试帧（观察者视角）：左板红、右板蓝；内侧（靠屏幕）一列白；最上一行绿。"""
    out = {}
    for board, base in (("left", [60, 0, 0, 0]), ("right", [0, 0, 60, 0])):
        arr = [[list(base) for _ in range(cols)] for _ in range(rows)]
        inner = cols - 1 if board == "left" else 0
        for r in range(rows):
            arr[r][inner] = [0, 0, 0, 60]
        for c in range(cols):
            arr[0][c] = [0, 60, 0, 0]
        out[board] = arr
    return out


def save_record(rec_dir: Path, stem: str, image_path: Path | None, result: dict | None, sent: dict) -> Path:
    """验收记录（09_algo C-6）：取景画面 + 下发参数 JSON；灯光照片由人补拍放进同一目录。"""
    d = rec_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}"
    d.mkdir(parents=True, exist_ok=True)
    if image_path is not None and image_path.exists():
        (d / f"input{image_path.suffix}").write_bytes(image_path.read_bytes())
    if result is not None:
        r = {k: v for k, v in result.items() if k != "_internal"}
        (d / "algo_result.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "bench_payload.json").write_text(json.dumps(sent["payload"], ensure_ascii=False), encoding="utf-8")
    meta = {k: v for k, v in sent.items() if k != "payload"}
    (d / "bench_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "灯光照片放这里.txt").write_text("把灯板实拍照片放进本目录，连同 input 与 bench_payload.json 一起附到飞书任务「过程记录」。\n",
                                     encoding="utf-8")
    return d
