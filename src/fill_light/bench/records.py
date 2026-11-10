"""验收记录：取景画面 + 算法结果 + 下发参数存进一个目录，灯光照片由人补拍放进去。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .link import SendReport


def save_record(rec_dir: Path, stem: str, image_path: Path | None, result: dict | None, sent: SendReport) -> Path:
    d = rec_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}"
    d.mkdir(parents=True, exist_ok=True)
    if image_path is not None and image_path.exists():
        (d / f"input{image_path.suffix}").write_bytes(image_path.read_bytes())
    if result is not None:
        (d / "algo_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "bench_payload.json").write_text(json.dumps(sent.payload, ensure_ascii=False), encoding="utf-8")
    (d / "bench_meta.json").write_text(json.dumps(sent.meta(), ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "灯光照片放这里.txt").write_text(
        "把灯板实拍照片放进本目录，连同 input 与 bench_payload.json 一起附到飞书任务「过程记录」。\n", encoding="utf-8")
    return d
