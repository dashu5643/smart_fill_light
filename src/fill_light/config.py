"""配置加载与项目路径。所有阈值、色彩、规则都在 configs/strategy.yaml，代码里只做引擎。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]            # 项目根目录
DEFAULT_CONFIG = ROOT / "configs" / "strategy.yaml"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    """配置里的相对路径一律相对项目根目录。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
