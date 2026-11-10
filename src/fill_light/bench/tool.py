"""载入台架工具 ss2light.py（在 ss2-light-bench 仓库里，不属于本项目，所以按路径动态加载）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ..config import resolve

UCI_NAME = "lightboard-v1.uci"       # 灯板配置文件，和 ss2light.py 在同一目录


def load_ss2light(tool_dir: str | Path):
    """返回 (ss2light 模块, 工具目录)。"""
    p = resolve(tool_dir).resolve()
    f = p / "ss2light.py"
    if not f.exists():
        raise FileNotFoundError(f"找不到台架工具 {f}；在 configs/bench.yaml 的 tool_dir 里改路径")
    if "ss2light" not in sys.modules:
        spec = importlib.util.spec_from_file_location("ss2light", f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ss2light"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["ss2light"], p
