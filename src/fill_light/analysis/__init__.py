"""② 分析层：观测 → 语义特征（环境光、肤色、光线方向、瑕疵、距离）。"""
from .analyzer import analyze
from .types import Analysis, CastInfo, FaceAnalysis, SkinInfo

__all__ = ["analyze", "Analysis", "CastInfo", "FaceAnalysis", "SkinInfo"]
