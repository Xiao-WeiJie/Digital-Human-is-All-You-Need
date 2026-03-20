# -*- coding: utf-8 -*-
"""
FR 指标计算模块
用于计算面部驱动模型的 5 项基���指标: FRCorr, FRdist, FRDiv, FRDvs, FRVar
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FRMetrics:
    """FR 系列指标结果"""
    FRCorr: float = 0.0
    FRdist: float = float('inf')
    FRDiv: float = 0.0
    FRDvs: float = 0.0
    FRVar: float = 0.0

    def to_dict(self) -> dict:
        return {
            'FRCorr': self.FRCorr,
            'FRdist': self.FRdist,
            'FRDiv': self.FRDiv,
            'FRDvs': self.FRDvs,
            'FRVar': self.FRVar
        }


@dataclass
class FRScore:
    """各项评分"""
    FRCorr_score: float = 0.0
    FRdist_score: float = 0.0
    FRDiv_score: float = 0.0
    FRDvs_score: float = 0.0
    FRVar_score: float = 0.0
    total_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            'FRCorr_score': self.FRCorr_score,
            'FRdist_score': self.FRdist_score,
            'FRDiv_score': self.FRDiv_score,
            'FRDvs_score': self.FRDvs_score,
            'FRVar_score': self.FRVar_score,
            'total_score': self.total_score
        }


class FRMetricsCalculator:
    """
    FR 指标计算器

    基准值和评分公式来自 val_dataset_analysis.md
    """

    # 基准值常量 (达到此值可获得满分)
    BENCHMARK = {
        'FRCorr': 0.09,      # 相关性
        'FRdist': 91.07,     # 距离
        'FRDiv': 3.4e-2,     # 多样性
        'FRDvs': 3.22e-2,    # 分布多样性
        'FRVar': 2.02e-2,    # 动态方差
    }

    # 满分值
    MAX_SCORE = {
        'FRCorr': 10,
        'FRdist': 10,
        'FRDiv': 10,
        'FRDvs': 10,
        'FRVar': 10,
    }

    @staticmethod
    def compute_FRCorr(gen_features: np.ndarray, gt_features: np.ndarray) -> float:
        """
        计算相关性指标
        衡量生成的人脸行为与目标驱动的匹配程度

        Args:
            gen_features: 生成视频的面部特征 (N_frames, N_features)
            gt_features: Ground Truth 视频的面部特征 (N_frames, N_features)

        Returns:
            float: 皮尔逊相关系数 (0-1)
        """
        # 对齐帧数
        min_frames = min(len(gen_features), len(gt_features))
        if min_frames == 0:
            return 0.0

        gen_flat = gen_features[:min_frames].flatten()
        gt_flat = gt_features[:min_frames].flatten()

        # 计算皮尔逊相关系数
        correlation = np.corrcoef(gen_flat, gt_flat)[0, 1]

        # 处理 NaN 情况
        if np.isnan(correlation):
            return 0.0

        return max(0, correlation)  # 确保非负

    @staticmethod
    def compute_FRdist(gen_features: np.ndarray, gt_features: np.ndarray) -> float:
        """
        计算距离指标
        衡量生成结果与参考目标之间的差异

        Args:
            gen_features: 生成视频的面部特征 (N_frames, N_features)
            gt_features: Ground Truth 视频的面部特征 (N_frames, N_features)

        Returns:
            float: 平均欧氏距离 * 100
        """
        min_frames = min(len(gen_features), len(gt_features))
        if min_frames == 0:
            return float('inf')

        gen_aligned = gen_features[:min_frames]
        gt_aligned = gt_features[:min_frames]

        # 欧氏距离
        distances = np.sqrt(np.sum((gen_aligned - gt_aligned) ** 2, axis=-1))

        return np.mean(distances) * 100  # 放大系数

    @staticmethod
    def compute_FRDiv(features: np.ndarray) -> float:
        """
        计算整体多样性
        衡量表情和动作是否丰富

        Args:
            features: 面部特征序列 (N_frames, N_features)

        Returns:
            float: 帧间差异的标准差
        """
        if len(features) < 2:
            return 0.0

        # 计算帧间差异的标准差
        diffs = np.diff(features, axis=0)
        diversity = np.std(diffs)

        return float(diversity)

    @staticmethod
    def compute_FRDvs(features_list: List[np.ndarray]) -> float:
        """
        计算分布多样性
        衡量不同样本中的丰富程度

        Args:
            features_list: 多个样本的面部特征列表

        Returns:
            float: 样本间均值的标准差
        """
        if len(features_list) < 2:
            return 0.0

        # 过滤空样本
        valid_features = [f for f in features_list if len(f) > 0]
        if len(valid_features) < 2:
            return 0.0

        # 计算各样本均值，然后计算样本间差异
        means = [np.mean(f, axis=0) for f in valid_features]
        means_arr = np.array(means)
        diversity = np.std(means_arr)

        return float(diversity)

    @staticmethod
    def compute_FRVar(features: np.ndarray) -> float:
        """
        计算动态方差
        衡量面部动作是否有自然波动

        Args:
            features: 面部特征序列 (N_frames, N_features)

        Returns:
            float: 整体方差
        """
        if len(features) == 0:
            return 0.0

        return float(np.var(features))


class FRScorer:
    """FR 评分计算器"""

    @classmethod
    def compute_score(cls, metrics: FRMetrics) -> FRScore:
        """
        根据 FR 指标计算各项评分

        评分公式:
        - ↑ 越大越好: score = min((value / benchmark) * max_score, max_score)
        - ↓ 越小越好: score = min((benchmark / value) * max_score, max_score)
        """
        score = FRScore()

        # FRCorr (↑)
        if metrics.FRCorr > 0:
            score.FRCorr_score = min(
                (metrics.FRCorr / 0.09) * 10, 10
            )
        else:
            score.FRCorr_score = 0

        # FRdist (↓)
        if metrics.FRdist > 0 and metrics.FRdist != float('inf'):
            score.FRdist_score = min(
                (91.07 / metrics.FRdist) * 10, 10
            )
        else:
            score.FRdist_score = 0

        # FRDiv (↑)
        if metrics.FRDiv > 0:
            score.FRDiv_score = min(
                (metrics.FRDiv / 3.4e-2) * 10, 10
            )
        else:
            score.FRDiv_score = 0

        # FRDvs (↑)
        if metrics.FRDvs > 0:
            score.FRDvs_score = min(
                (metrics.FRDvs / 3.22e-2) * 10, 10
            )
        else:
            score.FRDvs_score = 0

        # FRVar (↑)
        if metrics.FRVar > 0:
            score.FRVar_score = min(
                (metrics.FRVar / 2.02e-2) * 10, 10
            )
        else:
            score.FRVar_score = 0

        # 总分 (Phase 1 只有 5 项基础指标，满分 50)
        score.total_score = (
            score.FRCorr_score +
            score.FRdist_score +
            score.FRDiv_score +
            score.FRDvs_score +
            score.FRVar_score
        )

        return score


def aggregate_metrics(metrics_list: List[FRMetrics]) -> FRMetrics:
    """汇总多个样本的指标 (取均值)"""
    if not metrics_list:
        return FRMetrics()

    avg_metrics = FRMetrics()
    valid_frdist = [m.FRdist for m in metrics_list if m.FRdist != float('inf')]

    avg_metrics.FRCorr = np.mean([m.FRCorr for m in metrics_list])
    avg_metrics.FRdist = np.mean(valid_frdist) if valid_frdist else float('inf')
    avg_metrics.FRDiv = np.mean([m.FRDiv for m in metrics_list])
    avg_metrics.FRVar = np.mean([m.FRVar for m in metrics_list])

    # FRDvs 需要所有样本一起计算
    return avg_metrics


def print_metrics_report(metrics: FRMetrics, score: FRScore, sample_count: int):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("           FasterLivePortrait FR 指标评估报告 (Phase 1)")
    print("=" * 70)

    print(f"\n样本数量: {sample_count}")

    print("\n" + "-" * 70)
    print(f"{'指标':<12} {'方向':<8} {'实测值':<18} {'得分':<10} {'满分':<10} {'基准值':<12}")
    print("-" * 70)

    benchmarks = {
        'FRCorr': ('0.09', '↑'),
        'FRdist': ('91.07', '↓'),
        'FRDiv': ('3.4e-2', '↑'),
        'FRDvs': ('3.22e-2', '↑'),
        'FRVar': ('2.02e-2', '↑'),
    }

    rows = [
        ('FRCorr', metrics.FRCorr, score.FRCorr_score, 10, '↑', '0.09'),
        ('FRdist', metrics.FRdist, score.FRdist_score, 10, '↓', '91.07'),
        ('FRDiv', metrics.FRDiv, score.FRDiv_score, 10, '↑', '3.4e-2'),
        ('FRDvs', metrics.FRDvs, score.FRDvs_score, 10, '↑', '3.22e-2'),
        ('FRVar', metrics.FRVar, score.FRVar_score, 10, '↑', '2.02e-2'),
    ]

    for name, value, score_val, max_score, direction, benchmark in rows:
        if value == float('inf'):
            value_str = "N/A"
        elif value < 0.001:
            value_str = f"{value:.2e}"
        else:
            value_str = f"{value:.6f}"
        print(f"{name:<12} {direction:<8} {value_str:<18} {score_val:<10.2f} {max_score:<10} {benchmark:<12}")

    print("-" * 70)
    print(f"{'总分':<12} {'':<8} {'':<18} {score.total_score:<10.2f} {'50':<10}")
    print("=" * 70)

    # 评分等级
    if score.total_score >= 45:
        grade = "优秀 (Excellent)"
    elif score.total_score >= 35:
        grade = "良好 (Good)"
    elif score.total_score >= 25:
        grade = "中等 (Average)"
    elif score.total_score >= 15:
        grade = "及格 (Pass)"
    else:
        grade = "不及格 (Fail)"

    print(f"\n综合评级: {grade}")
    print(f"注: Phase 1 仅包含 5 项基础指标，满分 50 分")
