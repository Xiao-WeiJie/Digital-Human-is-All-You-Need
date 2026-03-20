# -*- coding: utf-8 -*-
"""
FR 评估模块
用于评估 FasterLivePortrait 面部驱动模型的性能
"""

from .fr_metrics import (
    FRMetrics,
    FRScore,
    FRMetricsCalculator,
    FRScorer,
    aggregate_metrics,
    print_metrics_report
)

from .fr_evaluator import (
    EvaluationConfig,
    FeatureExtractor,
    FasterLivePortraitEvaluator,
    run_evaluation
)

__all__ = [
    'FRMetrics',
    'FRScore',
    'FRMetricsCalculator',
    'FRScorer',
    'aggregate_metrics',
    'print_metrics_report',
    'EvaluationConfig',
    'FeatureExtractor',
    'FasterLivePortraitEvaluator',
    'run_evaluation'
]
