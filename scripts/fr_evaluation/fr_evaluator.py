# -*- coding: utf-8 -*-
"""
FasterLivePortrait 评估器模块
用于批量评估面部驱动模型的性能
"""

import os
import sys
import cv2
import time
import shutil
import tempfile
import platform
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "FasterLivePortrait"))

try:
    import mediapipe as mp
except ImportError:
    print("警告: mediapipe 未安装，请运行: pip install mediapipe")
    mp = None

from .fr_metrics import (
    FRMetrics, FRScore, FRMetricsCalculator, FRScorer,
    aggregate_metrics, print_metrics_report
)


@dataclass
class EvaluationConfig:
    """评估配置"""
    val_dir: str = "val/"                          # 测试数据集目录
    output_dir: str = "results/fr_eval/"           # 输出目录
    sample_limit: int = 20                         # 测试样本数量限制
    save_generated_videos: bool = False            # 是否保存生成的视频
    use_first_frame_as_source: bool = True         # 使用 GT 视频第一帧作为源图像

    # FasterLivePortrait 配置
    flp_config_path: str = "FasterLivePortrait/configs/trt_infer.yaml"
    flp_checkpoint_dir: str = "FasterLivePortrait/checkpoints"


class FeatureExtractor:
    """面部特征提取器 (使用 MediaPipe Face Mesh)"""

    def __init__(self):
        if mp is None:
            raise ImportError("mediapipe 未安装，请运行: pip install mediapipe")

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        从视频提取面部特征

        Args:
            video_path: 视频文件路径

        Returns:
            np.ndarray: 特征矩阵 (N_frames, N_features)
        """
        cap = cv2.VideoCapture(video_path)
        features_list = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # RGB 转换
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 检测面部关键点
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # 获取第一个检测到的面部
                landmarks = results.multi_face_landmarks[0]
                # 提取 478 个关键点的 (x, y, z) 坐标
                coords = np.array([[p.x, p.y, p.z] for p in landmarks.landmark])
                features_list.append(coords.flatten())
            else:
                # 如果没有检测到面部，使用前一帧的特征或零向量
                if features_list:
                    features_list.append(features_list[-1])
                else:
                    features_list.append(np.zeros(478 * 3))

            frame_count += 1

        cap.release()

        if not features_list:
            return np.array([])

        return np.array(features_list)

    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """从图像提取面部特征"""
        frame = cv2.imread(image_path)
        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            coords = np.array([[p.x, p.y, p.z] for p in landmarks.landmark])
            return coords.flatten()

        return None

    def close(self):
        """释放资源"""
        self.face_mesh.close()


class FasterLivePortraitEvaluator:
    """FasterLivePortrait 面部驱动模型评估器"""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.pipeline = None
        self.feature_extractor = None
        self.calculator = FRMetricsCalculator()
        self.scorer = FRScorer()

        # 统计信息
        self.total_samples = 0
        self.success_samples = 0
        self.failed_samples = 0

    def _init_pipeline(self):
        """初始化 FasterLivePortrait Pipeline"""
        if self.pipeline is not None:
            return

        print("正在初始化 FasterLivePortrait Pipeline...")

        from omegaconf import OmegaConf
        from FasterLivePortrait.src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline

        cfg_path = PROJECT_ROOT / self.config.flp_config_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

        cfg = OmegaConf.load(str(cfg_path))

        # 更新 checkpoint 路径
        checkpoint_dir = PROJECT_ROOT / self.config.flp_checkpoint_dir
        if checkpoint_dir.exists():
            self._update_checkpoint_paths(cfg, str(checkpoint_dir))

        self.pipeline = GradioLivePortraitPipeline(cfg=cfg)
        print("FasterLivePortrait Pipeline 初始化完成")

    def _update_checkpoint_paths(self, cfg, checkpoint_dir: str):
        """更新配置中的 checkpoint 路径"""
        for model_name in cfg.models:
            if isinstance(cfg.models[model_name].model_path, str):
                cfg.models[model_name].model_path = cfg.models[model_name].model_path.replace(
                    "./checkpoints", checkpoint_dir
                )
            elif isinstance(cfg.models[model_name].model_path, list):
                for i in range(len(cfg.models[model_name].model_path)):
                    cfg.models[model_name].model_path[i] = cfg.models[model_name].model_path[i].replace(
                        "./checkpoints", checkpoint_dir
                    )

    def _init_feature_extractor(self):
        """初始化特征提取器"""
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()

    def get_test_pairs(self, limit: int = None) -> List[Tuple[str, str]]:
        """
        获取测试数据对 (音频, GT视频)

        Args:
            limit: 限制返回数量

        Returns:
            List of (audio_path, gt_video_path) tuples
        """
        val_path = PROJECT_ROOT / self.config.val_dir
        audio_dir = val_path / "Audio_files"

        if not audio_dir.exists():
            raise FileNotFoundError(f"测试数据集目录不存在: {audio_dir}")

        pairs = []

        # 遍历音频文件
        for audio_path in audio_dir.rglob("*.wav"):
            # 构建对应的 GT 视频路径
            rel_path = audio_path.relative_to(audio_dir)
            gt_video_path = val_path / "Video_files" / rel_path.with_suffix(".mp4")

            if gt_video_path.exists():
                pairs.append((str(audio_path), str(gt_video_path)))

                if limit and len(pairs) >= limit:
                    break

        return pairs

    def get_first_frame(self, video_path: str, save_dir: str) -> Optional[str]:
        """提取视频第一帧并保存为图像"""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # 保存第一帧
        frame_name = Path(video_path).stem + "_source.jpg"
        save_path = os.path.join(save_dir, frame_name)
        cv2.imwrite(save_path, frame)

        return save_path

    def evaluate_single(
        self,
        audio_path: str,
        gt_video_path: str,
        save_dir: str
    ) -> Tuple[Optional[FRMetrics], Optional[str]]:
        """
        评估单个样本

        Args:
            audio_path: 音频文件路径
            gt_video_path: Ground Truth 视频路径
            save_dir: 临时保存目录

        Returns:
            (metrics, generated_video_path) 或
        """
        try:
            # 1. 获取源图像 (GT 视频第一帧)
            if self.config.use_first_frame_as_source:
                source_image = self.get_first_frame(gt_video_path, save_dir)
                if source_image is None:
                    print(f"  警告: 无法提取源图像 - {gt_video_path}")
                    return None, None
            else:
                raise NotImplementedError("目前仅支持使用 GT 视频第一帧作为源图像")

            # 2. 运行音频驱动
            print(f"  正在生成视频...")
            gen_video_path, _, gen_time = self.pipeline.run_audio_driving(
                driving_audio_path=audio_path,
                source_path=source_image,
                save_dir=save_dir
            )

            if not os.path.exists(gen_video_path):
                print(f"  警告: 生成视频失败")
                return None, None

            print(f"  生成完成，耗时: {gen_time:.2f}s")

            # 3. 提取特征
            print(f"  正在提取特征...")
            gen_features = self.feature_extractor.extract_from_video(gen_video_path)
            gt_features = self.feature_extractor.extract_from_video(gt_video_path)

            if len(gen_features) == 0 or len(gt_features) == 0:
                print(f"  警告: 特征提取失败")
                return None, None

            print(f"  生成帧数: {len(gen_features)}, GT帧数: {len(gt_features)}")

            # 4. 计算指标
            metrics = FRMetrics()
            metrics.FRCorr = self.calculator.compute_FRCorr(gen_features, gt_features)
            metrics.FRdist = self.calculator.compute_FRdist(gen_features, gt_features)
            metrics.FRDiv = self.calculator.compute_FRDiv(gen_features)
            metrics.FRVar = self.calculator.compute_FRVar(gen_features)

            # 不保存生成的视频 (除非配置要求)
            if not self.config.save_generated_videos:
                if os.path.exists(gen_video_path):
                    os.remove(gen_video_path)
                gen_video_path = None

            return metrics, gen_video_path

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def evaluate_batch(self) -> Dict:
        """
        批量评估

        Returns:
            评估结果字典
        """
        # 初始化
        self._init_pipeline()
        self._init_feature_extractor()

        # 创建输出目录
        output_dir = PROJECT_ROOT / self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取测试数据
        print(f"\n正在扫描测试数据...")
        test_pairs = self.get_test_pairs(limit=self.config.sample_limit)
        print(f"找到 {len(test_pairs)} 个测试样本")

        if not test_pairs:
            print("错误: 没有找到测试数据")
            return {}

        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="fr_eval_")

        try:
            all_metrics = []
            all_features = []  # 用于计算 FRDvs

            for i, (audio_path, gt_video_path) in enumerate(tqdm(test_pairs, desc="评估进度")):
                self.total_samples += 1

                print(f"\n[{i+1}/{len(test_pairs)}] 处理: {Path(audio_path).name}")

                # 创建样本专属目录
                sample_dir = os.path.join(temp_dir, f"sample_{i}")
                os.makedirs(sample_dir, exist_ok=True)

                metrics, gen_video = self.evaluate_single(
                    audio_path, gt_video_path, sample_dir
                )

                if metrics is not None:
                    all_metrics.append(metrics)
                    self.success_samples += 1

                    # 收集特征用于 FRDvs 计算
                    if gen_video and os.path.exists(gen_video):
                        features = self.feature_extractor.extract_from_video(gen_video)
                        if len(features) > 0:
                            all_features.append(features)
                else:
                    self.failed_samples += 1

            # 计算 FRDvs (分布多样性)
            if len(all_features) >= 2:
                frdvs = self.calculator.compute_FRDvs(all_features)
                for m in all_metrics:
                    m.FRDvs = frdvs

            # 汇总结果
            if all_metrics:
                avg_metrics = aggregate_metrics(all_metrics)
                avg_score = self.scorer.compute_score(avg_metrics)
            else:
                avg_metrics = FRMetrics()
                avg_score = FRScore()

            # 打印报告
            print_metrics_report(avg_metrics, avg_score, self.success_samples)

            # 保存结果
            result = {
                'config': {
                    'val_dir': self.config.val_dir,
                    'sample_limit': self.config.sample_limit,
                },
                'statistics': {
                    'total_samples': self.total_samples,
                    'success_samples': self.success_samples,
                    'failed_samples': self.failed_samples,
                },
                'metrics': avg_metrics.to_dict(),
                'score': avg_score.to_dict(),
            }

            # 保存 JSON
            import json
            result_file = output_dir / "evaluation_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {result_file}")

            return result

        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"\n已清理临时目录")

            # 释放资源
            if self.feature_extractor:
                self.feature_extractor.close()


def run_evaluation(
    val_dir: str = "val/",
    output_dir: str = "results/fr_eval/",
    sample_limit: int = 20,
    save_videos: bool = False
) -> Dict:
    """
    运行评估的便捷函数

    Args:
        val_dir: 测试数据集目录
        output_dir: 输出目录
        sample_limit: 样本数量限制
        save_videos: 是否保存生成的视频

    Returns:
        评估结果字典
    """
    config = EvaluationConfig(
        val_dir=val_dir,
        output_dir=output_dir,
        sample_limit=sample_limit,
        save_generated_videos=save_videos
    )

    evaluator = FasterLivePortraitEvaluator(config)
    return evaluator.evaluate_batch()
