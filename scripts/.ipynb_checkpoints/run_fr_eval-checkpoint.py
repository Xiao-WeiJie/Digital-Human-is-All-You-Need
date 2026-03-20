#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FasterLivePortrait FR 指标评估脚本 (Phase 1)

用途:
    评估 FasterLivePortrait 面部驱动模型在 val/ 测试数据集上的性能

测试指标 (5 项基础指标, 满分 50):
    - FRCorr: 相关性 (10分)
    - FRdist: 距离 (10分)
    - FRDiv:  多样性 (10分)
    - FRDvs:  分布多样性 (10分)
    - FRVar:  动态方差 (10分)

使用方法:
    python scripts/run_fr_eval.py [选项]

示例:
    # 使用默认配置 (20 个样本)
    python scripts/run_fr_eval.py

    # 指定样本数量
    python scripts/run_fr_eval.py --samples 50

    # 指定数据集目录
    python scripts/run_fr_eval.py --val-dir val/ --samples 30

    # 保存生成的视频
    python scripts/run_fr_eval.py --save-videos

前置准备:
    1. 确保 val/ 目录存在且包含测试数据
    2. 确保 FasterLivePortrait 模型已下载
    3. 安装必要的依赖: pip install mediapipe
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fr_evaluation import (
    EvaluationConfig,
    FasterLivePortraitEvaluator,
    run_evaluation
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="FasterLivePortrait FR 指标评估脚本 (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/run_fr_eval.py                          # 默认 20 个样本
    python scripts/run_fr_eval.py --samples 50             # 评估 50 个样本
    python scripts/run_fr_eval.py --val-dir ./val/         # 指定数据集目录
    python scripts/run_fr_eval.py --save-videos            # 保存生成的视频

评估指标:
    FRCorr (↑): 相关性，生成行为与驱动信号的匹配程度
    FRdist (↓): 距离，生成结果与参考目标的差异
    FRDiv  (↑): 多样性，表情动作的丰富性
    FRDvs  (↑): 分布多样性，不同样本间的丰富程度
    FRVar  (↑): 动态方差，面部动作的自然波动
        """
    )

    parser.add_argument(
        '--val-dir',
        type=str,
        default='/root/autodl-tmp/val/',
        help='测试数据集目录 (默认: val/)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/root/autodl-tmp/results/fr_eval/',
        help='评估结果输出目录 (默认: results/fr_eval/)'
    )

    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=20,
        help='评估样本数量限制 (默认: 20)'
    )

    parser.add_argument(
        '--save-videos',
        action='store_true',
        help='保存生成的视频 (默认不保存)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='/root/autodl-tmp/FasterLivePortrait/configs/trt_infer.yaml',
        help='FasterLivePortrait 配置文件路径'
    )

    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='检查依赖是否安装'
    )

    return parser.parse_args()


def check_dependencies():
    """检查必要的依赖"""
    print("=" * 60)
    print("依赖检查")
    print("=" * 60)

    dependencies = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('torch', 'torch'),
        ('omegaconf', 'omegaconf'),
        ('tqdm', 'tqdm'),
    ]

    all_ok = True
    for module_name, pip_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  ✗ {pip_name} (未安装)")
            all_ok = False

    print("=" * 60)

    if all_ok:
        print("所有依赖已安装!")
    else:
        print("\n请安装缺失的依赖:")
        print("  pip install numpy opencv-python mediapipe torch omegaconf tqdm")

    return all_ok


def check_data_dir(val_dir: str) -> bool:
    """检查测试数据目录"""
    val_path = Path(val_dir)

    if not val_path.exists():
        print(f"错误: 测试数据目录不存在: {val_path}")
        return False

    audio_dir = val_path / "Audio_files"
    video_dir = val_path / "Video_files"

    if not audio_dir.exists():
        print(f"错误: 音频目录不存在: {audio_dir}")
        return False

    if not video_dir.exists():
        print(f"错误: 视频目录不存在: {video_dir}")
        return False

    # 统计文件数量
    audio_count = len(list(audio_dir.rglob("*.wav")))
    video_count = len(list(video_dir.rglob("*.mp4")))

    print(f"\n测试数据统计:")
    print(f"  音频文件: {audio_count}")
    print(f"  视频文件: {video_count}")

    if audio_count == 0:
        print("错误: 没有找到音频文件")
        return False

    return True


def check_model_files(config_path: str) -> bool:
    """检查模型文件"""
    print("\n模型文件检查:")

    cfg_path = PROJECT_ROOT / config_path
    if not cfg_path.exists():
        print(f"  ✗ 配置文件不存在: {cfg_path}")
        return False

    print(f"  ✓ 配置文件: {cfg_path}")

    # 检查 checkpoint 目录
    checkpoint_dir = PROJECT_ROOT / "FasterLivePortrait" / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"  ✗ Checkpoint 目录不存在: {checkpoint_dir}")
        print(f"    请先下载模型:")
        print(f"    huggingface-cli download warmshao/FasterLivePortrait --local-dir FasterLivePortrait/checkpoints")
        return False

    # 检查关键模型
    key_models = [
        "liveportrait_onnx/motion_extractor.trt",
        "liveportrait_onnx/warping_spade-fix.trt",
        "JoyVASA/motion_generator/motion_generator_hubert_chinese.pt",
    ]

    all_ok = True
    for model_rel in key_models:
        model_path = checkpoint_dir / model_rel.replace("liveportrait_onnx/", "")
        # 也检查 .onnx 版本
        onnx_path = checkpoint_dir / model_rel.replace(".trt", ".onnx")

        if model_path.exists() or onnx_path.exists():
            print(f"  ✓ {model_rel}")
        else:
            print(f"  ? {model_rel} (可能需要转换)")
            all_ok = False

    return True


def main():
    """主函数"""
    args = parse_args()

    print("=" * 70)
    print("       FasterLivePortrait FR 指标评估 (Phase 1)")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查依赖
    if args.check_deps:
        check_dependencies()
        return

    # 前置检查
    print("\n前置检查...")

    if not check_dependencies():
        print("\n错误: 请先安装缺失的依赖")
        sys.exit(1)

    if not check_data_dir(args.val_dir):
        print("\n错误: 请先准备测试数据")
        sys.exit(1)

    if not check_model_files(args.config):
        print("\n警告: 部分模型文件可能缺失，评估可能失败")

    # 显示配置
    print("\n" + "-" * 70)
    print("评估配置:")
    print(f"  测试数据目录: {args.val_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  样本数量: {args.samples}")
    print(f"  保存视频: {'是' if args.save_videos else '否'}")
    print("-" * 70)

    # 确认运行
    try:
        response = input("\n是否开始评估? [Y/n]: ").strip().lower()
        if response and response != 'y':
            print("已取消")
            return
    except EOFError:
        pass

    # 运行评估
    try:
        config = EvaluationConfig(
            val_dir=args.val_dir,
            output_dir=args.output_dir,
            sample_limit=args.samples,
            save_generated_videos=args.save_videos,
            flp_config_path=args.config
        )

        evaluator = FasterLivePortraitEvaluator(config)
        result = evaluator.evaluate_batch()

        print("\n" + "=" * 70)
        print("评估完成!")
        print("=" * 70)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if result:
            print(f"\n结果已保存到: {PROJECT_ROOT / args.output_dir}")

    except KeyboardInterrupt:
        print("\n\n评估被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
