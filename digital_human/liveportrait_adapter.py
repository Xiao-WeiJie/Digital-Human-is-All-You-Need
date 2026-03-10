# -*- coding: utf-8 -*-
"""
Digital Human Phase 1 - FasterLivePortrait Adapter

封装 FasterLivePortrait 动画生成功能，复用现有 pipeline。
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# 添加 FasterLivePortrait 目录到路径
FLP_DIR = Path(__file__).parent.parent / "FasterLivePortrait"
if str(FLP_DIR) not in sys.path:
    sys.path.insert(0, str(FLP_DIR))

from omegaconf import OmegaConf

from .config import FasterLivePortraitConfig


class LivePortraitAdapter:
    """FasterLivePortrait 适配器"""

    def __init__(self, config: Optional[FasterLivePortraitConfig] = None):
        self.config = config or FasterLivePortraitConfig()
        self.pipeline = None
        self.joyvasa_pipeline = None
        self._initialized = False
        self._ffmpeg = self._get_ffmpeg_path()

    def _get_ffmpeg_path(self) -> str:
        """获取 ffmpeg 路径"""
        import platform
        if platform.system().lower() == 'windows':
            return str(FLP_DIR / "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe")
        return "ffmpeg"

    def initialize(self):
        """初始化模型（延迟加载）"""
        if self._initialized:
            return

        print("[LivePortrait] 正在初始化模型...")
        t0 = time.time()

        # 加载配置
        config_path = FLP_DIR / self.config.config_path
        if not config_path.exists():
            # 尝试 ONNX 配置
            config_path = FLP_DIR / "configs/onnx_infer.yaml"

        infer_cfg = OmegaConf.load(str(config_path))

        # ============================================================
        # 应用所有配置参数
        # ============================================================

        # ★ 核心动画参数
        infer_cfg.infer_params.cfg_scale = self.config.cfg_scale
        infer_cfg.infer_params.driving_multiplier = self.config.driving_multiplier

        # ★ 唇部相关参数
        infer_cfg.infer_params.flag_normalize_lip = self.config.flag_normalize_lip
        infer_cfg.infer_params.lip_normalize_threshold = self.config.lip_normalize_threshold

        # 拼接/融合参数
        infer_cfg.infer_params.flag_pasteback = self.config.flag_pasteback
        infer_cfg.infer_params.flag_stitching = self.config.flag_stitching

        # 运动模式参数
        infer_cfg.infer_params.flag_relative_motion = self.config.flag_relative_motion
        infer_cfg.infer_params.flag_do_crop = self.config.flag_do_crop
        infer_cfg.infer_params.flag_do_rot = self.config.flag_do_rot

        # 眼睛/嘴唇重定向参数
        infer_cfg.infer_params.flag_eye_retargeting = self.config.flag_eye_retargeting
        infer_cfg.infer_params.flag_lip_retargeting = self.config.flag_lip_retargeting
        infer_cfg.infer_params.flag_source_video_eye_retargeting = self.config.flag_source_video_eye_retargeting
        infer_cfg.infer_params.source_video_eye_retargeting_threshold = self.config.source_video_eye_retargeting_threshold

        # 平滑参数
        infer_cfg.infer_params.driving_smooth_observation_variance = self.config.driving_smooth_observation_variance

        # 高级参数
        infer_cfg.infer_params.animation_region = self.config.animation_region
        infer_cfg.infer_params.flag_crop_driving_video = self.config.flag_crop_driving_video
        infer_cfg.infer_params.flag_video_editing_head_rotation = self.config.flag_video_editing_head_rotation
        infer_cfg.infer_params.anchor_frame = self.config.anchor_frame
        infer_cfg.infer_params.cfg_mode = self.config.cfg_mode

        # 裁剪参数
        if not hasattr(infer_cfg, 'crop_params'):
            infer_cfg.crop_params = {}
        infer_cfg.crop_params.src_dsize = self.config.src_dsize
        infer_cfg.crop_params.src_scale = self.config.src_scale
        infer_cfg.crop_params.src_vx_ratio = self.config.src_vx_ratio
        infer_cfg.crop_params.src_vy_ratio = self.config.src_vy_ratio
        infer_cfg.crop_params.dri_scale = self.config.dri_scale
        infer_cfg.crop_params.dri_vx_ratio = self.config.dri_vx_ratio
        infer_cfg.crop_params.dri_vy_ratio = self.config.dri_vy_ratio

        print(f"[LivePortrait] 动画参数:")
        print(f"    driving_multiplier={self.config.driving_multiplier}")
        print(f"    cfg_scale={self.config.cfg_scale}")
        print(f"    lip_scale={self.config.lip_scale}")
        print(f"    flag_normalize_lip={self.config.flag_normalize_lip}")
        print(f"    flag_stitching={self.config.flag_stitching}")
        print(f"    flag_pasteback={self.config.flag_pasteback}")

        # 初始化 pipeline
        from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline

        self.pipeline = GradioLivePortraitPipeline(cfg=infer_cfg)

        # 设置 JoyVASA 模型路径（自动查找）
        self._joyvasa_motion_model = Path(self.config.joyvasa_motion_model)
        if not self._joyvasa_motion_model.exists():
            self._joyvasa_motion_model = self._find_model_path(
                "joyvasa_motion_model.pth",
                [
                    "checkpoints/JoyVASA/joyvasa_motion_model.pth",
                    "checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt",
                    "checkpoints/joyvasa_motion_model.pth",
                ]
            )

        self._joyvasa_audio_model = Path(self.config.joyvasa_audio_model)
        if not self._joyvasa_audio_model.exists():
            self._joyvasa_audio_model = self._find_model_path(
                "wav2vec2-base-960h",
                [
                    "checkpoints/wav2vec2-base-960h",
                    "checkpoints/chinese-hubert-base",
                ],
                is_dir=True
            )

        self._joyvasa_motion_template = Path(self.config.joyvasa_motion_template)
        if not self._joyvasa_motion_template.exists():
            self._joyvasa_motion_template = self._find_model_path(
                "motion_template.pkl",
                [
                    "checkpoints/motion_template.pkl",
                    "checkpoints/JoyVASA/motion_template.pkl",
                ]
            )

        print(f"[LivePortrait] JoyVASA 模型路径:")
        print(f"    Motion Model: {self._joyvasa_motion_model}")
        print(f"    Audio Model: {self._joyvasa_audio_model}")
        print(f"    Motion Template: {self._joyvasa_motion_template}")

        print(f"[LivePortrait] 模型初始化完成，耗时 {time.time() - t0:.2f}s")
        self._initialized = True

    def _find_model_path(self, default_name: str, candidates: list, is_dir: bool = False) -> Path:
        """查找模型文件/目录路径"""
        for candidate in candidates:
            path = FLP_DIR / candidate
            if is_dir:
                if path.exists() and path.is_dir():
                    return path
            else:
                if path.exists() and path.is_file():
                    return path

        # 如果都没找到，返回默认路径（会在后续报错）
        return FLP_DIR / "checkpoints" / default_name

    def _init_joyvasa(self):
        """初始化 JoyVASA pipeline"""
        if self.joyvasa_pipeline is not None:
            return

        from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

        self.joyvasa_pipeline = JoyVASAAudio2MotionPipeline(
            motion_model_path=str(self._joyvasa_motion_model),
            audio_model_path=str(self._joyvasa_audio_model),
            motion_template_path=str(self._joyvasa_motion_template),
            cfg_mode="incremental",
            cfg_scale=self.config.cfg_scale,
        )

    def prepare_source(self, source_image_path: str) -> bool:
        """
        准备源图像

        Args:
            source_image_path: 源图像路径

        Returns:
            bool: 是否成功
        """
        if not self._initialized:
            self.initialize()

        # 初始化变量
        self.pipeline.init_vars()

        # 准备源图像
        ret = self.pipeline.prepare_source(source_image_path, realtime=False)
        if not ret:
            print(f"[LivePortrait] 源图像处理失败: {source_image_path}")
            return False

        print(f"[LivePortrait] 源图像准备完成")
        return True

    
    def _stabilize_initial_motion(self, motion_data: dict):
        """
        稳定起始几帧的上半脸 / 视线 / 头姿，避免视频开头突然下视
        只处理前几帧，且尽量不动嘴部区域
        """
        import numpy as np

        motion_list = motion_data.get("motion", [])
        if not motion_list or len(motion_list) < 3:
            return

        warmup_frames = min(8, len(motion_list))
        first = motion_list[0]

        first_exp = first.get("exp")
        if first_exp is None:
            return

        first_exp = np.asarray(first_exp, dtype=np.float32)

        # 只压上半脸/眼周，不碰嘴部关键区域
        upper_face_idx = [
            1, 2, 3, 4, 5,
            7, 9, 10, 11,
            13, 15, 16, 18
        ]

        pose_keys = ["pitch", "yaw", "roll"]

        first_pose = {}
        for k in pose_keys:
            if k in first:
                first_pose[k] = float(first[k])

        first_scale = float(first["scale"]) if "scale" in first else None
        first_t = np.asarray(first["t"], dtype=np.float32) if "t" in first else None

        for i in range(1, warmup_frames):
            frame = motion_list[i]
            alpha = i / float(warmup_frames - 1)
            ease = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep

            exp = frame.get("exp")
            if exp is not None:
                exp = np.asarray(exp, dtype=np.float32)
                frame["exp"] = exp

                for idx in upper_face_idx:
                    exp[0, idx, :] = first_exp[0, idx, :] * (1.0 - ease) + exp[0, idx, :] * ease

            for k in pose_keys:
                if k in frame and k in first_pose:
                    frame[k] = first_pose[k] * (1.0 - ease) + float(frame[k]) * ease

            if "scale" in frame and first_scale is not None:
                frame["scale"] = first_scale * (1.0 - ease) + float(frame["scale"]) * ease

            if "t" in frame and first_t is not None:
                t_cur = np.asarray(frame["t"], dtype=np.float32)
                frame["t"] = first_t * (1.0 - ease) + t_cur * ease

    def _smooth_global_pitch(self, motion_data: dict):
        """
        对整个序列的 pitch 做平滑和突变抑制，防止视线突然下移。
        同时对 yaw/roll 做轻度平滑以减少头部抖动。
        """
        import numpy as np

        motion_list = motion_data.get("motion", [])
        if not motion_list or len(motion_list) < 5:
            return

        smooth_win = int(getattr(self.config, "pitch_smooth_window", 11))
        max_delta = float(getattr(self.config, "pitch_max_delta", 1.8))

        # 提取 pitch 序列
        pitches = []
        for frame in motion_list:
            pitches.append(float(frame.get("pitch", 0.0)))
        pitches = np.asarray(pitches, dtype=np.float64)

        # 第一步：限制相邻帧的最大 pitch 变化
        clamped = pitches.copy()
        for i in range(1, len(clamped)):
            delta = clamped[i] - clamped[i - 1]
            if abs(delta) > max_delta:
                clamped[i] = clamped[i - 1] + np.sign(delta) * max_delta
        # 反向再做一遍，消除单向累积偏差
        for i in range(len(clamped) - 2, -1, -1):
            delta = clamped[i] - clamped[i + 1]
            if abs(delta) > max_delta:
                clamped[i] = clamped[i + 1] + np.sign(delta) * max_delta

        # 第二步：均值滤波平滑
        if smooth_win > 1 and len(clamped) > smooth_win:
            kernel = np.ones(smooth_win, dtype=np.float64) / smooth_win
            # 边界用 "reflect" 模式避免首尾跳变
            padded = np.pad(clamped, (smooth_win // 2, smooth_win // 2), mode="reflect")
            smoothed = np.convolve(padded, kernel, mode="valid")[:len(clamped)]
        else:
            smoothed = clamped

        # 第三步：写回，同时对 yaw 做轻度平滑
        yaws = np.asarray([float(f.get("yaw", 0.0)) for f in motion_list], dtype=np.float64)
        if smooth_win > 1 and len(yaws) > smooth_win:
            yaw_kernel_size = max(3, smooth_win // 2)
            yaw_kernel = np.ones(yaw_kernel_size, dtype=np.float64) / yaw_kernel_size
            yaw_padded = np.pad(yaws, (yaw_kernel_size // 2, yaw_kernel_size // 2), mode="reflect")
            yaw_smoothed = np.convolve(yaw_padded, yaw_kernel, mode="valid")[:len(yaws)]
            # yaw 只做 30% 混合，保留头部自然转动
            yaws = yaws * 0.7 + yaw_smoothed * 0.3

        for i, frame in enumerate(motion_list):
            frame["pitch"] = float(smoothed[i])
            frame["yaw"] = float(yaws[i])

        print(f"[LivePortrait] pitch 平滑完成: 原始范围[{pitches.min():.2f}, {pitches.max():.2f}] "
              f"-> 平滑后[{smoothed.min():.2f}, {smoothed.max():.2f}]")
            
    def _apply_lip_scale(self, motion_data: dict, scale: float, audio_path: Optional[str] = None):
        """
        目标：
        1. 静音时保持自然微张，不闭死
        2. 正常说话时开口明显增大，但不过分夸张
        3. 聚唇/圆唇音时抑制“中线内收 + 前凸过强”，避免尖嘴/吸嘴
        4. 所有 X 修正围绕嘴中心进行，避免嘴斜
        """
        motion_list = motion_data.get("motion", [])
        if not motion_list:
            return

        import json
        import numpy as np

        UPPER_LIP = 6
        JAW = 8
        CORNER_L = 12
        CORNER_R = 14
        LIP_LEFT = 17
        LOWER_LIP = 19
        LIP_RIGHT = 20

        LIP_POINTS = [UPPER_LIP, CORNER_L, CORNER_R, LIP_LEFT, LOWER_LIP, LIP_RIGHT]

        baseline_frames = int(getattr(self.config, "lip_baseline_frames", 12))
        open_threshold = float(getattr(self.config, "lip_open_threshold", 0.0022))
        min_open_delta = float(getattr(self.config, "lip_min_open_delta", 0.0052))
        jaw_follow = float(getattr(self.config, "lip_jaw_follow", 0.62))
        silence_threshold_cfg = float(getattr(self.config, "lip_audio_silence_threshold", 0.010))
        z_soft_limit = float(getattr(self.config, "lip_z_soft_limit", 0.0075))

        # -------------------------
        # 1) 构造更稳的 baseline
        # -------------------------
        valid_exps = []
        open_list = []
        max_n = max(1, min(baseline_frames, len(motion_list)))

        for frame in motion_list[:max_n]:
            exp = frame.get("exp")
            if exp is None:
                continue
            arr = np.asarray(exp, dtype=np.float32)
            valid_exps.append(arr)
            upper_y = float(arr[0, UPPER_LIP, 1])
            lower_y = float(arr[0, LOWER_LIP, 1])
            open_list.append(lower_y - upper_y)

        if not valid_exps:
            return

        valid_exps = np.stack(valid_exps, axis=0)
        open_arr = np.asarray(open_list, dtype=np.float32)

        # 用较开的一组前几帧估 baseline，避免被首帧闭嘴拖死
        # 取 80 分位数（比 70 更偏向张开状态），筛选门槛放宽到 0.90
        q = float(np.quantile(open_arr, 0.80)) if len(open_arr) > 0 else 0.0
        keep_idx = [i for i, v in enumerate(open_arr) if v >= q * 0.90] if len(open_arr) > 0 else list(range(len(valid_exps)))
        if not keep_idx:
            keep_idx = list(range(len(valid_exps)))

        base_exp = np.mean(valid_exps[keep_idx], axis=0)

        base_values = {}
        for idx in LIP_POINTS:
            base_values[idx] = {
                "x": float(base_exp[0, idx, 0]),
                "y": float(base_exp[0, idx, 1]),
                "z": float(base_exp[0, idx, 2]),
            }

        base_jaw_y = float(base_exp[0, JAW, 1])
        raw_base_open = base_values[LOWER_LIP]["y"] - base_values[UPPER_LIP]["y"]

        # 静音自然唇缝 —— idle_open 上限需 >= idle_min_gap，否则 gap 保底会被 ceiling 截断
        idle_min_gap = float(getattr(self.config, "lip_idle_min_gap", 0.0075))
        idle_open = min(max(raw_base_open, idle_min_gap), 0.0095)

        # baseline 嘴宽与中心
        base_width = (
            abs(base_values[CORNER_R]["x"] - base_values[CORNER_L]["x"]) +
            abs(base_values[LIP_RIGHT]["x"] - base_values[LIP_LEFT]["x"])
        ) * 0.5
        base_center_x = 0.5 * (base_values[CORNER_L]["x"] + base_values[CORNER_R]["x"])

        # -------------------------
        # 2) 构建音频 mask
        # -------------------------
        audio_rms = np.zeros(len(motion_list), dtype=np.float32)
        active_mask = np.zeros(len(motion_list), dtype=bool)
        speaking_mask = np.zeros(len(motion_list), dtype=bool)

        if audio_path:
            try:
                import torchaudio

                wav, sr = torchaudio.load(audio_path)
                wav = wav.mean(dim=0).numpy().astype(np.float32)

                if wav.size > 0:
                    fps = float(motion_data.get("output_fps") or motion_data.get("fps") or 25.0)
                    hop = max(1, int(sr / fps))
                    win = max(hop, int(sr * 0.05))

                    rms_vals = []
                    for start in range(0, len(wav), hop):
                        seg = wav[start:start + win]
                        if seg.size == 0:
                            break
                        rms_vals.append(float(np.sqrt(np.mean(seg * seg) + 1e-8)))

                    if rms_vals:
                        rms_vals = np.asarray(rms_vals, dtype=np.float32)
                        x_old = np.linspace(0.0, 1.0, num=len(rms_vals), endpoint=True)
                        x_new = np.linspace(0.0, 1.0, num=len(motion_list), endpoint=True)
                        audio_rms = np.interp(x_new, x_old, rms_vals).astype(np.float32)

                        kernel = np.ones(7, dtype=np.float32) / 7.0
                        audio_rms = np.convolve(audio_rms, kernel, mode="same")

                        rms_mean = float(np.mean(audio_rms))
                        rms_p50 = float(np.quantile(audio_rms, 0.50))
                        rms_p68 = float(np.quantile(audio_rms, 0.68))

                        active_thr = max(silence_threshold_cfg, min(rms_p50, rms_mean * 1.00))
                        speaking_thr = max(active_thr * 1.06, min(rms_p68, rms_mean * 1.08))

                        active_mask = audio_rms >= active_thr
                        speaking_mask = audio_rms >= speaking_thr
            except Exception as e:
                print(f"[LivePortrait] lip_debug audio load failed: {e}")

        rms_max = float(np.max(audio_rms)) if audio_rms.size else 0.0
        openness_values = []

        # -------------------------
        # 3) 逐帧修正
        # -------------------------
        for i, frame in enumerate(motion_list):
            exp = frame.get("exp")
            if exp is None:
                continue

            exp = np.asarray(exp, dtype=np.float32)
            frame["exp"] = exp

            # 初始状态
            upper_y = float(exp[0, UPPER_LIP, 1])
            lower_y = float(exp[0, LOWER_LIP, 1])
            current_open = lower_y - upper_y

            cur_width = (
                abs(float(exp[0, CORNER_R, 0]) - float(exp[0, CORNER_L, 0])) +
                abs(float(exp[0, LIP_RIGHT, 0]) - float(exp[0, LIP_LEFT, 0]))
            ) * 0.5

            # A. 发声帧：保底 + 乘性增强双管齐下
            if active_mask[i]:
                r = float(audio_rms[i])
                r_norm = (r / rms_max) if rms_max > 1e-6 else 0.0

                # A-1. 保底：确保最小开口
                # 提高上限 0.020 → 0.028，让大音量帧有更大的保底开口
                target_min_open = idle_open + min_open_delta * (0.60 + 0.95 * r_norm)
                target_min_open = min(target_min_open, 0.028)

                current_open = float(exp[0, LOWER_LIP, 1]) - float(exp[0, UPPER_LIP, 1])
                if current_open < target_min_open:
                    deficit = target_min_open - current_open
                    exp[0, UPPER_LIP, 1] -= deficit * 0.20
                    exp[0, LOWER_LIP, 1] += deficit * 0.80
                    exp[0, JAW, 1] += deficit * jaw_follow * 0.7

                # A-2. 乘性增强：不管是否已达到保底，都额外放大已有的开口
                # 提高乘数范围：1.15~1.50 → 1.25~1.70
                current_open = float(exp[0, LOWER_LIP, 1]) - float(exp[0, UPPER_LIP, 1])
                if current_open > idle_open:
                    extra_open = current_open - idle_open
                    speak_mult = 1.25 + 0.45 * r_norm
                    boosted_open = idle_open + extra_open * speak_mult
                    new_deficit = boosted_open - current_open
                    if new_deficit > 0:
                        exp[0, UPPER_LIP, 1] -= new_deficit * 0.18
                        exp[0, LOWER_LIP, 1] += new_deficit * 0.82
                        exp[0, JAW, 1] += new_deficit * jaw_follow * 0.5

            # 重新计算
            current_open = float(exp[0, LOWER_LIP, 1]) - float(exp[0, UPPER_LIP, 1])
            open_delta = current_open - idle_open
            is_opening = open_delta > open_threshold

            # B. 在 A 的基础上，用 lip_scale 做额外的 delta 放大
            if is_opening and scale > 1.0:
                y_gain_u = 1.0 + (scale - 1.0) * 0.30
                y_gain_l = 1.0 + (scale - 1.0) * 0.75
                jaw_gain = 1.0 + (scale - 1.0) * 0.45

                upper_delta_y = float(exp[0, UPPER_LIP, 1]) - base_values[UPPER_LIP]["y"]
                lower_delta_y = float(exp[0, LOWER_LIP, 1]) - base_values[LOWER_LIP]["y"]
                jaw_delta_y = float(exp[0, JAW, 1]) - base_jaw_y

                exp[0, UPPER_LIP, 1] = base_values[UPPER_LIP]["y"] + upper_delta_y * y_gain_u
                exp[0, LOWER_LIP, 1] = base_values[LOWER_LIP]["y"] + lower_delta_y * y_gain_l
                exp[0, JAW, 1] = base_jaw_y + jaw_delta_y * jaw_gain

            # 重新计算当前状态
            current_open = float(exp[0, LOWER_LIP, 1]) - float(exp[0, UPPER_LIP, 1])
            cur_width = (
                abs(float(exp[0, CORNER_R, 0]) - float(exp[0, CORNER_L, 0])) +
                abs(float(exp[0, LIP_RIGHT, 0]) - float(exp[0, LIP_LEFT, 0]))
            ) * 0.5
            width_ratio = cur_width / max(base_width, 1e-6)

            cur_center_x = 0.5 * (float(exp[0, CORNER_L, 0]) + float(exp[0, CORNER_R, 0]))

            upper_dz = float(exp[0, UPPER_LIP, 2]) - base_values[UPPER_LIP]["z"]
            lower_dz = float(exp[0, LOWER_LIP, 2]) - base_values[LOWER_LIP]["z"]
            left_dz = float(exp[0, LIP_LEFT, 2]) - base_values[LIP_LEFT]["z"]
            right_dz = float(exp[0, LIP_RIGHT, 2]) - base_values[LIP_RIGHT]["z"]
            lip_forward = max(upper_dz, lower_dz, left_dz, right_dz)

            # C. 聚唇识别（适度放宽检测，加强矫正力度）
            # 放宽 mild 阈值以更早捕获"文化"等聚唇音的起始帧
            mild_pursing = (
                (width_ratio < 0.97 and lip_forward > 0.002) or
                (width_ratio < 0.95 and current_open < idle_open + 0.004)
            )
            strong_pursing = (
                (width_ratio < 0.94 and lip_forward > 0.003) or
                (width_ratio < 0.92)
            )

            if mild_pursing:
                # 围绕嘴中心做 X 宽度恢复
                cl_dx = float(exp[0, CORNER_L, 0]) - cur_center_x
                cr_dx = float(exp[0, CORNER_R, 0]) - cur_center_x
                ll_dx = float(exp[0, LIP_LEFT, 0]) - cur_center_x
                lr_dx = float(exp[0, LIP_RIGHT, 0]) - cur_center_x
                ul_dx = float(exp[0, UPPER_LIP, 0]) - cur_center_x
                dl_dx = float(exp[0, LOWER_LIP, 0]) - cur_center_x

                # 提高恢复目标和 gain 上限
                target_width_ratio = 0.98 if strong_pursing else 1.00
                if width_ratio < target_width_ratio:
                    width_gain = target_width_ratio / max(width_ratio, 1e-6)
                    width_gain = min(width_gain, 1.25 if strong_pursing else 1.14)

                    exp[0, CORNER_L, 0] = cur_center_x + cl_dx * width_gain
                    exp[0, CORNER_R, 0] = cur_center_x + cr_dx * width_gain
                    exp[0, LIP_LEFT, 0] = cur_center_x + ll_dx * (1.0 + (width_gain - 1.0) * 0.95)
                    exp[0, LIP_RIGHT, 0] = cur_center_x + lr_dx * (1.0 + (width_gain - 1.0) * 0.95)
                    exp[0, UPPER_LIP, 0] = cur_center_x + ul_dx * (1.0 + (width_gain - 1.0) * 0.70)
                    exp[0, LOWER_LIP, 0] = cur_center_x + dl_dx * (1.0 + (width_gain - 1.0) * 0.85)

                # 聚唇时保开口
                current_open2 = float(exp[0, LOWER_LIP, 1]) - float(exp[0, UPPER_LIP, 1])
                min_purse_open = idle_open + (0.006 if strong_pursing else 0.004)
                if current_open2 < min_purse_open:
                    deficit = min_purse_open - current_open2
                    exp[0, UPPER_LIP, 1] -= deficit * 0.16
                    exp[0, LOWER_LIP, 1] += deficit * 0.84
                    exp[0, JAW, 1] += deficit * 0.25

                # Z 轴前凸压制——更激进的限制，消灭尖嘴
                # strong: z_limit = 0.007*0.35=0.00245, mild: 0.007*0.55=0.00385
                z_limit = z_soft_limit * (0.35 if strong_pursing else 0.55)
                z_ratio = 0.03 if strong_pursing else 0.08
                for idx in [UPPER_LIP, LOWER_LIP, LIP_LEFT, LIP_RIGHT, CORNER_L, CORNER_R]:
                    dz = float(exp[0, idx, 2]) - base_values.get(idx, {"z": 0.0})["z"]
                    base_z = base_values.get(idx, {"z": float(exp[0, idx, 2])})["z"]
                    if dz > z_limit:
                        exp[0, idx, 2] = base_z + z_limit + (dz - z_limit) * z_ratio

                # 拉回中心
                new_center_x = 0.5 * (float(exp[0, CORNER_L, 0]) + float(exp[0, CORNER_R, 0]))
                center_shift = (base_center_x - new_center_x) * 0.30
                for idx in [CORNER_L, CORNER_R, LIP_LEFT, LIP_RIGHT, UPPER_LIP, LOWER_LIP]:
                    exp[0, idx, 0] += center_shift

            # 注意：已删除原来的 C2 全局 Z 轴压制！
            # 之前对所有帧做 Z 限制 (z_soft_limit=0.003) 是导致嘴巴打不开的主因之一
            # Z 轴是嘴巴张开运动的必要分量，不能在全局压死
            # 现在 Z 限制只在上面的聚唇分支(mild_pursing)内执行

            # D. 静音帧：保持自然微张，防止上下唇贴合/重叠
            if not active_mask[i]:
                current_open = float(exp[0, LOWER_LIP, 1]) - float(exp[0, UPPER_LIP, 1])
                if current_open < idle_open:
                    deficit = idle_open - current_open
                    # 加大上唇上提比例（0.25→0.35），避免只靠下唇撑开导致不自然
                    exp[0, UPPER_LIP, 1] -= deficit * 0.35
                    exp[0, LOWER_LIP, 1] += deficit * 0.65
                    exp[0, JAW, 1] += deficit * 0.18

            # E. 全局对称修正
            global_asym_thr = float(getattr(self.config, "lip_global_asym_threshold", 0.06))
            gc_x = 0.5 * (float(exp[0, CORNER_L, 0]) + float(exp[0, CORNER_R, 0]))
            g_cl = abs(float(exp[0, CORNER_L, 0]) - gc_x)
            g_cr = abs(float(exp[0, CORNER_R, 0]) - gc_x)
            g_ll = abs(float(exp[0, LIP_LEFT, 0]) - gc_x)
            g_lr = abs(float(exp[0, LIP_RIGHT, 0]) - gc_x)
            g_left_span = g_cl + g_ll
            g_right_span = g_cr + g_lr
            g_asym = abs(g_left_span - g_right_span) / max(g_left_span + g_right_span, 1e-6)

            if g_asym > global_asym_thr:
                blend = min((g_asym - global_asym_thr) / 0.08, 0.75)
                avg_corner = 0.5 * (g_cl + g_cr)
                avg_side = 0.5 * (g_ll + g_lr)
                exp[0, CORNER_L, 0] = float(exp[0, CORNER_L, 0]) * (1 - blend) + (gc_x - avg_corner) * blend
                exp[0, CORNER_R, 0] = float(exp[0, CORNER_R, 0]) * (1 - blend) + (gc_x + avg_corner) * blend
                exp[0, LIP_LEFT, 0] = float(exp[0, LIP_LEFT, 0]) * (1 - blend) + (gc_x - avg_side) * blend
                exp[0, LIP_RIGHT, 0] = float(exp[0, LIP_RIGHT, 0]) * (1 - blend) + (gc_x + avg_side) * blend

            # F. 安全网：上下唇 Y 最小间距 + 重叠修复
            final_gap = float(exp[0, LOWER_LIP, 1]) - float(exp[0, UPPER_LIP, 1])
            if final_gap < idle_min_gap:
                fix = idle_min_gap - final_gap
                # 如果是重叠（gap < 0），需要更大力度修复
                if final_gap < 0:
                    # 重叠情况：强制拉开，上唇上移多一些
                    exp[0, UPPER_LIP, 1] -= fix * 0.40
                    exp[0, LOWER_LIP, 1] += fix * 0.60
                    exp[0, JAW, 1] += fix * 0.15
                else:
                    exp[0, UPPER_LIP, 1] -= fix * 0.30
                    exp[0, LOWER_LIP, 1] += fix * 0.70

            openness_values.append(float(exp[0, LOWER_LIP, 1]) - float(exp[0, UPPER_LIP, 1]))

        if openness_values:
            lip_debug = {
                "idle_open": round(float(idle_open), 6),
                "open_min": round(float(np.min(openness_values)), 6),
                "open_max": round(float(np.max(openness_values)), 6),
                "open_mean": round(float(np.mean(openness_values)), 6),
                "speaking_frames": int(np.sum(speaking_mask)),
                "active_frames": int(np.sum(active_mask)),
            }
            print(f"lip_debug={json.dumps(lip_debug, ensure_ascii=False)}")

    def generate_motion_from_audio(self, audio_path: str) -> dict:
        """
        从音频生成运动序列

        Args:
            audio_path: 音频文件路径

        Returns:
            dict: 运动序列数据
        """
        if not self._initialized:
            self.initialize()

        self._init_joyvasa()

        print(f"[LivePortrait] 从音频生成运动序列: {audio_path}")
        t0 = time.time()

        # 调试：检查音频文件
        import torchaudio
        try:
            audio, sr = torchaudio.load(audio_path)
            print(f"[LivePortrait] 音频信息: 采样率={sr}, 形状={audio.shape}, 时长={audio.shape[1]/sr:.2f}s")
        except Exception as e:
            print(f"[LivePortrait] 无法读取音频: {e}")

        # 调试：检查 pipeline 参数
        print(
            f"[LivePortrait] JoyVASA 参数: n_motions={self.joyvasa_pipeline.n_motions}, "
            f"fps={self.joyvasa_pipeline.fps}, cfg_scale={self.config.cfg_scale}"
        )

        motion_data = self.joyvasa_pipeline.gen_motion_sequence(audio_path)

        # 先稳定开头几帧的上半脸 / 视线 / 头姿
        self._stabilize_initial_motion(motion_data)

        # 全序列 pitch 平滑，防止视线突变
        self._smooth_global_pitch(motion_data)

        # 再做唇部修正
        self._apply_lip_scale(motion_data, self.config.lip_scale, audio_path)

        if self.config.lip_scale != 1.0:
            print(f"[LivePortrait] 已应用唇部缩放: {self.config.lip_scale}")

        print(f"[LivePortrait] 运动序列生成完成，耗时 {time.time() - t0:.2f}s")
        print(f"[LivePortrait] 帧数: {motion_data['n_frames']}, FPS: {motion_data['output_fps']}")

        if motion_data['n_frames'] == 0:
            print(f"[LivePortrait] WARNING: 生成了0帧!")
            print(f"[LivePortrait] motion list 长度: {len(motion_data.get('motion', []))}")

        return motion_data

    def render_video_from_motion(
        self,
        motion_data: dict,
        source_image_path: str,
        output_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> str:
        """
        从运动序列渲染视频

        Args:
            motion_data: 运动序列数据
            source_image_path: 源图像路径
            output_path: 输出视频路径（可选）
            audio_path: 音频文件路径（用于合成音画）

        Returns:
            str: 输出视频路径
        """
        if not self._initialized:
            self.initialize()

        t0 = time.time()

        # 准备源图像
        if not self.prepare_source(source_image_path):
            raise Exception("无法准备源图像")

        # 创建临时目录
        save_dir = Path(self.config.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存运动数据到临时文件
        motion_pkl_path = save_dir / f"motion_{int(time.time())}.pkl"
        with open(motion_pkl_path, "wb") as f:
            pickle.dump(motion_data, f)

        try:
            # 使用 pipeline 的 pickle 驱动功能
            video_org_path, video_crop_path, _ = self.pipeline.run_pickle_driving(
                str(motion_pkl_path),
                source_image_path,
                save_dir=str(save_dir),
            )

            # 如果有音频，合成音画
            if audio_path and os.path.exists(audio_path):
                video_with_audio = self._merge_audio_video(
                    video_org_path,
                    audio_path,
                    output_path or str(save_dir / "output_audio.mp4")
                )
                # 删除中间视频文件，只保留有声音的视频
                for temp_video in [video_org_path, video_crop_path]:
                    if temp_video and os.path.exists(temp_video):
                        os.remove(temp_video)
                        print(f"[LivePortrait] 已删除中间文件: {temp_video}")
                print(f"[LivePortrait] 视频渲染完成（含音频��，耗时 {time.time() - t0:.2f}s")
                return video_with_audio

            # 重命名输出
            if output_path:
                import shutil
                shutil.move(video_org_path, output_path)
                return output_path

            print(f"[LivePortrait] 视频渲染完成，耗时 {time.time() - t0:.2f}s")
            return video_org_path

        finally:
            # 重命名为固定的调试文件名（不删除，用于调试）
            if motion_pkl_path.exists():
                debug_pkl_path = save_dir / "debug_motion.pkl"
                if debug_pkl_path.exists():
                    debug_pkl_path.unlink()
                motion_pkl_path.rename(debug_pkl_path)
                print(f"[LivePortrait] 运动数据已保存: {debug_pkl_path}")

    def _merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """合并音频和视频"""
        import os

        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"[LivePortrait] 视频文件不存在: {video_path}")
            return video_path
        if not os.path.exists(audio_path):
            print(f"[LivePortrait] 音频文件不存在: {audio_path}")
            return video_path

        video_size = os.path.getsize(video_path)
        audio_size = os.path.getsize(audio_path)
        print(f"[LivePortrait] 合并音视频: 视频={video_size}bytes, 音频={audio_size}bytes")

        # 检查 ffmpeg 是否可用
        ffmpeg_path = self._ffmpeg
        print(f"[LivePortrait] FFmpeg 路径: {ffmpeg_path}")

        # 先测试 ffmpeg 是否能正常运行
        try:
            test_result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, timeout=5)
            if test_result.returncode != 0:
                print(f"[LivePortrait] FFmpeg 不可用: {test_result.stderr}")
                return video_path
            print(f"[LivePortrait] FFmpeg 版本: {test_result.stdout.split()[2] if test_result.stdout else 'unknown'}")
        except FileNotFoundError:
            print(f"[LivePortrait] FFmpeg 未找到: {ffmpeg_path}")
            return video_path
        except Exception as e:
            print(f"[LivePortrait] FFmpeg 检查失败: {e}")
            return video_path

        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-i", audio_path,
            "-b:v", "10M",
            "-c:v", "libx264",
            "-map", "0:v",
            "-map", "1:a",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
            "-y",
        ]

        print(f"[LivePortrait] FFmpeg 命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[LivePortrait] FFmpeg 错误 (返回码 {result.returncode}):")
            print(f"[LivePortrait] stderr: {result.stderr}")
            print(f"[LivePortrait] stdout: {result.stdout}")
            return video_path  # 返回原视频

        # 验证输出文件
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            print(f"[LivePortrait] 合并完成: 输出={output_size}bytes")
            return output_path
        else:
            print(f"[LivePortrait] 输出文件不存在: {output_path}")
            return video_path

    def generate_from_audio(
        self,
        audio_path: str,
        source_image_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        从音频生成动画视频（完整流程）

        Args:
            audio_path: 音频文件路径
            source_image_path: 源图像路径
            output_path: 输出视频路径（可选）

        Returns:
            str: 输出视频路径
        """
        if not self._initialized:
            self.initialize()

        # 生成运动序列
        motion_data = self.generate_motion_from_audio(audio_path)

        # 渲染视频
        return self.render_video_from_motion(
            motion_data,
            source_image_path,
            output_path,
            audio_path,
        )

    def generate_from_text(
        self,
        text: str,
        source_image_path: str,
        audio_path: Optional[str] = None,
        output_path: Optional[str] = None,
        voice_name: str = "af",
    ) -> Tuple[str, str]:
        """
        从文本生成动画视频

        Args:
            text: 输入文本
            source_image_path: 源图像路径
            audio_path: 音频输出路径（可选）
            output_path: 视频输出路径（可选）
            voice_name: 声音名称（用于 Kokoro）

        Returns:
            Tuple[str, str]: (视频路径, 音频路径)
        """
        if not self._initialized:
            self.initialize()

        # 准备输出目录
        save_dir = Path(self.config.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())

        # 使用 pipeline 的文本驱动功能（如果可用）
        if audio_path is None:
            audio_path = str(save_dir / f"audio_{timestamp}.wav")

        # 生成音频
        # 注意：这里使用内置的 Kokoro，也可以外部生成音频后传入
        video_org, video_crop, _ = self.pipeline.run_text_driving(
            text,
            voice_name,
            source_image_path,
            save_dir=str(save_dir),
        )

        # 如果需要特定输出路径
        if output_path and output_path != video_org:
            import shutil
            shutil.move(video_org, output_path)
            return output_path, audio_path

        return video_org, audio_path

    def __call__(
        self,
        audio_or_text: str,
        source_image_path: str,
        is_text: bool = False,
        **kwargs,
    ) -> str:
        """使适配器可调用"""
        if is_text:
            video_path, _ = self.generate_from_text(
                audio_or_text,
                source_image_path,
                **kwargs
            )
            return video_path
        else:
            return self.generate_from_audio(
                audio_or_text,
                source_image_path,
                **kwargs
            )