###############################################################################
#  批量视频生成器 - 方案B核心模块
#  生成完整的数字人回复视频（JoyVASA + FLIP + FFmpeg）
###############################################################################

import os
import sys
import time
import tempfile
import subprocess
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2

# 添加 FasterLivePortrait 路径
fasterlp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FasterLivePortrait")
if fasterlp_path not in sys.path:
    sys.path.insert(0, fasterlp_path)

from logger import logger

# 导入 Motion 后处理器
try:
    from FasterLivePortrait.src.utils.motion_post_processor import MotionPostProcessor, get_motion_post_processor
    MOTION_POST_PROCESSOR_AVAILABLE = True
except ImportError:
    logger.warning("[BatchVideo] MotionPostProcessor not available, emotion features disabled")
    MOTION_POST_PROCESSOR_AVAILABLE = False


@dataclass
class VideoGenerationResult:
    """视频生成结果"""
    video_path: str           # 最终视频文件路径
    total_time: float         # 总耗时（秒）
    metrics: Dict[str, float] # 各阶段耗时明细
    audio_duration: float     # 音频时长（秒）
    frame_count: int          # 生成的帧数


class BatchVideoGenerator:
    """
    批量视频生成器 - 生成完整视频后推送

    流程:
    1. JoyVASA: 音频 → 运动序列
    2. FasterLivePortrait: 运动序列 → 视频帧
    3. FFmpeg: 音视频合成
    """

    def __init__(
        self,
        pipeline,
        joyvasa_pipeline,
        ffmpeg_path: str = "ffmpeg",
        output_dir: str = None,
        width: int = 450,
        height: int = 450,
        fps: int = 25
    ):
        """
        初始化批量视频生成器

        Args:
            pipeline: GradioLivePortraitPipeline 实例
            joyvasa_pipeline: JoyVASAAudio2MotionPipeline 实例
            ffmpeg_path: FFmpeg 可执行文件路径
            output_dir: 输出目录，默认为临时目录
            width: 视频宽度
            height: 视频高度
            fps: 帧率
        """
        self.pipeline = pipeline
        self.joyvasa = joyvasa_pipeline
        self.ffmpeg = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps

        # 输出目录
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), "digital_human_videos")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 临时文件目录
        self.temp_dir = Path(tempfile.mkdtemp(prefix="batch_video_"))

        logger.info(f"BatchVideoGenerator initialized, output_dir: {self.output_dir}")

    def generate_video(
        self,
        audio_path: str,
        source_image_path: str,
        output_path: str = None,
        session_id: str = None
    ) -> VideoGenerationResult:
        """
        生成完整的数字人回复视频

        Args:
            audio_path: 输入音频文件路径
            source_image_path: 源图像路径（数字人形象）
            output_path: 输出视频路径，默认自动生成
            session_id: 会话ID，用于生成唯一文件名

        Returns:
            VideoGenerationResult: 生成结果
        """
        start_time = time.time()
        metrics = {}

        # 生成输出路径
        if output_path is None:
            timestamp = int(time.time() * 1000)
            filename = f"video_{session_id or timestamp}.mp4"
            output_path = str(self.output_dir / filename)

        # 获取音频时长
        audio_duration = self._get_audio_duration(audio_path)
        logger.info(f"[BatchVideo] Audio duration: {audio_duration:.2f}s")

        # ========== Step 1: 准备源图像 ==========
        t1 = time.time()
        logger.info("[BatchVideo] Step 1: Preparing source image...")
        success = self.pipeline.prepare_source(source_image_path, realtime=False)
        if not success:
            raise RuntimeError(f"Failed to prepare source: {source_image_path}")

        src_img = self.pipeline.src_imgs[0]
        src_info = self.pipeline.src_infos[0][0]
        metrics['source_prep_time'] = time.time() - t1
        logger.info(f"[BatchVideo] Source prepared in {metrics['source_prep_time']:.2f}s")

        # ========== Step 2: JoyVASA 生成运动序列 ==========
        t2 = time.time()
        logger.info("[BatchVideo] Step 2: JoyVASA generating motion sequence...")
        motion_info = self.joyvasa.gen_motion_sequence(audio_path)
        metrics['joyvasa_time'] = time.time() - t2
        n_frames = motion_info.get('n_frames', 0)
        logger.info(f"[BatchVideo] JoyVASA generated {n_frames} frames in {metrics['joyvasa_time']:.2f}s")

        # ========== Step 3: 保存运动序列到临时文件 ==========
        motion_pickle = str(self.temp_dir / f"motion_{time.time()}.pkl")
        with open(motion_pickle, 'wb') as f:
            pickle.dump(motion_info, f)

        # ========== Step 4: FasterLivePortrait 渲染视频 ==========
        t3 = time.time()
        logger.info("[BatchVideo] Step 3: FasterLivePortrait rendering...")

        # 使用 pipeline 的 run_pickle_driving 方法
        video_path, video_concat_path, render_time = self.pipeline.run_pickle_driving(
            driving_pickle_path=motion_pickle,
            source_path=source_image_path,
            output_path=str(self.temp_dir / "raw_video.mp4")
        )
        metrics['flip_time'] = time.time() - t3
        logger.info(f"[BatchVideo] FLIP rendered in {metrics['flip_time']:.2f}s, video: {video_path}")

        # ========== Step 5: FFmpeg 合成音视频 ==========
        t4 = time.time()
        logger.info("[BatchVideo] Step 4: FFmpeg merging audio and video...")
        final_video = self._merge_audio_video(video_path, audio_path, output_path)
        metrics['ffmpeg_time'] = time.time() - t4
        logger.info(f"[BatchVideo] FFmpeg merged in {metrics['ffmpeg_time']:.2f}s")

        # ========== 清理临时文件 ==========
        self._cleanup_temp_files([motion_pickle])

        # 计算总耗时
        metrics['total_time'] = time.time() - start_time
        logger.info(f"[BatchVideo] Total time: {metrics['total_time']:.2f}s")

        return VideoGenerationResult(
            video_path=final_video,
            total_time=metrics['total_time'],
            metrics=metrics,
            audio_duration=audio_duration,
            frame_count=n_frames
        )

    def generate_video_from_frames(
        self,
        frames: list,
        audio_path: str,
        output_path: str = None,
        session_id: str = None
    ) -> VideoGenerationResult:
        """
        从已渲染的帧生成视频

        Args:
            frames: 视频帧列表 (BGR numpy arrays)
            audio_path: 音频文件路径
            output_path: 输出视频路径
            session_id: 会话ID

        Returns:
            VideoGenerationResult: 生成结果
        """
        start_time = time.time()
        metrics = {}

        # 生成输出路径
        if output_path is None:
            timestamp = int(time.time() * 1000)
            filename = f"video_{session_id or timestamp}.mp4"
            output_path = str(self.output_dir / filename)

        # 写入临时视频文件
        temp_video = str(self.temp_dir / f"frames_{time.time()}.mp4")

        t1 = time.time()
        self._write_frames_to_video(frames, temp_video)
        metrics['frame_write_time'] = time.time() - t1

        # FFmpeg 合成
        t2 = time.time()
        final_video = self._merge_audio_video(temp_video, audio_path, output_path)
        metrics['ffmpeg_time'] = time.time() - t2

        # 清理
        self._cleanup_temp_files([temp_video])

        metrics['total_time'] = time.time() - start_time

        return VideoGenerationResult(
            video_path=final_video,
            total_time=metrics['total_time'],
            metrics=metrics,
            audio_duration=self._get_audio_duration(audio_path),
            frame_count=len(frames)
        )

    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            return 0.0

    def _merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        使用 FFmpeg 合成音视频

        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            output_path: 输出文件路径

        Returns:
            str: 输出文件路径
        """
        cmd = [
            self.ffmpeg,
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-y',  # 覆盖已存在的文件
            output_path
        ]

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")

        return output_path

    def _write_frames_to_video(self, frames: list, output_path: str):
        """
        将帧列表写入视频文件

        Args:
            frames: 帧列表 (BGR numpy arrays)
            output_path: 输出路径
        """
        if not frames:
            raise ValueError("No frames to write")

        # 获取帧尺寸
        height, width = frames[0].shape[:2]

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def _cleanup_temp_files(self, file_paths: list):
        """清理临时文件"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {path}: {e}")

    def cleanup(self):
        """清理所有临时文件"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir: {e}")


class BatchVideoGeneratorSimple:
    """
    简化版批量视频生成器
    直接使用 FasterLivePortrait 的 run_audio_driving 方法
    """

    def __init__(
        self,
        pipeline,
        joyvasa_pipeline,
        output_dir: str = None,
        ffmpeg_path: str = "ffmpeg",
        emotion_config=None
    ):
        self.pipeline = pipeline
        self.joyvasa = joyvasa_pipeline
        self.ffmpeg = ffmpeg_path
        self.emotion_config = emotion_config

        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), "digital_human_videos")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = Path(tempfile.mkdtemp(prefix="batch_simple_"))

        # 初始化 Motion 后处理器
        self.motion_post_processor = None
        if MOTION_POST_PROCESSOR_AVAILABLE:
            try:
                self.motion_post_processor = get_motion_post_processor(emotion_config)
                logger.info("[BatchVideo] MotionPostProcessor initialized successfully")
            except Exception as e:
                logger.warning(f"[BatchVideo] Failed to initialize MotionPostProcessor: {e}")

        logger.info(f"BatchVideoGeneratorSimple initialized, output: {self.output_dir}")

    def generate(
        self,
        audio_path: str,
        source_image_path: str,
        session_id: str = None,
        emotion: str = "default"
    ) -> VideoGenerationResult:
        """
        生成数字人视频

        直接使用 pipeline 的 run_audio_driving 方法

        Args:
            audio_path: 音频文件路径
            source_image_path: 源图像路径
            session_id: 会话ID
            emotion: 情感标签 (happy/sad/calm/question/default)
        """
        start_time = time.time()
        metrics = {}

        # 检查依赖是否正确初始化
        if self.joyvasa is None:
            raise RuntimeError("JoyVASA pipeline is not initialized! Check batch_pipeline_manager.initialize()")
        if self.pipeline is None:
            raise RuntimeError("FasterLivePortrait pipeline is not initialized! Check batch_pipeline_manager.initialize()")

        # 生成输出路径
        timestamp = int(time.time() * 1000)
        base_name = f"video_{session_id or timestamp}"
        output_video = str(self.output_dir / f"{base_name}.mp4")

        # Step 1: JoyVASA
        t1 = time.time()
        logger.info("[SimpleBatch] Running JoyVASA...")

        # 验证音频文件存在且有效
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio file not found: {audio_path}")

        audio_file_size = os.path.getsize(audio_path)
        logger.info(f"[SimpleBatch] Audio file size: {audio_file_size} bytes")

        if audio_file_size == 0:
            raise RuntimeError(f"Audio file is empty: {audio_path}")

        # 检查音频文件信息
        audio_duration = 0
        try:
            import soundfile as sf
            audio_info = sf.info(audio_path)
            audio_duration = audio_info.duration
            logger.info(f"[SimpleBatch] Audio file: {audio_path}, duration={audio_duration:.2f}s, samplerate={audio_info.samplerate}, format={audio_info.format}")

            if audio_duration < 0.1:
                raise RuntimeError(f"Audio duration too short: {audio_duration:.2f}s")
        except Exception as e:
            logger.error(f"[SimpleBatch] Failed to read audio info: {e}")
            raise RuntimeError(f"Invalid audio file: {e}")

        try:
            motion_info = self.joyvasa.gen_motion_sequence(audio_path)
            logger.info(f"[SimpleBatch] JoyVASA returned motion_info keys: {list(motion_info.keys()) if motion_info else 'None'}")
        except Exception as e:
            logger.error(f"[SimpleBatch] JoyVASA failed: {e}")
            raise RuntimeError(f"JoyVASA motion generation failed: {e}")

        metrics['joyvasa_time'] = time.time() - t1
        n_frames = motion_info.get('n_frames', 0) if motion_info else 0
        logger.info(f"[SimpleBatch] JoyVASA: {n_frames} frames in {metrics['joyvasa_time']:.2f}s")

        if n_frames == 0:
            logger.error(f"[SimpleBatch] JoyVASA generated 0 frames for audio: {audio_path}, duration: {audio_duration:.2f}s")
            raise RuntimeError(f"JoyVASA generated 0 frames. Audio file: {audio_path}, duration: {audio_duration:.2f}s")

        # ========== 新增: Motion 后处理 ==========
        t_post = time.time()
        if self.motion_post_processor is not None and emotion != "default":
            logger.info(f"[SimpleBatch] Applying motion post-processing with emotion: {emotion}")
            try:
                motion_info = self.motion_post_processor.process(motion_info, emotion=emotion)
                logger.info(f"[SimpleBatch] Motion post-processing completed in {time.time() - t_post:.3f}s")
            except Exception as e:
                logger.warning(f"[SimpleBatch] Motion post-processing failed: {e}, using original motion")
        metrics['motion_post_time'] = time.time() - t_post
        # ========== Motion 后处理结束 ==========

        # Step 2: 准备源图像
        t2 = time.time()
        logger.info("[SimpleBatch] Preparing source image...")
        success = self.pipeline.prepare_source(source_image_path, realtime=False)
        if not success:
            raise RuntimeError(f"Failed to prepare source image: {source_image_path}")
        src_img = self.pipeline.src_imgs[0]

        # 调试：打印 src_infos 结构
        logger.info(f"[SimpleBatch] src_infos length: {len(self.pipeline.src_infos)}")
        logger.info(f"[SimpleBatch] src_infos[0] length: {len(self.pipeline.src_infos[0])}")
        if len(self.pipeline.src_infos[0]) > 0:
            logger.info(f"[SimpleBatch] src_infos[0][0] length: {len(self.pipeline.src_infos[0][0])}")
            logger.info(f"[SimpleBatch] src_infos[0][0] types: {[type(x).__name__ for x in self.pipeline.src_infos[0][0]]}")

        src_info = self.pipeline.src_infos[0]
        metrics['prep_time'] = time.time() - t2
        logger.info(f"[SimpleBatch] Source prepared in {metrics['prep_time']:.2f}s")

        # Step 3: 渲染帧
        t3 = time.time()
        logger.info("[SimpleBatch] Rendering frames...")
        frames = []

        motion_list = motion_info.get("motion", [])
        for i, motion_frame in enumerate(motion_list):
            dri_motion_info = [motion_frame, None, None]

            out_crop, out_org = self.pipeline.run_with_pkl(
                dri_motion_info,
                src_img,
                src_info,
                first_frame=(i == 0)
            )

            if out_org is not None:
                frame = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                frames.append(frame)

            if (i + 1) % 50 == 0:
                logger.info(f"[SimpleBatch] Rendered {i + 1}/{n_frames} frames")

        metrics['render_time'] = time.time() - t3
        logger.info(f"[SimpleBatch] Rendered {len(frames)} frames in {metrics['render_time']:.2f}s")

        # Step 4: 写入视频
        t4 = time.time()
        temp_video = str(self.temp_dir / f"{base_name}_noaudio.mp4")
        self._write_frames_video(frames, temp_video)
        metrics['write_time'] = time.time() - t4

        # 获取音频时长（在验证之前）
        audio_duration = 0
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            audio_duration = info.duration
            logger.info(f"[SimpleBatch] Audio duration: {audio_duration:.2f}s")
        except Exception as e:
            logger.warning(f"[SimpleBatch] Failed to get audio duration: {e}")

        # Step 5: FFmpeg 合成
        t5 = time.time()
        self._merge_av(temp_video, audio_path, output_video)
        metrics['ffmpeg_time'] = time.time() - t5

        # 验证最终视频时长
        try:
            probe_cmd = [self.ffmpeg.replace('ffmpeg', 'ffprobe'), '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', output_video]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            final_duration = float(probe_result.stdout.strip())
            logger.info(f"[SimpleBatch] Final video duration: {final_duration:.2f}s, expected: {audio_duration:.2f}s")
            if abs(final_duration - audio_duration) > 0.5:
                logger.warning(f"[SimpleBatch] Duration mismatch! Video: {final_duration:.2f}s, Audio: {audio_duration:.2f}s")
        except Exception as e:
            logger.warning(f"[SimpleBatch] Failed to verify video duration: {e}")

        # 清理
        try:
            os.remove(temp_video)
        except:
            pass

        metrics['total_time'] = time.time() - start_time
        logger.info(f"[SimpleBatch] Total: {metrics['total_time']:.2f}s")

        return VideoGenerationResult(
            video_path=output_video,
            total_time=metrics['total_time'],
            metrics=metrics,
            audio_duration=audio_duration,
            frame_count=len(frames)
        )

    def _write_frames_video(self, frames: list, output_path: str, fps: int = 25):
        """写入视频文件 - 使用 FFmpeg 确保正确的时间戳"""
        if not frames:
            raise ValueError("No frames")

        h, w = frames[0].shape[:2]

        # 使用 FFmpeg 管道方式写入，确保正确的时间戳元数据
        cmd = [
            self.ffmpeg, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            stderr = process.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg video encoding failed: {stderr}")

    def _merge_av(self, video_path: str, audio_path: str, output_path: str):
        """FFmpeg 合成音视频"""
        # 使用 copy 模式避免重新编码，保留正确的时间戳
        cmd = [
            self.ffmpeg, '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # 直接复制视频流，避免重新编码导致的时间戳问题
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',  # 优化流式播放
            '-map', '0:v:0',  # 使用第一个输入的视频流
            '-map', '1:a:0',  # 使用第二个输入的音频流
            '-shortest',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"[SimpleBatch] FFmpeg copy mode failed, trying re-encode: {result.stderr[:500]}")
            # 回退到重新编码模式
            cmd_fallback = [
                self.ffmpeg, '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-shortest',
                output_path
            ]
            result = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        return output_path
