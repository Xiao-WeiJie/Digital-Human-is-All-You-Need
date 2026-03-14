###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
#
#  Adapted for FasterLivePortrait integration
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import time
import copy
import os
import sys
import numpy as np
import cv2
import torch
import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from basereal import BaseReal
from logger import logger

# Add FasterLivePortrait to path
fasterlp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FasterLivePortrait")
if fasterlp_path not in sys.path:
    sys.path.insert(0, fasterlp_path)


def load_model(cfg_path=None):
    """
    Load FasterLivePortrait models.

    Args:
        cfg_path: Path to config file. If None, uses default.

    Returns:
        Tuple of (pipeline, joyvasa_pipeline)
    """
    from omegaconf import OmegaConf
    from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline
    from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

    # Load configuration
    if cfg_path is None:
        cfg_path = os.path.join(fasterlp_path, "configs", "trt_infer.yaml")

    cfg = OmegaConf.load(cfg_path)

    # Update model paths to absolute paths
    for name in cfg.models:
        if isinstance(cfg.models[name].model_path, str):
            cfg.models[name].model_path = cfg.models[name].model_path.replace(
                "./checkpoints", os.path.join(fasterlp_path, "checkpoints")
            )

    # Update JoyVASA paths
    if hasattr(cfg, 'joyvasa_models'):
        for key in cfg.joyvasa_models:
            if isinstance(cfg.joyvasa_models[key], str):
                cfg.joyvasa_models[key] = cfg.joyvasa_models[key].replace(
                    "./checkpoints", os.path.join(fasterlp_path, "checkpoints")
                )

    # Initialize pipeline
    logger.info("Loading FasterLivePortrait pipeline...")
    pipeline = GradioLivePortraitPipeline(cfg=cfg)

    # Initialize JoyVASA
    logger.info("Loading JoyVASA audio-to-motion pipeline...")
    joyvasa_pipeline = JoyVASAAudio2MotionPipeline(
        motion_model_path=getattr(cfg.joyvasa_models, 'motion_model_path', ''),
        audio_model_path=getattr(cfg.joyvasa_models, 'audio_model_path', ''),
        motion_template_path=getattr(cfg.joyvasa_models, 'motion_template_path', ''),
        cfg_mode=getattr(cfg.infer_params, 'cfg_mode', 'incremental'),
        cfg_scale=getattr(cfg.infer_params, 'cfg_scale', 2.8)
    )

    logger.info("FasterLivePortrait models loaded successfully")
    return pipeline, joyvasa_pipeline


def load_avatar(avatar_id, pipeline=None):
    """
    Load avatar data for FasterLivePortrait.

    Args:
        avatar_id: Avatar identifier or path to avatar image
        pipeline: GradioLivePortraitPipeline instance

    Returns:
        Tuple of (src_img, src_info, frame_list) for rendering
    """
    # Determine if avatar_id is a path or identifier
    if os.path.exists(avatar_id):
        source_path = avatar_id
    else:
        # Look in Human_Choice directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        human_choice_dir = os.path.join(base_dir, "Human_Choice")

        # Try common avatar patterns
        possible_paths = [
            os.path.join(human_choice_dir, avatar_id),
            os.path.join(human_choice_dir, f"Human_1/{avatar_id}"),
            os.path.join(human_choice_dir, f"Human_2/{avatar_id}"),
            os.path.join(human_choice_dir, f"{avatar_id}.png"),
            os.path.join(human_choice_dir, f"{avatar_id}.jpg"),
        ]

        source_path = None
        for path in possible_paths:
            if os.path.exists(path):
                source_path = path
                break

        if source_path is None:
            # Try to find in avatar config
            config_path = os.path.join(human_choice_dir, "avatar_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                avatar_info = config.get("avatars", {}).get(avatar_id)
                if avatar_info:
                    img_path = avatar_info.get("source_image")
                    if img_path:
                        source_path = os.path.join(human_choice_dir, img_path)

        if source_path is None or not os.path.exists(source_path):
            raise FileNotFoundError(f"Avatar not found: {avatar_id}")

    logger.info(f"Loading avatar from {source_path}")

    # Prepare source using pipeline
    if pipeline is not None:
        success = pipeline.prepare_source(source_path, realtime=True)
        if not success:
            raise RuntimeError(f"Failed to prepare source: {source_path}")

        src_img = pipeline.src_imgs[0]
        src_info = pipeline.src_infos[0][0]  # First face, first frame

        # For idle animation, we create a single frame list
        frame_list = [cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)]

        return frame_list, src_info, src_img

    return None, None, None


@torch.no_grad()
def warm_up(batch_size, pipeline, avatar_data):
    """
    Warm up the FasterLivePortrait model.

    Args:
        batch_size: Batch size for inference
        pipeline: GradioLivePortraitPipeline instance
        avatar_data: Tuple of (frame_list, src_info, src_img)
    """
    logger.info('Warming up FasterLivePortrait model...')

    frame_list, src_info, src_img = avatar_data
    if src_info is None or src_img is None:
        logger.warning("No avatar data for warmup, skipping")
        return

    # Create a dummy motion frame from source info
    # Extract motion info from src_info
    x_s_info = src_info[0]  # First element is x_s_info dict

    dummy_motion = {
        "exp": x_s_info['exp'].astype(np.float32),
        "scale": x_s_info['scale'].astype(np.float32),
        "R": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "t": x_s_info['t'].astype(np.float32),
        "pitch": x_s_info['pitch'].astype(np.float32),
        "yaw": x_s_info['yaw'].astype(np.float32),
        "roll": x_s_info['roll'].astype(np.float32)
    }

    dri_motion_info = [dummy_motion, None, None]

    # Run inference
    try:
        out_crop, out_org = pipeline.run_with_pkl(
            dri_motion_info,
            src_img,
            src_info,
            first_frame=True
        )
        logger.info('FasterLivePortrait warmup completed')
    except Exception as e:
        logger.warning(f"Warmup failed (may be normal): {e}")


@torch.no_grad()
def inference(quit_event, batch_size, src_img, src_info, motion_queue, audio_out_queue, res_frame_queue, pipeline):
    """
    Inference thread for FasterLivePortrait.

    Args:
        quit_event: Event to signal thread termination
        batch_size: Number of frames to process per batch
        src_img: Source image (RGB numpy array)
        src_info: Source information dict from pipeline
        motion_queue: Queue receiving motion sequences from ASR
        audio_out_queue: Queue for audio output synchronization
        res_frame_queue: Queue for output video frames
        pipeline: GradioLivePortraitPipeline instance
    """
    logger.info('FasterLivePortrait inference thread started')
    index = 0
    count = 0
    counttime = 0

    while not quit_event.is_set():
        starttime = time.perf_counter()

        try:
            # Get motion chunks from ASR
            motion_chunks = motion_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        # Check if all silence
        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size * 2):
            frame, type, eventpoint = audio_out_queue.get()
            audio_frames.append((frame, type, eventpoint))
            if type == 0:
                is_all_silence = False

        if is_all_silence:
            # Output idle frames
            for i in range(batch_size):
                res_frame_queue.put((None, index, audio_frames[i * 2:i * 2 + 2]))
                index += 1
        else:
            # Process motion frames
            t = time.perf_counter()

            for i, motion_info in enumerate(motion_chunks[:batch_size]):
                try:
                    dri_motion_info = motion_info  # [motion_dict, eye_ratio, lip_ratio]

                    # Run inference
                    out_crop, out_org = pipeline.run_with_pkl(
                        dri_motion_info,
                        src_img,
                        src_info,
                        first_frame=(index == 0)
                    )

                    # Convert to BGR for output
                    if out_org is not None:
                        out_frame = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                    else:
                        out_frame = None

                    counttime += (time.perf_counter() - t)
                    count += 1

                    if count >= 100:
                        logger.info(f"------FasterLivePortrait avg infer fps: {count / counttime:.4f}")
                        count = 0
                        counttime = 0

                    res_frame_queue.put((out_frame, index, audio_frames[i * 2:i * 2 + 2]))
                    index += 1

                except Exception as e:
                    logger.warning(f"Frame inference error: {e}")
                    continue

    logger.info('FasterLivePortrait inference thread stopped')


class FasterLivePortraitReal(BaseReal):
    """
    FasterLivePortrait implementation for LiveTalking framework.

    This class integrates FasterLivePortrait's pipeline into LiveTalking's
    real-time rendering framework, supporting WebRTC streaming and audio sync.
    """

    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        """
        Initialize FasterLivePortraitReal.

        Args:
            opt: Options containing session configuration
            model: Tuple of (pipeline, joyvasa_pipeline)
            avatar: Tuple of (frame_list, src_info, src_img)
        """
        super().__init__(opt)

        self.fps = opt.fps  # 50 fps for audio, 25 fps for video

        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size * 2)

        # Unpack model and avatar data
        self.pipeline, self.joyvasa_pipeline = model
        self.frame_list, self.src_info, self.src_img = avatar

        # Initialize ASR adapter
        from fasterlpasr import FasterLPASR
        self.asr = FasterLPASR(opt, self, self.joyvasa_pipeline)
        self.asr.warm_up()

        self.render_event = mp.Event()

        logger.info(f"FasterLivePortraitReal initialized for session {self.sessionid}")

    def __mirror_index(self, index):
        """Get mirrored index for looping idle animation."""
        size = len(self.frame_list)
        if size == 0:
            return 0
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def paste_back_frame(self, pred_frame, idx: int):
        """
        Process the predicted frame for output.

        For FasterLivePortrait, the pipeline already handles paste_back,
        so we just return the frame directly or use the idle frame.

        Args:
            pred_frame: Predicted frame from inference (can be None for idle)
            idx: Frame index for idle animation

        Returns:
            Processed frame ready for display
        """
        if pred_frame is not None:
            # Frame already processed by pipeline with paste_back
            return pred_frame
        else:
            # Return idle frame
            mirindex = self.__mirror_index(idx)
            if mirindex < len(self.frame_list):
                return self.frame_list[mirindex]
            else:
                return self.frame_list[0] if self.frame_list else None

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        """
        Main render loop.

        Starts three threads:
        1. TTS thread - generates audio from text
        2. Inference thread - generates video frames from motion
        3. Process frames thread - outputs synchronized audio/video

        Args:
            quit_event: Event to signal termination
            loop: Asyncio event loop for WebRTC
            audio_track: WebRTC audio track
            video_track: WebRTC video track
        """
        self.init_customindex()
        self.tts.render(quit_event)

        # Start inference thread
        infer_quit_event = Event()
        infer_thread = Thread(
            target=inference,
            args=(
                infer_quit_event,
                self.batch_size,
                self.src_img,
                self.src_info,
                self.asr.feat_queue,
                self.asr.output_queue,
                self.res_frame_queue,
                self.pipeline
            )
        )
        infer_thread.start()

        # Start frame processing thread
        process_quit_event = Event()
        process_thread = Thread(
            target=self.process_frames,
            args=(process_quit_event, loop, audio_track, video_track)
        )
        process_thread.start()

        # Main ASR processing loop
        count = 0
        totaltime = 0

        while not quit_event.is_set():
            t = time.perf_counter()
            self.asr.run_step()

            # Throttle if output queue is getting full
            if video_track and video_track._queue.qsize() >= 1.5 * self.batch_size:
                logger.debug(f'Throttling, queue size: {video_track._queue.qsize()}')
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)

        logger.info(f'FasterLivePortraitReal render thread stopping for session {self.sessionid}')

        # Cleanup threads
        infer_quit_event.set()
        infer_thread.join()

        process_quit_event.set()
        process_thread.join()

    def switch_avatar(self, avatar_id):
        """
        Switch to a different avatar during runtime.

        Args:
            avatar_id: Avatar identifier or path

        Returns:
            True if successful, False otherwise
        """
        try:
            frame_list, src_info, src_img = load_avatar(avatar_id, self.pipeline)
            if frame_list is not None and src_info is not None:
                self.frame_list = frame_list
                self.src_info = src_info
                self.src_img = src_img
                self.idx = 0  # Reset frame index

                # 更新当前数字人ID（用于TTS参考音频）
                # 如果传入的是路径，尝试从配置中查找对应的ID
                if os.path.exists(avatar_id):
                    # 传入的是路径，保持原ID或尝试匹配
                    pass
                else:
                    # 传入的是ID（如 human_1, human_2）
                    self.avatar_id = avatar_id

                logger.info(f"Switched to avatar: {avatar_id}, current avatar_id: {self.avatar_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to switch avatar: {e}")
        return False
