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
import math
import os
import tempfile
import numpy as np
import soundfile as sf
import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from baseasr import BaseASR
from logger import logger


class FasterLPASR(BaseASR):
    """
    ASR adapter for FasterLivePortrait that accumulates audio frames
    and generates motion sequences using JoyVASA.

    Key differences from MuseASR:
    - Instead of extracting Whisper features, generates motion sequences
    - Accumulates audio until minimum duration (0.5s) for JoyVASA
    - Outputs motion sequences to feat_queue for inference thread
    """

    def __init__(self, opt, parent, joyvasa_pipeline):
        """
        Initialize FasterLPASR.

        Args:
            opt: Options containing batch_size, l, r, fps
            parent: Parent BaseReal instance
            joyvasa_pipeline: JoyVASAAudio2MotionPipeline instance
        """
        super().__init__(opt, parent)

        self.joyvasa_pipeline = joyvasa_pipeline

        # JoyVASA parameters
        self.n_motions = joyvasa_pipeline.n_motions  # typically 16 frames per chunk
        self.audio_unit = joyvasa_pipeline.audio_unit  # samples per frame
        self.target_fps = joyvasa_pipeline.fps  # typically 25 fps

        # Audio accumulation parameters
        # Need to accumulate at least n_motions frames worth of audio
        # At 16kHz and 25fps, each frame is 640 samples (40ms)
        self.min_audio_samples = int(self.n_motions * self.audio_unit)  # ~10240 samples for 16 frames
        self.min_audio_duration = self.min_audio_samples / 16000  # ~0.64 seconds

        # Buffer for audio accumulation
        self.audio_buffer = []
        self.total_samples = 0

        # Motion queue - stores motion sequences for inference
        self.motion_queue = mp.Queue(4)

        # Create temp directory for audio files
        self.temp_dir = tempfile.mkdtemp(prefix="fasterlp_asr_")

        logger.info(f"FasterLPASR initialized: n_motions={self.n_motions}, "
                    f"min_audio_duration={self.min_audio_duration:.2f}s")

    def run_step(self):
        """
        Process audio frames and generate motion sequences.

        This method:
        1. Collects audio frames (20ms chunks)
        2. Accumulates until minimum duration reached
        3. Calls JoyVASA to generate motion sequence
        4. Puts motion chunks into feat_queue for inference
        """
        start_time = time.time()

        # Collect audio frames for output (passthrough)
        for _ in range(self.batch_size * 2):
            audio_frame, type, eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, type, eventpoint))

        # Check if we have enough context
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return

        # Accumulate audio buffer
        audio_data = np.concatenate(self.frames)
        self.audio_buffer.append(audio_data)
        self.total_samples += len(audio_data)

        # Generate motion when we have enough audio
        if self.total_samples >= self.min_audio_samples:
            self._generate_motion()

        # Discard old frames to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

    def _generate_motion(self):
        """Generate motion sequence from accumulated audio buffer."""
        if self.total_samples < self.min_audio_samples:
            return

        try:
            # Concatenate audio buffer
            audio_data = np.concatenate(self.audio_buffer)
            audio_duration = len(audio_data) / 16000

            # Save to temp file for JoyVASA (it expects a file path)
            temp_audio_path = os.path.join(self.temp_dir, f"audio_{time.time()}.wav")
            sf.write(temp_audio_path, audio_data, 16000)

            # Generate motion sequence
            t0 = time.time()
            motion_dict = self.joyvasa_pipeline.gen_motion_sequence(temp_audio_path)
            gen_time = time.time() - t0

            logger.debug(f"JoyVASA generated {motion_dict['n_frames']} frames in {gen_time:.3f}s")

            # Clean up temp file
            try:
                os.remove(temp_audio_path)
            except:
                pass

            # Split motion into batch-sized chunks
            motion_list = motion_dict["motion"]
            n_frames = len(motion_list)

            # Create motion chunks matching batch_size
            # Each chunk contains batch_size motion frames
            chunk_size = self.batch_size

            for i in range(0, n_frames, chunk_size):
                end_idx = min(i + chunk_size, n_frames)
                chunk_motions = []

                for j in range(i, end_idx):
                    motion_frame = motion_list[j]
                    # Format: [motion_dict, eye_ratio, lip_ratio]
                    # JoyVASA doesn't provide eye/lip ratios, use None
                    chunk_motions.append([motion_frame, None, None])

                # Pad to batch_size if needed
                while len(chunk_motions) < chunk_size:
                    # Repeat last motion for padding
                    if chunk_motions:
                        chunk_motions.append(chunk_motions[-1].copy())
                    else:
                        # Empty chunk, shouldn't happen
                        break

                # Put into motion queue (will be read by inference thread)
                try:
                    self.feat_queue.put(chunk_motions, block=False)
                except queue.Full:
                    logger.warning("Motion queue full, dropping frames")
                    break

            # Reset buffer
            self.audio_buffer = []
            self.total_samples = 0

        except Exception as e:
            logger.error(f"Error generating motion: {e}")
            import traceback
            traceback.print_exc()

    def flush_talk(self):
        """Clear all queues and buffers."""
        super().flush_talk()
        self.audio_buffer = []
        self.total_samples = 0

        # Clear motion queue
        while not self.motion_queue.empty():
            try:
                self.motion_queue.get_nowait()
            except queue.Empty:
                break

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
