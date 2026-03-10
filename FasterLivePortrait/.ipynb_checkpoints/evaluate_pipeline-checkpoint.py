#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_pipeline.py
====================
End-to-end evaluation pipeline for FasterLivePortrait.

Pipeline
--------
  Step 1  [Generation]   Drive FasterLivePortrait with val/Video_files/*.mp4
                          (or val/Audio_files/*.wav in audio mode) via subprocess.
  Step 2  [Extraction]   Extract 58-dim 3D facial features from each generated
                          video.  *** See SECTION 3 — replace the placeholder. ***
  Step 3  [Evaluation]   Compare extracted features against val/3D_FV_files/*_full.npy
                          (Ground Truth) using 8 FR metrics.
  Step 4  [Scoring]      Aggregate per scenario; S_final = max_i(S_i).

Quick start
-----------
  # video-drive mode (default)
  python evaluate_pipeline.py --drive_mode video --src_image assets/examples/source/s12.jpg

  # audio-drive mode (requires JoyVASA checkpoints)
  python evaluate_pipeline.py --drive_mode audio --src_image assets/examples/source/s12.jpg

  # skip generation, only re-run metrics on cached features
  python evaluate_pipeline.py --skip_generation

  # force re-extraction of 58-dim features
  python evaluate_pipeline.py --skip_generation --force_extract
"""

import os
import sys
import json
import math
import shutil
import warnings
import datetime
import argparse
import subprocess
import platform
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from scipy import stats
from scipy.linalg import sqrtm
from scipy.signal import correlate
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0  TOP-LEVEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── Drive-mode switch ──────────────────────────────────────────────────────────
#   "video"  →  reads val/Video_files/**/*.mp4 as --dri_video
#   "audio"  →  reads val/Audio_files/**/*.wav, runs JoyVASA first
DRIVE_MODE = "video"

# ── Directory paths ────────────────────────────────────────────────────────────
VAL_ROOT    = "val"
OUTPUT_ROOT = "eval_output"
SRC_IMAGE   = "../FasterLivePortrait/assets/examples/source/s12.jpg"  # source portrait for FLP
INFER_CFG   = "configs/onnx_infer.yaml"           # swap to trt_infer.yaml for TRT
RUN_SCRIPT  = "run.py"

# ── FFmpeg binary ──────────────────────────────────────────────────────────────
if platform.system().lower() == "windows":
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"

# ── JoyVASA checkpoints (required only when DRIVE_MODE = "audio") ──────────────
JOYVASA_MOTION_MODEL = "checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"
JOYVASA_AUDIO_MODEL  = "checkpoints/chinese-hubert-base"
JOYVASA_TEMPLATE     = "checkpoints/JoyVASA/motion_template/motion_template.pkl"

# ── Ground-truth frame rate ────────────────────────────────────────────────────
#   3D_FV_files were extracted at this FPS.
#   Generated videos MUST be at the same FPS — the pipeline enforces this.
GT_FPS = 25

# ── Jaw-open feature dimension ─────────────────────────────────────────────────
#   In the 58-dim output of the recommended extractor:
#     dims  0-51 : 52 ARKit BlendShape coefficients in standard ARKit order
#                  (jawOpen sits at index 25 in the canonical ARKit ordering)
#     dims 52-57 : 6-DoF head pose  [Rx, Ry, Rz (deg), Tx, Ty, Tz (mm)]
#   Adjust JAW_OPEN_DIM if your extractor uses a different layout.
JAW_OPEN_DIM = 25

# ── FRSyn calibration constant ─────────────────────────────────────────────────
#   FRSyn = (1.0 - peak_normalised_xcorr + ε) × FRSYN_SCALE
#   A baseline model with peak_xcorr ≈ 0.58  →  FRSyn ≈ 41.99
#   (consistent with the "41.99 / FRSyn" term in the scoring formula)
FRSYN_SCALE = 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1  DATA DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def discover_pairs(val_root: str, drive_mode: str) -> List[Dict]:
    """
    Walk val_root and return a list of clip records.

    Each record contains:
      drive_path   str   path to .mp4 (video mode) or .wav (audio mode)
      gt_path      str   path to the matching *_full.npy Ground Truth
      audio_path   str|None  path to .wav for FRSyn (audio mode or matched wav)
      scenario_id  str   e.g. "RECOLA/group-2/P41"
                         or  "NoXI/017_2016-04-05_Paris/Expert_video"
      clip_id      str   e.g. "1", "2", …
    """
    ext        = ".mp4"        if drive_mode == "video" else ".wav"
    drive_sub  = "Video_files" if drive_mode == "video" else "Audio_files"
    drive_root = Path(val_root) / drive_sub
    gt_root    = Path(val_root) / "3D_FV_files"
    audio_root = Path(val_root) / "Audio_files"

    pairs: List[Dict] = []
    for path in sorted(drive_root.rglob(f"*{ext}")):
        rel      = path.parent.relative_to(drive_root)
        stem     = path.stem
        gt_path  = gt_root / rel / (stem + "_full.npy")
        if not gt_path.exists():
            continue
        # Matching .wav for FRSyn cross-correlation (available even in video mode)
        wav_path = audio_root / rel / (stem + ".wav")
        pairs.append({
            "drive_path":  str(path),
            "gt_path":     str(gt_path),
            "audio_path":  str(wav_path) if wav_path.exists() else None,
            "scenario_id": str(rel).replace(os.sep, "/"),
            "clip_id":     stem,
        })
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2  VIDEO GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_video_fps(video_path: str) -> float:
    """Return the FPS stored in the video container via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else GT_FPS


def resample_video_to_fps(src: str, tgt_fps: float, dst: str) -> bool:
    """
    Re-encode src at tgt_fps via ffmpeg → dst.

    This is the FPS-alignment step that ensures frame counts of the generated
    video match the GT npy array (both at GT_FPS), making per-frame metrics valid.
    """
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    cmd = [
        FFMPEG, "-y", "-i", src,
        "-vf", f"fps={tgt_fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        dst,
    ]
    ret = subprocess.run(cmd, capture_output=True)
    return ret.returncode == 0 and os.path.exists(dst)


def _find_output_video(stdout: str) -> Optional[str]:
    """
    Parse run.py stdout for the generated video path.
    run.py prints the absolute output path as a bare .mp4 line to stdout.
    Scan in reverse to pick up the last such line (org video, not crop).
    """
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.endswith(".mp4") and os.path.isfile(line):
            return line
    return None


def generate_video_from_video(
    drive_path: str,
    src_image:  str,
    out_video:  str,
    cfg:        str,
    gt_fps:     float = GT_FPS,
) -> bool:
    """
    Drive FasterLivePortrait with a .mp4 file.

    FPS-alignment strategy
    ----------------------
    1. If drive_path FPS != gt_fps  →  resample drive to gt_fps before calling run.py
       (prevents run.py from producing a video at the wrong FPS).
    2. After run.py finishes, verify generated FPS; resample output if still wrong.
    """
    if os.path.exists(out_video) and os.path.getsize(out_video) > 0:
        return True

    # ── FPS alignment: pre-process driving video ───────────────────────────────
    actual_drive = drive_path
    src_fps = get_video_fps(drive_path)
    if abs(src_fps - gt_fps) > 0.5:
        resampled = str(Path(out_video).parent / (Path(drive_path).stem + f"_fps{int(gt_fps)}.mp4"))
        ok = resample_video_to_fps(drive_path, gt_fps, resampled)
        if ok:
            actual_drive = resampled
        else:
            warnings.warn(
                f"Pre-resample failed for {drive_path} "
                f"(src={src_fps:.1f} fps → {gt_fps} fps). Using original."
            )

    # ── Call run.py as subprocess ──────────────────────────────────────────────
    cmd = [
        sys.executable, RUN_SCRIPT,
        "--src_image", src_image,
        "--dri_video", actual_drive,
        "--cfg",       cfg,
        "--paste_back",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    if result.returncode != 0:
        warnings.warn(f"run.py failed for {drive_path}:\n{result.stderr[:500]}")
        return False

    raw_path = _find_output_video(result.stdout)
    if raw_path is None:
        warnings.warn(
            f"Could not locate generated video in run.py stdout for {drive_path}.\n"
            f"stdout:\n{result.stdout[:400]}"
        )
        return False

    # ── FPS alignment: post-process generated video ────────────────────────────
    gen_fps = get_video_fps(raw_path)
    if abs(gen_fps - gt_fps) > 0.5:
        ok = resample_video_to_fps(raw_path, gt_fps, out_video)
        if not ok:
            shutil.copy2(raw_path, out_video)
    else:
        shutil.copy2(raw_path, out_video)

    return os.path.exists(out_video) and os.path.getsize(out_video) > 0


def generate_video_from_audio(
    audio_path: str,
    src_image:  str,
    out_video:  str,
    cfg:        str,
    gt_fps:     float = GT_FPS,
) -> bool:
    """
    Drive FasterLivePortrait with a .wav file (JoyVASA audio-to-motion pipeline).

    Step A  audio (.wav) → JoyVASA → motion template (.pkl)
    Step B  motion (.pkl) + source image → run.py → generated video (.mp4)
    """
    if os.path.exists(out_video) and os.path.getsize(out_video) > 0:
        return True

    pkl_path = str(Path(out_video).with_suffix("")) + "_motion.pkl"

    # ── Step A: audio → motion pkl ────────────────────────────────────────────
    if not (os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 0):
        # 修改点：强制将JoyVASA的输出包装为 run.py 要求的字典格式
        joyvasa_script = (
            "import sys, pickle\n"
            "sys.path.insert(0, '.')\n"
            "from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline\n"
            f"pipe = JoyVASAAudio2MotionPipeline(\n"
            f"    motion_model_path=r'{JOYVASA_MOTION_MODEL}',\n"
            f"    audio_model_path=r'{JOYVASA_AUDIO_MODEL}',\n"
            f"    motion_template_path=r'{JOYVASA_TEMPLATE}')\n"
            f"motion_res = pipe.gen_motion_sequence(r'{audio_path}')\n"
            
            # 以下为新增的数据格式化包装逻辑
            "if isinstance(motion_res, dict):\n"
            "    template_dct = motion_res\n"
            "    if 'output_fps' not in template_dct: template_dct['output_fps'] = 25\n"
            "else:\n"
            "    motion_lst = motion_res[0] if isinstance(motion_res, tuple) else motion_res\n"
            "    if hasattr(motion_lst, 'tolist'): motion_lst = motion_lst.tolist()\n"
            "    template_dct = {\n"
            "        'n_frames': len(motion_lst),\n"
            "        'output_fps': 25,\n"
            "        'motion': motion_lst,\n"
            "        'c_eyes_lst': [[0.0]] * len(motion_lst),\n"
            "        'c_lip_lst': [[0.0]] * len(motion_lst)\n"
            "    }\n"
            
            f"with open(r'{pkl_path}', 'wb') as _f:\n"
            "    pickle.dump(template_dct, _f)\n"
            "print('JOYVASA_PKL_OK')\n"
        )
        ret = subprocess.run([sys.executable, "-c", joyvasa_script],
                             capture_output=True, text=True)
        if ret.returncode != 0 or "JOYVASA_PKL_OK" not in ret.stdout:
            warnings.warn(
                f"JoyVASA pkl generation failed for {audio_path}:\n{ret.stderr[:500]}"
            )
            return False

    # ── Step B: pkl → video ───────────────────────────────────────────────────
    cmd = [
        sys.executable, RUN_SCRIPT,
        "--src_image", src_image,
        "--dri_video", pkl_path,
        "--cfg",       cfg,
        "--paste_back",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    if result.returncode != 0:
        warnings.warn(f"run.py (pkl) failed for {audio_path}:\n{result.stderr[:500]}")
        return False

    raw_path = _find_output_video(result.stdout)
    if raw_path is None:
        warnings.warn(f"Cannot find generated video for {audio_path}.")
        return False

    gen_fps = get_video_fps(raw_path)
    if abs(gen_fps - gt_fps) > 0.5:
        ok = resample_video_to_fps(raw_path, gt_fps, out_video)
        if not ok:
            shutil.copy2(raw_path, out_video)
    else:
        shutil.copy2(raw_path, out_video)

    return os.path.exists(out_video) and os.path.getsize(out_video) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3  FEATURE EXTRACTION  *** PLACEHOLDER — REPLACE BEFORE REAL EVAL ***
# ═══════════════════════════════════════════════════════════════════════════════

def extract_58d_features_from_video(
    video_path:  str,
    target_fps:  float = GT_FPS,
) -> Optional[np.ndarray]:
    """
    Extract per-frame 58-dimensional 3D facial features from a video.

    OUTPUT SHAPE:  (N, 1, 58)   where N = number of frames at target_fps.

    Feature layout (must match the GT 3D_FV_files convention):
    ┌───────────────────────────────────────────────────────────┐
    │  dims  0 – 51  │  52 ARKit BlendShape coefficients        │
    │                │  canonical order (Apple ARKit standard)   │
    │                │  index 25 = jawOpen  ← used by FRSyn      │
    ├───────────────────────────────────────────────────────────┤
    │  dims 52 – 57  │  6-DoF head pose                         │
    │                │  [Rx, Ry, Rz (deg), Tx, Ty, Tz (mm)]     │
    └───────────────────────────────────────────────────────────┘

    ── RECOMMENDED IMPLEMENTATION (MediaPipe + ARKit regression) ──────────────

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp

    1. Load pre-trained regression weights (478×3 landmarks → 52 ARKit dims).
       A ready-made weight file can be exported from an Apple device stream or
       trained on a paired (MediaPipe landmarks, ARKit stream) dataset.

    2. Per frame:
         a. Run mp.solutions.face_mesh (478 3-D landmarks, normalised coords).
         b. Multiply by the (1434, 52) regression matrix  →  arkit_52  (float32)
         c. Solve 6-DoF head pose with cv2.solvePnP on 6 stable landmarks and
            the MediaPipe canonical 3-D face model  →  pose_6  [Rx,Ry,Rz,Tx,Ty,Tz]
         d. Stack:  feat = np.concatenate([arkit_52, pose_6])  →  shape (58,)

    3. Collect into array of shape (N_raw, 1, 58).

    4. FPS alignment — CRITICAL:
         If the video was not perfectly at target_fps after generation, resample:
           from scipy.interpolate import interp1d
           t_src = np.linspace(0, 1, N_raw)
           t_tgt = np.linspace(0, 1, N_target)   # N_target = round(duration * target_fps)
           feat_resampled = interp1d(t_src, feat[:,0,:], axis=0)(t_tgt)
           feat_resampled = feat_resampled[:, np.newaxis, :]

    ── ALTERNATIVE: DECA / EMOCA ──────────────────────────────────────────────
       DECA encode() → FLAME exp (50-dim) + pose (6-dim) = 56-dim; pad to 58.
       Note: only valid if GT npy was also produced with DECA.

    ── IMPLEMENTATION: MediaPipe FaceLandmarker (mediapipe >= 0.10) ───────────
    Outputs ARKit 52 BlendShapes natively — no regression matrix needed.
    Pose convention matches the GT: rotation in radians (solvePnP rvec),
    translation in normalised image coordinates (solvePnP tvec / focal_len).
    jawOpen sits at dim 16 in standard ARKit order; use --jaw_dim 16 at runtime.
    """
    # ── Import mediapipe (optional dependency) ──────────────────────────────
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python import BaseOptions
    except ImportError:
        warnings.warn(
            "mediapipe >= 0.10 is required for feature extraction.\n"
            "Install with:  pip install mediapipe\n"
            "Returning None.",
            UserWarning, stacklevel=2,
        )
        return None

    # ── MediaPipe canonical 3-D face model (6 stable landmarks for solvePnP) ──
    # Indices: nose-tip(1), left-eye-outer(33), right-eye-outer(263),
    #          left-mouth(61), right-mouth(291), chin(199)
    _CANONICAL_3D = np.array([
        [ 0.000000,  0.000000,  0.000000],
        [-0.101450,  0.091160, -0.047540],
        [ 0.101450,  0.091160, -0.047540],
        [-0.078910, -0.056930, -0.043930],
        [ 0.078910, -0.056930, -0.043930],
        [ 0.000000, -0.170900, -0.026990],
    ], dtype=np.float64)
    _POSE_IDXS = [1, 33, 263, 61, 291, 199]

    # ── Locate face_landmarker.task model file ──────────────────────────────
    _task_candidates = [
        "face_landmarker.task",
        "checkpoints/face_landmarker.task",
        str(Path(__file__).parent / "face_landmarker.task"),
    ]
    task_path = next((p for p in _task_candidates if os.path.exists(p)), None)
    if task_path is None:
        warnings.warn(
            "face_landmarker.task not found. Download it with:\n"
            "  curl -O https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task\n"
            "Place the file in the project root or checkpoints/.",
            UserWarning, stacklevel=2,
        )
        return None

    # ── Initialise FaceLandmarker ───────────────────────────────────────────
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=task_path),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        landmarker.close()
        warnings.warn(f"Cannot open video: {video_path}", UserWarning, stacklevel=2)
        return None

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or target_fps
    all_feats = []
    last_feat = np.zeros(58, dtype=np.float32)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        if not result.face_blendshapes or not result.face_landmarks:
            # No face detected: repeat last known feature vector
            all_feats.append(last_feat.copy())
            continue

        # ── 52 ARKit BlendShapes (direct output, no regression matrix needed) ──
        arkit_52 = np.array(
            [c.score for c in result.face_blendshapes[0]],
            dtype=np.float32,
        )  # shape (52,), values in [0, 1]

        # ── 6-DoF head pose via solvePnP ────────────────────────────────────
        # GT convention: rvec in radians, tvec in normalised image coordinates.
        lmk   = result.face_landmarks[0]
        pts2d = np.array(
            [[lmk[i].x * w, lmk[i].y * h] for i in _POSE_IDXS],
            dtype=np.float64,
        )
        cam_mat = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)
        _, rvec, tvec = cv2.solvePnP(
            _CANONICAL_3D, pts2d, cam_mat, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        pose6 = np.concatenate([
            rvec.flatten().astype(np.float32),   # Rx, Ry, Rz  (radians)
            tvec.flatten().astype(np.float32),   # Tx, Ty, Tz  (normalised)
        ])

        last_feat = np.concatenate([arkit_52, pose6])  # (58,)
        all_feats.append(last_feat.copy())

    cap.release()
    landmarker.close()

    if not all_feats:
        warnings.warn(f"No frames processed: {video_path}", UserWarning, stacklevel=2)
        return None

    feats = np.array(all_feats, dtype=np.float32)  # (N_raw, 58)

    # ── FPS alignment (GT was extracted at target_fps) ──────────────────────
    N_raw    = feats.shape[0]
    duration = N_raw / src_fps
    N_target = max(1, round(duration * target_fps))
    if abs(src_fps - target_fps) > 0.5 and N_raw > 1:
        t_src = np.linspace(0, 1, N_raw)
        t_tgt = np.linspace(0, 1, N_target)
        feats = interp1d(t_src, feats, axis=0, kind="linear")(t_tgt).astype(np.float32)

    return feats[:, np.newaxis, :]   # (N_target, 1, 58)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4  AUDIO PROCESSING  (for FRSyn)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_audio_rms_envelope(
    audio_path:  str,
    target_fps:  float = GT_FPS,
) -> Optional[np.ndarray]:
    """
    Compute a per-frame RMS energy envelope from a .wav file.

    Returns shape (N,)  where N ≈ audio_duration_seconds × target_fps.

    Preferred backend: librosa  (pip install librosa)
    Fallback backend:  scipy.io.wavfile  (no extra install required)
    """
    hop_seconds = 1.0 / target_fps     # one RMS value per video frame

    # ── librosa path ───────────────────────────────────────────────────────────
    try:
        import librosa
        y, sr      = librosa.load(audio_path, sr=None, mono=True)
        hop_samples = max(1, int(sr * hop_seconds))
        rms         = librosa.feature.rms(y=y, hop_length=hop_samples)[0]
        return rms.astype(np.float32)
    except ImportError:
        pass

    # ── scipy fallback ─────────────────────────────────────────────────────────
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Normalise integer PCM to float [-1, 1]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        else:
            data = data.astype(np.float32)

        hop_samples = max(1, int(sr * hop_seconds))
        n_frames    = len(data) // hop_samples
        rms_vals    = np.array(
            [float(np.sqrt(np.mean(data[i * hop_samples:(i + 1) * hop_samples] ** 2)))
             for i in range(n_frames)],
            dtype=np.float32,
        )
        return rms_vals
    except Exception as e:
        warnings.warn(f"Audio RMS computation failed for '{audio_path}': {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5  METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _to2d(arr: np.ndarray) -> np.ndarray:
    """Reshape (N, 1, 58) or (N, 58) → (N, 58)."""
    return arr.reshape(arr.shape[0], -1)


def _align(p: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate pred and GT to the same number of frames (min of the two)."""
    N = min(p.shape[0], g.shape[0])
    return p[:N], g[:N]


# ── FRCorr ─────────────────────────────────────────────────────────────────────
def metric_frcorr(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Global Pearson correlation on the fully-flattened pred/GT sequences.

    Captures overall temporal co-movement between all 58 predicted channels
    and their ground-truth counterparts.
    Baseline target: ≥ 0.09  (higher is better).
    """
    p, g = _align(_to2d(pred), _to2d(gt))
    pf, gf = p.flatten(), g.flatten()
    if np.std(pf) < 1e-8 or np.std(gf) < 1e-8:
        return 0.0
    r, _ = stats.pearsonr(pf, gf)
    return float(r)


# ── FRdist ─────────────────────────────────────────────────────────────────────
def metric_frdist(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Mean per-frame Euclidean distance (L2 norm in 58-dim space).

    Direct reconstruction fidelity; lower is better.
    Baseline target: ≤ 91.07  (formula inverts it).
    """
    p, g = _align(_to2d(pred), _to2d(gt))
    return float(np.mean(np.linalg.norm(p - g, axis=1)))


# ── FRDiv ──────────────────────────────────────────────────────────────────────
def metric_frdiv(pred: np.ndarray, n_pairs: int = 1000, seed: int = 42) -> float:
    """
    Intra-sequence diversity: mean L2 distance between randomly sampled frame pairs.

    High value  →  rich, non-repetitive motion within the clip.
    Baseline target: ≥ 3.4e-2  (higher is better).
    """
    p = _to2d(pred)
    N = p.shape[0]
    if N < 2:
        return 0.0
    rng  = np.random.default_rng(seed)
    k    = min(n_pairs, N * (N - 1) // 2)
    idx0 = rng.integers(0, N, k)
    idx1 = rng.integers(0, N, k)
    mask = idx0 != idx1
    idx0, idx1 = idx0[mask], idx1[mask]
    if len(idx0) == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(p[idx0] - p[idx1], axis=1)))


# ── FRDvs ─────────────────────────────────────────────────────────────────────
def metric_frdvs(pred_list: List[np.ndarray]) -> float:
    """
    Inter-sample diversity: mean L2 between frame-aligned pairs from different clips.

    Computed ONCE per scenario using all clips together.
    Measures whether the model produces varied responses to different drives.
    Baseline target: ≥ 3.22e-2  (higher is better).
    """
    if len(pred_list) < 2:
        return 0.0
    arrs  = [_to2d(p) for p in pred_list]
    dists: List[float] = []
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            L = min(arrs[i].shape[0], arrs[j].shape[0])
            dists.append(float(np.mean(np.linalg.norm(
                arrs[i][:L] - arrs[j][:L], axis=1
            ))))
    return float(np.mean(dists))


# ── FRVar ─────────────────────────────────────────────────────────────────────
def metric_frvar(pred: np.ndarray) -> float:
    """
    Motion liveliness via mean absolute 2nd-order temporal difference.

    Formula:  d2[t] = x[t+2] - 2·x[t+1] + x[t]   for t = 0 … N-3
              FRVar = mean( |d2| )  over all t and all 58 dims

    Analogous to the mean absolute acceleration of the expression motion.
    A large value indicates detail-rich, lively facial motion.
    Baseline target: ≥ 2.02e-2  (higher is better).
    """
    p  = _to2d(pred).astype(np.float64)   # (N, 58)
    d2 = np.diff(p, n=2, axis=0)          # (N-2, 58)  second-order diff
    return float(np.mean(np.abs(d2)))


# ── FRRea ─────────────────────────────────────────────────────────────────────
def metric_frrea(pred_all: np.ndarray, gt_all: np.ndarray) -> float:
    """
    Realism via Fréchet Distance (FD) in the 58-dim feature space.

    Fits a multivariate Gaussian to all predicted frames and all GT frames
    (concatenated across every clip in the scenario), then computes:

        FD = ||μ_pred - μ_gt||²  +  Tr(Σ_pred + Σ_gt - 2·(Σ_pred·Σ_gt)^{1/2})

    Lower FD  →  predicted distribution is closer to GT distribution  →  better.
    Baseline target: ≤ 69.33  (formula inverts it).

    NOTE: must be called once per scenario on the CONCATENATED feature arrays
    (not per-clip), to give stable Gaussian estimates.
    """
    p = _to2d(pred_all).astype(np.float64)
    g = _to2d(gt_all).astype(np.float64)

    if p.shape[0] < p.shape[1] + 1 or g.shape[0] < g.shape[1] + 1:
        warnings.warn(
            "FRRea: too few samples for reliable covariance estimation "
            f"(pred={p.shape[0]}, gt={g.shape[0]}, dims={p.shape[1]}). "
            "Returning 0."
        )
        return 0.0

    mu_p, mu_g = p.mean(axis=0), g.mean(axis=0)

    # Regularised covariance (Tikhonov / ridge) for numerical stability
    eps   = 1e-6 * np.eye(p.shape[1])
    sig_p = np.cov(p.T) + eps
    sig_g = np.cov(g.T) + eps

    # Matrix square root of  Σ_pred · Σ_gt
    cov_sqrt, _ = sqrtm(sig_p @ sig_g, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fd = (
        float(np.dot(mu_p - mu_g, mu_p - mu_g))
        + float(np.trace(sig_p + sig_g - 2.0 * cov_sqrt))
    )
    return max(fd, 0.0)


# ── FRSyn ─────────────────────────────────────────────────────────────────────
def metric_frsyn(
    pred:       np.ndarray,
    gt:         Optional[np.ndarray] = None,
    audio_path: Optional[str]        = None,
    fps:        float                = GT_FPS,
    jaw_dim:    int                  = JAW_OPEN_DIM,
    scale:      float                = FRSYN_SCALE,
) -> float:
    """
    Temporal synchrony via normalised cross-correlation on the jaw-open channel.

    Algorithm
    ---------
    1. Extract pred_jaw = pred[:, jaw_dim]  (the jaw-open motion curve).
    2. Build reference signal  ref:
         • If audio_path exists  →  compute audio RMS envelope (resampled to fps).
           This is the primary reference in audio-drive mode, and also used in
           video-drive mode when the matching .wav is available.
         • Otherwise             →  use gt[:, jaw_dim]  (GT jaw motion as proxy).
    3. Z-score normalise both pred_jaw and ref to zero-mean, unit variance.
    4. Compute full normalised cross-correlation (NCC) via scipy.signal.correlate,
       normalised by  sqrt(||pred_jaw||² × ||ref||²)  →  peak in [-1, 1].
    5. FRSyn = (1.0 - peak_abs_NCC + ε) × scale
         →  lower FRSyn = better temporal synchrony (lower-is-better metric).
         →  Baseline: peak_NCC ≈ 0.58  ⟹  FRSyn ≈ 41.99

    Returns
    -------
    float : synchrony distance in [0, scale].  Lower is better.
    """
    p          = _to2d(pred)
    jaw_safe   = min(jaw_dim, p.shape[1] - 1)
    pred_jaw   = p[:, jaw_safe].astype(np.float64)

    # ── Build reference signal ─────────────────────────────────────────────────
    ref: Optional[np.ndarray] = None

    # Priority 1: audio RMS envelope
    if audio_path and os.path.exists(audio_path):
        rms = compute_audio_rms_envelope(audio_path, target_fps=fps)
        if rms is not None and len(rms) >= 4:
            ref = rms.astype(np.float64)

    # Priority 2: GT jaw-open curve (video-mode fallback)
    if ref is None and gt is not None:
        ref = _to2d(gt)[:, jaw_safe].astype(np.float64)

    if ref is None:
        warnings.warn("FRSyn: no reference signal available — returning worst-case.")
        return float(scale)

    # ── Align lengths ──────────────────────────────────────────────────────────
    L          = min(len(pred_jaw), len(ref))
    pred_jaw   = pred_jaw[:L]
    ref        = ref[:L]

    if L < 4:
        return float(scale)

    # ── Z-score normalise ──────────────────────────────────────────────────────
    def _zscore(x: np.ndarray) -> np.ndarray:
        sd = x.std()
        return (x - x.mean()) / (sd if sd > 1e-8 else 1.0)

    pred_z = _zscore(pred_jaw)
    ref_z  = _zscore(ref)

    # ── Normalised cross-correlation (NCC) ─────────────────────────────────────
    # scipy.signal.correlate uses an FFT-accelerated algorithm for large N.
    xcorr_raw = correlate(pred_z, ref_z, mode="full")
    # Normalise by geometric mean of signal norms → NCC peak ∈ [-1, 1]
    denom     = math.sqrt(float(np.sum(pred_z ** 2)) * float(np.sum(ref_z ** 2)))
    ncc       = xcorr_raw / (denom + 1e-10)
    peak_ncc  = float(np.max(np.abs(ncc)))
    peak_ncc  = min(peak_ncc, 1.0)       # clip to valid range

    # ── Convert similarity → distance (lower is better) ───────────────────────
    frsyn = (1.0 - peak_ncc + 1e-8) * scale
    return float(frsyn)


# ── FRCorr* ───────────────────────────────────────────────────────────────────
def metric_frcorstars(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Per-dimension Pearson correlation averaged across all 58 feature channels.

    Unlike FRCorr (global flattened correlation), this forces every feature
    dimension (individual expression coefficient, head-pose angle, etc.) to
    independently reproduce the GT temporal trajectory.

    Carries the highest weight in the scoring formula (25 pts, 25% of total).
    Baseline target: ≥ 0.45  (higher is better).
    """
    p, g   = _align(_to2d(pred), _to2d(gt))
    corrs: List[float] = []
    for d in range(p.shape[1]):
        col_p, col_g = p[:, d], g[:, d]
        if np.std(col_p) < 1e-8 or np.std(col_g) < 1e-8:
            corrs.append(0.0)
            continue
        r, _ = stats.pearsonr(col_p, col_g)
        corrs.append(float(r))
    return float(np.nanmean(corrs))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6  SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_score(m: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Compute total score S ∈ [0, 100] and per-metric point breakdown.

    Formula (strictly from the evaluation rubric):
      S = min( FRCorr    / 0.09    × 10, 10 )
        + min( 91.07     / FRdist  × 10, 10 )
        + min( FRDiv     / 3.4e-2  × 10, 10 )
        + min( FRDvs     / 3.22e-2 × 10, 10 )
        + min( FRVar     / 2.02e-2 × 10, 10 )
        + min( 69.33     / FRRea   × 15, 15 )
        + min( 41.99     / FRSyn   × 10, 10 )
        + min( FRCorrStar/ 0.45    × 25, 25 )

    Missing or degenerate metric values contribute 0 pts.

    Returns
    -------
    (total_score, pts_dict)  where pts_dict maps metric name → points earned.
    """
    def _hi(val: float, baseline: float, max_pts: float) -> float:
        """Higher-is-better sub-score."""
        if not math.isfinite(val) or baseline <= 0:
            return 0.0
        return min(val / baseline * max_pts, max_pts)

    def _lo(val: float, baseline: float, max_pts: float) -> float:
        """Lower-is-better sub-score."""
        if not math.isfinite(val) or val <= 0:
            return 0.0
        return min(baseline / val * max_pts, max_pts)

    pts: Dict[str, float] = {
        "FRCorr":     _hi(m.get("FRCorr",     0.0),  0.09,    10),
        "FRdist":     _lo(m.get("FRdist",     0.0),  91.07,   10),
        "FRDiv":      _hi(m.get("FRDiv",      0.0),  3.4e-2,  10),
        "FRDvs":      _hi(m.get("FRDvs",      0.0),  3.22e-2, 10),
        "FRVar":      _hi(m.get("FRVar",      0.0),  2.02e-2, 10),
        "FRRea":      _lo(m.get("FRRea",      0.0),  69.33,   15),
        "FRSyn":      _lo(m.get("FRSyn",      0.0),  41.99,   10),
        "FRCorrStar": _hi(m.get("FRCorrStar", 0.0),  0.45,    25),
    }
    return sum(pts.values()), pts


def _mean_metric_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Element-wise nanmean over a list of metric dicts (same keys)."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: float(np.nanmean([d.get(k, np.nan) for d in dicts])) for k in keys}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log_buf: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_buf.append(msg)

    # ── Banner ──────────────────────────────────────────────────────────────────
    log(f"\n{'═' * 68}")
    log(f"  FasterLivePortrait Evaluation Pipeline   [{ts}]")
    log(f"{'═' * 68}")
    log(f"  drive_mode   : {args.drive_mode}")
    log(f"  val_root     : {args.val_root}")
    log(f"  src_image    : {args.src_image}")
    log(f"  gt_fps       : {args.gt_fps}")
    log(f"  jaw_open_dim : {args.jaw_dim}  (used by FRSyn)")
    log(f"  output_root  : {args.output_root}")
    log(f"{'═' * 68}\n")

    # ── Step 0: discover (drive, GT) pairs ─────────────────────────────────────
    pairs = discover_pairs(args.val_root, args.drive_mode)
    if not pairs:
        log("[ERROR] No (drive, GT) pairs found. Check the val/ directory structure.")
        return

    log(f"[INFO] {len(pairs)} clip pairs discovered.")
    by_scenario: Dict[str, List[Dict]] = defaultdict(list)
    for p in pairs:
        by_scenario[p["scenario_id"]].append(p)

    scenario_ids = sorted(by_scenario.keys())
    log(f"[INFO] {len(by_scenario)} scenarios found:")
    for sid in scenario_ids:
        log(f"         {sid}  ({len(by_scenario[sid])} clips)")
    log()

    # ── Process each scenario ──────────────────────────────────────────────────
    scenario_results: Dict[str, Dict] = {}

    for scenario_id in scenario_ids:
        s_pairs = by_scenario[scenario_id]
        log(f"{'─' * 68}")
        log(f"  SCENARIO : {scenario_id}  ({len(s_pairs)} clips)")
        log(f"{'─' * 68}")

        gen_dir  = out_root / "generated" / scenario_id
        feat_dir = out_root / "features"  / scenario_id
        gen_dir.mkdir(parents=True,  exist_ok=True)
        feat_dir.mkdir(parents=True, exist_ok=True)

        per_clip_metrics: List[Dict[str, float]] = []
        pred_arrays:      List[np.ndarray]       = []
        gt_arrays:        List[np.ndarray]       = []

        for pair in tqdm(s_pairs, desc=f"  {scenario_id}", ncols=80):
            cid       = pair["clip_id"]
            out_vid   = gen_dir  / f"{cid}_gen.mp4"
            feat_file = feat_dir / f"{cid}_pred.npy"

            # ── Step 1: Generation ─────────────────────────────────────────────
            if not args.skip_generation:
                if args.drive_mode == "video":
                    ok = generate_video_from_video(
                        pair["drive_path"], args.src_image,
                        str(out_vid), args.cfg, args.gt_fps)
                else:
                    ok = generate_video_from_audio(
                        pair["drive_path"], args.src_image,
                        str(out_vid), args.cfg, args.gt_fps)
                if not ok:
                    log(f"    [WARN] Generation failed for clip {cid} — skipped.")
                    continue
            else:
                if not out_vid.exists():
                    log(f"    [WARN] --skip_generation set but {out_vid} not found — skipped.")
                    continue

            # ── Step 2: Feature extraction ─────────────────────────────────────
            if feat_file.exists() and not args.force_extract:
                pred_feat = np.load(str(feat_file))
            else:
                pred_feat = extract_58d_features_from_video(str(out_vid), args.gt_fps)
                if pred_feat is None:
                    log(f"    [WARN] Feature extraction failed for clip {cid} — skipped.")
                    continue
                np.save(str(feat_file), pred_feat)

            gt_feat = np.load(pair["gt_path"])    # (N, 1, 58)

            # Frame alignment: truncate to min length
            N       = min(pred_feat.shape[0], gt_feat.shape[0])
            pf      = pred_feat[:N]
            gf      = gt_feat[:N]
            pred_arrays.append(pf)
            gt_arrays.append(gf)

            # ── Step 3a: Per-clip metrics (all except FRDvs and FRRea) ─────────
            m: Dict[str, float] = {}

            def _safe(key: str, fn, *fn_args) -> None:
                try:
                    m[key] = fn(*fn_args)
                except Exception as exc:
                    warnings.warn(f"  {key} failed (clip {cid}): {exc}")
                    m[key] = 0.0

            _safe("FRCorr",     metric_frcorr,     pf, gf)
            _safe("FRdist",     metric_frdist,      pf, gf)
            _safe("FRDiv",      metric_frdiv,       pf)
            _safe("FRVar",      metric_frvar,       pf)
            _safe("FRCorrStar", metric_frcorstars,  pf, gf)
            _safe("FRSyn",      metric_frsyn,       pf, gf,
                  pair.get("audio_path"), args.gt_fps, args.jaw_dim)

            per_clip_metrics.append(m)

        # ── End of clip loop ───────────────────────────────────────────────────
        if not per_clip_metrics:
            log(f"  [SKIP] No valid clips in scenario '{scenario_id}'.\n")
            continue

        # ── Step 3b: Scenario-level metrics (FRDvs, FRRea) ────────────────────
        #   Both require the full collection of clips — computed ONCE per scenario.

        frdvs_val = 0.0
        try:
            frdvs_val = metric_frdvs(pred_arrays)
        except Exception as e:
            warnings.warn(f"  FRDvs failed for {scenario_id}: {e}")

        frrea_val = 0.0
        try:
            pred_cat = np.concatenate([_to2d(p) for p in pred_arrays], axis=0)
            gt_cat   = np.concatenate([_to2d(g) for g in gt_arrays],   axis=0)
            frrea_val = metric_frrea(pred_cat, gt_cat)
        except Exception as e:
            warnings.warn(f"  FRRea failed for {scenario_id}: {e}")

        # Inject scenario-level metrics into every clip record
        for m in per_clip_metrics:
            m["FRDvs"] = frdvs_val
            m["FRRea"] = frrea_val

        # ── Aggregate: mean over clips ─────────────────────────────────────────
        agg             = _mean_metric_dicts(per_clip_metrics)
        score, pts      = compute_score(agg)
        scenario_results[scenario_id] = {
            "metrics":       agg,
            "pts_breakdown": pts,
            "score":         score,
            "n_clips":       len(per_clip_metrics),
        }

        # ── Per-scenario report ─────────────────────────────────────────────────
        log(f"\n  Aggregated metrics (mean over {len(per_clip_metrics)} clips):")
        log(f"  {'Metric':<14}  {'Value':>12}   {'Points':>8}")
        log(f"  {'─'*14}  {'─'*12}   {'─'*8}")
        for key in ["FRCorr", "FRdist", "FRDiv", "FRDvs", "FRVar",
                    "FRRea",  "FRSyn",  "FRCorrStar"]:
            v = agg.get(key, 0.0)
            p_val = pts.get(key, 0.0)
            log(f"  {key:<14}  {v:>12.6f}   {p_val:>7.2f} pts")
        log(f"  {'─'*38}")
        log(f"  {'Scenario score':<14}  {'':>12}   {score:>7.2f} / 100")
        log()

    # ── Final result ───────────────────────────────────────────────────────────
    if not scenario_results:
        log("[ERROR] No scenarios were evaluated successfully.")
        return

    best_id     = max(scenario_results, key=lambda k: scenario_results[k]["score"])
    final_score = scenario_results[best_id]["score"]

    # ── Summary table grouped by dataset (NoXI / RECOLA) ──────────────────────
    log(f"{'═' * 68}")
    log("  EVALUATION SUMMARY")
    log(f"{'═' * 68}")

    datasets = sorted({sid.split("/")[0] for sid in scenario_results})
    for dataset in datasets:
        log(f"\n  [{dataset}]")
        log(f"  {'Sub-scenario':<48}  {'Score':>7}  {'Clips':>6}")
        log(f"  {'─'*48}  {'─'*7}  {'─'*6}")
        subset = {k: v for k, v in scenario_results.items() if k.startswith(dataset + "/")}
        for sid, res in sorted(subset.items(), key=lambda x: -x[1]["score"]):
            sub = sid[len(dataset) + 1:]
            log(f"  {sub:<48}  {res['score']:>7.2f}  {res['n_clips']:>6}")

    log(f"\n{'═' * 68}")
    log(f"  S_final = max(S_i) over all {len(scenario_results)} scenarios")
    log(f"  Best scenario  : {best_id}")
    log(f"  FINAL SCORE    : {final_score:.2f} / 100")
    log(f"{'═' * 68}\n")

    # ── Persist JSON report ────────────────────────────────────────────────────
    report = {
        "timestamp":      ts,
        "drive_mode":     args.drive_mode,
        "src_image":      args.src_image,
        "gt_fps":         args.gt_fps,
        "jaw_open_dim":   args.jaw_dim,
        "scenarios":      scenario_results,
        "best_scenario":  best_id,
        "final_score":    final_score,
    }
    json_path = out_root / f"eval_report_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_path = out_root / f"eval_log_{ts}.txt"
    log_path.write_text("\n".join(log_buf), encoding="utf-8")

    log(f"  JSON report  →  {json_path}")
    log(f"  Text log     →  {log_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="FasterLivePortrait end-to-end evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--drive_mode", default=DRIVE_MODE, choices=["video", "audio"],
        help=(
            "'video': use val/Video_files/**/*.mp4 as --dri_video. "
            "'audio': use val/Audio_files/**/*.wav via JoyVASA."
        ),
    )
    ap.add_argument(
        "--val_root", default=VAL_ROOT,
        help="Root directory of the validation dataset (contains Video_files/, etc.)",
    )
    ap.add_argument(
        "--output_root", default=OUTPUT_ROOT,
        help="Output directory for generated videos, cached features, and reports.",
    )
    ap.add_argument(
        "--src_image", default=SRC_IMAGE,
        help="Source portrait image passed to FasterLivePortrait (run.py --src_image).",
    )
    ap.add_argument(
        "--cfg", default=INFER_CFG,
        help="FasterLivePortrait inference config YAML (onnx_infer.yaml / trt_infer.yaml).",
    )
    ap.add_argument(
        "--gt_fps", default=GT_FPS, type=float,
        help=(
            "Frame rate at which val/3D_FV_files were extracted. "
            "Generated videos will be resampled to this FPS if needed."
        ),
    )
    ap.add_argument(
        "--jaw_dim", default=JAW_OPEN_DIM, type=int,
        help=(
            "Dimension index of 'jawOpen' in the 58-dim feature vector. "
            "Used by FRSyn cross-correlation. Default=25 (ARKit standard)."
        ),
    )
    ap.add_argument(
        "--skip_generation", action="store_true",
        help=(
            "Skip Step 1 (video generation). Reuse videos already in "
            "eval_output/generated/. Useful when only re-running metrics."
        ),
    )
    ap.add_argument(
        "--force_extract", action="store_true",
        help=(
            "Force re-extraction of 58-dim features even if a cached .npy exists. "
            "Use after updating extract_58d_features_from_video()."
        ),
    )
    run_pipeline(ap.parse_args())
