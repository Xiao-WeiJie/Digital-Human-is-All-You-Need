# -*- coding: utf-8 -*-
"""
舌诊识别适配器

封装 tongue_diagnosis 中的单帧 YOLO + SAM + ResNet 推理能力，供 Web API
和桌面摄像头演示复用。
"""

import base64
import logging
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TONGUE_ROOT = PROJECT_ROOT / "tongue_diagnosis"
WEIGHTS_DIR = TONGUE_ROOT / "application" / "net" / "weights"
MODEL_DIR = TONGUE_ROOT / "application" / "net" / "model"


class TongueDetectionError(Exception):
    """可返回给前端的舌诊业务错误。"""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class TongueResult:
    """舌诊识别结果。"""

    tongue_color: str
    coat_color: str
    thickness: str
    greasiness: str
    confidence: float
    advice: str = "本次结果仅供辅助参考"
    timestamp: float = field(default_factory=time.time)
    box: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的响应字典。"""
        return {
            "tongue_color": self.tongue_color,
            "coat_color": self.coat_color,
            "thickness": self.thickness,
            "greasiness": self.greasiness,
            "confidence": round(float(self.confidence), 4),
            "advice": self.advice,
            "timestamp": self.timestamp,
        }

    def to_runtime_dict(self) -> Dict[str, Any]:
        """转换为桌面演示绘制逻辑兼容的字典。"""
        return {
            "success": True,
            "tongue_color": self.tongue_color,
            "coat_color": self.coat_color,
            "thickness": self.thickness,
            "greasiness": self.greasiness,
            "box": self.box,
            "mask": self.mask,
            "conf": float(self.confidence),
        }


class TongueAdapter:
    """单帧舌诊适配器，模型按需懒加载并在后续请求中复用。"""

    TONGUE_COLORS = ["淡白", "淡红", "红", "紫", "青"]
    COAT_COLORS = ["白", "黄", "灰黑"]
    THICKNESS = ["薄", "厚"]
    GREASINESS = ["不腻", "腻"]

    def __init__(
        self,
        yolo_path: Optional[str] = None,
        sam_path: Optional[str] = None,
        resnet_paths: Optional[list] = None,
        device: str = "cpu",
    ):
        custom_resnet_paths = list(resnet_paths or [])
        default_resnet_paths = [
            WEIGHTS_DIR / "tongue_color.pth",
            WEIGHTS_DIR / "tongue_coat_color.pth",
            WEIGHTS_DIR / "thickness.pth",
            WEIGHTS_DIR / "rot_and_greasy.pth",
        ]

        self.yolo_path = self._resolve_path(yolo_path, WEIGHTS_DIR / "yolov5.pt")
        self.sam_path = self._resolve_path(sam_path, WEIGHTS_DIR / "sam_vit_b_01ec64.pth")
        self.resnet_paths = [
            self._resolve_path(custom_resnet_paths[index] if index < len(custom_resnet_paths) else None, default)
            for index, default in enumerate(default_resnet_paths)
        ]
        self.device = device

        self.yolo = None
        self.sam = None
        self.sam_predictor = None
        self.resnet = None
        self._loaded = False
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @staticmethod
    def _resolve_path(path: Optional[str], default: Path) -> Path:
        if not path:
            return default
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        candidate = TONGUE_ROOT / path_obj
        if candidate.exists():
            return candidate
        return PROJECT_ROOT / path_obj

    @staticmethod
    def _load_yolo_model(yolo_path: Path, device: str):
        import torch
        from yolov5 import load

        original_torch_load = torch.load
        original_windows_path = getattr(pathlib, "WindowsPath", None)
        original_posix_path = getattr(pathlib, "PosixPath", None)

        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat
        if os.name != "nt" and original_windows_path is not None and original_posix_path is not None:
            pathlib.WindowsPath = pathlib.PosixPath
        elif os.name == "nt" and original_windows_path is not None and original_posix_path is not None:
            pathlib.PosixPath = pathlib.WindowsPath

        try:
            return load(str(yolo_path), device=device)
        finally:
            torch.load = original_torch_load
            if original_windows_path is not None:
                pathlib.WindowsPath = original_windows_path
            if original_posix_path is not None:
                pathlib.PosixPath = original_posix_path

    def _ensure_model_files(self) -> None:
        missing_paths = [
            path for path in [self.yolo_path, self.sam_path, *self.resnet_paths]
            if not path.exists()
        ]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise TongueDetectionError(-5, f"tongue model file not found: {missing}")

    def _ensure_models_loaded(self) -> None:
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            self._ensure_model_files()
            logger.info("[TongueAdapter] Loading tongue diagnosis models on %s", self.device)

            try:
                import torch
                from segment_anything import SamPredictor, sam_model_registry

                if str(MODEL_DIR) not in sys.path:
                    sys.path.insert(0, str(MODEL_DIR))
                from resnet import ResNetPredictor

                self.yolo = self._load_yolo_model(self.yolo_path, self.device)
                self.sam = sam_model_registry["vit_b"](checkpoint=str(self.sam_path)).to(torch.device(self.device))
                self.sam.eval()
                self.sam_predictor = SamPredictor(sam_model=self.sam)
                self.resnet = ResNetPredictor([str(path) for path in self.resnet_paths])
                self._loaded = True

                logger.info("[TongueAdapter] Tongue diagnosis models loaded")
            except TongueDetectionError:
                raise
            except Exception as exc:
                logger.exception("[TongueAdapter] Failed to load tongue models")
                raise TongueDetectionError(-5, f"tongue model load failed: {exc}") from exc

    def detect_from_base64(self, base64_image: str, session_id: str = "default") -> TongueResult:
        """从 Base64 图片执行单帧舌诊。"""
        del session_id

        if not base64_image:
            raise TongueDetectionError(-1, "image parameter is required")

        try:
            payload = base64_image.split(",", 1)[1] if "," in base64_image else base64_image
            image_data = base64.b64decode(payload)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise TongueDetectionError(-2, "invalid image") from exc

        if frame is None or frame.size == 0:
            raise TongueDetectionError(-2, "invalid image")

        return self.detect_from_frame(frame)

    def detect_from_frame(self, frame: np.ndarray, session_id: str = "default") -> TongueResult:
        """从 OpenCV BGR 图像执行单帧舌诊。"""
        del session_id

        if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
            raise TongueDetectionError(-2, "invalid image")

        self._ensure_models_loaded()

        try:
            import torch

            with self._inference_lock:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)

                self.yolo.eval()
                with torch.no_grad():
                    prediction = self.yolo(pil_image)

                detected_count = len(prediction.xyxy[0])
                if detected_count < 1:
                    raise TongueDetectionError(-3, "未检测到舌体")
                if detected_count > 1:
                    raise TongueDetectionError(-4, "检测到多个舌体，请重新取景")

                raw_box = prediction.xyxy[0][0, :4].detach().cpu().numpy()
                confidence = float(prediction.xyxy[0][0, 4].detach().cpu().item())
                height, width = rgb_image.shape[:2]
                x1, y1, x2, y2 = self._clamp_box(raw_box, width, height)

                if x2 <= x1 or y2 <= y1:
                    raise TongueDetectionError(-5, "invalid tongue box")

                self.sam_predictor.set_image(rgb_image)
                with torch.no_grad():
                    masks, _, _ = self.sam_predictor.predict(box=np.array([x1, y1, x2, y2]))

                if masks is None or len(masks) == 0:
                    raise TongueDetectionError(-5, "tongue segmentation failed")

                mask = masks[0]
                segmented = rgb_image * mask[:, :, np.newaxis]
                tongue_crop = Image.fromarray(segmented.astype(np.uint8)).crop((x1, y1, x2, y2)).convert("RGB")
                indices = self.resnet.predict(np.array(tongue_crop))

                return TongueResult(
                    tongue_color=self._safe_label(self.TONGUE_COLORS, indices[0]),
                    coat_color=self._safe_label(self.COAT_COLORS, indices[1]),
                    thickness=self._safe_label(self.THICKNESS, indices[2]),
                    greasiness=self._safe_label(self.GREASINESS, indices[3]),
                    confidence=confidence,
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    mask=mask,
                )
        except TongueDetectionError:
            raise
        except Exception as exc:
            logger.exception("[TongueAdapter] Tongue detection failed")
            raise TongueDetectionError(-5, f"tongue detection failed: {exc}") from exc

    @staticmethod
    def _clamp_box(box: np.ndarray, width: int, height: int):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        return x1, y1, x2, y2

    @staticmethod
    def _safe_label(labels, index: int) -> str:
        try:
            return labels[int(index)]
        except Exception:
            return "未知"


_tongue_adapter_instance: Optional[TongueAdapter] = None
_tongue_adapter_lock = Lock()


def get_tongue_adapter(
    yolo_path: Optional[str] = None,
    sam_path: Optional[str] = None,
    resnet_paths: Optional[list] = None,
    device: str = "cpu",
) -> TongueAdapter:
    """获取舌诊适配器单例。"""
    global _tongue_adapter_instance

    if _tongue_adapter_instance is None:
        with _tongue_adapter_lock:
            if _tongue_adapter_instance is None:
                _tongue_adapter_instance = TongueAdapter(
                    yolo_path=yolo_path,
                    sam_path=sam_path,
                    resnet_paths=resnet_paths,
                    device=device,
                )

    return _tongue_adapter_instance
