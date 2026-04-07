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

import os
import json
import pickle
import torch
import numpy as np
import cv2
from typing import Dict, Optional, Any
from logger import logger


class AvatarManager:
    """
    Manages multiple digital human avatars for FasterLivePortrait.

    Avatar data structure:
    - source_image: Path to the avatar source image
    - src_info: Preprocessed source information (landmarks, features, etc.)
    - src_img: RGB source image array
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the AvatarManager.

        Args:
            config_path: Path to avatar_config.json. If None, uses default path.
        """
        if config_path is None:
            # Default config path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "Human_Choice", "avatar_config.json")

        self.config_path = config_path
        self.config = self._load_config()
        self.avatars: Dict[str, Dict[str, Any]] = {}
        self.current_avatar_id: Optional[str] = None
        self.pipeline = None  # Will be set when pipeline is available

    def _load_config(self) -> Dict:
        """Load avatar configuration from JSON file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Avatar config not found at {self.config_path}, creating default config")
            return self._create_default_config()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded avatar config from {self.config_path}")
        return config

    def _create_default_config(self) -> Dict:
        """Create a default avatar configuration."""
        default_config = {
            "avatars": {
                "human_1": {
                    "name": "Human 1",
                    "source_image": "Human_1/qys.png",
                    "description": "Default digital human avatar"
                },
                "human_2": {
                    "name": "Human 2",
                    "source_image": "Human_2/dy.jpg",
                    "description": "Second digital human avatar"
                }
            },
            "default_avatar": "human_1"
        }

        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        logger.info(f"Created default avatar config at {self.config_path}")
        return default_config

    def set_pipeline(self, pipeline):
        """Set the FasterLivePortrait pipeline for preprocessing."""
        self.pipeline = pipeline

    def list_avatars(self) -> list:
        """Return list of available avatar IDs."""
        return list(self.config.get("avatars", {}).keys())

    def get_avatar_info(self, avatar_id: str) -> Optional[Dict]:
        """Get avatar information by ID."""
        return self.config.get("avatars", {}).get(avatar_id)

    def load_avatar(self, avatar_id: str, pipeline=None) -> bool:
        """
        Load and preprocess an avatar.

        Args:
            avatar_id: The avatar identifier
            pipeline: FasterLivePortrait pipeline instance. If None, uses self.pipeline

        Returns:
            True if successful, False otherwise
        """
        if pipeline is not None:
            self.pipeline = pipeline

        if self.pipeline is None:
            logger.error("Pipeline not set. Call set_pipeline() first or pass pipeline parameter.")
            return False

        avatar_info = self.get_avatar_info(avatar_id)
        if avatar_info is None:
            logger.error(f"Avatar '{avatar_id}' not found in config")
            return False

        # Get source image path
        source_image = avatar_info.get("source_image")
        if source_image is None:
            logger.error(f"No source_image defined for avatar '{avatar_id}'")
            return False

        # Make path absolute if relative
        if not os.path.isabs(source_image):
            base_dir = os.path.dirname(self.config_path)
            source_image = os.path.join(base_dir, source_image)

        if not os.path.exists(source_image):
            logger.error(f"Source image not found: {source_image}")
            return False

        logger.info(f"Loading avatar '{avatar_id}' from {source_image}")

        # Prepare source using pipeline
        success = self.pipeline.prepare_source(source_image, realtime=True)
        if not success:
            logger.error(f"Failed to prepare source for avatar '{avatar_id}'")
            return False

        # Store avatar data
        self.avatars[avatar_id] = {
            "source_path": source_image,
            "src_imgs": self.pipeline.src_imgs,
            "src_infos": self.pipeline.src_infos,
            "info": avatar_info
        }

        self.current_avatar_id = avatar_id
        logger.info(f"Successfully loaded avatar '{avatar_id}'")
        return True

    def get_current_avatar(self) -> Optional[Dict]:
        """Get the currently loaded avatar data."""
        if self.current_avatar_id is None:
            return None
        return self.avatars.get(self.current_avatar_id)

    def get_avatar_data(self, avatar_id: str = None) -> Optional[tuple]:
        """
        Get avatar data for rendering.

        Args:
            avatar_id: Avatar ID. If None, uses current avatar.

        Returns:
            Tuple of (src_img, src_info) or None if not available
        """
        if avatar_id is None:
            avatar_id = self.current_avatar_id

        if avatar_id is None:
            return None

        avatar_data = self.avatars.get(avatar_id)
        if avatar_data is None:
            return None

        src_imgs = avatar_data.get("src_imgs", [])
        src_infos = avatar_data.get("src_infos", [])

        if len(src_imgs) == 0 or len(src_infos) == 0:
            return None

        # Return first face data
        return src_imgs[0], src_infos[0][0] if src_infos else None

    def switch_avatar(self, avatar_id: str, pipeline=None) -> bool:
        """
        Switch to a different avatar.

        Args:
            avatar_id: The avatar to switch to
            pipeline: Pipeline instance if not already set

        Returns:
            True if successful, False otherwise
        """
        if avatar_id in self.avatars:
            # Already loaded
            self.current_avatar_id = avatar_id
            logger.info(f"Switched to preloaded avatar '{avatar_id}'")
            return True

        # Need to load
        return self.load_avatar(avatar_id, pipeline)

    def preload_all_avatars(self, pipeline=None) -> Dict[str, bool]:
        """
        Preload all configured avatars.

        Args:
            pipeline: FasterLivePortrait pipeline instance

        Returns:
            Dict mapping avatar_id to load success status
        """
        results = {}
        for avatar_id in self.list_avatars():
            results[avatar_id] = self.load_avatar(avatar_id, pipeline)
        return results

    def clear_avatar_cache(self, avatar_id: str = None):
        """
        Clear cached avatar data.

        Args:
            avatar_id: Specific avatar to clear, or None to clear all
        """
        if avatar_id is None:
            self.avatars.clear()
            self.current_avatar_id = None
            logger.info("Cleared all avatar cache")
        elif avatar_id in self.avatars:
            del self.avatars[avatar_id]
            if self.current_avatar_id == avatar_id:
                self.current_avatar_id = None
            logger.info(f"Cleared cache for avatar '{avatar_id}'")


# Singleton instance
_avatar_manager_instance: Optional[AvatarManager] = None


def get_avatar_manager(config_path: str = None) -> AvatarManager:
    """
    Get or create the singleton AvatarManager instance.

    Args:
        config_path: Path to avatar config. Only used on first call.

    Returns:
        AvatarManager instance
    """
    global _avatar_manager_instance
    if _avatar_manager_instance is None:
        _avatar_manager_instance = AvatarManager(config_path)
    return _avatar_manager_instance
