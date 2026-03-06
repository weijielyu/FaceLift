# Copyright 2025 Adobe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import os
from PIL import Image
from typing import Dict, List, Union

class FixViewDataset(Dataset):
    """Dataset for multi-view diffusion training with fixed viewpoints."""
    
    def __init__(self, config, split: str):
        super().__init__()
        self.config = config
        self.split = split

        # Dataset parameters
        self.img_wh = self.config.get("img_wh", 512)
        self.n_views = self.config.get("n_views", 6)

        # Load data paths and background color settings
        if self.split == "train":
            with open(self.config.train_dataset.path, 'r') as f:
                self.all_data_paths = f.read().strip().split("\n")
            self.bg_color = self.config.train_dataset.bg_color
        elif self.split == "val":
            with open(self.config.validation_dataset.path, 'r') as f:
                self.all_data_paths = f.read().strip().split("\n")
            self.bg_color = self.config.validation_dataset.bg_color
        else:
            raise NotImplementedError(f"Split '{split}' is not supported")

        # Filter empty paths
        self.all_data_paths = [path for path in self.all_data_paths if len(path.strip()) > 0]
        self.all_data_paths = pd.array(self.all_data_paths, dtype="string")

        # Shuffle the validation set
        if self.split == "val":
            random.shuffle(self.all_data_paths)

        # Camera view configuration
        self.view_type_to_idx = {
            "front": 0,
            "front_right": 1,
            "right": 2,
            "back": 3,
            "left": 4,
            "front_left": 5,
        }

        self.view_types = ["front", "front_right", "right", "back", "left", "front_left"]
        self.target_view_idx = [self.view_type_to_idx[view] for view in self.view_types]

        # Load pre-computed color prompt embeddings
        self.color_prompt_embedding = torch.load("mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt")

        # Precompute background color choices for efficiency
        self._bg_color_choices = {
            'white': np.array([1., 1., 1.], dtype=np.float32),
            'black': np.array([0., 0., 0.], dtype=np.float32),
            'gray': np.array([0.5, 0.5, 0.5], dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.all_data_paths)

    def get_bg_color(self) -> np.ndarray:
        """Generate background color based on configuration."""
        if self.bg_color in self._bg_color_choices:
            return self._bg_color_choices[self.bg_color].copy()
        elif self.bg_color == 'random':
            return np.random.rand(3).astype(np.float32)
        elif self.bg_color == 'three_choices':
            return random.choice(list(self._bg_color_choices.values())).copy()
        elif isinstance(self.bg_color, (int, float)):
            return np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError(f"Background color '{self.bg_color}' is not supported")

    def load_image(self, image_path: str, bg_color: np.ndarray) -> torch.Tensor:
        """
        Load and process an image with background compositing.
        
        Args:
            image_path: Path to the RGBA image
            bg_color: Background color as RGB array
            
        Returns:
            Processed image as torch tensor
        """
        rgba = Image.open(image_path).convert('RGBA')

        # Load and resize image
        # rgba = Image.open(image_path).convert('RGBA')
        if rgba.size != (self.img_wh, self.img_wh):
            rgba = rgba.resize((self.img_wh, self.img_wh), Image.LANCZOS)

        # Convert to numpy and normalize
        rgba = np.array(rgba, dtype=np.float32) / 255.0
        image = rgba[..., :3]
        alpha = rgba[..., 3:4]
        
        # Validate alpha channel
        if alpha.sum() <= 1e-8:
            raise ValueError(f"Image {image_path} has no foreground content")
        
        # Composite with background
        image = image * alpha + bg_color[None, None, :] * (1 - alpha)
        
        return torch.from_numpy(image)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing input images, output images, and embeddings
        """
        data_path = self.all_data_paths[index].strip()
        bg_color = self.get_bg_color()

        # Load conditioning (input) image - front view
        cond_view_idx = self.view_type_to_idx["front"]
        input_image_path = os.path.join(data_path, f"cam_{cond_view_idx:03d}.png")
        
        # Replicate input image for all views
        input_image = self.load_image(input_image_path, bg_color).permute(2, 0, 1)
        input_images = torch.stack([input_image] * self.n_views, dim=0)

        # Load target images for all target views
        target_images = []
        for view_idx in self.target_view_idx:
            target_image_path = os.path.join(data_path, f"cam_{view_idx:03d}.png")
            target_image = self.load_image(target_image_path, bg_color).permute(2, 0, 1)
            target_images.append(target_image)
        
        target_images = torch.stack(target_images, dim=0)

        return {
            'imgs_in': input_images.float(),      # (n_views, 3, H, W)
            'imgs_out': target_images.float(),    # (n_views, 3, H, W)
            'color_prompt_embeddings': self.color_prompt_embedding,
        }