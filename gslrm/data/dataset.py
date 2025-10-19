# Copyright (C) 2025, FaceLift Research Group
# https://github.com/weijielyu/FaceLift
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact: wlyu3@ucmerced.edu

import json
import random
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import torch
import os

from torch.utils.data import Dataset

def pil_to_np(pil_image):
    if pil_image.mode == "RGBA":
        # if directly convert to np.array, alpha=0 pixels will be black
        r, g, b, a = pil_image.split()
        r, g, b, a = np.asarray(r), np.asarray(g), np.asarray(b), np.asarray(a)
        image = np.stack([r, g, b, a], axis=2)
    else:
        image = np.asarray(pil_image)
    return image


def get_bg_color(bg_color_config):
    """
    Generate background color based on configuration.
    
    Args:
        bg_color_config: Configuration for background color. Can be:
            - 'white': White background [1, 1, 1]
            - 'black': Black background [0, 0, 0]
            - 'gray': Gray background [0.5, 0.5, 0.5]
            - 'random': Random RGB values
            - 'three_choices': Randomly choose from white, black, or gray
            - float: Grayscale value applied to all channels
    
    Returns:
        torch.Tensor: RGB color tensor of shape (3,) with values in [0, 1]
    
    Raises:
        ValueError: If bg_color_config is not a supported type or value
    """
    # Predefined color constants
    COLORS = {
        'white': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'black': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'gray': np.array([0.5, 0.5, 0.5], dtype=np.float32)
    }
    
    if isinstance(bg_color_config, str):
        if bg_color_config in COLORS:
            bg_color = COLORS[bg_color_config]
        elif bg_color_config == 'random':
            bg_color = np.random.rand(3).astype(np.float32)
        elif bg_color_config == 'three_choices':
            bg_color = random.choice(list(COLORS.values()))
        else:
            raise ValueError(f"Unsupported string background color: '{bg_color_config}'. "
                           f"Supported options: {list(COLORS.keys()) + ['random', 'three_choices']}")
    elif isinstance(bg_color_config, (int, float)):
        if not 0 <= bg_color_config <= 1:
            raise ValueError(f"Numeric background color must be in range [0, 1], got {bg_color_config}")
        bg_color = np.array([bg_color_config] * 3, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported background color type: {type(bg_color_config)}. "
                        "Expected str, int, or float.")

    return torch.from_numpy(bg_color)

class RandomViewDataset(Dataset):
    """
    Dataset for loading multi-view images with random view sampling.
    
    This dataset loads images from multiple viewpoints and applies various preprocessing
    including background handling, resizing, and camera parameter normalization.
    
    Args:
        config: Configuration object containing dataset parameters
        split: Dataset split ('train' or 'val')
    """
    
    def __init__(self, config, split: str):
        super().__init__()
        self.config = config
        self.split = split

        # Load dataset paths based on split
        if self.split == "train":
            dataset_path = self.config.training.dataset.dataset_path
        elif self.split == "val":
            dataset_path = self.config.validation.dataset_path
        else:
            raise NotImplementedError(f"Split '{split}' is not supported")

        # Load dataset paths from local file
        with open(dataset_path, 'r') as f:
            self.all_data_paths = f.read().strip().split("\n")

        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.all_data_paths = pd.array(
            [s for s in self.all_data_paths if len(s) > 0], dtype="string"
        )

        # Extract dataset configuration
        dataset_config = self.config.training.dataset
        self.bg_color = dataset_config.get("background_color", "white")
        self.maximize_view_overlap = dataset_config.get("maximize_view_overlap", False)
        self.remove_alpha = dataset_config.get("remove_alpha", False)
        self.num_views = dataset_config.get("num_views", self.config.training.get("num_views", 8))
        self.num_input_views = dataset_config.get("num_input_views", self.config.training.get("num_input_views", 4))
        self.target_has_input = dataset_config.get("target_has_input", True)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.all_data_paths)

    def select_max_overlap_views(self, cameras, viewangle_threshold=60):
        """
        Select views that maximize overlap by choosing views within viewing angle threshold.
        
        Returns:
            list or None: Selected view indices, or None if insufficient overlapping views
        """
        # Extract camera positions
        cam_positions = []
        for frame in cameras:
            c2w = np.linalg.inv(np.array(frame["w2c"]))
            cam_positions.append(c2w[:3, 3])
        cam_positions = np.stack(cam_positions, axis=0)  # [N, 3]

        # Randomly select input views
        all_indices = list(range(len(cameras)))
        input_indices = random.sample(all_indices, self.num_input_views)
        selected_indices = input_indices.copy()

        # Select additional views with good overlap
        num_additional = self.num_views - self.num_input_views
        if num_additional > 0:
            # Find views within angle threshold of input views
            dot_products = cam_positions @ cam_positions[input_indices].T  # [N, num_input]
            best_overlap = np.max(dot_products, axis=1)  # best overlap with any input view
            
            angle_threshold = np.cos(np.deg2rad(viewangle_threshold))
            valid_mask = best_overlap >= angle_threshold
            valid_mask[input_indices] = False  # exclude already selected views

            candidates = np.array(all_indices)[valid_mask].tolist()
            if len(candidates) < num_additional:
                print(f"Warning: Need {num_additional} views, found {len(candidates)} "
                      f"within {viewangle_threshold}° threshold")
                return None
                
            selected_indices.extend(random.sample(candidates, num_additional))

        return selected_indices

    def _process_image_channels(self, image, bg_color_255):
        """
        Process image channels, handling RGBA and other formats.
        
        Args:
            image: PIL Image to process
            bg_color_255: Background color as RGB tuple (0-255 range)
            
        Returns:
            PIL Image: Processed image in RGB or RGBA format
        """
        if image.mode == "RGBA":
            # Composite RGBA image onto background color
            background = Image.new("RGB", image.size, bg_color_255)
            alpha_mask = image.split()[3]
            background.paste(image, mask=alpha_mask)
            
            if self.remove_alpha:
                return background
            else:
                background.putalpha(alpha_mask)
                return background
        elif image.mode != "RGB":
            # Convert other modes to RGB
            return image.convert("RGB")
        else:
            return image

    def __getitem__(self, idx):
        """
        Load and preprocess a multi-view sample.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            dict: Contains 'image', 'c2w', 'fxfycxcy', 'index', and 'bg_color'
        """
        try:
        # if True:
            data_json_path = os.path.join(self.all_data_paths[idx].strip(), "opencv_cameras.json")
            data_path = os.path.dirname(data_json_path)
            
            # Load camera data from local file
            with open(data_json_path, 'r') as f:
                    data_json = json.load(f)

            cameras = data_json["frames"]

            bg_color = get_bg_color(self.bg_color)
            bg_color_255 = (int(bg_color[0] * 255), int(bg_color[1] * 255), int(bg_color[2] * 255))

            # Select views based on configuration
            if self.maximize_view_overlap:
                image_choices = self.select_max_overlap_views(cameras)
                if image_choices is None:
                    return self.__getitem__(random.randint(0, len(self) - 1))
            else:
                image_choices = random.sample(
                    range(len(cameras)), self.num_views
                )

            # Sort view indices for deterministic behavior during evaluation
            if self.config.get("evaluation", False) or self.config.get("inference", False):
                input_choices = sorted(image_choices[:self.num_input_views])
                target_choices = sorted(image_choices[self.num_input_views:])
                image_choices = input_choices + target_choices

            # Extract selected camera data
            selected_cameras = [cameras[idx] for idx in image_choices]
            selected_image_paths = [data_path + "/" + cameras[idx]["file_path"] for idx in image_choices]

            # Initialize data collection lists
            input_images = []
            input_fxfycxcy = []
            input_c2ws = []
            for idx_chosen, (camera, image_path) in enumerate(
                zip(selected_cameras, selected_image_paths)
            ):
                # Load and validate image
                image = Image.open(image_path)

                assert image.size[0] == image.size[1], f"Image {image_path} is not square: {image.size}"

                # Resize image if needed
                target_size = self.config.model.image_tokenizer.image_size
                resize_ratio = target_size / image.size[0]
                if image.size[0] != target_size:
                    image = image.resize((target_size, target_size), resample=Image.LANCZOS)

                # Process image channels
                image = self._process_image_channels(image, bg_color_255)

                # Extract and adjust camera intrinsics
                intrinsics = np.array([camera["fx"], camera["fy"], camera["cx"], camera["cy"]])
                intrinsics *= resize_ratio
                
                # Extract camera pose
                c2w = np.linalg.inv(np.array(camera["w2c"]))
                
                # Convert image to tensor
                image_tensor = pil_to_np(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)  # (3, h, w)

                # Collect processed data
                input_images.append(image_tensor)
                input_fxfycxcy.append(intrinsics)
                input_c2ws.append(c2w)

            # Stack all data into tensors/arrays
            input_images = torch.stack(input_images, dim=0)  # [num_views, 3, height, width]
            input_fxfycxcy = np.array(input_fxfycxcy)  # [num_views, 4]
            input_c2ws = np.array(input_c2ws)  # [num_views, 4, 4]

        except Exception as e:
            traceback.print_exc()
            print(f"Error loading data from {data_path}: {str(e)}")
            # Fallback to a random sample to avoid training interruption
            return self.__getitem__(random.randint(0, len(self) - 1))

        input_c2ws = torch.from_numpy(input_c2ws).float()  # [v, 4, 4]
        input_fxfycxcy = torch.from_numpy(input_fxfycxcy).float()  # [v, 4]

        image_indices = (
            torch.from_numpy(np.array(image_choices)).long().unsqueeze(-1)
        )  # [v, 1]
        scene_indices = (
            torch.tensor(idx).long().unsqueeze(0).expand_as(image_indices)
        )  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_fxfycxcy,
            "index": indices,
            "bg_color": bg_color,
        }