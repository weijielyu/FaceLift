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

# gslrm.py is under the Adobe Research License. Copyright 2025 Adobe Inc.

"""
GSLRM (Gaussian Splatting Large Reconstruction Model)

This module implements a transformer-based model for generating 3D Gaussian splats
from multi-view images. The model uses a combination of image tokenization,
transformer processing, and Gaussian splatting for novel view synthesis.

Classes:
    Renderer: Handles Gaussian splatting rendering operations
    GaussiansUpsampler: Converts transformer tokens to Gaussian parameters
    LossComputer: Computes various loss functions for training
    TransformTarget: Handles target image transformations (cropping, etc.)
    GSLRM: Main model class that orchestrates the entire pipeline
"""

import copy
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange
from einops.layers.torch import Rearrange
from PIL import Image

# Local imports
from .utils_losses import PerceptualLoss, SsimLoss
from .gaussians_renderer import (
    GaussianModel,
    RGB2SH,
    deferred_gaussian_render,
    imageseq2video,
    render_opencv_cam,
    render_turntable,
)
from .transform_data import SplitData, TransformInput, TransformTarget
from .utils_transformer import (
    TransformerBlock,
    _init_weights,
)

class Renderer(nn.Module):
    """
    Handles Gaussian splatting rendering operations.
    
    Supports both deferred rendering (for training with gradients) and
    standard rendering (for inference).
    """
    
    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        
        # Initialize Gaussian model with scaling modifier
        self.scaling_modifier = config.model.gaussians.get("scaling_modifier", None)
        self.gaussians_model = GaussianModel(
            config.model.gaussians.sh_degree, 
            self.scaling_modifier
        )
        
        print(f"Renderer initialized with scaling_modifier: {self.scaling_modifier}")

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        xyz: torch.Tensor,           # [b, n_gaussians, 3]
        features: torch.Tensor,      # [b, n_gaussians, (sh_degree+1)^2, 3]
        scaling: torch.Tensor,       # [b, n_gaussians, 3]
        rotation: torch.Tensor,      # [b, n_gaussians, 4]
        opacity: torch.Tensor,       # [b, n_gaussians, 1]
        height: int,
        width: int,
        C2W: torch.Tensor,          # [b, v, 4, 4]
        fxfycxcy: torch.Tensor,     # [b, v, 4]
        deferred: bool = True,
    ) -> torch.Tensor:              # [b, v, 3, height, width]
        """
        Render Gaussian splats to images.
        
        Args:
            xyz: Gaussian positions
            features: Gaussian spherical harmonic features
            scaling: Gaussian scaling parameters
            rotation: Gaussian rotation quaternions
            opacity: Gaussian opacity values
            height: Output image height
            width: Output image width
            C2W: Camera-to-world transformation matrices
            fxfycxcy: Camera intrinsics (fx, fy, cx, cy)
            deferred: Whether to use deferred rendering (maintains gradients)
            
        Returns:
            Rendered images
        """
        if deferred:
            return deferred_gaussian_render(
                xyz, features, scaling, rotation, opacity,
                height, width, C2W, fxfycxcy, self.scaling_modifier
            )
        else:
            return self._render_sequential(
                xyz, features, scaling, rotation, opacity,
                height, width, C2W, fxfycxcy
            )
    
    def _render_sequential(
        self, xyz, features, scaling, rotation, opacity,
        height, width, C2W, fxfycxcy
    ) -> torch.Tensor:
        """Sequential rendering without gradient support (used for inference)."""
        b, v = C2W.size(0), C2W.size(1)
        renderings = torch.zeros(
            b, v, 3, height, width, dtype=torch.float32, device=xyz.device
        )
        
        for i in range(b):
            pc = self.gaussians_model.set_data(
                xyz[i], features[i], scaling[i], rotation[i], opacity[i]
            )
            for j in range(v):
                renderings[i, j] = render_opencv_cam(
                    pc, height, width, C2W[i, j], fxfycxcy[i, j]
                )["render"]
                
        return renderings


class GaussiansUpsampler(nn.Module):
    """
    Converts transformer output tokens to Gaussian splatting parameters.
    
    Takes high-dimensional transformer features and projects them to the
    concatenated Gaussian parameter space (xyz + features + scaling + rotation + opacity).
    """
    
    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        
        # Layer normalization before final projection
        self.layernorm = nn.LayerNorm(config.model.transformer.d, bias=False)
        
        # Calculate output dimension for Gaussian parameters
        sh_dim = (config.model.gaussians.sh_degree + 1) ** 2 * 3
        gaussian_param_dim = 3 + sh_dim + 3 + 4 + 1  # xyz + features + scaling + rotation + opacity
        
        # Check upsampling factor (currently only supports 1x)
        upsample_factor = config.model.gaussians.upsampler.upsample_factor
        if upsample_factor > 1:
            raise NotImplementedError("GaussiansUpsampler only supports upsample_factor=1")
        
        # Linear projection to Gaussian parameters
        self.linear = nn.Linear(
            config.model.transformer.d,
            gaussian_param_dim,
            bias=False,
        )

    def forward(
        self, 
        gaussians: torch.Tensor,  # [b, n_gaussians, d]
        images: torch.Tensor      # [b, l, d] (unused but kept for interface compatibility)
    ) -> torch.Tensor:           # [b, n_gaussians, gaussian_param_dim]
        """
        Convert transformer tokens to Gaussian parameters.
        
        Args:
            gaussians: Transformer output tokens for Gaussians
            images: Image tokens (unused but kept for compatibility)
            
        Returns:
            Raw Gaussian parameters (before conversion to final format)
        """
        upsample_factor = self.config.model.gaussians.upsampler.upsample_factor
        if upsample_factor > 1:
            raise NotImplementedError("GaussiansUpsampler only supports upsample_factor=1")
        
        return self.linear(self.layernorm(gaussians))

    def to_gs(self, gaussians: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Convert raw Gaussian parameters to final format.
        
        Args:
            gaussians: Raw Gaussian parameters [b, n_gaussians, param_dim]
            
        Returns:
            Tuple of (xyz, features, scaling, rotation, opacity)
        """
        sh_dim = (self.config.model.gaussians.sh_degree + 1) ** 2 * 3
        
        # Split concatenated parameters
        xyz, features, scaling, rotation, opacity = gaussians.split(
            [3, sh_dim, 3, 4, 1], dim=2
        )
        
        # Reshape features to proper spherical harmonics format
        features = features.reshape(
            features.size(0),
            features.size(1),
            (self.config.model.gaussians.sh_degree + 1) ** 2,
            3,
        )
        
        # Apply activation functions with specific biases
        # Scaling: exp(x - 2.3) clamped to prevent too large values
        scaling = (scaling - 2.3).clamp(max=-1.20)
        
        # Opacity: sigmoid(x - 2.0) to get values in [0, 1]
        opacity = opacity - 2.0
        
        return xyz, features, scaling, rotation, opacity


class LossComputer(nn.Module):
    """
    Computes various loss functions for training the GSLRM model.
    
    Supports multiple loss types:
    - L2 (MSE) loss
    - LPIPS perceptual loss
    - Custom perceptual loss
    - SSIM loss
    - Pixel alignment loss
    - Point distance regularization loss
    """
    
    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        
        # Initialize loss modules based on config
        self._init_loss_modules()
    
    def _init_loss_modules(self):
        """Initialize the various loss computation modules."""
        # LPIPS loss
        if self.config.training.losses.lpips_loss_weight > 0.0:
            self.lpips_loss_module = lpips.LPIPS(net="vgg")
            self.lpips_loss_module.eval()
            # Freeze LPIPS parameters
            for param in self.lpips_loss_module.parameters():
                param.requires_grad = False

        # Perceptual loss
        if self.config.training.losses.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = PerceptualLoss()
            self.perceptual_loss_module.eval()
            # Freeze perceptual loss parameters
            for param in self.perceptual_loss_module.parameters():
                param.requires_grad = False

        # SSIM loss
        if self.config.training.losses.ssim_loss_weight > 0.0:
            self.ssim_loss_module = SsimLoss()
            self.ssim_loss_module.eval()
            # Freeze SSIM parameters
            for param in self.ssim_loss_module.parameters():
                param.requires_grad = False

    def forward(
        self,
        rendering: torch.Tensor,        # [b, v, 3, h, w]
        target: torch.Tensor,           # [b, v, 3, h, w]
        img_aligned_xyz: torch.Tensor,  # [b, v, 3, h, w]
        input: edict,
        result_softpa: Optional[edict] = None,
        create_visual: bool = False,
    ) -> edict:
        """
        Compute all losses between rendered and target images.
        
        Args:
            rendering: Rendered images in range [0, 1]
            target: Target images in range [0, 1]
            img_aligned_xyz: Image-aligned 3D positions
            input: Input data containing ray information
            result_softpa: Additional results (unused)
            create_visual: Whether to create visualization images
            
        Returns:
            Dictionary containing all loss values and metrics
        """
        b, v, _, h, w = rendering.size()
        rendering_flat = rendering.reshape(b * v, -1, h, w)
        target_flat = target.reshape(b * v, -1, h, w)
        
        # Handle alpha channel if present
        mask = None
        if target_flat.size(1) == 4:
            target_flat, mask = target_flat.split([3, 1], dim=1)
        
        # Compute individual losses
        losses = self._compute_all_losses(
            rendering_flat, target_flat, img_aligned_xyz, input, mask, b, v, h, w
        )
        
        # Compute total weighted loss
        total_loss = self._compute_total_loss(losses)
        
        # Create visualization if requested
        visual = self._create_visual(rendering_flat, target_flat, v) if create_visual else None
        
        # Compile loss metrics
        return self._compile_loss_metrics(losses, total_loss, visual)
    
    def _compute_all_losses(self, rendering, target, img_aligned_xyz, input, mask, b, v, h, w):
        """Compute all individual loss components."""
        losses = {}
        
        # L2 (MSE) loss
        losses['l2'] = self._compute_l2_loss(rendering, target)
        losses['psnr'] = -10.0 * torch.log10(losses['l2'])
        
        # LPIPS loss
        losses['lpips'] = self._compute_lpips_loss(rendering, target)
        
        # Perceptual loss
        losses['perceptual'] = self._compute_perceptual_loss(rendering, target)
        
        # SSIM loss
        losses['ssim'] = self._compute_ssim_loss(rendering, target)
        
        # Pixel alignment loss
        losses['pixelalign'] = self._compute_pixelalign_loss(
            img_aligned_xyz, input, mask, b, v, h, w
        )
        
        # Point distance loss
        losses['pointsdist'] = self._compute_pointsdist_loss(
            img_aligned_xyz, input, b, v, h, w
        )
        
        return losses
    
    def _compute_l2_loss(self, rendering, target):
        """Compute L2 (MSE) loss."""
        if self.config.training.losses.l2_loss_weight > 0.0:
            return F.mse_loss(rendering, target)
        return torch.tensor(1e-8, device=rendering.device)
    
    def _compute_lpips_loss(self, rendering, target):
        """Compute LPIPS perceptual loss."""
        if self.config.training.losses.lpips_loss_weight > 0.0:
            # LPIPS expects inputs in range [-1, 1]
            return self.lpips_loss_module(
                rendering * 2.0 - 1.0, target * 2.0 - 1.0
            ).mean()
        return torch.tensor(0.0, device=rendering.device)
    
    def _compute_perceptual_loss(self, rendering, target):
        """Compute custom perceptual loss."""
        if self.config.training.losses.perceptual_loss_weight > 0.0:
            return self.perceptual_loss_module(rendering, target)
        return torch.tensor(0.0, device=rendering.device)
    
    def _compute_ssim_loss(self, rendering, target):
        """Compute SSIM loss."""
        if self.config.training.losses.ssim_loss_weight > 0.0:
            return self.ssim_loss_module(rendering, target)
        return torch.tensor(0.0, device=rendering.device)
    
    def _compute_pixelalign_loss(self, img_aligned_xyz, input, mask, b, v, h, w):
        """Compute pixel alignment loss."""
        if self.config.training.losses.pixelalign_loss_weight > 0.0:
            # Compute orthogonal component to ray direction
            xyz_vec = img_aligned_xyz - input.ray_o
            ortho_vec = (
                xyz_vec
                - torch.sum(xyz_vec.detach() * input.ray_d, dim=2, keepdim=True)
                * input.ray_d
            )
            
            # Apply mask if enabled
            if self.config.training.losses.get("masked_pixelalign_loss", False):
                assert mask is not None, "mask is None but masked_pixelalign_loss is enabled"
                mask_reshaped = mask.view(b, v, 1, h, w)
                ortho_vec = ortho_vec * mask_reshaped
            
            return torch.mean(ortho_vec.norm(dim=2, p=2))
        
        return torch.tensor(0.0, device=img_aligned_xyz.device)
    
    def _compute_pointsdist_loss(self, img_aligned_xyz, input, b, v, h, w):
        """Compute point distance regularization loss."""
        if self.config.training.losses.pointsdist_loss_weight > 0.0:
            # Target mean distance (distance from origin to ray origin)
            target_mean_dist = torch.norm(input.ray_o, dim=2, p=2, keepdim=True)
            target_std_dist = 0.5
            
            # Predicted distance
            pred_dist = (img_aligned_xyz - input.ray_o).norm(dim=2, p=2, keepdim=True)
            
            # Normalize to target distribution
            pred_dist_detach = pred_dist.detach()
            pred_mean = pred_dist_detach.mean(dim=(2, 3, 4), keepdim=True)
            pred_std = pred_dist_detach.std(dim=(2, 3, 4), keepdim=True)
            
            target_dist = (pred_dist_detach - pred_mean) / (pred_std + 1e-8) * target_std_dist + target_mean_dist
            
            return torch.mean((pred_dist - target_dist) ** 2)
        
        return torch.tensor(0.0, device=img_aligned_xyz.device)
    
    def _compute_total_loss(self, losses):
        """Compute weighted sum of all losses."""
        weights = self.config.training.losses
        return (
            weights.l2_loss_weight * losses['l2']
            + weights.lpips_loss_weight * losses['lpips']
            + weights.perceptual_loss_weight * losses['perceptual']
            + weights.ssim_loss_weight * losses['ssim']
            + weights.pixelalign_loss_weight * losses['pixelalign']
            + weights.pointsdist_loss_weight * losses['pointsdist']
        )
    
    def _create_visual(self, rendering, target, v):
        """Create visualization by concatenating target and rendering."""
        visual = torch.cat((target, rendering), dim=3).detach().cpu()  # [b*v, c, h, w*2]
        visual = rearrange(visual, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2)
        return (visual.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    
    def _compile_loss_metrics(self, losses, total_loss, visual):
        """Compile all loss metrics into a dictionary."""
        l2_loss = losses['l2']
        
        return edict(
            loss=total_loss,
            l2_loss=l2_loss,
            psnr=losses['psnr'],
            lpips_loss=losses['lpips'],
            perceptual_loss=losses['perceptual'],
            ssim_loss=losses['ssim'],
            pixelalign_loss=losses['pixelalign'],
            pointsdist_loss=losses['pointsdist'],
            visual=visual,
            # Normalized losses for logging
            norm_perceptual_loss=losses['perceptual'] / l2_loss,
            norm_lpips_loss=losses['lpips'] / l2_loss,
            norm_ssim_loss=losses['ssim'] / l2_loss,
            norm_pixelalign_loss=losses['pixelalign'] / l2_loss,
            norm_pointsdist_loss=losses['pointsdist'] / l2_loss,
        )


class GSLRM(nn.Module):
    """
    Gaussian Splatting Large Reconstruction Model.
    
    A transformer-based model that generates 3D Gaussian splats from multi-view images.
    The model processes input images through tokenization, transformer layers, and
    generates Gaussian parameters for novel view synthesis.
    
    Architecture:
    1. Image tokenization with patch-based encoding
    2. Transformer processing with Gaussian positional embeddings
    3. Gaussian parameter generation and upsampling
    4. Rendering and loss computation
    """
    
    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        
        # Initialize data processing modules
        self._init_data_processors(config)
        
        # Initialize core model components
        self._init_tokenizer(config)
        self._init_positional_embeddings(config)
        self._init_transformer(config)
        self._init_gaussian_modules(config)
        self._init_rendering_modules(config)
        
        # Initialize training state management
        self._init_training_state(config)
    
    def _init_data_processors(self, config: edict) -> None:
        """Initialize data splitting and transformation modules."""
        self.data_splitter = SplitData(config)
        self.input_transformer = TransformInput(config)
        self.target_transformer = TransformTarget(config)
    
    def _init_tokenizer(self, config: edict) -> None:
        """Initialize image tokenization pipeline."""
        patch_size = config.model.image_tokenizer.patch_size
        input_channels = config.model.image_tokenizer.in_channels
        hidden_dim = config.model.transformer.d
        
        self.patch_embedder = nn.Sequential(
            Rearrange(
                "batch views channels (height patch_h) (width patch_w) -> (batch views) (height width) (patch_h patch_w channels)",
                patch_h=patch_size,
                patch_w=patch_size,
            ),
            nn.Linear(
                input_channels * (patch_size ** 2),
                hidden_dim,
                bias=False,
            ),
        )
        self.patch_embedder.apply(_init_weights)
    
    def _init_positional_embeddings(self, config: edict) -> None:
        """Initialize positional embeddings for reference/source markers and Gaussians."""
        hidden_dim = config.model.transformer.d
        
        # Optional reference/source view markers
        self.view_type_embeddings = None
        if config.model.get("add_refsrc_marker", False):
            self.view_type_embeddings = nn.Parameter(
                torch.randn(2, hidden_dim)  # [reference_marker, source_marker]
            )
            nn.init.trunc_normal_(self.view_type_embeddings, std=0.02)
        
        # Gaussian positional embeddings
        num_gaussians = config.model.gaussians.n_gaussians
        self.gaussian_position_embeddings = nn.Parameter(
            torch.randn(num_gaussians, hidden_dim)
        )
        nn.init.trunc_normal_(self.gaussian_position_embeddings, std=0.02)
    
    def _init_transformer(self, config: edict) -> None:
        """Initialize transformer architecture."""
        hidden_dim = config.model.transformer.d
        head_dim = config.model.transformer.d_head
        num_layers = config.model.transformer.n_layer
        
        self.input_layer_norm = nn.LayerNorm(hidden_dim, bias=False)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, head_dim)
            for _ in range(num_layers)
        ])
        self.transformer_layers.apply(_init_weights)
    
    def _init_gaussian_modules(self, config: edict) -> None:
        """Initialize Gaussian parameter generation modules."""
        hidden_dim = config.model.transformer.d
        patch_size = config.model.image_tokenizer.patch_size
        sh_degree = config.model.gaussians.sh_degree
        
        # Calculate output dimension for pixel-aligned Gaussians
        # Components: xyz(3) + sh_features((sh_degree+1)^2*3) + scaling(3) + rotation(4) + opacity(1)
        gaussian_param_dim = 3 + (sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
        
        # Gaussian upsampler for transformer tokens
        self.gaussian_upsampler = GaussiansUpsampler(config)
        self.gaussian_upsampler.apply(_init_weights)
        
        # Pixel-aligned Gaussian decoder
        self.pixel_gaussian_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim, bias=False),
            nn.Linear(
                hidden_dim,
                (patch_size ** 2) * gaussian_param_dim,
                bias=False,
            ),
        )
        self.pixel_gaussian_decoder.apply(_init_weights)
    
    def _init_rendering_modules(self, config: edict) -> None:
        """Initialize rendering and loss computation modules."""
        self.gaussian_renderer = Renderer(config)
        self.loss_calculator = LossComputer(config)
    
    def _init_training_state(self, config: edict) -> None:
        """Initialize training state management variables."""
        self.training_step = None
        self.training_start_step = None
        self.training_max_step = None
        self.original_config = copy.deepcopy(config)

    def set_training_step(self, current_step: int, start_step: int, max_step: int) -> None:
        """
        Update training step and dynamically adjust configuration based on training phase.
        
        Args:
            current_step: Current training step
            start_step: Starting step of training
            max_step: Maximum training steps
        """
        self.training_step = current_step
        self.training_start_step = start_step
        self.training_max_step = max_step

        # Determine if config modification is needed based on warmup settings
        needs_config_modification = self._should_modify_config_for_warmup(current_step)
        
        if needs_config_modification:
            # Always use original config as base for modifications
            self.config = copy.deepcopy(self.original_config)
            self._apply_warmup_modifications(current_step)
        else:
            # Restore original configuration
            self.config = self.original_config

        # Update loss calculator with current config
        self.loss_calculator.config = self.config
    
    def _should_modify_config_for_warmup(self, current_step: int) -> bool:
        """Check if configuration should be modified for warmup phases."""
        pointsdist_warmup = (
            self.config.training.losses.get("warmup_pointsdist", False) 
            and current_step < 1000
        )
        l2_warmup = (
            self.config.training.schedule.get("l2_warmup_steps", 0) > 0 
            and current_step < self.config.training.schedule.l2_warmup_steps
        )
        return pointsdist_warmup or l2_warmup
    
    def _apply_warmup_modifications(self, current_step: int) -> None:
        """Apply configuration modifications for warmup phases."""
        # Point distance warmup phase
        if (self.config.training.losses.get("warmup_pointsdist", False) 
            and current_step < 1000):
            self.config.training.losses.l2_loss_weight = 0.0
            self.config.training.losses.perceptual_loss_weight = 0.0
            self.config.training.losses.pointsdist_loss_weight = 0.1
            self.config.model.clip_xyz = False  # Disable xyz clipping during warmup

        # L2 loss warmup phase
        if (self.config.training.schedule.get("l2_warmup_steps", 0) > 0 
            and current_step < self.config.training.schedule.l2_warmup_steps):
            self.config.training.losses.perceptual_loss_weight = 0.0
            self.config.training.losses.lpips_loss_weight = 0.0
    
    def set_current_step(self, current_step: int, start_step: int, max_step: int) -> None:
        """Backward compatibility wrapper for set_training_step."""
        self.set_training_step(current_step, start_step, max_step)

    def train(self, mode: bool = True) -> None:
        """
        Override train method to keep frozen modules in eval mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
        """
        super().train(mode)
        # Keep loss calculator in eval mode to prevent training of frozen components
        if self.loss_calculator is not None:
            self.loss_calculator.eval()

    def get_parameter_overview(self) -> edict:
        """
        Get overview of trainable parameters in each module.
        
        Returns:
            Dictionary containing parameter counts for each major component
        """
        def count_trainable_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return edict(
            patch_embedder=count_trainable_params(self.patch_embedder),
            gaussian_position_embeddings=self.gaussian_position_embeddings.data.numel(),
            transformer_total=(
                count_trainable_params(self.transformer_layers) +
                count_trainable_params(self.input_layer_norm)
            ),
            gaussian_upsampler=count_trainable_params(self.gaussian_upsampler),
            pixel_gaussian_decoder=count_trainable_params(self.pixel_gaussian_decoder),
        )
    
    def get_overview(self) -> edict:
        """Backward compatibility wrapper for get_parameter_overview."""
        return self.get_parameter_overview()

    def _create_transformer_layer_runner(self, start_layer: int, end_layer: int):
        """
        Create a function to run a subset of transformer layers.
        
        Args:
            start_layer: Starting layer index
            end_layer: Ending layer index (exclusive)
            
        Returns:
            Function that processes tokens through specified layers
        """
        def run_transformer_layers(token_sequence: torch.Tensor) -> torch.Tensor:
            for layer_idx in range(start_layer, min(end_layer, len(self.transformer_layers))):
                token_sequence = self.transformer_layers[layer_idx](token_sequence)
            return token_sequence
        return run_transformer_layers
    
    def _create_posed_images_with_plucker(self, input_data: edict) -> torch.Tensor:
        """
        Create posed images by concatenating RGB with Plucker coordinates.
        
        Args:
            input_data: Input data containing images and ray information
            
        Returns:
            Posed images with Plucker coordinates [batch, views, channels, height, width]
        """
        # Normalize RGB to [-1, 1] range
        normalized_rgb = input_data.image[:, :, :3, :, :] * 2.0 - 1.0
        
        if self.config.model.get("use_custom_plucker", False):
            # Custom Plucker: RGB + ray_direction + nearest_points
            ray_origin_dot_direction = torch.sum(
                -input_data.ray_o * input_data.ray_d, dim=2, keepdim=True
            )
            nearest_points = input_data.ray_o + ray_origin_dot_direction * input_data.ray_d
            
            return torch.cat([
                normalized_rgb,
                input_data.ray_d,
                nearest_points,
            ], dim=2)
            
        elif self.config.model.get("use_aug_plucker", False):
            # Augmented Plucker: RGB + cross_product + ray_direction + nearest_points
            ray_cross_product = torch.cross(input_data.ray_o, input_data.ray_d, dim=2)
            ray_origin_dot_direction = torch.sum(
                -input_data.ray_o * input_data.ray_d, dim=2, keepdim=True
            )
            nearest_points = input_data.ray_o + ray_origin_dot_direction * input_data.ray_d
            
            return torch.cat([
                normalized_rgb,
                ray_cross_product,
                input_data.ray_d,
                nearest_points,
            ], dim=2)
            
        else:
            # Standard Plucker: RGB + cross_product + ray_direction
            ray_cross_product = torch.cross(input_data.ray_o, input_data.ray_d, dim=2)
            
            return torch.cat([
                normalized_rgb,
                ray_cross_product,
                input_data.ray_d,
            ], dim=2)
    
    def _add_view_type_embeddings(
        self, 
        image_tokens: torch.Tensor, 
        batch_size: int, 
        num_views: int, 
        num_patches: int, 
        hidden_dim: int
    ) -> torch.Tensor:
        """Add view type embeddings to distinguish reference vs source views."""
        image_tokens = image_tokens.reshape(batch_size, num_views, num_patches, hidden_dim)
        
        # Create view type markers: first view is reference, rest are source
        view_markers = [self.view_type_embeddings[0]] + [
            self.view_type_embeddings[1] for _ in range(1, num_views)
        ]
        view_markers = torch.stack(view_markers, dim=0)[None, :, None, :]  # [1, views, 1, hidden_dim]
        
        # Add markers to image tokens
        image_tokens = image_tokens + view_markers
        return image_tokens.reshape(batch_size, num_views * num_patches, hidden_dim)
    
    def _process_through_transformer(
        self, 
        gaussian_tokens: torch.Tensor, 
        image_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Process combined tokens through transformer with gradient checkpointing."""
        # Combine Gaussian and image tokens
        combined_tokens = torch.cat((gaussian_tokens, image_tokens), dim=1)
        combined_tokens = self.input_layer_norm(combined_tokens)
        
        # Process through transformer layers with gradient checkpointing
        checkpoint_interval = self.config.training.runtime.grad_checkpoint_every
        num_layers = len(self.transformer_layers)
        
        for start_idx in range(0, num_layers, checkpoint_interval):
            end_idx = start_idx + checkpoint_interval
            layer_runner = self._create_transformer_layer_runner(start_idx, end_idx)
            
            combined_tokens = torch.utils.checkpoint.checkpoint(
                layer_runner,
                combined_tokens,
                use_reentrant=False,
            )
        
        return combined_tokens
    
    def _apply_hard_pixel_alignment(
        self, 
        pixel_aligned_xyz: torch.Tensor, 
        input_data: edict
    ) -> torch.Tensor:
        """Apply hard pixel alignment to ensure Gaussians align with ray directions."""
        depth_bias = self.config.model.get("depth_preact_bias", 0.0)
        
        # Apply sigmoid activation to depth values
        depth_values = torch.sigmoid(
            pixel_aligned_xyz.mean(dim=2, keepdim=True) + depth_bias
        )
        
        # Apply different depth computation strategies
        if (self.config.model.get("use_aug_plucker", False) or 
            self.config.model.get("use_custom_plucker", False)):
            # For Plucker coordinates: use dot product offset
            ray_origin_dot_direction = torch.sum(
                -input_data.ray_o * input_data.ray_d, dim=2, keepdim=True
            )
            depth_values = (2.0 * depth_values - 1.0) * 1.8 + ray_origin_dot_direction
            
        elif (self.config.model.get("depth_min", -1.0) > 0.0 and 
              self.config.model.get("depth_max", -1.0) > 0.0):
            # Use explicit depth range
            depth_min = self.config.model.depth_min
            depth_max = self.config.model.depth_max
            depth_values = depth_values * (depth_max - depth_min) + depth_min
            
        elif self.config.model.get("depth_reference_origin", False):
            # Reference from ray origin norm
            ray_origin_norm = input_data.ray_o.norm(dim=2, p=2, keepdim=True)
            depth_values = (2.0 * depth_values - 1.0) * 1.8 + ray_origin_norm
            
        else:
            # Default depth computation
            depth_values = (2.0 * depth_values - 1.0) * 1.5 + 2.7
        
        # Compute final 3D positions along rays
        aligned_positions = input_data.ray_o + depth_values * input_data.ray_d
        
        # Apply coordinate clipping if enabled (only during training)
        if (self.config.model.get("clip_xyz", False) and 
            not self.config.inference):
            aligned_positions = aligned_positions.clamp(-1.0, 1.0)
        
        return aligned_positions
    
    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path: str, 
        config: edict, 
        map_location: Optional[str] = None
    ) -> 'GSLRM':
        """
        Load model from checkpoint with automatic legacy name translation.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            config: Model configuration
            map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')
            
        Returns:
            Loaded GSLRM model
        """
        # Create model instance
        model = cls(config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Extract state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict with automatic translation
        model.load_state_dict(state_dict)
        
        print(f"Successfully loaded model from {checkpoint_path}")
        return model
    
    def _create_gaussian_models_and_stats(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor, 
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        num_pixel_aligned: int,
        num_views: int,
        height: int,
        width: int,
        patch_size: int
    ) -> Tuple[List, torch.Tensor, List[float]]:
        """
        Create Gaussian models for each batch item and compute usage statistics.
        
        Returns:
            Tuple of (gaussian_models, pixel_aligned_positions, usage_statistics)
        """
        gaussian_models = []
        pixel_aligned_positions_list = []
        usage_statistics = []
        
        batch_size = xyz.size(0)
        opacity_threshold = 0.05
        
        for batch_idx in range(batch_size):
            # Create fresh Gaussian model for this batch item
            self.gaussian_renderer.gaussians_model.empty()
            gaussian_model = copy.deepcopy(self.gaussian_renderer.gaussians_model)
            
            # Set Gaussian data
            gaussian_model = gaussian_model.set_data(
                xyz[batch_idx].detach().float(),
                features[batch_idx].detach().float(),
                scaling[batch_idx].detach().float(),
                rotation[batch_idx].detach().float(),
                opacity[batch_idx].detach().float(),
            )
            gaussian_models.append(gaussian_model)

            # Compute usage statistics (fraction of Gaussians above opacity threshold)
            opacity_mask = gaussian_model.get_opacity > opacity_threshold
            usage_ratio = opacity_mask.sum() / opacity_mask.numel()
            if torch.is_tensor(usage_ratio):
                usage_ratio = usage_ratio.item()
            usage_statistics.append(usage_ratio)

            # Extract pixel-aligned positions and reshape
            pixel_xyz = gaussian_model.get_xyz[-num_pixel_aligned:, :]
            pixel_xyz_reshaped = rearrange(
                pixel_xyz,
                "(views height width patch_h patch_w) coords -> views coords (height patch_h) (width patch_w)",
                views=num_views,
                height=height // patch_size,
                width=width // patch_size,
                patch_h=patch_size,
                patch_w=patch_size,
            )
            pixel_aligned_positions_list.append(pixel_xyz_reshaped)
        
        # Stack pixel-aligned positions
        pixel_aligned_positions = torch.stack(pixel_aligned_positions_list, dim=0)
        
        return gaussian_models, pixel_aligned_positions, usage_statistics

    def forward(
        self, 
        batch_data: edict, 
        create_visual: bool = False, 
        split_data: bool = True
    ) -> edict:
        """
        Forward pass of the GSLRM model.
        
        Args:
            batch_data: Input batch containing:
                - image: Multi-view images [batch, views, channels, height, width]
                - fxfycxcy: Camera intrinsics [batch, views, 4]
                - c2w: Camera-to-world matrices [batch, views, 4, 4]
            create_visual: Whether to create visualization outputs
            split_data: Whether to split input/target data
            
        Returns:
            Dictionary containing model outputs including Gaussians, renders, and losses
        """
        with torch.no_grad():
            target_data = None
            if split_data:
                batch_data, target_data = self.data_splitter(
                    batch_data, self.config.training.dataset.target_has_input
                )
                target_data = self.target_transformer(target_data)

            input_data = self.input_transformer(batch_data)

            # Prepare posed images with Plucker coordinates [batch, views, channels, height, width]
            posed_images = self._create_posed_images_with_plucker(input_data)

        # Process images through tokenization and transformer
        batch_size, num_views, channels, height, width = posed_images.size()
        
        # Tokenize images into patches
        image_patch_tokens = self.patch_embedder(posed_images)  # [batch*views, num_patches, hidden_dim]
        _, num_patches, hidden_dim = image_patch_tokens.size()
        image_patch_tokens = image_patch_tokens.reshape(
            batch_size, num_views * num_patches, hidden_dim
        )  # [batch, views*patches, hidden_dim]

        # Add view type embeddings if enabled (reference vs source views)
        if self.view_type_embeddings is not None:
            image_patch_tokens = self._add_view_type_embeddings(
                image_patch_tokens, batch_size, num_views, num_patches, hidden_dim
            )

        # Prepare Gaussian tokens with positional embeddings
        gaussian_tokens = self.gaussian_position_embeddings.expand(batch_size, -1, -1)

        # Process through transformer with gradient checkpointing
        combined_tokens = self._process_through_transformer(
            gaussian_tokens, image_patch_tokens
        )

        # Split back into Gaussian and image tokens
        num_gaussians = self.config.model.gaussians.n_gaussians
        gaussian_tokens, image_patch_tokens = combined_tokens.split(
            [num_gaussians, num_views * num_patches], dim=1
        )
        
        # Generate Gaussian parameters from transformer outputs
        gaussian_params = self.gaussian_upsampler(gaussian_tokens, image_patch_tokens)

        # Generate pixel-aligned Gaussians from image tokens
        pixel_aligned_gaussian_params = self.pixel_gaussian_decoder(image_patch_tokens)
        
        # Calculate Gaussian parameter dimensions
        sh_degree = self.config.model.gaussians.sh_degree
        gaussian_param_dim = 3 + (sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
        
        pixel_aligned_gaussian_params = pixel_aligned_gaussian_params.reshape(
            batch_size, -1, gaussian_param_dim
        )  # [batch, views*pixels, gaussian_params]
        num_pixel_aligned_gaussians = pixel_aligned_gaussian_params.size(1)

        # Combine all Gaussian parameters
        all_gaussian_params = torch.cat((gaussian_params, pixel_aligned_gaussian_params), dim=1)
        
        # Convert to final Gaussian format
        xyz, features, scaling, rotation, opacity = self.gaussian_upsampler.to_gs(all_gaussian_params)

        # Extract pixel-aligned Gaussian positions for processing
        pixel_aligned_xyz = xyz[:, -num_pixel_aligned_gaussians:, :]
        patch_size = self.config.model.image_tokenizer.patch_size
        
        pixel_aligned_xyz = rearrange(
            pixel_aligned_xyz,
            "batch (views height width patch_h patch_w) coords -> batch views coords (height patch_h) (width patch_w)",
            views=num_views,
            height=height // patch_size,
            width=width // patch_size,
            patch_h=patch_size,
            patch_w=patch_size,
        )

        # Apply hard pixel alignment if enabled
        if self.config.model.hard_pixelalign:
            pixel_aligned_xyz = self._apply_hard_pixel_alignment(
                pixel_aligned_xyz, input_data
            )
            
            # Reshape back to flat format and update xyz
            pixel_aligned_xyz_flat = rearrange(
                pixel_aligned_xyz,
                "batch views coords (height patch_h) (width patch_w) -> batch (views height width patch_h patch_w) coords",
                patch_h=patch_size,
                patch_w=patch_size,
            )
            
            # Replace pixel-aligned Gaussians in the full xyz tensor
            xyz = torch.cat(
                (xyz[:, :-num_pixel_aligned_gaussians, :], pixel_aligned_xyz_flat), 
                dim=1
            )

        # Create Gaussian splatting result structure
        gaussian_splat_result = edict(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )

        # Perform rendering and loss computation if target data is available
        loss_metrics = None
        rendered_images = None
        
        if target_data is not None:
            target_height, target_width = target_data.image.size(3), target_data.image.size(4)
            
            # Render images using Gaussian splatting
            rendered_images = self.gaussian_renderer(
                xyz, features, scaling, rotation, opacity,
                target_height, target_width,
                C2W=target_data.c2w,
                fxfycxcy=target_data.fxfycxcy,
            )
            
            # Compute losses if rendered and target have matching dimensions
            if rendered_images.shape[1] == target_data.image.shape[1]:
                loss_metrics = self.loss_calculator(
                    rendered_images,
                    target_data.image,
                    pixel_aligned_xyz,
                    input_data,
                    create_visual=create_visual,
                    result_softpa=gaussian_splat_result,
                )

        # Create Gaussian models for each batch item and compute usage statistics
        gaussian_models, pixel_aligned_positions, usage_statistics = self._create_gaussian_models_and_stats(
            xyz, features, scaling, rotation, opacity, 
            num_pixel_aligned_gaussians, num_views, height, width, patch_size
        )

        # Add usage statistics to loss metrics for logging
        if loss_metrics is not None:
            loss_metrics.gaussians_usage = torch.tensor(
                np.mean(np.array(usage_statistics))
            ).float()

        # Compile final results
        return edict(
            input=input_data,
            target=target_data,
            gaussians=gaussian_models,
            pixelalign_xyz=pixel_aligned_positions,
            img_tokens=image_patch_tokens,
            loss_metrics=loss_metrics,
            render=rendered_images,
        )

    @torch.no_grad()
    def save_visualization_outputs(
        self, 
        output_directory: str, 
        model_results: edict, 
        batch_data: edict, 
        save_all_items: bool = False
    ) -> None:
        """
        Save visualization outputs including rendered images and Gaussian models.
        
        Args:
            output_directory: Directory to save outputs
            model_results: Results from model forward pass
            batch_data: Original batch data
            save_all_items: Whether to save all batch items or just the first
        """
        os.makedirs(output_directory, exist_ok=True)

        input_data, target_data = model_results.input, model_results.target
        
        # Save supervision visualization if available
        if (model_results.loss_metrics is not None and 
            model_results.loss_metrics.visual is not None):
            
            batch_uids = [
                target_data.index[b, 0, -1].item() 
                for b in range(target_data.index.size(0))
            ]

            uid_range = f"{batch_uids[0]:08}_{batch_uids[-1]:08}"
            
            # Save supervision comparison image
            Image.fromarray(model_results.loss_metrics.visual).save(
                os.path.join(output_directory, f"supervision_{uid_range}.jpg")
            )
            
            # Save UIDs for reference
            with open(os.path.join(output_directory, "uids.txt"), "w") as f:
                uid_string = "_".join([f"{uid:08}" for uid in batch_uids])
                f.write(uid_string)

            # Save input images
            input_visualization = rearrange(
                input_data.image, "batch views channels height width -> (batch height) (views width) channels"
            )
            input_visualization = (
                (input_visualization.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            )
            Image.fromarray(input_visualization[..., :3]).save(
                os.path.join(output_directory, f"input_{uid_range}.jpg")
            )

        # Process each batch item individually
        batch_size = input_data.image.size(0)
        for batch_idx in range(batch_size):
            item_uid = input_data.index[batch_idx, 0, -1].item()

            # Render turntable visualization
            turntable_image = render_turntable(model_results.gaussians[batch_idx])
            Image.fromarray(turntable_image).save(
                os.path.join(output_directory, f"turntable_{item_uid}.jpg")
            )

            # Save individual input images during inference
            if self.config.inference:
                individual_input = rearrange(
                    input_data.image[batch_idx], "views channels height width -> height (views width) channels"
                )
                individual_input = (
                    (individual_input.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                )
                Image.fromarray(individual_input[..., :3]).save(
                    os.path.join(output_directory, f"input_{item_uid}.jpg")
                )

            # Extract image dimensions and create opacity/depth visualizations
            _, num_views, _, img_height, img_width = input_data.image.size()
            patch_size = self.config.model.image_tokenizer.patch_size
            
            # Get opacity values for pixel-aligned Gaussians
            gaussian_opacity = model_results.gaussians[batch_idx].get_opacity
            pixel_opacity = gaussian_opacity[-num_views * img_height * img_width:]
            
            # Reshape opacity to image format
            opacity_visualization = rearrange(
                pixel_opacity,
                "(views height width patch_h patch_w) channels -> (height patch_h) (views width patch_w) channels",
                views=num_views,
                height=img_height // patch_size,
                width=img_width // patch_size,
                patch_h=patch_size,
                patch_w=patch_size,
            ).squeeze(-1).cpu().numpy()
            opacity_visualization = (opacity_visualization * 255.0).clip(0.0, 255.0).astype(np.uint8)

            # Get 3D positions and compute depth visualization
            gaussian_positions = model_results.gaussians[batch_idx].get_xyz
            pixel_positions = gaussian_positions[-num_views * img_height * img_width:]
            
            # Reshape positions to image format
            pixel_positions_reshaped = rearrange(
                pixel_positions,
                "(views height width patch_h patch_w) coords -> views coords (height patch_h) (width patch_w)",
                views=num_views,
                height=img_height // patch_size,
                width=img_width // patch_size,
                patch_h=patch_size,
                patch_w=patch_size,
            )
            
            # Compute distances from ray origins
            ray_distances = (pixel_positions_reshaped - input_data.ray_o[batch_idx]).norm(dim=1, p=2)
            distance_visualization = rearrange(ray_distances, "views height width -> height (views width)")
            distance_visualization = distance_visualization.cpu().numpy()
            
            # Normalize distances for visualization
            dist_min, dist_max = distance_visualization.min(), distance_visualization.max()
            distance_visualization = (distance_visualization - dist_min) / (dist_max - dist_min)
            distance_visualization = (distance_visualization * 255.0).clip(0.0, 255.0).astype(np.uint8)

            # Combine opacity and depth visualizations
            combined_visualization = np.concatenate([opacity_visualization, distance_visualization], axis=0)
            Image.fromarray(combined_visualization).save(
                os.path.join(output_directory, f"aligned_gs_opacity_depth_{item_uid}.jpg")
            )

            # Save unfiltered Gaussian model for small images during early training
            if (self.config.model.image_tokenizer.image_size <= 256 and 
                self.training_step is not None and self.training_step <= 5000):
                model_results.gaussians[batch_idx].save_ply(
                    os.path.join(output_directory, f"gaussians_{item_uid}_unfiltered.ply")
                )

            # Save filtered Gaussian model
            camera_origins = None  # Could use input_data.ray_o[batch_idx, :, :, 0, 0] if needed
            default_crop_box = [-0.91, 0.91, -0.91, 0.91, -0.91, 0.91]
            
            model_results.gaussians[batch_idx].apply_all_filters(
                opacity_thres=0.02,
                crop_bbx=default_crop_box,
                cam_origins=camera_origins,
                nearfar_percent=(0.0001, 1.0),
            ).save_ply(os.path.join(output_directory, f"gaussians_{item_uid}.ply"))
            
            print(f"Saved visualization for UID: {item_uid}")

            # Break after first item unless saving all
            if not save_all_items:
                break
    
    @torch.no_grad()
    def save_visuals(self, out_dir: str, result: edict, batch: edict, save_all: bool = False) -> None:
        """Backward compatibility wrapper for save_visualization_outputs."""
        self.save_visualization_outputs(out_dir, result, batch, save_all)

    @torch.no_grad()
    def save_evaluation_results(
        self, 
        output_directory: str, 
        model_results: edict, 
        batch_data: edict, 
        dataset
    ) -> None:
        """Save comprehensive evaluation results including metrics, visualizations, and 3D models."""
        from .utils_metrics import compute_psnr, compute_lpips, compute_ssim

        os.makedirs(output_directory, exist_ok=True)
        input_data, target_data = model_results.input, model_results.target
        
        for batch_idx in range(input_data.image.size(0)):
            item_uid = input_data.index[batch_idx, 0, -1].item()
            item_output_dir = os.path.join(output_directory, f"{item_uid:08d}")
            os.makedirs(item_output_dir, exist_ok=True)
            
            # Save input image
            input_image = rearrange(
                input_data.image[batch_idx], "views channels height width -> height (views width) channels"
            )
            input_image = (input_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(input_image[..., :3]).save(os.path.join(item_output_dir, "input.png"))

            # Save ground truth vs prediction comparison
            comparison_image = torch.stack((target_data.image[batch_idx], model_results.render[batch_idx]), dim=0)
            num_views = comparison_image.size(1)
            if num_views > 10:
                comparison_image = comparison_image[:, ::num_views // 10, :, :, :]
            comparison_image = rearrange(
                comparison_image, "comparison_type views channels height width -> (comparison_type height) (views width) channels"
            )
            comparison_image = (comparison_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(comparison_image).save(os.path.join(item_output_dir, "gt_vs_pred.png"))
            
            # Compute and save metrics
            per_view_psnr = compute_psnr(target_data.image[batch_idx], model_results.render[batch_idx])
            per_view_lpips = compute_lpips(target_data.image[batch_idx], model_results.render[batch_idx])
            per_view_ssim = compute_ssim(target_data.image[batch_idx], model_results.render[batch_idx])

            # Save per-view metrics
            view_ids = target_data.index[batch_idx, :, 0].cpu().numpy()
            with open(os.path.join(item_output_dir, "perview_metrics.txt"), "w") as f:
                for i in range(per_view_psnr.size(0)):
                    f.write(
                        f"view {view_ids[i]:0>6}, psnr: {per_view_psnr[i].item():.4f}, "
                        f"lpips: {per_view_lpips[i].item():.4f}, ssim: {per_view_ssim[i].item():.4f}\n"
                    )

            # Save average metrics
            avg_psnr = per_view_psnr.mean().item()
            avg_lpips = per_view_lpips.mean().item()
            avg_ssim = per_view_ssim.mean().item()
            
            with open(os.path.join(item_output_dir, "metrics.txt"), "w") as f:
                f.write(f"psnr: {avg_psnr:.4f}\nlpips: {avg_lpips:.4f}\nssim: {avg_ssim:.4f}\n")
            
            print(f"UID {item_uid}: PSNR={avg_psnr:.4f}, LPIPS={avg_lpips:.4f}, SSIM={avg_ssim:.4f}")
            
            # Save Gaussian model
            crop_box = None
            if self.config.model.get("clip_xyz", False):
                if self.config.model.get("half_bbx_size", None) is not None:
                    half_size = self.config.model.half_bbx_size
                    crop_box = [-half_size, half_size, -half_size, half_size, -half_size, half_size]
                else:
                    crop_box = [-0.91, 0.91, -0.91, 0.91, -0.91, 0.91]
            
            model_results.gaussians[batch_idx].apply_all_filters(
                opacity_thres=0.02, crop_bbx=crop_box, cam_origins=None, nearfar_percent=(0.0001, 1.0)
            ).save_ply(os.path.join(item_output_dir, "gaussians.ply"))
            
            # Create turntable visualization
            num_turntable_views = 150
            render_resolution = input_image.shape[0]
            
            turntable_frames = render_turntable(
                model_results.gaussians[batch_idx], rendering_resolution=render_resolution, num_views=num_turntable_views
            )
            turntable_frames = rearrange(
                turntable_frames, "height (views width) channels -> views height width channels", views=num_turntable_views
            )
            turntable_frames = np.ascontiguousarray(turntable_frames)
            
            # Save basic turntable video
            imageseq2video(turntable_frames, os.path.join(item_output_dir, "turntable.mp4"), fps=30)
            
            # Save description and preview if available
            try:
                description = dataset.get_description(item_uid)["prompt"]
                if len(description) > 0:
                    with open(os.path.join(item_output_dir, "description.txt"), "w") as f:
                        f.write(description)
                    
                    # Create preview image (subsample to 10 views)
                    preview_frames = turntable_frames[::num_turntable_views // 10]
                    preview_image = rearrange(preview_frames, "views height width channels -> height (views width) channels")
                    Image.fromarray(preview_image).save(os.path.join(item_output_dir, "turntable_preview.png"))
            except (AttributeError, KeyError):
                pass
            
            # Create turntable with input overlay
            border_width = 2
            target_width = render_resolution
            target_height = int(input_image.shape[0] / input_image.shape[1] * target_width)
            
            resized_input = cv2.resize(
                input_image, (target_width - border_width * 2, target_height - border_width * 2), interpolation=cv2.INTER_AREA
            )
            bordered_input = np.pad(
                resized_input, ((border_width, border_width), (border_width, border_width), (0, 0)), 
                mode="constant", constant_values=200
            )
            
            input_sequence = np.tile(bordered_input[None], (turntable_frames.shape[0], 1, 1, 1))
            combined_frames = np.concatenate((turntable_frames, input_sequence), axis=1)
            
            imageseq2video(combined_frames, os.path.join(item_output_dir, "turntable_with_input.mp4"), fps=30)
    
    @torch.no_grad()
    def save_evaluations(self, out_dir: str, result: edict, batch: edict, dataset) -> None:
        """Backward compatibility wrapper for save_evaluation_results."""
        self.save_evaluation_results(out_dir, result, batch, dataset)

    @torch.no_grad()
    def save_validation_results(
        self, 
        output_directory: str, 
        model_results: edict, 
        batch_data: edict, 
        dataset, 
        save_visualizations: bool = False
    ) -> Dict[str, float]:
        """Save validation results and compute aggregated metrics."""
        from .utils_metrics import compute_psnr, compute_lpips, compute_ssim

        os.makedirs(output_directory, exist_ok=True)
        input_data, target_data = model_results.input, model_results.target
        validation_metrics = {"psnr": [], "lpips": [], "ssim": []}

        for batch_idx in range(input_data.image.size(0)):
            item_uid = input_data.index[batch_idx, 0, -1].item()
            should_save_visuals = (batch_idx == 0) and save_visualizations
            
            # Compute metrics (RGB only)
            target_image = target_data.image[batch_idx][:, :3, ...]
            per_view_psnr = compute_psnr(target_image, model_results.render[batch_idx])
            per_view_lpips = compute_lpips(target_image, model_results.render[batch_idx])
            per_view_ssim = compute_ssim(target_image, model_results.render[batch_idx])
            
            avg_psnr = per_view_psnr.mean().item()
            avg_lpips = per_view_lpips.mean().item()
            avg_ssim = per_view_ssim.mean().item()
            
            validation_metrics["psnr"].append(avg_psnr)
            validation_metrics["lpips"].append(avg_lpips)
            validation_metrics["ssim"].append(avg_ssim)
            
            # Save visualizations only for first item if requested
            if should_save_visuals:
                item_output_dir = os.path.join(output_directory, f"{item_uid:08d}")
                os.makedirs(item_output_dir, exist_ok=True)
                
                # Save input image
                input_image = rearrange(
                    input_data.image[batch_idx][:, :3, ...], "views channels height width -> height (views width) channels"
                )
                input_image = (input_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                Image.fromarray(input_image).save(os.path.join(item_output_dir, "input.png"))
                
                # Save ground truth vs prediction comparison
                comparison_image = torch.stack((target_image, model_results.render[batch_idx]), dim=0)
                num_views = comparison_image.size(1)
                if num_views > 10:
                    comparison_image = comparison_image[:, ::num_views // 10, :, :, :]
                comparison_image = rearrange(
                    comparison_image, "comparison_type views channels height width -> (comparison_type height) (views width) channels"
                )
                comparison_image = (comparison_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                Image.fromarray(comparison_image).save(os.path.join(item_output_dir, "gt_vs_pred.png"))
                
                # Save per-view metrics
                view_ids = target_data.index[batch_idx, :, 0].cpu().numpy()
                with open(os.path.join(item_output_dir, "perview_metrics.txt"), "w") as f:
                    for i in range(per_view_psnr.size(0)):
                        f.write(
                            f"view {view_ids[i]:0>6}, psnr: {per_view_psnr[i].item():.4f}, "
                            f"lpips: {per_view_lpips[i].item():.4f}, ssim: {per_view_ssim[i].item():.4f}\n"
                        )
                
                # Save averaged metrics
                with open(os.path.join(item_output_dir, "metrics.txt"), "w") as f:
                    f.write(f"psnr: {avg_psnr:.4f}\nlpips: {avg_lpips:.4f}\nssim: {avg_ssim:.4f}\n")
                
                print(f"Validation UID {item_uid}: PSNR={avg_psnr:.4f}, LPIPS={avg_lpips:.4f}, SSIM={avg_ssim:.4f}")
                
                # Save Gaussian model
                crop_box = None
                if self.config.model.get("clip_xyz", False):
                    if self.config.model.get("half_bbx_size", None) is not None:
                        half_size = self.config.model.half_bbx_size
                        crop_box = [-half_size, half_size, -half_size, half_size, -half_size, half_size]
                    else:
                        crop_box = [-0.91, 0.91, -0.91, 0.91, -0.91, 0.91]
                
                model_results.gaussians[batch_idx].apply_all_filters(
                    opacity_thres=0.02, crop_bbx=crop_box, cam_origins=None, nearfar_percent=(0.0001, 1.0)
                ).save_ply(os.path.join(item_output_dir, "gaussians.ply"))
                
                # Create turntable visualization
                num_turntable_views = 150
                render_resolution = input_image.shape[0]
                
                turntable_frames = render_turntable(
                    model_results.gaussians[batch_idx], rendering_resolution=render_resolution, num_views=num_turntable_views
                )
                turntable_frames = rearrange(
                    turntable_frames, "height (views width) channels -> views height width channels", views=num_turntable_views
                )
                turntable_frames = np.ascontiguousarray(turntable_frames)
                
                imageseq2video(turntable_frames, os.path.join(item_output_dir, "turntable.mp4"), fps=30)
                
                # Create turntable with input overlay
                border_width = 2
                target_width = render_resolution
                target_height = int(input_image.shape[0] / input_image.shape[1] * target_width)
                
                resized_input = cv2.resize(
                    input_image, (target_width - border_width * 2, target_height - border_width * 2), interpolation=cv2.INTER_AREA
                )
                bordered_input = np.pad(
                    resized_input, ((border_width, border_width), (border_width, border_width), (0, 0)), 
                    mode="constant", constant_values=200
                )
                
                input_sequence = np.tile(bordered_input[None], (turntable_frames.shape[0], 1, 1, 1))
                combined_frames = np.concatenate((turntable_frames, input_sequence), axis=1)
                
                imageseq2video(combined_frames, os.path.join(item_output_dir, "turntable_with_input.mp4"), fps=30)
        
        # Return averaged metrics
        return {
            "psnr": torch.tensor(validation_metrics["psnr"]).mean().item(),
            "lpips": torch.tensor(validation_metrics["lpips"]).mean().item(),
            "ssim": torch.tensor(validation_metrics["ssim"]).mean().item(),
        }
    
    @torch.no_grad()
    def save_validations(
        self, 
        out_dir: str, 
        result: edict, 
        batch: edict, 
        dataset, 
        save_img: bool = False
    ) -> Dict[str, float]:
        """Backward compatibility wrapper for save_validation_results."""
        return self.save_validation_results(out_dir, result, batch, dataset, save_img)