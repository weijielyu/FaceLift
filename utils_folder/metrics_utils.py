# Copyright (C) 2025, FaceLift Research Group
# https://github.com/weijielyu/FaceLift
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact: wlyu3@ucmerced.edu

"""
Image quality metrics for 3D reconstruction evaluation.

This module provides implementations of common image quality metrics including
PSNR, LPIPS, and SSIM for evaluating reconstruction quality.
"""

from functools import lru_cache
from typing import List

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between ground truth and predicted images.
    
    Args:
        ground_truth: Ground truth images in range [0, 1]
        predicted: Predicted images in range [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@lru_cache(maxsize=None)
def get_lpips(device: torch.device) -> LPIPS:
    """
    Get cached LPIPS model for the specified device.
    
    Args:
        device: Target device for the LPIPS model
        
    Returns:
        LPIPS model instance
    """
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth_images: Float[Tensor, "batch channel height width"],
    predicted_images: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS) between images.
    
    Args:
        ground_truth_images: Ground truth images
        predicted_images: Predicted images
        
    Returns:
        LPIPS values for each image in the batch
    """
    lpips_model = get_lpips(predicted_images.device)
    
    # Process in batches to avoid memory issues
    processing_batch_size = 10
    lpips_scores: List[Tensor] = []
    
    for batch_start_idx in range(0, ground_truth_images.shape[0], processing_batch_size):
        batch_end_idx = batch_start_idx + processing_batch_size
        ground_truth_batch = ground_truth_images[batch_start_idx:batch_end_idx]
        predicted_batch = predicted_images[batch_start_idx:batch_end_idx]
        
        batch_lpips_score = lpips_model.forward(ground_truth_batch, predicted_batch, normalize=True)
        lpips_scores.append(batch_lpips_score)
    
    concatenated_scores = torch.cat(lpips_scores, dim=0)
    return concatenated_scores[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth_images: Float[Tensor, "batch channel height width"],
    predicted_images: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index (SSIM) between images.
    
    Args:
        ground_truth_images: Ground truth images
        predicted_images: Predicted images
        
    Returns:
        SSIM values for each image in the batch
    """
    ssim_scores = [
        structural_similarity(
            ground_truth_image.detach().cpu().numpy(),
            predicted_image.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for ground_truth_image, predicted_image in zip(ground_truth_images, predicted_images)
    ]
    return torch.tensor(ssim_scores, dtype=predicted_images.dtype, device=predicted_images.device)
