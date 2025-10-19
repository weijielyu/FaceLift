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

"""
Multi-view Diffusion Training Script

This script trains a multi-view diffusion model for 3D-aware image generation.
"""

import argparse
import copy
import logging
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import transformers
import diffusers
import accelerate
import wandb
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models import AutoencoderKL
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from easydict import EasyDict as edict
from einops import rearrange, repeat
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel

# Local imports
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.data.dataset import FixViewDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from utils_folder.metrics_utils import compute_psnr, compute_lpips, compute_ssim

logger = get_logger(__name__, log_level="INFO")


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Model and data parameters
    val_out_dir: str
    n_views: int
    img_wh: int
    
    # Pre-trained model paths
    pretrained_model_name_or_path: str
    pretrained_unet_path: Optional[str]
    revision: Optional[str]
    
    # Dataset configuration
    train_dataset: Dict
    validation_dataset: Dict
    
    # Training configuration
    output_dir: str
    checkpoint_prefix: str
    seed: Optional[int]
    train_batch_size: int
    validation_batch_size: int
    max_train_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    step_rules: Optional[str]
    lr_warmup_steps: int
    snr_gamma: Optional[float]
    use_8bit_adam: bool
    allow_tf32: bool
    use_ema: bool
    dataloader_num_workers: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: Optional[float]
    prediction_type: Optional[str]
    
    # Logging and validation
    logging_dir: str
    vis_dir: str
    mixed_precision: Optional[str]
    report_to: Optional[str]
    local_rank: int
    checkpointing_steps: int
    checkpoints_total_limit: Optional[int]
    resume_from_checkpoint: Optional[str]
    enable_xformers_memory_efficient_attention: bool
    validation_steps: int
    validation_sanity_check: bool
    tracker_project_name: str
    
    # Training specifics
    trainable_modules: Optional[list]
    use_classifier_free_guidance: bool
    condition_drop_rate: float
    scale_input_latents: bool
    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float
    drop_type: str
    
    # Wandb configuration
    wandb_exp_name: str
    wandb_group: str
    wandb_job_type: str

def noise_image_embeddings(
    image_embeds: torch.Tensor,
    noise_level: int,
    noise: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    image_normalizer: Optional[StableUnCLIPImageNormalizer] = None,
    image_noising_scheduler: Optional[DDPMScheduler] = None,
) -> torch.Tensor:
    """
    Add noise to image embeddings for stable unCLIP training.
    
    Args:
        image_embeds: Image embeddings to noise
        noise_level: Amount of noise to add
        noise: Optional pre-generated noise
        generator: Random number generator
        image_normalizer: Image normalizer for scaling
        image_noising_scheduler: Scheduler for adding noise
        
    Returns:
        Noised image embeddings with time embeddings appended
    """
    if noise is None:
        noise = randn_tensor(
            image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
        )
    noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

    image_embeds = image_normalizer.scale(image_embeds)
    image_embeds = image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)
    image_embeds = image_normalizer.unscale(image_embeds)

    noise_level = get_timestep_embedding(
        timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
    )

    # Cast to correct dtype
    noise_level = noise_level.to(image_embeds.dtype)
    image_embeds = torch.cat((image_embeds, noise_level), 1)
    return image_embeds


def compute_snr(timesteps: torch.Tensor, noise_scheduler: DDPMScheduler) -> torch.Tensor:
    """
    Compute SNR for min-SNR diffusion training.
    
    Args:
        timesteps: Timesteps tensor
        noise_scheduler: Noise scheduler
        
    Returns:
        SNR values for the given timesteps
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand tensors to match timesteps shape
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR
    snr = (alpha / sigma) ** 2
    return snr


def load_models(cfg: TrainingConfig):
    """
    Load all required models for training.
    
    Args:
        cfg: Training configuration
        
    Returns:
        Dictionary containing all loaded models
    """
    models = {}
    
    # Load CLIP models
    models['image_encoder'] = CLIPVisionModelWithProjection.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="image_encoder", revision=cfg.revision
    )
    models['feature_extractor'] = CLIPImageProcessor.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="feature_extractor", revision=cfg.revision
    )
    models['tokenizer'] = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision
    )
    models['text_encoder'] = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder='text_encoder', revision=cfg.revision
    )
    
    # Load diffusion models
    models['image_noising_scheduler'] = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="image_noising_scheduler"
    )
    models['image_normalizer'] = StableUnCLIPImageNormalizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="image_normalizer"
    )
    models['noise_scheduler'] = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler"
    )
    models['vae'] = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision
    )
    
    # Load UNet
    if cfg.pretrained_unet_path is None:
        models['unet'] = UNetMV2DConditionModel.from_pretrained_2d(
            cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs
        )
    else:
        logger.info(f'Loading pretrained UNet from {cfg.pretrained_unet_path}')
        models['unet'] = UNetMV2DConditionModel.from_pretrained_2d(
            cfg.pretrained_unet_path, subfolder="unet", revision=cfg.revision, **cfg.unet_from_pretrained_kwargs
        )
    
    # Set up EMA if needed
    if cfg.use_ema:
        models['ema_unet'] = EMAModel(models['unet'].parameters(), model_cls=UNetMV2DConditionModel, model_config=models['unet'].config)
    
    return models


def setup_model_training(models: Dict, cfg: TrainingConfig):
    """
    Configure models for training (freeze/unfreeze parameters, enable features).
    
    Args:
        models: Dictionary of loaded models
        cfg: Training configuration
    """
    # Freeze models that shouldn't be trained
    models['vae'].requires_grad_(False)
    models['image_encoder'].requires_grad_(False)
    models['image_normalizer'].requires_grad_(False)
    models['text_encoder'].requires_grad_(False)

    # Configure UNet training
    if cfg.trainable_modules is None:
        models['unet'].requires_grad_(True)
    else:
        models['unet'].requires_grad_(False)
        for name, module in models['unet'].named_modules():
            if name.endswith(tuple(cfg.trainable_modules)):
                for params in module.parameters():
                    params.requires_grad = True

    # Enable xformers if available
    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xFormers to at least 0.0.17."
                )
            models['unet'].enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory efficient attention")
        else:
            raise ValueError("xFormers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if cfg.gradient_checkpointing:
        models['unet'].enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


def setup_optimizer(models: Dict, cfg: TrainingConfig, accelerator: Accelerator):
    """
    Set up optimizer with different learning rates for different parameter groups.
    
    Args:
        models: Dictionary of loaded models
        cfg: Training configuration
        accelerator: Accelerator instance
        
    Returns:
        Configured optimizer and learning rate scheduler
    """
    # Scale learning rate if needed
    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )
    
    # Choose optimizer class
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
    else:
        optimizer_cls = torch.optim.AdamW

    # Group parameters by type for different learning rates
    params_base = []
    params_camera_embedding = []
    params_multiview_layers = []
    
    for name, param in models['unet'].named_parameters():
        if ('class_embedding' in name) or ('camera_embedding' in name):
            params_camera_embedding.append(param)
        elif ('attn_mv' in name) or ('norm_mv' in name):
            params_multiview_layers.append(param)
        else:
            params_base.append(param)
    
    # Create optimizer parameter groups
    optimizer_params = [{"params": params_base, "lr": cfg.learning_rate}]
    if len(params_camera_embedding) > 0:
        optimizer_params.append({
            "params": params_camera_embedding, 
            "lr": cfg.learning_rate * cfg.camera_embedding_lr_mult
        })
    if len(params_multiview_layers) > 0:
        optimizer_params.append({
            "params": params_multiview_layers, 
            "lr": cfg.learning_rate * cfg.camera_embedding_lr_mult
        })
    
    optimizer = optimizer_cls(
        optimizer_params,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        step_rules=cfg.step_rules,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )
    
    return optimizer, lr_scheduler


def setup_datasets_and_dataloaders(cfg: TrainingConfig):
    """
    Create datasets and dataloaders for training and validation.
    
    Args:
        cfg: Training configuration
        
    Returns:
        Training and validation dataloaders
    """
    # Create datasets
    train_dataset = FixViewDataset(cfg, split="train")
    validation_dataset = FixViewDataset(cfg, split="val")

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.train_batch_size, 
        shuffle=True, 
        num_workers=cfg.dataloader_num_workers,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=cfg.validation_batch_size, 
        shuffle=False, 
        num_workers=cfg.dataloader_num_workers
    )
    
    return train_dataloader, validation_dataloader


def setup_accelerator_and_logging(cfg: TrainingConfig):
    """
    Set up accelerator and logging configuration.
    
    Args:
        cfg: Training configuration
        
    Returns:
        Configured accelerator
    """
    # Override local_rank with environment variable if available
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank not in [-1, cfg.local_rank]:
        cfg.local_rank = env_local_rank

    # Set up directories
    logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)
    model_dir = os.path.join(cfg.checkpoint_prefix, cfg.output_dir)
    vis_dir = os.path.join(cfg.output_dir, cfg.vis_dir)
    
    # Create accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
    )
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed if provided
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Create directories
    if accelerator.is_main_process:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))
    
    return accelerator, model_dir, vis_dir

def process_training_batch(batch: Dict, cfg: TrainingConfig, models: Dict, accelerator: Accelerator, weight_dtype: torch.dtype, generator: torch.Generator):
    """
    Process a single training batch.
    
    Args:
        batch: Training batch data
        cfg: Training configuration
        models: Dictionary of loaded models
        accelerator: Accelerator instance
        weight_dtype: Weight data type for mixed precision
        generator: Random number generator
        
    Returns:
        MSE loss for the batch
    """
    # Extract and reshape input/output images
    input_images, target_images = batch['imgs_in'], batch['imgs_out']
    batch_size, num_views = input_images.shape[0], input_images.shape[1]
    
    # Reshape from (B, Nv, C, H, W) to (B*Nv, C, H, W)
    input_images = rearrange(input_images, "B Nv C H W -> (B Nv) C H W")
    target_images = rearrange(target_images, "B Nv C H W -> (B Nv) C H W")
    input_images, target_images = input_images.to(weight_dtype), target_images.to(weight_dtype)
    
    # Process prompt embeddings
    prompt_embeddings = batch['color_prompt_embeddings']
    prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
    prompt_embeddings = prompt_embeddings.to(weight_dtype)
    
    # Process input images for CLIP encoder
    feature_extractor = models['feature_extractor']
    input_images_processed = TF.resize(
        input_images, 
        (feature_extractor.crop_size['height'], feature_extractor.crop_size['width']), 
        interpolation=InterpolationMode.BICUBIC
    )
    
    # Normalize for CLIP (in float32 for precision)
    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(accelerator.device, dtype=torch.float32)
    clip_image_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(accelerator.device, dtype=torch.float32)
    input_images_processed = ((input_images_processed.float() - clip_image_mean) / clip_image_std).to(weight_dtype)
    
    # Get image embeddings
    image_embeddings = models['image_encoder'](input_images_processed).image_embeds
    
    # Add noise to image embeddings
    noise_level = torch.tensor([0], device=accelerator.device)
    image_embeddings = noise_image_embeddings(
        image_embeddings, noise_level, generator=generator, 
        image_normalizer=models['image_normalizer'], 
        image_noising_scheduler=models['image_noising_scheduler']
    ).to(weight_dtype)
    
    # Encode input images with VAE
    conditional_vae_embeddings = models['vae'].encode(input_images * 2.0 - 1.0).latent_dist.mode()
    if cfg.scale_input_latents:
        conditional_vae_embeddings *= models['vae'].config.scaling_factor
    
    # Encode target images with VAE and add noise
    latents = models['vae'].encode(target_images * 2.0 - 1.0).latent_dist.sample() * models['vae'].config.scaling_factor
    noise = torch.randn_like(latents)
    batch_latent_size = latents.shape[0]
    
    # Generate timesteps (same noise for different views of the same object)
    timesteps = torch.randint(0, models['noise_scheduler'].num_train_timesteps, (batch_latent_size // cfg.n_views,), device=latents.device)
    timesteps = repeat(timesteps, "b -> (b v)", v=cfg.n_views)
    timesteps = timesteps.long()
    
    noisy_latents = models['noise_scheduler'].add_noise(latents, noise, timesteps)
    
    # Apply conditioning dropout for classifier-free guidance
    if cfg.use_classifier_free_guidance and cfg.condition_drop_rate > 0.:
        if cfg.drop_type == 'drop_as_a_whole':
            # Drop conditioning for entire objects
            random_p = torch.rand(batch_size, device=latents.device, generator=generator)
            
            # Create masks for VAE conditioning
            image_mask = 1 - (
                (random_p >= cfg.condition_drop_rate).to(conditional_vae_embeddings.dtype)
                * (random_p < 3 * cfg.condition_drop_rate).to(conditional_vae_embeddings.dtype)
            )
            image_mask = image_mask.reshape(batch_size, 1, 1, 1, 1).repeat(1, num_views, 1, 1, 1)
            image_mask = rearrange(image_mask, "B Nv C H W -> (B Nv) C H W")
            conditional_vae_embeddings = image_mask * conditional_vae_embeddings
            
            # Create masks for CLIP conditioning
            clip_mask = 1 - ((random_p < 2 * cfg.condition_drop_rate).to(image_embeddings.dtype))
            clip_mask = clip_mask.reshape(batch_size, 1, 1).repeat(1, num_views, 1)
            clip_mask = rearrange(clip_mask, "B Nv C -> (B Nv) C")
            image_embeddings = clip_mask * image_embeddings
            
        elif cfg.drop_type == 'drop_independent':
            # Drop conditioning independently for each view
            random_p = torch.rand(batch_latent_size, device=latents.device, generator=generator)
            
            # VAE conditioning mask
            image_mask = 1 - (
                (random_p >= cfg.condition_drop_rate).to(conditional_vae_embeddings.dtype)
                * (random_p < 3 * cfg.condition_drop_rate).to(conditional_vae_embeddings.dtype)
            )
            image_mask = image_mask.reshape(batch_latent_size, 1, 1, 1)
            conditional_vae_embeddings = image_mask * conditional_vae_embeddings
            
            # CLIP conditioning mask
            clip_mask = 1 - ((random_p < 2 * cfg.condition_drop_rate).to(image_embeddings.dtype))
            clip_mask = clip_mask.reshape(batch_latent_size, 1, 1)
            image_embeddings = clip_mask * image_embeddings
    
    # Prepare input for UNet
    latent_model_input = torch.cat([noisy_latents, conditional_vae_embeddings], dim=1)
    
    # Forward pass through UNet
    model_output = models['unet'](
        latent_model_input,
        timesteps,
        encoder_hidden_states=prompt_embeddings,
        class_labels=image_embeddings,
        vis_max_min=False
    )
    
    model_pred = model_output.sample
    
    # Compute target based on prediction type
    if models['noise_scheduler'].config.prediction_type == "epsilon":
        target = noise
    elif models['noise_scheduler'].config.prediction_type == "v_prediction":
        target = models['noise_scheduler'].get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {models['noise_scheduler'].config.prediction_type}")
    
    # Compute MSE loss
    if cfg.snr_gamma is None:
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean").to(weight_dtype)
    else:
        # Apply SNR weighting
        snr = compute_snr(timesteps, models['noise_scheduler'])
        mse_loss_weights = (
            torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        mse_loss = loss.mean().to(weight_dtype)
    
    return mse_loss


def log_validation(dataloader, vae, feature_extractor, image_encoder, image_normalizer, image_noising_scheduler, tokenizer, text_encoder, 
                   unet, cfg:TrainingConfig, accelerator, weight_dtype, global_step, name, val_out_dir):
    """Run validation and log results."""
    logger.info(f"Running {name} ... ")

    pipeline = StableUnCLIPImg2ImgPipeline(
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        image_normalizer=image_normalizer,
        image_noising_scheduler=image_noising_scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=accelerator.unwrap_model(unet),
        scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
        **cfg.pipe_kwargs
    )

    pipeline.set_progress_bar_config(disable=True)

    if cfg.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()    

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)
    
    images_cond, images_gt, images_pred = [], [], defaultdict(list)

    val_metrics = {}
    for guidance_scale in cfg.validation_guidance_scales:
        val_metrics[f"psnr-{guidance_scale:.1f}"] = []
        val_metrics[f"lpips-{guidance_scale:.1f}"] = []
        val_metrics[f"ssim-{guidance_scale:.1f}"] = []

    for i, batch in enumerate(dataloader):
        # For fast validation, only run the first batch
        if i > 0 and cfg.validation_sanity_check:
            break
            
        input_images, target_images = batch['imgs_in'], batch['imgs_out']
        batch_size, num_views = input_images.shape[0], input_images.shape[1]

        images_cond.append(input_images[:, 0, :, :, :])
        
        input_images = rearrange(input_images, "B Nv C H W -> (B Nv) C H W")
        target_images = rearrange(target_images, "B Nv C H W -> (B Nv) C H W")
        images_gt.append(target_images)

        prompt_embeddings = batch['color_prompt_embeddings']
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
        prompt_embeddings = prompt_embeddings.to(weight_dtype)
        
        with torch.autocast("cuda"):
            # Save input and ground truth images for first batch
            if i == 0:
                os.makedirs(os.path.join(val_out_dir, f"global_step_{global_step:04d}"), exist_ok=True)
                input_images_batch = rearrange(input_images, "(B Nv) C H W -> B Nv C H W", B=batch_size, Nv=num_views)
                target_images_batch = rearrange(target_images, "(B Nv) C H W -> B Nv C H W", B=batch_size, Nv=num_views)
                
                for b in range(batch_size):
                    os.makedirs(os.path.join(val_out_dir, f"global_step_{global_step:04d}", f"{b:04d}"), exist_ok=True)
                    
                    input_image = rearrange(input_images_batch[b], "Nv C H W -> H (Nv W) C")
                    input_image = (input_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(input_image[..., :3]).save(
                        os.path.join(val_out_dir, f"global_step_{global_step:04d}", f"{b:04d}", "input.png")
                    )

                    gt_image = rearrange(target_images_batch[b], "Nv C H W -> H (Nv W) C")
                    gt_image = (gt_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(gt_image[..., :3]).save(
                        os.path.join(val_out_dir, f"global_step_{global_step:04d}", f"{b:04d}", "gt.png")
                    )
                    
            # Generate predictions for each guidance scale
            for guidance_scale in cfg.validation_guidance_scales:
                out = pipeline(
                    input_images,
                    None,
                    prompt_embeds=prompt_embeddings,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    output_type='pt',
                    num_images_per_prompt=1,
                    **cfg.pipe_validation_kwargs
                ).images
                
                target_images = target_images.to(out.device)
                images_pred[f"{name}-sample_cfg{guidance_scale:.1f}"].append(out)

                # Compute metrics
                val_metrics[f"psnr-{guidance_scale:.1f}"].append(compute_psnr(target_images, out))
                val_metrics[f"lpips-{guidance_scale:.1f}"].append(compute_lpips(target_images, out))
                val_metrics[f"ssim-{guidance_scale:.1f}"].append(compute_ssim(target_images, out))
                
                # Save predictions for first batch
                if i == 0:
                    out_batch = rearrange(out, "(B Nv) C H W -> B Nv C H W", B=batch_size, Nv=num_views)
                    target_images_batch = target_images_batch.to(out_batch.device)
                    
                    for b in range(batch_size):
                        pred_image = rearrange(out_batch[b], "Nv C H W -> H (Nv W) C")
                        pred_image = (pred_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(pred_image[..., :3]).save(
                            os.path.join(val_out_dir, f"global_step_{global_step:04d}", f"{b:04d}", f"pred_gs_{guidance_scale:.1f}.png")
                        )

                        # Compute and save metrics for this sample
                        psnr = compute_psnr(target_images_batch[b], out_batch[b]).mean().item()
                        lpips = compute_lpips(target_images_batch[b], out_batch[b]).mean().item()
                        ssim = compute_ssim(target_images_batch[b], out_batch[b]).mean().item()

                        with open(os.path.join(val_out_dir, f"global_step_{global_step:04d}", f"{b:04d}", f"metrics_gs_{guidance_scale:.1f}.txt"), "w") as f:
                            metrics_txt = f"psnr: {psnr}\nlpips: {lpips}\nssim: {ssim}\n"
                            f.write(metrics_txt)

    # Calculate and log overall metrics
    metrics_txt = ""
    for guidance_scale in cfg.validation_guidance_scales:
        psnr = torch.stack(val_metrics[f"psnr-{guidance_scale:.1f}"]).mean().item()
        lpips = torch.stack(val_metrics[f"lpips-{guidance_scale:.1f}"]).mean().item()
        ssim = torch.stack(val_metrics[f"ssim-{guidance_scale:.1f}"]).mean().item()

        wandb_log_dict = {
            f"val/psnr-guidance_scale-{guidance_scale:.1f}": psnr,
            f"val/lpips-guidance_scale-{guidance_scale:.1f}": lpips,
            f"val/ssim-guidance_scale-{guidance_scale:.1f}": ssim
        }

        if accelerator.is_main_process:
            wandb.log(wandb_log_dict, step=global_step)
            metrics_txt += f"guidance_scale: {guidance_scale:.1f}\n psnr: {psnr}\nlpips: {lpips}\nssim: {ssim}\n"

    val_out_dir = os.path.join(val_out_dir, f"global_step_{global_step:04d}")

    if accelerator.is_main_process:
        with open(os.path.join(val_out_dir, f"{name}-metrics.txt"), "w") as f:
            f.write(metrics_txt)
    
    torch.cuda.empty_cache()


def main(cfg: TrainingConfig):
    """Main training function."""
    # Set up accelerator and logging
    accelerator, model_dir, vis_dir = setup_accelerator_and_logging(cfg)
    
    # Load all models
    models = load_models(cfg)
    
    # Configure models for training
    setup_model_training(models, cfg)
    
    # Set up accelerator hooks for model saving/loading
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models_to_save, weights, output_dir):
            if cfg.use_ema:
                models['ema_unet'].save_pretrained(os.path.join(cfg.checkpoint_prefix, output_dir, "unet_ema"))
            
            for i, model in enumerate(models_to_save):
                model.save_pretrained(os.path.join(cfg.checkpoint_prefix, output_dir, "unet"))
                weights.pop()

        def load_model_hook(models_to_load, input_dir):
            if cfg.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(cfg.checkpoint_prefix, input_dir, "unet_ema"), UNetMV2DConditionModel
                )
                models['ema_unet'].load_state_dict(load_model.state_dict())
                models['ema_unet'].to(accelerator.device)
                del load_model

            for i in range(len(models_to_load)):
                model = models_to_load.pop()
                load_model = UNetMV2DConditionModel.from_pretrained(
                    os.path.join(cfg.checkpoint_prefix, input_dir), subfolder="unet"
                )
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Set up optimizer and learning rate scheduler
    optimizer, lr_scheduler = setup_optimizer(models, cfg, accelerator)
    
    # Set up datasets and dataloaders
    train_dataloader, validation_dataloader = setup_datasets_and_dataloaders(cfg)
    
    # Prepare everything with accelerator
    models['unet'], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        models['unet'], optimizer, train_dataloader, lr_scheduler
    )

    if cfg.use_ema:
        models['ema_unet'].to(accelerator.device)
    
    # Set up weight dtype for mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        cfg.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        cfg.mixed_precision = accelerator.mixed_precision

    # Move models to device with correct dtype
    models['image_encoder'].to(accelerator.device, dtype=weight_dtype)
    models['image_normalizer'].to(accelerator.device)
    models['text_encoder'].to(accelerator.device, dtype=weight_dtype)
    models['vae'].to(accelerator.device, dtype=weight_dtype)

    # Calculate training parameters
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = {}
        accelerator.init_trackers(project_name=cfg.tracker_project_name, config=tracker_config, \
            init_kwargs={"wandb": {
                "name": cfg.wandb_exp_name, 
                "job_type": cfg.wandb_job_type,
                "group": cfg.wandb_group}})

    # Set up training
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if specified
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            if os.path.exists(os.path.join(model_dir, "checkpoint")):
                path = "checkpoint"
            else:
                dirs = os.listdir(model_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run.")
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(model_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # Main training loop
    for epoch in range(first_epoch, num_train_epochs):
        models['unet'].train()
        train_mse_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(models['unet']):
                # Process training batch (simplified - no pose losses since random_input_view is False)
                mse_loss = process_training_batch(
                    batch, cfg, models, accelerator, weight_dtype, generator
                )
                
                # Gather losses across processes
                avg_mse_loss = accelerator.gather(mse_loss.repeat(cfg.train_batch_size)).mean()
                train_mse_loss += avg_mse_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate (only MSE loss since no pose losses)
                accelerator.backward(mse_loss)

                if accelerator.sync_gradients and cfg.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(models['unet'].parameters(), cfg.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress and log metrics
            if accelerator.sync_gradients:
                if cfg.use_ema:
                    models['ema_unet'].step(models['unet'])
                
                progress_bar.update(1)
                global_step += 1

                # Log training metrics
                accelerator.log({"train_mse_loss": train_mse_loss}, step=global_step)
                train_mse_loss = 0.0

                # Save checkpoint
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # Clean up old checkpoints if limit is set
                        if cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(model_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(model_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        save_path = os.path.join(model_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Run validation
                if (global_step % cfg.validation_steps == 0 or 
                    (cfg.validation_sanity_check and global_step == 1)): # Make sure val
                    if accelerator.is_main_process:
                        if cfg.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters
                            models['ema_unet'].store(models['unet'].parameters())
                            models['ema_unet'].copy_to(models['unet'].parameters())
                        
                        torch.cuda.empty_cache()
                        log_validation(
                            validation_dataloader,
                            models['vae'],
                            models['feature_extractor'],
                            models['image_encoder'],
                            models['image_normalizer'],
                            models['image_noising_scheduler'],
                            models['tokenizer'],
                            models['text_encoder'],
                            models['unet'],
                            cfg,
                            accelerator,
                            weight_dtype,
                            global_step,
                            'validation',
                            cfg.val_out_dir
                        )           

                        if cfg.use_ema:
                            # Switch back to the original UNet parameters
                            models['ema_unet'].restore(models['unet'].parameters())

            # Update progress bar with current loss and learning rate
            logs = {"step_loss": mse_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Create final pipeline and save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(models['unet'])
        if cfg.use_ema:
            models['ema_unet'].copy_to(unet.parameters())
            
        pipeline = StableUnCLIPImg2ImgPipeline(
            image_encoder=models['image_encoder'],
            feature_extractor=models['feature_extractor'],
            image_normalizer=models['image_normalizer'],
            image_noising_scheduler=models['image_noising_scheduler'],
            tokenizer=models['tokenizer'],
            text_encoder=models['text_encoder'],
            vae=models['vae'], 
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
            **cfg.pipe_kwargs
        )            
        
        pipeline_dir = os.path.join(model_dir, "pipeckpts")
        os.makedirs(pipeline_dir, exist_ok=True)
        pipeline.save_pretrained(pipeline_dir)

    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    
    main(cfg)