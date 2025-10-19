# Copyright (C) 2025, FaceLift Research Group
# https://github.com/weijielyu/FaceLift
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact: wlyu3@ucmerced.edu

import inspect
import os
import traceback

import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from rich import print
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

def print_rank0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def configure_optimizer(model, weight_decay, learning_rate, betas):
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    
    # Separate params: 2D+ gets weight decay, 1D doesn't (biases, norms)
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    
    print_rank0(f"Decay params: {len(decay_params)}, No decay: {len(nodecay_params)}")
    
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    
    # Use fused AdamW if available and on CUDA
    use_fused = ("fused" in inspect.signature(torch.optim.AdamW).parameters and 
                 next(model.parameters()).is_cuda)
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    
    return optimizer, param_dict, all_param_dict


def configure_lr_scheduler(optimizer, total_train_steps, warm_up_steps, scheduler_type="cosine"):
    schedulers = {
        "linear": lambda: get_linear_schedule_with_warmup(
            optimizer, warm_up_steps, total_train_steps),
        "cosine": lambda: get_cosine_schedule_with_warmup(
            optimizer, warm_up_steps, total_train_steps),
        "constant": lambda: get_constant_schedule_with_warmup(
            optimizer, warm_up_steps),
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return schedulers[scheduler_type]()


def checkpoint_job(out_dir, model, optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step):
    """Save model and optimizer states."""
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model.module
    
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"ckpt_{fwdbwd_pass_step:016}.pt")
    
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": fwdbwd_pass_step,
        "param_update_step": param_update_step,
    }, ckpt_path)
    
    print(f"Saved checkpoint to {os.path.abspath(ckpt_path)}")


def find_checkpoints(out_dir):
    """Find all checkpoints in output directory."""
    ckpts = [x for x in os.listdir(out_dir) if x.startswith("ckpt_") and x.endswith(".pt")]
    ckpts = sorted(ckpts, key=lambda x: x[5:-3])  # Sort by step number
    return [os.path.join(out_dir, ckpt) for ckpt in ckpts]


def resume_job(load_path, model, optimizer, lr_scheduler, job_overview, warmup,
               reset_lr=False, reset_weight_decay=False, reset_training_state=False):
    """Resume training from checkpoint. Returns (optimizer, lr_scheduler, fwdbwd_step, param_step)."""
    
    # Find checkpoint paths
    if os.path.isdir(load_path):
        ckpt_paths = find_checkpoints(load_path)
        if not ckpt_paths:
            return optimizer, lr_scheduler, 0, 0
    else:
        if not load_path.endswith(".pt"):
            return optimizer, lr_scheduler, 0, 0
        ckpt_paths = [load_path]
    
    # Load checkpoint (try in reverse order to avoid corrupted last checkpoint)
    checkpoint = None
    for ckpt_path in ckpt_paths[::-1]:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            break
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_path}, trying next...")
    
    if checkpoint is None:
        print(f"Failed to load any checkpoint from {load_path}")
        return optimizer, lr_scheduler, 0, 0
    
    # Load model
    if model is not None:
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module
        status = model.load_state_dict(checkpoint["model"], strict=False)
        print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}: {status}")
    
    if reset_training_state:
        print_rank0("Reset training state")
        return optimizer, lr_scheduler, 0, 0
    
    # Load optimizer
    try:
        if reset_lr:
            for ckpt_pg, pg in zip(checkpoint["optimizer"]["param_groups"], optimizer.param_groups):
                ckpt_pg["lr"] = pg["lr"]
                ckpt_pg["initial_lr"] = pg["initial_lr"]
            print_rank0(f"Reset learning rate to {ckpt_pg['initial_lr']}")
        
        if reset_weight_decay:
            for ckpt_pg, pg in zip(checkpoint["optimizer"]["param_groups"], optimizer.param_groups):
                if ckpt_pg["weight_decay"] > 0.0:
                    ckpt_pg["weight_decay"] = pg["weight_decay"]
            print_rank0(f"Reset weight_decay to {ckpt_pg['weight_decay']}")
        
        optimizer.load_state_dict(checkpoint["optimizer"])
        print_rank0(f"Loaded optimizer, lr={optimizer.param_groups[0]['lr']}")
    except:
        traceback.print_exc()
    
    # Load scheduler
    try:
        if reset_lr:
            total_steps = job_overview.num_param_updates - checkpoint["param_update_step"]
            lr_scheduler = configure_lr_scheduler(optimizer, total_steps, warmup, "cosine")
            print_rank0(f"Reset scheduler: warmup={warmup}, total_steps={total_steps}")
        else:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print_rank0(f"Loaded scheduler from {os.path.abspath(ckpt_path)}")
    except:
        traceback.print_exc()
    
    return optimizer, lr_scheduler, checkpoint["fwdbwd_pass_step"], checkpoint["param_update_step"]


def get_job_overview(num_gpus, num_epochs, num_train_samples, batch_size_per_gpu,
                     gradient_accumulation_steps, max_fwdbwd_passes=int(1e10)):
    """Compute training steps overview."""
    batch_per_fwdbwd = batch_size_per_gpu * num_gpus
    fwdbwd_per_epoch = max(1, int(num_train_samples / batch_per_fwdbwd))
    batch_per_update = batch_per_fwdbwd * gradient_accumulation_steps
    updates_per_epoch = int(fwdbwd_per_epoch / gradient_accumulation_steps)
    
    num_epochs = min(num_epochs, int(max_fwdbwd_passes / fwdbwd_per_epoch) + 1)
    
    return edict(
        batch_size_per_fwdbwd_pass=batch_per_fwdbwd,
        batch_size_per_param_update=batch_per_update,
        num_fwdbwd_passes_per_epoch=fwdbwd_per_epoch,
        num_param_updates_per_epoch=updates_per_epoch,
        num_fwdbwd_passes=fwdbwd_per_epoch * num_epochs,
        num_param_updates=updates_per_epoch * num_epochs,
        num_epochs=num_epochs,
    )

