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

# utils_losses.py is under the Adobe Research License. Copyright 2025 Adobe Inc.

"""
Perceptual Loss Implementation using VGG19 and SSIM Loss Implementation.

Adapted from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
"""
import os
import hashlib
import urllib.request
from typing import List, Tuple, Union, Optional
from pathlib import Path

import scipy.io
import torch
import torch.nn as nn
from pytorch_msssim import SSIM

# VGG19 ImageNet normalization constants
IMAGENET_MEAN = [123.6800, 116.7790, 103.9390]

# VGG19 layer configuration
VGG19_LAYER_INDICES = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
VGG19_LAYER_NAMES = [
    "conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8",
    "conv9", "conv10", "conv11", "conv12", "conv13", "conv14", "conv15", "conv16"
]
VGG19_CHANNEL_SIZES = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

# Perceptual loss weighting factors
LAYER_WEIGHTS = [1.0, 1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5]

# VGG19 weights download URL and MD5 checksum
VGG19_WEIGHTS_URL = "https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
VGG19_WEIGHTS_MD5 = "106118b7cf60435e6d8e04f6a6dc3657"


def _download_vgg19_weights(cache_dir: Optional[str] = None) -> str:
    """
    Download VGG19 weights to cache directory.
    
    Args:
        cache_dir: Directory to cache the weights. If None, uses ~/.cache/openfacelift
        
    Returns:
        Path to the downloaded weights file
        
    Raises:
        RuntimeError: If download fails or MD5 checksum doesn't match
    """
    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "openfacelift")
    
    os.makedirs(cache_dir, exist_ok=True)
    weight_file = os.path.join(cache_dir, "imagenet-vgg-verydeep-19.mat")
    
    # If file exists and has correct MD5, return it
    if os.path.isfile(weight_file):
        with open(weight_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash == VGG19_WEIGHTS_MD5:
            return weight_file
        else:
            print(f"Existing file has incorrect MD5 checksum. Re-downloading...")
    
    # Download the file
    print(f"Downloading VGG19 weights from {VGG19_WEIGHTS_URL}...")
    print(f"This may take a few minutes (file size: ~548 MB)")
    
    try:
        urllib.request.urlretrieve(VGG19_WEIGHTS_URL, weight_file)
    except Exception as e:
        raise RuntimeError(f"Failed to download VGG19 weights: {e}")
    
    # Verify MD5 checksum
    with open(weight_file, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    if file_hash != VGG19_WEIGHTS_MD5:
        os.remove(weight_file)
        raise RuntimeError(
            f"Downloaded file has incorrect MD5 checksum.\n"
            f"Expected: {VGG19_WEIGHTS_MD5}\n"
            f"Got: {file_hash}"
        )
    
    print(f"VGG19 weights successfully downloaded to {weight_file}")
    return weight_file


class VGG19(nn.Module):
    """
    VGG19 network implementation for perceptual loss computation.
    
    This class implements the VGG19 architecture with specific layer outputs
    used for computing perceptual losses at different scales.
    """
    
    def __init__(self) -> None:
        """Initialize VGG19 network layers."""
        super(VGG19, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.max1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.max2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.max3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.max4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu14 = nn.ReLU(inplace=True)

        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu15 = nn.ReLU(inplace=True)

        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu16 = nn.ReLU(inplace=True)
        self.max5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, return_style: int) -> Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Forward pass through VGG19 network.
        
        Args:
            x: Input tensor of shape [B, 3, H, W]
            return_style: If > 0, return style features as list; otherwise return content features as tuple
            
        Returns:
            Either a list of style features or tuple of content features from different layers
        """
        out1 = self.conv1(x)
        out2 = self.relu1(out1)

        out3 = self.conv2(out2)
        out4 = self.relu2(out3)
        out5 = self.max1(out4)

        out6 = self.conv3(out5)
        out7 = self.relu3(out6)
        out8 = self.conv4(out7)
        out9 = self.relu4(out8)
        out10 = self.max2(out9)
        out11 = self.conv5(out10)
        out12 = self.relu5(out11)
        out13 = self.conv6(out12)
        out14 = self.relu6(out13)
        out15 = self.conv7(out14)
        out16 = self.relu7(out15)
        out17 = self.conv8(out16)
        out18 = self.relu8(out17)
        out19 = self.max3(out18)
        out20 = self.conv9(out19)
        out21 = self.relu9(out20)
        out22 = self.conv10(out21)
        out23 = self.relu10(out22)
        out24 = self.conv11(out23)
        out25 = self.relu11(out24)
        out26 = self.conv12(out25)
        out27 = self.relu12(out26)
        out28 = self.max4(out27)
        out29 = self.conv13(out28)
        out30 = self.relu13(out29)
        out31 = self.conv14(out30)
        out32 = self.relu14(out31)

        if return_style > 0:
            return [out2, out7, out12, out21, out30]
        else:
            return out4, out9, out14, out23, out32


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss module using pre-trained VGG19.
    
    This class implements perceptual loss by comparing features extracted from
    different layers of a pre-trained VGG19 network. It computes weighted
    differences across multiple scales to capture both low-level and high-level
    visual differences between images.
    """
    
    def __init__(self, device: str = "cpu", weight_file: Optional[str] = None) -> None:
        """
        Initialize PerceptualLoss module.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
            weight_file: Path to VGG19 weight file. If None, uses default path or environment variable.
            
        Raises:
            FileNotFoundError: If weight file is not found
            RuntimeError: If weight file cannot be loaded
        """
        super().__init__()
        self.device = device
        self.net = VGG19()

        # Determine weight file path
        if weight_file is None:
            # Check environment variable first
            weight_file = os.environ.get('VGG19_WEIGHTS_PATH')
            if weight_file is None or not os.path.isfile(weight_file):
                # Download weights to cache directory
                weight_file = _download_vgg19_weights()
        
        # Verify the file exists
        if not os.path.isfile(weight_file):
            raise FileNotFoundError(f"VGG19 weight file not found: {weight_file}")
        
        # Load VGG19 weights
        try:
            vgg_rawnet = scipy.io.loadmat(weight_file)
            vgg_layers = vgg_rawnet["layers"][0]
        except Exception as e:
            raise RuntimeError(f"Failed to load VGG19 weights from {weight_file}: {e}")

        # Load pre-trained weights into the network
        self._load_pretrained_weights(vgg_layers)
        
        # Set network to evaluation mode and freeze parameters
        self.net = self.net.eval().to(device)
        for param in self.net.parameters():
            param.requires_grad = False
            
    def _load_pretrained_weights(self, vgg_layers) -> None:
        """Load pre-trained VGG19 weights into the network."""
        for layer_idx in range(len(VGG19_LAYER_NAMES)):
            layer_name = VGG19_LAYER_NAMES[layer_idx]
            mat_layer_idx = VGG19_LAYER_INDICES[layer_idx]
            channel_size = VGG19_CHANNEL_SIZES[layer_idx]
            
            # Extract weights and biases from MATLAB format
            layer_weights = torch.from_numpy(
                vgg_layers[mat_layer_idx][0][0][2][0][0]
            ).permute(3, 2, 0, 1)
            layer_biases = torch.from_numpy(
                vgg_layers[mat_layer_idx][0][0][2][0][1]
            ).view(channel_size)
            
            # Assign to network
            getattr(self.net, layer_name).weight = nn.Parameter(layer_weights)
            getattr(self.net, layer_name).bias = nn.Parameter(layer_biases)

    def _compute_l1_error(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 (Mean Absolute Error) between two tensors.
        
        Args:
            truth: Ground truth tensor
            pred: Predicted tensor
            
        Returns:
            L1 error as a scalar tensor
        """
        return torch.mean(torch.abs(truth - pred))

    def forward(self, pred_img: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and real images.
        
        Args:
            pred_img: Predicted image tensor of shape [B, 3, H, W] in range [0, 1]
            real_img: Real image tensor of shape [B, 3, H, W] in range [0, 1]
            
        Returns:
            Perceptual loss as a scalar tensor
        """
        # Convert to ImageNet normalization (RGB -> BGR and subtract mean)
        imagenet_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=pred_img.device)
        imagenet_mean = imagenet_mean.view(1, 3, 1, 1)

        # Scale to [0, 255] and apply ImageNet normalization
        real_img_normalized = real_img * 255.0 - imagenet_mean
        pred_img_normalized = pred_img * 255.0 - imagenet_mean

        # Extract features from both images
        real_features = self.net(real_img_normalized, return_style=0)
        pred_features = self.net(pred_img_normalized, return_style=0)

        # Compute weighted L1 losses at different scales
        losses = []
        
        # Raw image loss
        raw_loss = self._compute_l1_error(real_img_normalized, pred_img_normalized)
        losses.append(raw_loss * LAYER_WEIGHTS[0])
        
        # Feature losses at different VGG layers
        for i, (real_feat, pred_feat) in enumerate(zip(real_features, pred_features)):
            feature_loss = self._compute_l1_error(real_feat, pred_feat)
            losses.append(feature_loss * LAYER_WEIGHTS[i + 1])

        # Combine all losses and normalize
        total_loss = sum(losses) / 255.0
        return total_loss

class SsimLoss(nn.Module):
    """
    SSIM Loss module that computes 1 - SSIM for image similarity.
    
    Args:
        data_range: Range of input data (default: 1.0 for [0,1] range)
    """
    
    def __init__(self, data_range: float = 1.0) -> None:
        super().__init__()
        self.data_range = data_range
        self.ssim_module = SSIM(
            win_size=11,
            win_sigma=1.5,
            data_range=self.data_range,
            size_average=True,
            channel=3,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss between two image tensors.
        
        Args:
            x: Image tensor of shape (N, C, H, W)
            y: Image tensor of shape (N, C, H, W)
            
        Returns:
            SSIM loss (1 - SSIM similarity)
        """
        return 1.0 - self.ssim_module(x, y)
