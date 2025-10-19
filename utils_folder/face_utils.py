# Copyright (C) 2025, FaceLift Research Group
# https://github.com/weijielyu/FaceLift
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact: wlyu3@ucmerced.edu

"""
Face detection and cropping utilities for 3D face reconstruction.

This module provides functions for face detection, cropping, and preprocessing
to align faces with training data specifications.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from rembg import remove

# Training set face parameters (derived from training data statistics)
TRAINING_SET_FACE_SIZE = 194.2749650813705
TRAINING_SET_FACE_CENTER = [251.83270369057132, 280.0133630862363]

# Public constants for external use
FACE_SIZE = TRAINING_SET_FACE_SIZE
FACE_CENTER = TRAINING_SET_FACE_CENTER
DEFAULT_BACKGROUND_COLOR = (255, 255, 255)
DEFAULT_IMG_SIZE = 512

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default face detector instance
FACE_DETECTOR = MTCNN(
    image_size=512, 
    margin=0, 
    min_face_size=20, 
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=True, 
    device=DEVICE
)

def select_face(detected_bounding_boxes: Optional[np.ndarray], confidence_scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Select the largest face from detected faces with confidence above threshold.
    
    Args:
        detected_bounding_boxes: Detected bounding boxes in xyxy format
        confidence_scores: Detection confidence probabilities
        
    Returns:
        Selected bounding box or None if no suitable face found
    """
    if detected_bounding_boxes is None or confidence_scores is None:
        return None
        
    # Filter faces with confidence > 0.8
    high_confidence_faces = [
        detected_bounding_boxes[i] for i in range(len(detected_bounding_boxes)) 
        if confidence_scores[i] > 0.8
    ]
    
    if not high_confidence_faces:
        return None

    # Return the largest face (by area)
    return max(high_confidence_faces, key=lambda bbox: (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))

def crop_face(
    input_image_array: np.ndarray, 
    face_detector: MTCNN = FACE_DETECTOR, 
    target_face_size: float = FACE_SIZE, 
    target_face_center: list = FACE_CENTER, 
    output_image_size: int = 512, 
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Crop and align face in image to match training data specifications.
    
    Args:
        input_image_array: Input image as numpy array (H, W, C)
        face_detector: MTCNN face detector instance
        target_face_size: Target face size from training data
        target_face_center: Target face center from training data
        output_image_size: Output image size
        background_color: Background color for padding
        
    Returns:
        Tuple of (cropped_image, crop_parameters)
        
    Raises:
        ValueError: If no face is detected in the image
    """
    image_height, image_width, _ = input_image_array.shape
    
    # Handle RGBA images by compositing with background color
    if input_image_array.shape[2] == 4:
        rgba_pil_image = Image.fromarray(input_image_array)
        background_image = Image.new("RGB", rgba_pil_image.size, background_color)
        rgb_composite_image = Image.alpha_composite(background_image.convert("RGBA"), rgba_pil_image).convert("RGB")
        processed_image_array = np.array(rgb_composite_image)
    else:
        processed_image_array = input_image_array[:, :, :3]  # Ensure RGB format

    # Detect and select face
    detected_bounding_boxes, confidence_scores = face_detector.detect(processed_image_array)
    selected_face_bbox = select_face(detected_bounding_boxes, confidence_scores)
    if selected_face_bbox is None:
        raise ValueError("No face detected in the image")

    # Calculate detected face properties
    detected_face_size = 0.5 * (selected_face_bbox[2] - selected_face_bbox[0] + selected_face_bbox[3] - selected_face_bbox[1])
    detected_face_center = (
        0.5 * (selected_face_bbox[0] + selected_face_bbox[2]), 
        0.5 * (selected_face_bbox[1] + selected_face_bbox[3])
    )

    # Scale image to match training face size
    scale_ratio = target_face_size / detected_face_size
    scaled_width, scaled_height = int(image_width * scale_ratio), int(image_height * scale_ratio)
    scaled_pil_image = Image.fromarray(processed_image_array).resize((scaled_width, scaled_height))
    scaled_face_center = (
        int(detected_face_center[0] * scale_ratio), 
        int(detected_face_center[1] * scale_ratio)
    )

    # Create output image with background
    output_image = Image.new("RGB", (output_image_size, output_image_size), color=background_color)

    # Calculate alignment offsets
    horizontal_offset = target_face_center[0] - scaled_face_center[0]
    vertical_offset = target_face_center[1] - scaled_face_center[1]

    # Calculate crop boundaries
    crop_left_boundary = int(max(0, -horizontal_offset))
    crop_top_boundary = int(max(0, -vertical_offset))
    crop_right_boundary = int(min(scaled_width, output_image_size - horizontal_offset))
    crop_bottom_boundary = int(min(scaled_height, output_image_size - vertical_offset))

    # Crop and paste
    cropped_face_image = scaled_pil_image.crop((crop_left_boundary, crop_top_boundary, crop_right_boundary, crop_bottom_boundary))
    paste_coordinates = (int(max(0, horizontal_offset)), int(max(0, vertical_offset)))
    output_image.paste(cropped_face_image, paste_coordinates)

    crop_parameters = {
        'resize_ratio': scale_ratio,
        'x_offset_left': horizontal_offset,
        'y_offset_top': vertical_offset,
    }

    return output_image, crop_parameters

def prepare_foreground_with_rembg(input_image_array: np.ndarray) -> np.ndarray:
    """
    Prepare foreground image using rembg for background removal.
    
    Args:
        input_image_array: Input image as numpy array (H, W, C)
        
    Returns:
        RGBA image as numpy array with background removed
    """
    pil_image = Image.fromarray(input_image_array)
    background_removed_image = remove(pil_image)
    processed_image_array = np.array(background_removed_image)
    
    # Ensure RGBA format
    if processed_image_array.shape[2] == 4:
        return processed_image_array
    elif processed_image_array.shape[2] == 3:
        height, width = processed_image_array.shape[:2]
        alpha_channel = np.full((height, width), 255, dtype=np.uint8)
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[:, :, :3] = processed_image_array
        rgba_image[:, :, 3] = alpha_channel
        return rgba_image
    
    return processed_image_array

def preprocess_image(
    original_image_array: np.ndarray, 
    target_image_size: int = DEFAULT_IMG_SIZE, 
    background_color: Tuple[int, int, int] = DEFAULT_BACKGROUND_COLOR
) -> Image.Image:
    """
    Preprocess image with background removal and face cropping.
    
    Args:
        original_image_array: Input image as numpy array
        target_image_size: Target image size
        background_color: Background color for compositing
        
    Returns:
        Processed PIL Image
    """
    processed_image_array = prepare_foreground_with_rembg(original_image_array)
    
    # Convert RGBA to RGB with specified background
    if processed_image_array.shape[2] == 4:
        rgba_pil_image = Image.fromarray(processed_image_array)
        background_image = Image.new("RGB", rgba_pil_image.size, background_color)
        rgb_composite_image = Image.alpha_composite(background_image.convert("RGBA"), rgba_pil_image).convert("RGB")
        processed_image_array = np.array(rgb_composite_image)
    
    cropped_image, crop_parameters = crop_face(
        processed_image_array,
        FACE_DETECTOR,
        FACE_SIZE, 
        FACE_CENTER,
        target_image_size, 
        background_color
    )
    return cropped_image

def preprocess_image_without_cropping(
    original_image_array: np.ndarray, 
    target_image_size: int = DEFAULT_IMG_SIZE, 
    background_color: Tuple[int, int, int] = DEFAULT_BACKGROUND_COLOR
) -> Image.Image:
    """
    Preprocess image with background removal, without face cropping.
    
    Args:
        original_image_array: Input image as numpy array
        target_image_size: Target image size
        background_color: Background color for compositing
        
    Returns:
        Processed PIL Image
    """
    processed_image_array = prepare_foreground_with_rembg(original_image_array)
    
    resized_image = Image.fromarray(processed_image_array).resize((target_image_size, target_image_size))
    background_image = Image.new("RGBA", (target_image_size, target_image_size), background_color)
    composite_image = Image.alpha_composite(background_image, resized_image).convert("RGB")
    return composite_image
