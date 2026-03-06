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
Script for generating fixed text embeddings for multi-view diffusion models.
Generates CLIP text embeddings for different view angles and types.

copied from https://github.com/pengHTYX/Era3D/blob/main/mvdiffusion/data/generate_fixed_text_embeds.py
"""

from typing import List, Optional
import os
import torch
from transformers import CLIPTokenizer, CLIPTextModel


class TextEmbeddingGenerator:
    """Generator for fixed text embeddings using CLIP."""
    
    def __init__(self, 
                 model_name: str = 'stabilityai/stable-diffusion-2-1-unclip',
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float16):
        """
        Initialize the text embedding generator.
        
        Args:
            model_name: Pretrained model name or path
            device: Device to run on (defaults to cuda:0 if available)
            dtype: Data type for computations
        """
        self.model_name = model_name
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')
        self.text_encoder = self.text_encoder.to(self.device, dtype=self.dtype)
    
    def _encode_text_prompts(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts into embeddings.
        
        Args:
            text_prompts: List of text prompts to encode
            
        Returns:
            Text embeddings tensor
        """
        print(f"Encoding prompts: {text_prompts}")
        
        # Tokenize
        text_inputs = self.tokenizer(
            text_prompts, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        text_input_ids = text_inputs.input_ids
        
        # Check for truncation
        untruncated_ids = self.tokenizer(text_prompts, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            print("Warning: Some prompts were truncated during tokenization")
        
        # Handle attention mask
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None
        
        # Generate embeddings
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids.to(self.device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0].detach().cpu()
        
        print(f"Generated embeddings shape: {prompt_embeds.shape}")
        return prompt_embeds
    
    def generate_multiview_embeds(self, output_path: str = './fixed_prompt_embeds_6view') -> None:
        """
        Generate text embeddings for multi-view rendering prompts.
        
        Args:
            output_path: Directory to save the embeddings
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Define view angles
        views = ["front", "front_right", "right", "back", "left", "front_left"]
        
        # Generate prompts for color and normal maps
        color_prompts = [f"a rendering image of 3D models, {view} view, color map." for view in views]
        normal_prompts = [f"a rendering image of 3D models, {view} view, normal map." for view in views]
        
        prompt_types = [
            (color_prompts, "clr_embeds.pt"),
            (normal_prompts, "normal_embeds.pt")
        ]
        
        for prompts, filename in prompt_types:
            embeds = self._encode_text_prompts(prompts)
            save_path = os.path.join(output_path, filename)
            torch.save(embeds, save_path)
            print(f"Saved embeddings to {save_path}")
        
        print("Multi-view embeddings generation completed")
    

def main():
    """Main function to generate all embeddings."""
    generator = TextEmbeddingGenerator()
    
    # Generate multi-view embeddings
    generator.generate_multiview_embeds()


if __name__ == "__main__":
    main()