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
FaceLift: Single Image 3D Face Reconstruction
Generates 3D head models from single images using multi-view diffusion and GS-LRM.
"""

import json
from pathlib import Path
from datetime import datetime

import gradio as gr
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from einops import rearrange
from PIL import Image
from huggingface_hub import snapshot_download

from gslrm.model.gaussians_renderer import render_turntable, imageseq2video
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from utils_folder.face_utils import preprocess_image, preprocess_image_without_cropping

# HuggingFace repository configuration
HF_REPO_ID = "wlyu/OpenFaceLift"

def download_weights_from_hf() -> Path:
    """Download model weights from HuggingFace if not already present.
    
    Returns:
        Path to the downloaded repository
    """
    workspace_dir = Path(__file__).parent
    
    # Check if weights already exist locally
    mvdiffusion_path = workspace_dir / "checkpoints/mvdiffusion/pipeckpts"
    gslrm_path = workspace_dir / "checkpoints/gslrm/ckpt_0000000000021125.pt"
    prompt_embeds_path = workspace_dir / "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"
    
    if mvdiffusion_path.exists() and gslrm_path.exists() and prompt_embeds_path.exists():
        print("Using local model weights")
        return workspace_dir
    
    print(f"Downloading model weights from HuggingFace: {HF_REPO_ID}")
    print("This may take a few minutes on first run...")
    
    # Download to checkpoints directory
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(workspace_dir / "checkpoints"),
        local_dir_use_symlinks=False,
    )
    
    print("Model weights downloaded successfully!")
    return workspace_dir

class FaceLiftPipeline:
    """Pipeline for FaceLift 3D head generation from single images."""
    
    def __init__(self):
        # Download weights from HuggingFace if needed
        workspace_dir = download_weights_from_hf()
        
        # Setup paths
        self.output_dir = workspace_dir / "outputs"
        self.examples_dir = workspace_dir / "examples"
        self.output_dir.mkdir(exist_ok=True)
        
        # Parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = 512
        self.camera_indices = [2, 1, 0, 5, 4, 3]
        
        # Load models
        print("Loading models...")
        self.mvdiffusion_pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
            str(workspace_dir / "checkpoints/mvdiffusion/pipeckpts"),
            torch_dtype=torch.float16,
        )
        self.mvdiffusion_pipeline.unet.enable_xformers_memory_efficient_attention()
        self.mvdiffusion_pipeline.to(self.device)
        
        with open(workspace_dir / "configs/gslrm.yaml", "r") as f:
            config = edict(yaml.safe_load(f))
        
        module_name, class_name = config.model.class_name.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        ModelClass = getattr(module, class_name)
        
        self.gs_lrm_model = ModelClass(config)
        checkpoint = torch.load(
            workspace_dir / "checkpoints/gslrm/ckpt_0000000000021125.pt",
            map_location="cpu"
        )
        self.gs_lrm_model.load_state_dict(checkpoint["model"])
        self.gs_lrm_model.to(self.device)
        
        self.color_prompt_embedding = torch.load(
            workspace_dir / "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt",
            map_location=self.device
        )
        
        with open(workspace_dir / "utils_folder/opencv_cameras.json", 'r') as f:
            self.cameras_data = json.load(f)["frames"]
        
        print("Models loaded successfully!")
    
    def generate_3d_head(self, image_path, auto_crop=True, guidance_scale=3.0, 
                         random_seed=4, num_steps=50):
        """Generate 3D head from single image."""
        try:
            # Setup output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_dir / timestamp
            output_dir.mkdir(exist_ok=True)
            
            # Preprocess input
            original_img = np.array(Image.open(image_path))
            input_image = preprocess_image(original_img) if auto_crop else \
                         preprocess_image_without_cropping(original_img)
            
            if input_image.size != (self.image_size, self.image_size):
                input_image = input_image.resize((self.image_size, self.image_size))
            
            input_path = output_dir / "input.png"
            input_image.save(input_path)
            
            # Generate multi-view images
            generator = torch.Generator(device=self.mvdiffusion_pipeline.unet.device)
            generator.manual_seed(random_seed)
            
            result = self.mvdiffusion_pipeline(
                input_image, None,
                prompt_embeds=self.color_prompt_embedding,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                num_inference_steps=num_steps,
                generator=generator,
                eta=1.0,
            )
            
            selected_views = result.images[:6]
            
            # Save multi-view composite
            multiview_image = Image.new("RGB", (self.image_size * 6, self.image_size))
            for i, view in enumerate(selected_views):
                multiview_image.paste(view, (self.image_size * i, 0))
            
            multiview_path = output_dir / "multiview.png"
            multiview_image.save(multiview_path)
            
            # Prepare 3D reconstruction input
            view_arrays = [np.array(view) for view in selected_views]
            lrm_input = torch.from_numpy(np.stack(view_arrays, axis=0)).float()
            lrm_input = lrm_input[None].to(self.device) / 255.0
            lrm_input = rearrange(lrm_input, "b v h w c -> b v c h w")
            
            # Prepare camera parameters
            selected_cameras = [self.cameras_data[i] for i in self.camera_indices]
            fxfycxcy_list = [[c["fx"], c["fy"], c["cx"], c["cy"]] for c in selected_cameras]
            c2w_list = [np.linalg.inv(np.array(c["w2c"])) for c in selected_cameras]
            
            fxfycxcy = torch.from_numpy(np.stack(fxfycxcy_list, axis=0).astype(np.float32))
            c2w = torch.from_numpy(np.stack(c2w_list, axis=0).astype(np.float32))
            fxfycxcy = fxfycxcy[None].to(self.device)
            c2w = c2w[None].to(self.device)
            
            batch_indices = torch.stack([
                torch.zeros(lrm_input.size(1)).long(),
                torch.arange(lrm_input.size(1)).long(),
            ], dim=-1)[None].to(self.device)
            
            batch = edict({
                "image": lrm_input,
                "c2w": c2w,
                "fxfycxcy": fxfycxcy,
                "index": batch_indices,
            })
            
            # Run 3D reconstruction
            with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
                result = self.gs_lrm_model.forward(batch, create_visual=False, split_data=True)
            
            comp_image = result.render[0].unsqueeze(0).detach()
            gaussians = result.gaussians[0]
            
            # Save filtered gaussians
            filtered_gaussians = gaussians.apply_all_filters(
                cam_origins=None,
                opacity_thres=0.04,
                scaling_thres=0.2,
                floater_thres=0.75,
                crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
                nearfar_percent=(0.0001, 1.0),
            )
            
            ply_path = output_dir / "gaussians.ply"
            filtered_gaussians.save_ply(str(ply_path))
            
            # Save output image
            comp_image = rearrange(comp_image, "x v c h w -> (x h) (v w) c")
            comp_image = (comp_image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            output_path = output_dir / "output.png"
            Image.fromarray(comp_image).save(output_path)
            
            # Generate turntable video
            turntable_frames = render_turntable(gaussians, rendering_resolution=self.image_size, 
                                               num_views=180)
            turntable_frames = rearrange(turntable_frames, "h (v w) c -> v h w c", v=180)
            turntable_frames = np.ascontiguousarray(turntable_frames)
            
            turntable_path = output_dir / "turntable.mp4"
            imageseq2video(turntable_frames, str(turntable_path), fps=30)
            
            return str(input_path), str(multiview_path), str(output_path), \
                   str(turntable_path), str(ply_path)
            
        except Exception as e:
            raise gr.Error(f"Generation failed: {str(e)}")


def main():
    """Run the FaceLift application."""
    pipeline = FaceLiftPipeline()
    
    # Load examples
    examples = []
    if pipeline.examples_dir.exists():
        examples = [[str(f)] for f in sorted(pipeline.examples_dir.iterdir()) 
                   if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    
    # Create interface
    demo = gr.Interface(
        fn=pipeline.generate_3d_head,
        title="FaceLift: Single Image 3D Face Reconstruction",
        description="""
        Transform a single portrait image into a complete 3D head model.
        
        **Tips:**
        - Use high-quality portrait images with clear facial features
        - If face detection fails, try disabling auto-cropping and manually crop to square
        """,
        inputs=[
            gr.Image(type="filepath", label="Input Portrait Image"),
            gr.Checkbox(value=True, label="Auto Cropping"),
            gr.Slider(1.0, 10.0, 3.0, step=0.1, label="Guidance Scale"),
            gr.Number(value=4, label="Random Seed"),
            gr.Slider(10, 100, 50, step=5, label="Generation Steps"),
        ],
        outputs=[
            gr.Image(label="Processed Input"),
            gr.Image(label="Multi-view Generation"),
            gr.Image(label="3D Reconstruction"),
            gr.Video(label="Turntable Animation"),
            gr.File(label="3D Model (.ply)"),
        ],
        examples=examples,
        allow_flagging="never",
    )
    
    demo.queue(max_size=10)
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()