# FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads

### 🌺 ICCV 2025 🌺

[Weijie Lyu](https://weijielyu.github.io/), [Yi Zhou](https://zhouyisjtu.github.io/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Zhixin Shu](https://zhixinshu.github.io/)  
University of California, Merced - Adobe Research

[![Website](https://img.shields.io/badge/Website-FaceLift?logo=googlechrome&logoColor=hsl(204%2C%2086%25%2C%2053%25)&label=FaceLift&labelColor=%23f5f5dc&color=hsl(204%2C%2086%25%2C%2053%25))](https://weijielyu.github.io/FaceLift)
[![Paper](https://img.shields.io/badge/Paper-arXiv?logo=arxiv&logoColor=%23B31B1B&label=arXiv&labelColor=%23f5f5dc&color=%23B31B1B)](https://arxiv.org/abs/2412.17812)
[![Video](https://img.shields.io/badge/Video-YouTube?logo=youtube&logoColor=%23FF0000&label=YouTube&labelColor=%23f5f5dc&color=%23FF0000)](https://youtu.be/lf0Gck9UOcU)

<div align='center'>
<img alt="image" src='media/teaser.png'>
</div>

> *FaceLift* transforms a single facial image into a high-fidelity 3D Gaussian head representation, and it generalizes remarkably well to real-world human images.

This is a self-reimplementation of *FaceLift*.

## 🔧 Prerequisites

### Model Checkpoints

Model checkpoints will be automatically downloaded from [HuggingFace](https://huggingface.co/wlyu/OpenFaceLift) on first run.

Alternatively, you can manually place the checkpoints in the `checkpoints/` directory:
- `checkpoints/mvdiffusion/pipeckpts/` - Multi-view diffusion model
- `checkpoints/gslrm/ckpt_0000000000021125.pt` - GS-LRM model checkpoint

### Environment Setup

```bash
bash setup_env.sh
```

## 🚀 Inference

### Command Line Interface

Process images from a directory:

```bash
python inference.py --input_dir examples/ --output_dir outputs/
```

**Available Arguments:**
| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input_dir` | `-i` | `examples/` | Input directory containing images |
| `--output_dir` | `-o` | `outputs/` | Output directory for results |
| `--auto_crop` | - | `True` | Automatically crop faces |
| `--seed` | - | `4` | Random seed for reproducible results |
| `--guidance_scale_2D` | - | `3.0` | Guidance scale for multi-view generation |
| `--step_2D` | - | `50` | Number of diffusion steps |

### Web Interface

Launch the interactive Gradio web interface:

```bash
python gradio_app.py
```

Open your browser and navigate to `http://localhost:7860` to use the web interface. If running on a server, use the provided public link.

## 🎓 Training

### Data Structure

We sincerely apologize, but due to company policy, our training data will not be publicly available.

If you have your own data, organize your data directory following the structure in `FaceLift/data_sample/`:

**Multi-view Diffusion Data:**
```
data_sample/
├── mvdiffusion/
│   ├── data_mvdiff_train.txt          # Training data list
│   ├── data_mvdiff_val.txt            # Validation data list
│   └── sample_000/
│       ├── cam_000.png                # Front view (RGBA, 512×512)
│       ├── cam_001.png                # Front-right view
│       ├── cam_002.png                # Right view
│       ├── cam_003.png                # Back view
│       ├── cam_004.png                # Left view
│       └── cam_005.png                # Front-left view
```

**GS-LRM Data:**
```
data_sample/
├── gslrm/
│   ├── data_gslrm_train.txt           # Training data list
│   ├── data_gslrm_val.txt             # Validation data list
│   └── sample_000/
│       ├── images/
│       │   ├── cam_000.png            # Multi-view images (RGBA, 512×512)
│       │   ├── cam_001.png
│       │   ├── ...
│       │   └── cam_031.png            # 32 views total
│       └── opencv_cameras.json        # Camera parameters
```

### Multi-view Diffusion Training

```bash
accelerate launch --config_file mvdiffusion/node_config/8gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion.yaml
```

### Gaussian Reconstructor Training

Our Gaussian Reconstructor is based on GS-LRM and uses pre-trained weights from Objaverse data.

- Stage I: 256 resolution on Objaverse - `gslrm_pretrain_256.yaml`
- Stage II: 512 resolution on Objaverse - `gslrm_pretrain_512.yaml`
- Stage III: 512 resolution on synthetic heads data - `gslrm.yaml`

```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id ${JOB_UUID} --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    train_gslrm.py --config configs/gslrm.yaml
```

## 📝 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@misc{lyu2025facelift,
    title={FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads}, 
    author={Weijie Lyu and Yi Zhou and Ming-Hsuan Yang and Zhixin Shu},
    year={2025},
    eprint={2412.17812},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.17812}
}
```

## 📄 License

Copyright 2025 Adobe Inc.

Model weights are licensed from Adobe Inc. under the Adobe Research License.

<div align="center">

**🌟 If you find this project helpful, please give it a star! 🌟**

</div>