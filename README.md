# FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads

[Weijie Lyu](https://weijielyu.github.io/), [Yi Zhou](https://zhouyisjtu.github.io/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Zhixin Shu](https://zhixinshu.github.io/)<br>
University of California, Merced - Adobe Research

[![Website](https://img.shields.io/badge/Website-FaceLift?logo=googlechrome&logoColor=hsl(204%2C%2086%25%2C%2053%25)&label=FaceLift&labelColor=%23f5f5dc&color=hsl(204%2C%2086%25%2C%2053%25))](https://weijielyu.github.io/FaceLift)
[![Paper](https://img.shields.io/badge/Paper-arXiv?logo=arxiv&logoColor=%23B31B1B&label=arXiv&labelColor=%23f5f5dc&color=%23B31B1B)](https://arxiv.org/abs/2412.17812)
[![Video](https://img.shields.io/badge/Video-YouTube?logo=youtube&logoColor=%23FF0000&label=YouTube&labelColor=%23f5f5dc&color=%23FF0000)](https://youtu.be/lf0Gck9UOcU)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fweijielyu%2FFaceLift&count_bg=%2379C83D&title_bg=%23F5F5DC&icon=github.svg&icon_color=%2379C83D&title=🔎&edge_flat=false)](https://hits.seeyoufarm.com)

<div align='center'>
<img alt="image" src='media/teaser.png'>
</div>

> *FaceLift* transforms a single facial image into a high-fidelity 3D Gaussian head representation. Trained exclusively on synthetic 3D data, our pipeline first generates sparse, identity-preserving multiview images of the input head using a diffusion model. These sparse generated views are then fed into a transformer-based 3D Gaussian Splats reconstructor, producing complete and detailed 3D head representations that generalize remarkably well to real-world human images.

## Citation

If you find our work useful for your project, please consider citing our paper.

```
@misc{lyu2024facelift,
      title={FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads}, 
      author={Weijie Lyu and Yi Zhou and Ming-Hsuan Yang and Zhixin Shu},
      year={2024},
      eprint={2412.17812},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
