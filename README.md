# ProlificDreamer

Official implementation of *[ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation](https://arxiv.org/abs/2305.16213)*, published in NeurIPS 2023 (Spotlight).


<p align="center">
    <img src="teaser.png">
</p>

## Installation

The codebase is built on [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion). For installation, 
```
pip install -r requirements.txt
```

## Training
ProlificDreamer includes 3 stages for high-fidelity text-to-3d generation.
```
# --------- Stage 1 (NeRF, VSD guidance) --------- #
# This costs approximately 27GB GPU memory, with rendering resolution of 512x512
CUDA_VISIBLE_DEVICES=0 python main.py --text "A pineapple." --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 512  --w 512 --workspace exp-nerf-stage1/
# If you find the result is foggy, you can increase the --lambda_entropy. For example
CUDA_VISIBLE_DEVICES=0 python main.py --text "A pineapple." --iters 25000 --lambda_entropy 100 --scale 7.5 --n_particles 1 --h 512  --w 512 --workspace exp-nerf-stage1/
# Generate with multiple particles. Notice that generating with multiple particles is only supported in Stage 1.
CUDA_VISIBLE_DEVICES=0 python main.py --text "A pineapple." --iters 100000 --lambda_entropy 10 --scale 7.5 --n_particles 4 --h 512  --w 512 --t5_iters 20000 --workspace exp-nerf-stage1/

# --------- Stage 2 (Geometry Refinement) --------- #
# This costs <20GB GPU memory
CUDA_VISIBLE_DEVICES=0 python main.py --text "A pineapple." --iters 15000 --scale 100 --dmtet --mesh_idx 0  --init_ckpt /path/to/stage1/ckpt --normal True --sds True --density_thresh 0.1 --lambda_normal 5000 --workspace exp-dmtet-stage2/
# If the results are with maney floaters, you can increase --density_thresh. Notice that the value of --density_thresh must be consistent in stage2 and stage3.
CUDA_VISIBLE_DEVICES=0 python main.py --text "A pineapple." --iters 15000 --scale 100 --dmtet --mesh_idx 0  --init_ckpt /path/to/stage1/ckpt --normal True --sds True --density_thresh 0.4 --lambda_normal 5000 --workspace exp-dmtet-stage2/

# --------- Stage 3 (Texturing, VSD guidance) --------- #
# texturing with 512x512 rasterization
CUDA_VISIBLE_DEVICES=0 python main.py --text "A pineapple." --iters 30000 --scale 7.5 --dmtet --mesh_idx 0  --init_ckpt /path/to/stage2/ckpt --density_thresh 0.1 --finetune True --workspace exp-dmtet-stage3/
```

We also provide a script that can automatically run these 3 stages.
```
bash run.sh gpu_id text_prompt
```

For example, 
```
bash run.sh 0 "A pineapple."
```

**Limitations:** (1) Our work ultilizes the original Stable Diffusion without any 3D data, thus the multi-face Janus problem is prevalent in the results. Ultilizing text-to-image diffusion which has been finetuned on multi-view images will alleviate this problem.
(2) If the results are not satisfactory, try different seeds. This is helpful if the results have a good quality but suffer from the multi-face Janus problem.

## TODO List
- [x] Release our code.
- [ ] Combine MVDream with VSD to alleviate the multi-face problem.

## Related Links
- ProlificDreamer is also integrated in [Threestudio](https://github.com/threestudio-project/threestudio) library ❤️.
- [DreamCraft3D](https://mrtornado24.github.io/DreamCraft3D/)
- [Fantasia3D](https://fantasia3d.github.io/)
- [Magic3D](https://research.nvidia.com/labs/dir/magic3d/)
- [DreamFusion](https://dreamfusion3d.github.io/)
- [SJC](https://pals.ttic.edu/p/score-jacobian-chaining)
- [Latent-NeRF](https://github.com/eladrich/latent-nerf)

## BibTeX
If you find our work useful for your project, please consider citing the following paper.

```
@inproceedings{wang2023prolificdreamer,
  title={ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation},
  author={Zhengyi Wang and Cheng Lu and Yikai Wang and Fan Bao and Chongxuan Li and Hang Su and Jun Zhu},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
