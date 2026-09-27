<div align="center">

# :fire: Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2408.15994)
[![IEEE TIP](https://img.shields.io/badge/Publication-IEEE%20TIP-blue)](https://doi.org/10.1109/TIP.2025.3566300)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://house-yuyu.github.io/Perceive-IR/)
[![Weights](https://img.shields.io/badge/Weights-Baidu%20Netdisk-blue)](https://pan.baidu.com/s/1r04w7uSwXZ6V8bE1X9BbMQ?pwd=k4uc)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)

</div>

---

PyTorch implementation of the paper:

> **Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration**<br>
> [Xu Zhang<sup>1*</sup>](https://house-yuyu.github.io/), [Jiaqi Ma<sup>1,2*</sup>](https://leonmakise.github.io/), [Guoli Wang<sup>3</sup>](https://scholar.google.com.hk/citations?user=z-25fk0AAAAJ&hl=zh-CN), [Qian Zhang<sup>3</sup>](https://scholar.google.com.hk/citations?user=pCY-bikAAAAJ&hl=zh-CN), [Huan Zhang<sup>4</sup>](https://scholar.google.com.hk/citations?user=bJjd_kMAAAAJ&hl=zh-CN), [Lefei Zhang<sup>1📧</sup>](https://scholar.google.com.hk/citations?user=BLKHwNwAAAAJ&hl=zh-CN)<br>
> <sup>1</sup>Wuhan University, <sup>2</sup>Mohamed bin Zayed University of Artificial Intelligence,<br>
> <sup>3</sup>Horizon Robotics, <sup>4</sup>Guangdong University of Technology<br>
> <sup>*</sup>Equal contribution; <sup>📧</sup>Corresponding author.

![Perceive-IR restoration framework](https://raw.githubusercontent.com/House-yuyu/Perceive-IR/main/fig/PerceiveIR_Stage2.png)

:star: If you find Perceive-IR useful, please consider starring this repository.

## Overview

Perceive-IR learns to distinguish image quality across different degradation types and severity levels. Its two-stage framework first learns quality prompts in the CLIP feature space, then uses the learned quality perceiver to guide image restoration. Semantic guidance and degradation representations provide complementary information for the restoration network.

This repository provides the following all-in-one settings:

| Setting | Restoration tasks | Configuration |
| --- | --- | --- |
| Three-task | Denoising, dehazing, deraining | [perceiveIR_3task.yaml](configs/perceiveIR_3task.yaml) |
| Five-task | Denoising, dehazing, deraining, deblurring, low-light enhancement | [perceiveIR_5task.yaml](configs/perceiveIR_5task.yaml) |

The Python package is named `perceiveIR`, and the restoration model is `PerceiveIR`.

## Installation

Use **Python 3.10 or 3.11** with **PyTorch 2.4.1** and **torchvision 0.19.1**. CUDA training is intended for Linux; the launch scripts require Bash.

```bash
git clone https://github.com/House-yuyu/Perceive-IR.git
cd Perceive-IR
conda create -n perceiveir python=3.10 -y
conda activate perceiveir

# Install torch==2.4.1 and torchvision==0.19.1 for your CUDA runtime first.
# Install the shared dependencies (Git is required for OpenAI CLIP):
python -m pip install -r requirements.txt
```

The single `requirements.txt` covers training, evaluation, analysis tools, and tests. Multi-proxy training also requires the external model repositories and their dependencies; follow [third_party/README.md](third_party/README.md), including the CUDA extension setup for MambaIR.

All paths in the provided configurations are relative to the repository root. Run the commands below from this directory. The shell scripts also accept a `PYTHON` environment variable to select an interpreter.

## Dataset Preparation

Download the datasets separately from their original sources. Dataset files are not included in this repository or the source archive.

| Task | Training datasets | Test datasets |
| --- | --- | --- |
| Denoising | BSD400 + WED | CBSD68; Urban100 and Kodak24 are also used for training validation |
| Dehazing | RESIDE-OTS | SOTS-Outdoor |
| Deraining | Train100L | Rain100L |
| Deblurring | GoPro training split | GoPro test split |
| Low-light enhancement | LOL-v1 training split | LOL-v1 test split |

**Download links, the complete directory tree, and image-pair naming rules are provided in [data/README.md](data/README.md).** Place the prepared data under `data/AiOIR/`, or update `data.root` in the relevant YAML files:

```text
data/AiOIR/
├── train/
│   ├── Denoise/gt/
│   ├── Dehaze/{input,gt}/
│   ├── Derain/{input,gt}/
│   ├── Deblur/{input,gt}/
│   └── Enhance/{input,gt}/
└── test/
    ├── denoise/{bsd68,urban100,kodak24}/target/
    ├── dehaze/{input,target}/
    ├── derain/Rain100L/{input,target}/
    ├── deblur/gopro/{input,target}/
    └── enhance/lol/{input,target}/
```

Denoising uses Gaussian noise with standard deviations of 15, 25, and 50. The default three-task evaluator expects 804 cases, and the five-task evaluator expects 1930 cases; see the dataset guide for the exact splits.

## Pretrained Weights

Download the weights from **Baidu Netdisk**:

- **Shared folder:** Perceive-IR
- **Download:** [Baidu Netdisk](https://pan.baidu.com/s/1r04w7uSwXZ6V8bE1X9BbMQ?pwd=k4uc)
- **Extraction code:** `k4uc`

Place the downloaded files in **`weight/`**. The expected layout is:

```text
weight/
├── perceiveIR_3task.pth
├── perceiveIR_5task.pth
├── dinov2-base/                 # DINOv2 config.json and model.safetensors
├── clip/ViT-B-32.pt
├── vgg16-397923af.pth
├── stage1/
│   ├── quality_prompts_100000.pth
│   ├── holdout_0/medium_020000.pth
│   └── holdout_1/medium_020000.pth
└── proxies/                    # Training proxy checkpoints and task embeddings
```

**Evaluation requires only the selected restoration checkpoint and DINOv2.** The CLIP, VGG16, first-stage, and proxy assets are used for training. The full filenames, upstream download sources, and optional assets are listed in [weight/README.md](weight/README.md). Check the files required for your workflow after downloading:

```bash
python scripts/check_perceiveIR_assets.py --profile inference3
python scripts/check_perceiveIR_assets.py --profile inference5
# For the complete training asset list:
python scripts/check_perceiveIR_assets.py --profile train
```

## Training

### Stage 1: Quality Prompt Learning

Use [configs/perceiveIR_stage1.yaml](configs/perceiveIR_stage1.yaml) to train the two medium-quality models, render the low/medium/high-quality triplets, and learn the quality prompts:

```bash
# Train each fold to the 20K checkpoint used by the restoration configurations.
bash scripts/train_perceiveIR_stage1.sh medium --heldout-fold 0 --total-iters 20000
bash scripts/train_perceiveIR_stage1.sh medium --heldout-fold 1 --total-iters 20000
```

Place the medium checkpoints under `weight/stage1/`, then render the triplets and train the prompts:

```bash
mkdir -p weight/stage1/holdout_0 weight/stage1/holdout_1
cp experiments/perceiveIR/stage1/medium_models/holdout_0/checkpoints/medium_020000.pth weight/stage1/holdout_0/
cp experiments/perceiveIR/stage1/medium_models/holdout_1/checkpoints/medium_020000.pth weight/stage1/holdout_1/
bash scripts/prepare_perceiveIR_stage1.sh
bash scripts/train_perceiveIR_stage1.sh prompts --batch-size-per-gpu 32
cp experiments/perceiveIR/stage1/quality_prompts/checkpoints/quality_prompts_100000.pth weight/stage1/
```

The prompt launcher enables task-balanced sampling and trains for 100K iterations by default. The current first-stage pipeline uses all five training tasks. Fold assignment is computed from image paths, so keep the dataset paths unchanged between medium-model training and triplet generation. Historical medium checkpoints require their original sample-to-fold mapping; moving the data and recomputing folds does not preserve the held-out split.

### Stage 2: Image Restoration

Prepare the first-stage assets and third-party proxies, then generate the validation manifest for the setting you want to train:

```bash
python -m perceiveIR.validation --config configs/perceiveIR_3task.yaml
# Or, for the five-task setting:
python -m perceiveIR.validation --config configs/perceiveIR_5task.yaml
```

The provided restoration configurations use AdamW, a learning rate of `2e-4`, `256 × 256` crops, and 500K iterations:

```bash
# Three-task setting: 4 GPUs, total batch size 8.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_perceiveIR.sh configs/perceiveIR_3task.yaml

# Five-task setting: 2 GPUs, total batch size 4.
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_perceiveIR.sh configs/perceiveIR_5task.yaml
```

For a different GPU count, set `NPROC_PER_NODE` and provide one batch size per process. For example:

```bash
NPROC_PER_NODE=1 bash scripts/train_perceiveIR.sh configs/perceiveIR_3task.yaml --batch-sizes 2
```

Checkpoints and logs are saved under the configuration's `output` directory in `experiments/perceiveIR/`. Resume a compatible run with:

```bash
bash scripts/train_perceiveIR.sh configs/perceiveIR_3task.yaml \
  --resume experiments/perceiveIR/stage2/proxy_all_500k_bs8/last.pth
```

Full training checkpoints contain optimizer state, proxy paths, and a DPL probe hash. Resume with the same dataset and proxy configuration; a model-only inference checkpoint is insufficient for resuming training. If data paths change, regenerate the triplet and validation manifests. The optional [specialist-proxy configuration](configs/perceiveIR_5task_special_proxy.yaml) additionally requires offline NAFNet and Retinexformer proxy images prepared with the rendering and manifest tools in `scripts/`.

## Inference and Evaluation

After preparing the test datasets and the required weights, evaluate the three-task or five-task model:

```bash
bash scripts/test_perceiveIR.sh 3
bash scripts/test_perceiveIR.sh 5
```

To use a custom checkpoint or dataset location:

```bash
bash scripts/test_perceiveIR.sh 3 \
  --checkpoint weight/perceiveIR_3task.pth \
  --data-root /path/to/AiOIR \
  --output results/my_three_task_test
```

The evaluator runs whole-image inference in FP32 and reports RGB PSNR/SSIM. By default, metric records and summaries are written to `results/perceiveIR_3task/` or `results/perceiveIR_5task/`. Use a separate output directory for each checkpoint. Add `--device cpu` for CPU evaluation.

If a validation subset was used to select the checkpoint, pass its manifest through `--validation-manifest` to also report results excluding those scenes. The summary includes both the image-weighted average and the equal-weight mean across benchmark columns.

## Implementation

- **Quality perception:** CLIP ViT-B/32 with learned prompts for three quality levels.
- **Semantic guidance:** a frozen DINOv2 encoder supplies multi-level semantic features to the restoration network.
- **Degradation representation:** the Compact Feature Extractor learns degradation features with a contrastive objective.
- **Quality-aware learning:** CLIP guidance and a difficulty-adaptive VGG16 perceptual loss supervise restoration. The provided training configurations use multiple frozen proxy models.

The four shell entry points cover first-stage training, triplet preparation, restoration training, and evaluation. Additional diagnostic and rendering tools are available under `scripts/`.

## Acknowledgments

We thank the authors of [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), [Restormer](https://github.com/swz30/Restormer), [PromptIR](https://github.com/va1shn9v/PromptIR), [InstructIR](https://github.com/mv-lab/InstructIR), [FSNet](https://github.com/c-yn/FSNet), [MambaIR](https://github.com/csguoh/MambaIR), and [AdaIR](https://github.com/c-yn/AdaIR) for their publicly available resources. Optional specialist proxies use [NAFNet](https://github.com/megvii-research/NAFNet) and [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer).

## :book: Citation

If you use Perceive-IR in your research, please cite the paper. The entry below uses the final journal volume and page metadata associated with the [DOI](https://doi.org/10.1109/TIP.2025.3566300).

```bibtex
@article{zhang2026perceiveir,
  title   = {Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration},
  author  = {Zhang, Xu and Ma, Jiaqi and Wang, Guoli and Zhang, Qian and Zhang, Huan and Zhang, Lefei},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  volume  = {35},
  pages   = {2018--2033},
  doi     = {10.1109/TIP.2025.3566300}
}
```

## :postbox: Contact

For questions about Perceive-IR, please contact [Xu Zhang](mailto:zhangx0802@whu.edu.cn).
