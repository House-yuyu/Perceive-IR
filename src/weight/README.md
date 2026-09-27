# Pretrained Weights

Model weights are distributed separately through Baidu Netdisk and are not included in the source repository.

- **Shared folder:** Perceive-IR
- **Download:** [Baidu Netdisk](https://pan.baidu.com/s/1r04w7uSwXZ6V8bE1X9BbMQ?pwd=k4uc)
- **Extraction code:** `k4uc`

Place downloaded files under **`weight/`** at the repository root, following the paths below. Avoid an extra `weight/weight/` directory. Download metadata is also recorded in [manifest.json](manifest.json). Missing third-party assets can be obtained from the upstream sources listed here.

## Required Files

| Path relative to `weight/` | Used for | Source / format |
| --- | --- | --- |
| `perceiveIR_3task.pth` | Three-task inference | Released restoration checkpoint |
| `perceiveIR_5task.pth` | Five-task inference | Released restoration checkpoint |
| `dinov2-base/config.json` and `model.safetensors` | Restoration training and inference | Local model directory from [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base/tree/main) |
| `clip/ViT-B-32.pt` | Prompt learning and restoration training | [OpenAI CLIP](https://github.com/openai/CLIP), ViT-B/32 |
| `vgg16-397923af.pth` | Perceptual loss during training | [Torchvision VGG16](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.vgg16.html), [official weights](https://download.pytorch.org/models/vgg16-397923af.pth) |
| `stage1/quality_prompts_100000.pth` | Restoration training | Task-balanced quality prompts trained for 100K iterations |
| `stage1/holdout_0/medium_020000.pth` | Triplet rendering and dynamic denoising proxy | First-stage checkpoint with `heldout_fold: 0` |
| `stage1/holdout_1/medium_020000.pth` | Triplet rendering and dynamic denoising proxy | First-stage checkpoint with `heldout_fold: 1` |
| `proxies/promptir/model.ckpt` | Training proxy | [PromptIR](https://github.com/va1shn9v/PromptIR) all-in-one checkpoint containing a Lightning `state_dict` |
| `proxies/instructir/im_instructir-7d.pt` | Training proxy | [InstructIR](https://github.com/mv-lab/InstructIR) 7D image model |
| `proxies/instructir/task_embeddings.pt` | Three-task training | 256-dimensional vectors for denoising, dehazing, and deraining; a complete five-task vector file is also accepted |
| `proxies/instructir/task_embeddings_five.pt` | Five-task training | One 256-dimensional vector for each of the five tasks |
| `proxies/fsnet/ots.pkl` | Dehazing training proxy | [FSNet OTS](https://github.com/c-yn/FSNet/tree/main/Dehazing/OTS), checkpoint containing `model` |
| `proxies/mambair/denoise15.pth` | Denoising training proxy | [MambaIR v1](https://github.com/csguoh/MambaIR), renamed from `ColorDN_MambaIR_level15.pth` |
| `proxies/mambair/denoise25.pth` | Denoising training proxy | MambaIR v1 `ColorDN_MambaIR_level25.pth` |
| `proxies/mambair/denoise50.pth` | Denoising training proxy | MambaIR v1 `ColorDN_MambaIR_level50.pth` |
| `proxies/nafnet/NAFNet-GoPro-width32.pth` | Optional offline deblurring proxy | [NAFNet](https://github.com/megvii-research/NAFNet), width32 GoPro model |
| `proxies/retinexformer/LOL_v1.pth` | Optional offline low-light proxy | [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer), LOL-v1 model |

Evaluation needs only the selected restoration model and DINOv2. Training also uses the medium models, quality prompts, task embeddings, and proxy models. Proxy weights require their matching source repositories under `third_party/`; follow the [source setup guide](../third_party/README.md). Use the specified MambaIR v1 color-denoising models and their matching noise levels.

To regenerate InstructIR task vectors offline, also obtain `proxies/instructir/lm_instructir-7d.pt` and a complete local `proxies/instructir/bge-micro-v2/` directory containing the model configuration, weights, and tokenizer. These are available from [InstructIR](https://github.com/mv-lab/InstructIR) and [TaylorAI/bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2). They are unnecessary when using existing task vectors. The preparation tool is `python -m scripts.prepare_perceiveIR_instructir_embeddings`; use `--help` for its arguments.

## Check Downloaded Assets

```bash
python scripts/check_perceiveIR_assets.py --profile inference3
python scripts/check_perceiveIR_assets.py --profile inference5
python scripts/check_perceiveIR_assets.py --profile train
python scripts/check_perceiveIR_assets.py --profile specialist
python scripts/check_perceiveIR_assets.py --profile all --json
```

The `train` profile checks the files referenced by both restoration configurations, including both task-vector files. The `all` profile also includes optional assets. Missing or empty files produce exit code 1. This checks file presence, not tensor contents, datasets, or installed third-party dependencies.

## Checkpoint Format

Released restoration checkpoints must retain `model`, `config`, and `iteration`. Renaming a file does not change its state-dictionary keys. The training output `best_model.pth` can be used for inference; resuming training requires the full optimizer, scheduler, and training state. Medium checkpoints must also retain `heldout_fold`, and prompt checkpoints must retain their complete original training format.

Publish the SHA256 checksum, source, actual training iteration, and matching validation manifest for each checkpoint. Historical medium models require their original sample-to-fold mapping because fold assignment depends on image paths; see the [training instructions](../README.md#training).

Retain upstream attribution and license information when distributing third-party weights. Git ignore rules and the source packager exclude model files.
