# Dataset Preparation

This repository does not distribute datasets. Download them from the sources below and follow their original terms of use. The download entries follow the [AdaIR dataset guide](https://github.com/c-yn/AdaIR/blob/main/INSTALL.md); organize the extracted files according to the layout used by this project.

| Task | Training data | Test data |
| --- | --- | --- |
| Denoising | [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view), [WED](https://drive.google.com/file/d/1e62XGdi5c6IbvkZ70LFq0KLRhFvih7US/view) | [CBSD68 color images](https://github.com/clausmichele/CBSD68-dataset/tree/master/CBSD68/original), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u), [Kodak24](https://r0k.us/graphics/kodak/) |
| Deraining | [Train100L / Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f) training split | Rain100L test split from the same download |
| Dehazing | [RESIDE-OTS](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) | [SOTS-Outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) |
| Deblurring | [GoPro](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view) training split | GoPro test split from the same download |
| Low-light enhancement | [LOL-v1](https://daooshee.github.io/BMVC2018website/) training split | LOL-v1 test split from the same release |

BSD400 and WED contain clean images. The code synthesizes Gaussian noise with standard deviations of 15, 25, and 50 on the 0-255 intensity scale. Dehazing uses OTS for training and SOTS-Outdoor for testing. If a download link changes, consult the original release page or the AdaIR guide for an updated entry.

## Directory Layout

```text
data/AiOIR/
|-- train/
|   |-- Denoise/gt/             # BSD400 + WED
|   |-- Dehaze/input/           # OTS hazy images; nested shards are supported
|   |-- Dehaze/gt/              # OTS clear images
|   |-- Derain/input/           # rain-1.png, ...
|   |-- Derain/gt/              # norain-1.png, ...
|   |-- Deblur/input/           # GoPro blur
|   |-- Deblur/gt/              # GoPro sharp
|   |-- Enhance/input/          # LOL-v1 low
|   `-- Enhance/gt/             # LOL-v1 high / normal
`-- test/
    |-- denoise/bsd68/target/
    |-- denoise/urban100/target/
    |-- denoise/kodak24/target/
    |-- dehaze/input/
    |-- dehaze/target/
    |-- derain/Rain100L/input/
    |-- derain/Rain100L/target/
    |-- deblur/gopro/input/
    |-- deblur/gopro/target/
    |-- enhance/lol/input/
    `-- enhance/lol/target/
```

The default `data.root` is `data/AiOIR`. A nested `AiO/train` and `AiO/test` layout is also supported. Existing datasets can be connected through directory symlinks. Point the configuration to extracted images, not a directory containing only download archives. The legacy spelling `train/Denosie/gt` remains supported; use `Denoise` for new datasets.

## Pairing and Filenames

- Training rain images use `rain-X` paired with `norain-X`. Test rain images must have matching input/target stems, such as `1.png` on both sides.
- Hazy images are paired by the portion before the first underscore: `0001_0.8_0.2.jpg` matches `0001.png`.
- GoPro and LOL-v1 pairs use matching filename stems. When collecting GoPro frames from different sequences, add the same sequence prefix to both input and target filenames to avoid repeated stems such as `000001`.
- Target stems must be unique within each task, and paired images must have the same dimensions. Extensions may differ. Preserve the original test resolution and do not rename only one side of a pair.
- The first-stage pipeline reads all five training tasks. Three-task data alone supports three-task restoration training and testing, but not the complete first-stage pipeline.

## Evaluation and Validation Splits

The whole-image evaluator expects 68 CBSD68 images at three noise levels, 500 SOTS-Outdoor pairs, and 100 Rain100L pairs: 804 cases in total. The five-task setting adds 1111 GoPro pairs and 15 LOL-v1 pairs, giving 1930 cases. Other dataset variants, such as a 492-image SOTS subset, do not satisfy the current enumeration checks.

Training validation also uses Urban100 and Kodak24 and selects scenes from the test datasets. Keep the validation manifest with its checkpoint. Pass it to evaluation using `--validation-manifest path/to/manifest.json` to report additional results excluding those scenes. Without a manifest, the exclusion statistic is marked unavailable. Regenerate training and triplet manifests when dataset paths change, and follow the fold requirements in the [training instructions](../README.md#training).
