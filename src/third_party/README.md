# Third-Party Model Sources

Restoration evaluation does not require these source repositories. Multi-proxy training uses the first four projects below, while optional specialist rendering uses the last two. Keep source code under `third_party/` and weights under `weight/proxies/`. Third-party code is not bundled with this repository.

```bash
git clone https://github.com/va1shn9v/PromptIR.git third_party/promptir
git clone https://github.com/mv-lab/InstructIR.git third_party/instructir
git clone https://github.com/c-yn/FSNet.git third_party/fsnet
git clone https://github.com/csguoh/MambaIR.git third_party/mambair
# Optional offline specialist proxies:
git clone https://github.com/megvii-research/NAFNet.git third_party/nafnet
git clone https://github.com/caiyuanhao1998/Retinexformer.git third_party/retinexformer
```

| Project | Source loaded by this repository | Setup notes |
| --- | --- | --- |
| [PromptIR](https://github.com/va1shn9v/PromptIR) | `net/model.py` | Install the model dependencies described by the upstream project |
| [InstructIR](https://github.com/mv-lab/InstructIR) | `models/instructir.py`; optional `text/models.py` | 7D image model with 256-dimensional task vectors |
| [FSNet](https://github.com/c-yn/FSNet) | `Dehazing/OTS/models/FSNet.py` | Use the OTS dehazing model |
| [MambaIR](https://github.com/csguoh/MambaIR) | `basicsr/archs/mambair_arch.py` | Use v1; install upstream dependencies and CUDA-compatible `mamba_ssm` / `causal_conv1d` builds |
| [NAFNet](https://github.com/megvii-research/NAFNet) | `basicsr/models/archs/NAFNet_arch.py` | width32 GoPro model; run in a separate rendering process |
| [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) | `basicsr/models/archs/RetinexFormer_arch.py` | LOL-v1 model; run in a separate rendering process |

Install the project's shared dependencies from the root `requirements.txt`. Upstream repositories may have different historical environments; install their additional model dependencies without replacing this project's entire environment. MambaIR and NAFNet use their own `basicsr` source trees, so an unrelated installed package with the same name can load the wrong architecture.

Once the sources and weights are ready, check the assets and run the GPU proxy smoke test:

```bash
python scripts/check_perceiveIR_assets.py --profile train
python -m scripts.smoke_perceiveIR_proxy_all --config configs/perceiveIR_3task.yaml --output results/proxy_smoke.json
```

Record the actual source commits alongside released weights. The original commit list for the main proxy models is not included here, so compatibility with arbitrary upstream revisions is unverified. The historical specialist tools recorded NAFNet commit `2b4af71ebe098a92a75910c233a3965a3e93ede4` and Retinexformer commit `1e9a0efce4b306b6701b824768370ff26066c32a`; these can be checked out in their respective clones.

Keep each project's original licenses, citations, and attribution with its source and weights.
