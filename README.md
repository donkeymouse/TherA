<div align="center">

# TherA: Thermal-Aware Visual-Language Prompting for<br>Controllable RGB-to-Thermal Infrared Translation

[**Dong-Guw Lee**](https://scholar.google.com/citations?user=u6VDnlgAAAAJ&hl=ko)<sup>1*</sup>&emsp;
[**Tai Hyoung Rhee**](https://scholar.google.com/citations?user=PF8EfdYAAAAJ&hl=en&oi=ao)<sup>1*</sup>&emsp;
[**Hyunsoo Jang**](https://rpm.snu.ac.kr)<sup>1</sup><br>
[**Young-Sik Shin**](https://scholar.google.com/citations?user=gGfBRawAAAAJ&hl=en&oi=ao)<sup>2</sup>&emsp;
[**Ukcheol Shin**](https://scholar.google.com/citations?user=ZvxI80EAAAAJ&hl=ko&oi=ao)<sup>3</sup>&emsp;
[**Ayoung Kim**](https://ayoungk.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>Seoul National University&emsp;
<sup>2</sup>Kyungpook National University&emsp;
<sup>3</sup>KENTECH  
<sup>*</sup> Equal Contribution&emsp;
<sup>&dagger;</sup> Corresponding Author

**CVPR 2026**

[![Project Page](https://img.shields.io/badge/Project_Page-TherA-blue)](https://donkeymouse.github.io/thera_cvpr26/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.19430-b31b1b.svg)](https://arxiv.org/abs/2602.19430)
[![GitHub](https://img.shields.io/badge/GitHub-donkeymouse%2FTherA-black)](https://github.com/donkeymouse/TherA)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-yellow)](https://huggingface.co/donkeymouse/TherA/tree/main)
[![Dataset](https://img.shields.io/badge/HuggingFace-R2T2-orange)](https://huggingface.co/datasets/donkeymouse/TherA-R2T2)
[![Docker](https://img.shields.io/badge/Docker-donkeymouse%2Fthera-2496ED)](https://hub.docker.com/r/donkeymouse/thera)

</div>

<p align="center">
  <img src="assets/method.png" width="90%" alt="TherA method overview">
</p>


---

## News

- **2026-04-03**: TherA github repo opening
- **2026-05-22**: TherA inference code and R2T2 dataset release.

---

## Overview

**TherA** is a controllable RGB-to-thermal infrared translation framework. Given an RGB image, TherA synthesizes a long-wave thermal infrared image using a latent-diffusion translator conditioned on thermal-aware visual-language features.

TherA is designed for:

- **RGB → TIR translation** for thermal perception research.
- **Thermal-aware VLM conditioning** using LLaVA hidden-state features.
- **Scene- and object-level controllability** across weather, time of day, and object state.
- **Reference-cache inference**, allowing deployment without loading LLaVA at runtime.


<div align="center">
  <a href="https://www.youtube.com/watch?v=X60UxjGKQkg">
    <img src="https://img.youtube.com/vi/X60UxjGKQkg/0.jpg" alt="TherA demo video" width="720">
  </a>
</div>


---

## Key Idea

TherA does **not** condition directly on raw text during diffusion inference. Instead, it uses a **4096-dimensional LLaVA hidden state**, either:

1. loaded from a precomputed `.pt` reference cache, or
2. extracted on the fly using LLaVA.



For resource limited environments, we recommend **reference-cache mode**. This mode uses precomputed LLaVA features such as `SUNNY.pt`, `CLOUDY.pt`, `RAINY.pt`, or `NIGHT.pt`, and therefore does **not** require loading LLaVA weights at runtime. An alternative would be to compute pre-computed LLaVA feature first followed by inferencing with reference-cache mode (upcoming feature). 

---
## Repository Layout

```text
TherA/
├── infer_custom.py             # Batch RGB → TIR inference on a folder
├── infer_example_guided.py     # Single-image / example-guided inference
├── infer_palette.sh        # Run multiple weather/style palettes
├── lavi_ip2p/                  # UNet 8-channel + adapter wrapper
├── LaVi-Bridge/modules/        # TextAdapter architecture
├── llava/                      # LLaVA code, only needed for on-the-fly mode
├── thera_paths.py              # Default local weight paths
├── thera_llava.py              # Lazy LLaVA loader
└── weights/                    # Download weights here; not tracked by git
    ├── checkpoint/               # TherA Model (contains model.pt)
    ├── merged_models/          # Initialization model
    │   ├── unet/
    │   └── adapter/
    ├── stable-diffusion/
    │   ├── vae/
    │   └── scheduler/
    ├── reference_caches/
    │   ├── SUNNY.pt
    │   ├── CLOUDY.pt
    │   ├── RAINY.pt
    │   └── NIGHT.pt
    ├── llava-1.5-7b-hf/          # LLaVA base model (optional, for on-the-fly mode)
    └── TherA-VLM/                # LoRA adapter weights (optional, for on-the-fly mode)
        ├── adaptor_config.json
        ├── adapter_model.safetensors
        ├── config.json
        ├── non_lora_trainables.bin
        └── trainer_state.json
```

---

## Installation

### Option 1: Local Python Environment

```bash
git clone https://github.com/donkeymouse/TherA.git
cd TherA

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Recommended environment**

- Python 3.10+
- CUDA-capable GPU
- 16 GB+ VRAM recommended for comfortable inference

---

### Option 2: Docker

A prebuilt Docker image is available at:

```bash
docker pull donkeymouse/thera:latest
```

Example interactive run:

```bash
docker run --gpus all --rm -it \
  -v "$(pwd)":/workspace/TherA \
  -w /workspace/TherA \
  donkeymouse/thera:latest \
  bash
```

Then run inference commands from inside the container.

---

## Download Weights

TherA weights are hosted on Hugging Face:

```bash
pip install -U huggingface_hub

huggingface-cli download donkeymouse/TherA \
  --local-dir weights
```

After downloading, your `weights/` directory should contain:

| Path | Description | Required? |
|---|---|---|
| `weights/model.pt` | TherA trained UNet and adapter checkpoint | Yes |
| `weights/merged_models/unet/` | UNet architecture/config files | Yes |
| `weights/merged_models/adapter/` | TextAdapter architecture/config files | Yes |
| `weights/stable-diffusion/vae/` | Stable Diffusion VAE | Yes |
| `weights/stable-diffusion/scheduler/` | DDIM scheduler config | Yes |
| `weights/reference_caches/*.pt` | Precomputed LLaVA hidden states for inference palettes | Recommended |
| `weights/TherA-VLM/` | LLaVA weights for on-the-fly feature extraction | Optional |

---

## Optional: Download LLaVA Weights

LLaVA is only required for **on-the-fly feature extraction** or **two-image guided mode**. It is not required for reference-cache inference.

```bash
huggingface-cli download llava-hf/llava-1.5-7b-hf \
  --local-dir weights/llava-1.5-7b-hf
```


---

## Quick Start

---

## Full RGB-TIR translation using TherA-VLM

Use this mode if you want to extract hidden states from TherA directly at runtime from an RGB image and prompt.

```bash
python infer_custom.py \
  --rgb-dir examples/rgb \
  --output-dir preds \
  --llava-base-path weights/llava-1.5-7b-hf \
  --llava-lora-path weights/TherA-VLM \
  --llava-prompt "How would this RGB scene appear in long-wave thermal infrared spectrum."
```

This mode is more expensive because it loads LLaVA during inference.



### Reference-Guided Image Translation Mode

This mode extracts LLaVA features from a reference RGB image and applies them to a target RGB image.

```bash
python infer_example_guided.py \
  --mode two-image \
  --reference-image examples/ref/rgb.jpg \
  --input-image examples/rgb/scene.jpg \
  --output preds/scene_tir.png \
  --llava-base-path weights/llava-1.5-7b-hf \
  --llava-lora-path weights/TherA-VLM
```

---

### Recursive Folder Inference

```bash
python infer_custom.py \
  --rgb-dir /path/to/dataset/RGB \
  --output-dir preds \
  --reference-cache weights/reference_caches/SUNNY.pt \
  --recursive
```

When `--recursive` is used, the output folder preserves the input directory structure.

---
---

## Reference-cache Mode
Reference-cache mode is the recommended if you are lacking GPU memory. It does not load LLaVA at runtime.

```bash
python infer_custom.py \
  --rgb-dir examples/rgb \
  --output-dir preds/sunny \
  --reference-cache weights/reference_caches/SUNNY.pt
```

The script reads all images in `examples/rgb` and writes translated TIR images to `preds/sunny`.

A lighter version of the text-guided image translation module. 

Example palette caches:

```text
weights/reference_caches/SUNNY.pt
weights/reference_caches/CLOUDY.pt
weights/reference_caches/RAINY.pt
weights/reference_caches/NIGHT.pt
```

You can use different pallete cache to achieve different translation effects. 

---

## Inference Modes

| Mode | Main flag / script | LLaVA weights needed? | Recommended use |
|---|---|---:|---|
| Reference cache | `--reference-cache path.pt` | No | Default deployment and fast inference |
| Per-image cache directory | `--cache-dir dir/` | No | Precomputed feature per image |
| Full RGB-TIR translation| `--llava-base-path ...` | Yes | Runtime prompt/image conditioning |
| Reference image-guided translation| `infer_example_guided.py --mode two-image` | Yes | Apply reference-image conditioning |

---

## Reference Cache Format

Reference caches are precomputed LLaVA hidden states saved as `.pt` files.

Supported tensor shapes:

```text
[1, L, 4096]
[L, 4096]
```

A single reference cache can be applied to all input images as a global thermal/weather/style condition.

---

## Important Arguments

| Argument | Default | Description |
|---|---:|---|
| `--checkpoint` | `weights/checkpoint` | Directory containing `model.pt` |
| `--merged-model-path` | `weights/merged_models` | Directory containing UNet and adapter configs |
| `--pretrained-sd` | `weights/stable-diffusion` | Directory containing VAE and scheduler |
| `--rgb-dir` | Required | Folder of RGB images for batch inference |
| `--output-dir` | `custom_predictions` | Output folder for predictions |
| `--reference-cache` | `None` | Single `.pt` cache used for all images |
| `--cache-dir` | `None` | Folder of per-image `.pt` caches matched by filename stem |
| `--llava-base-path` | `None` | Base LLaVA model path for on-the-fly mode |
| `--llava-lora-path` | `None` | Optional LLaVA LoRA path |
| `--llava-prompt` | thermal prompt | Prompt used for default inference/text-guided translation|
| `--num-steps` | `100` | DDIM sampling steps |
| `--cfg-text` | `3.5` | Text/VLM guidance strength |
| `--cfg-image` | `1.5` | Image guidance strength |
| `--target-size` | Auto | Resize image to this square size; otherwise dimensions are rounded to multiples of 32 |
| `--recursive` | Off | Recursively process subdirectories |
| `--device` | `cuda` | Device for inference |

---

## Architecture

```text
RGB image
   │
   ▼
VAE encoder ──► RGB latents ───────────────────┐
                                                │
                                                ├──► 8-channel diffusion UNet ──► VAE decoder ──► TIR image
                                                │
LLaVA hidden state, 4096-d ──► TextAdapter ─────┘
                              768-d cross-attention tokens
```

TherA uses dual classifier-free guidance at inference by combining:

- full conditioning,
- image-only conditioning,
- text/VLM-only conditioning.

---

## R2T2 Dataset

TherA is trained with **R2T2**, a large-scale RGB–TIR–Text dataset.

R2T2 includes:

- **112,970 aligned triplets**: RGB image, TIR image, and canonical thermal schema.
- Scene diversity across driving, CCTV, aerial, and ego-view settings.
- Temporal diversity across day/night and diurnal transitions.
- Environmental diversity across weather, season, and illumination.
- Material- and object-level annotations with structured canonicalization.
- Data compiled from multiple aligned RGB–TIR datasets with additional pseudo-aligned pairs.

Dataset page:

```text
https://huggingface.co/datasets/donkeymouse/TherA-R2T2
```

Example structure:

```text
R2T2/
├── ${DATASET_NAME}/
│   └── ${SEQUENCE_NAME}/
│       ├── RGB/
│       │   ├── 1.jpg
│       │   └── ...
│       └── TIR/
│           ├── 1.jpg
│           └── ...
├── ViVID/
│   ├── img_campus_day1/
│   │   ├── RGB/
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── TIR/
│   │       ├── 000001.png
│   │       └── ...
│   └── ...
└── ...
```

---

## Troubleshooting

### `Checkpoint not found: weights/model.pt`

Download the TherA weights and make sure `model.pt` is located at:

```text
weights/model.pt
```

---

### `OSError: ... stable-diffusion/vae`

Make sure the Stable Diffusion VAE and scheduler folders are present:

```text
weights/stable-diffusion/vae/
weights/stable-diffusion/scheduler/
```

---

### Outputs look identical across palettes

Try increasing text/VLM guidance:

```bash
--cfg-text 7.5
```

Also verify that your reference cache files are distinct:

```text
SUNNY.pt
CLOUDY.pt
RAINY.pt
NIGHT.pt
```

---

### CUDA out of memory

Try reducing the image size:

```bash
--target-size 512
```

You can also run one image at a time with:

```bash
python infer_example_guided.py
```

---

### LLaVA import or loading errors

Use reference-cache mode if you do not need runtime LLaVA extraction:

```bash
--reference-cache weights/reference_caches/SUNNY.pt
```

For on-the-fly mode, make sure the LLaVA base model and TherA weights are correctly loaded`.

---

## TODOs
- [x] inference code and R2T2 dataset
- [] Upload cache extraction code
- [] Improve text-guidance 


## Citation

If you find TherA useful for your research, please cite:

```bibtex
@inproceedings{lee2026thera,
  title     = {TherA: Thermal-Aware Visual-Language Prompting for Controllable RGB-to-Thermal Infrared Translation},
  author    = {Lee, Dong-Guw and Rhee, Tai Hyoung and Jang, Hyunsoo and Shin, Young-Sik and Shin, Ukcheol and Kim, Ayoung},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

You may also cite the arXiv version:

```bibtex
@article{lee2026thera_arxiv,
  title   = {TherA: Thermal-Aware Visual-Language Prompting for Controllable RGB-to-Thermal Infrared Translation},
  author  = {Lee, Dong-Guw and Rhee, Tai Hyoung and Jang, Hyunsoo and Shin, Young-Sik and Shin, Ukcheol and Kim, Ayoung},
  journal = {arXiv preprint arXiv:2602.19430},
  year    = {2026}
}
```

---

## Acknowledgements

TherA builds on open-source components from the vision-language and diffusion communities, including LLaVA, Stable Diffusion, Diffusers, and LaVi-Bridge-style adapter architectures.

---

## License

See `LICENSE` for details.

Third-party models, datasets, and libraries retain their own licenses. Please review the licenses for LLaVA, Stable Diffusion, Hugging Face model files, and any external datasets before use.

## Contact
If you have any questions, contact here please
```
donkeymouse@snu.ac.kr
```
