# BlurProof Overview

**BlurProof** is a deep learning framework for *radio interferometric image deconvolution*.
It generates realistic dirty images by convolving clean galaxy images with physically accurate point spread functions (PSFs) derived from ALMA, VLA, and ngVLA array configurations.
The model learns to reconstruct the original clean image, providing a data-driven approach to deconvolution.

> **Note:** The model architecture, data preprocessing, and PSF generation pipeline are fully implemented.
> However, the model itself has **not been trained yet** — all scripts and configurations are in place and ready for training.

---

## Contents

1. [Installation](installation.md)
2. [Usage](usage.md)
3. [Dataset](dataset.md)
4. [Model Architecture](model.md)
5. [API Reference](api_reference.md)

---

## Scientific Context

In radio interferometry, incomplete sampling of the spatial frequency plane produces a *dirty image*:

[
I_\text{dirty} = I_\text{true} * B_\text{dirty}
]

where (B_\text{dirty}) is the point spread function.
Traditional algorithms such as **CLEAN** (Högbom 1974, Clark 1980) iteratively approximate (I_\text{true}).
**BlurProof** replaces this iterative process with a learned U-Net model trained on synthetic dirty–clean pairs.

---

## Quick Start

```bash
git clone https://github.com/harry353/BlurProof.git
cd BlurProof
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python utilities/create_psf.py
python train.py
```
