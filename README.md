# JiT-Flow-Matching-PyTorch
# JiT (Just image Transformers) - Consumer GPU Edition

A lightweight, modernized PyTorch implementation of He Kaiming's **JiT (Just image Transformers)** paper, adapted for consumer hardware (e.g., RTX 4060 Ti) and Windows environments.

Unlike the original paper which uses standard diffusion, this repo implements **Flow Matching (SiT style)** for better convergence and quality on smaller datasets like CIFAR-10.

## 🚀 Key Features

* **Consumer Hardware Friendly**: Tuned for **8GB VRAM** (Batch Size 64 + Gradient Accumulation).
* **Modern Architecture**: Includes `RoPE` (Rotary Positional Embeddings), `SwiGLU`, and `RMSNorm`.
* **Windows Optimized**: Solves common `torch.compile` / Triton errors on Windows.
* **Single File**: `train_4060.py` contains everything you need (Model, Data, Training Loop).
* **Flow Matching**: Uses velocity-based training target ($v$-prediction) instead of noise prediction.

## 📊 Performance & Requirements

Designed for consumer NVIDIA GPUs.

| VRAM | Batch Size | Accumulation Steps | Effective Batch | Status |
| :--- | :--- | :--- | :--- | :--- |
| **8GB** (Default) | 64 | 4 | 256 | ✅ Tested (Stable) |
| **16GB** (Unlock) | 128 | 2 | 256 | 🚀 Faster |
| **24GB+** | 256 | 1 | 256 | 🔥 Max Speed |

**Note**: The script defaults to 8GB settings to prevent OOM errors. 16GB users can edit `Config` to increase speed.


## ⚡ Quick Start

1.  Clone the repo:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/JiT-Flow-Matching-PyTorch.git](https://github.com/YOUR_USERNAME/JiT-Flow-Matching-PyTorch.git)
    cd JiT-Flow-Matching-PyTorch
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchvision tqdm einops triton-windows==3.4.0.post21
    ```

3.  Run training:
    ```bash
    python train_4060.py
    ```

## 📝 Windows vs Linux

The script automatically detects your OS:
* **Linux**: Enables `num_workers=4` for faster data loading.
* **Windows**: Disables `torch.compile` (to avoid Triton errors) and sets `num_workers=0` for stability.

## 📊 Performance

On an **RTX 4060 Ti (16GB)**:
* Speed: ~9-10 it/s
* VRAM Usage: ~6-7 GB
* Precision: BFloat16 (Mixed Precision)

## 📄 References

* [Back to Basics: Let Denoising Generative Models Denoise (JiT)](https://arxiv.org/abs/2511.13720)
* [Scalable Interpolant Transformers (SiT)](https://arxiv.org/abs/2401.08740)
