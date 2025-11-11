# API: train.py

This module defines the main **training pipeline** for the BlurProof model, including the dataset, U-Net architecture, and training routine. It dynamically generates dirty images by convolving clean inputs with PSFs at each iteration.

---

## Classes

### `OnTheFlyDeconvDataset`

Dataset class that generates dirty–clean pairs dynamically at load time.

**Parameters**

* `clean_dir` (*str*): Directory containing clean images.
* `psf_dir` (*str*): Directory containing PSFs (`.npy` files).
* `n_samples` (*int*, optional): Number of synthetic samples to produce (default: 100000).

**Methods**

* `__getitem__(idx)`: Randomly selects a clean image and PSF, performs FFT convolution, and returns `(dirty, clean)` tensors.
* `__len__()`: Returns the virtual dataset length.

**Returns**

* `tuple[torch.Tensor, torch.Tensor]`: Dirty and clean image tensors, normalized to [0, 1].

**Notes**

* PSFs are normalized and inverted if necessary to enforce positive peaks.
* Clean images are resized to 256×256 if not already.

---

### `UNet256`

Deep convolutional **U-Net** model optimized for 256×256 grayscale images.

**Parameters**

* `in_channels` (*int*, default=1): Number of input channels.
* `out_channels` (*int*, default=1): Number of output channels.
* `base_channels` (*int*, default=64): Number of filters in the first convolutional layer.

**Architecture Overview**

* Encoder: 4 downsampling blocks (Conv → BatchNorm → ReLU → Pool)
* Bottleneck: 1024 filters for deepest features
* Decoder: 4 upsampling blocks with skip connections
* Output: 1×1 convolution with **sigmoid** activation

**Forward Pass**

```python
out = model(dirty_image)
```

**Returns**

* `torch.Tensor`: Reconstructed clean image of shape (1, 256, 256).

---

## Functions

### `train(model, loader, epochs=20, lr=1e-3, device='cpu')`

Train the U-Net model using **MSE loss** with **PSNR** and **SSIM** metrics.

**Parameters**

* `model` (*torch.nn.Module*): The neural network to train.
* `loader` (*torch.utils.data.DataLoader*): DataLoader yielding `(dirty, clean)` pairs.
* `epochs` (*int*, optional): Number of training epochs (default: 20).
* `lr` (*float*, optional): Learning rate (default: 1e-3).
* `device` (*str*, optional): Target device (`'cpu'` or `'cuda'`).

**Behavior**

1. Uses **Adam optimizer** and **MSELoss**.
2. Computes PSNR and SSIM for each batch.
3. Saves model weights and loss curves to `checkpoints/`.

**Outputs**

* Model checkpoints: `unet_epochXX.pth`
* Metrics plot: `training_metrics.png`

---

### `main()`

Run the full training loop using the on-the-fly dataset.

**Process**

1. Initialize dataset and DataLoader.
2. Create a U-Net model instance.
3. Train the model for the specified number of epochs.

**Example**

```bash
python train.py
```

---

## Notes

* All convolutions use FFT to simulate the interferometric response.
* The model has not yet been trained — all code is ready for execution.
