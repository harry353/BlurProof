#!/usr/bin/env python3
import os
import random
import numpy as np
from scipy.signal import fftconvolve
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage.transform import resize
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------- PATHS --------------------
clean_dir = "/home/haris/Documents/BlurProof/clean_images"
psf_dir = "/home/haris/Documents/BlurProof/psfs"
checkpoint_dir = "/home/haris/Documents/BlurProof/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


# -------------------- DATASET --------------------
class OnTheFlyDeconvDataset(Dataset):
    """
    Dataset class that generates dirty images dynamically by convolving
    clean images with randomly selected PSFs.

    This produces effectively infinite training data by pairing random
    clean images with random PSFs at runtime.

    Parameters
    ----------
    clean_dir : str
        Directory containing clean images (.jpg, .png, or .npy).
    psf_dir : str
        Directory containing PSF files in `.npy` format.
    n_samples : int, optional
        Number of virtual samples to generate (default: 100000).

    Notes
    -----
    - Images are resized to 256×256.
    - PSFs are normalized and used for FFT-based convolution.
    """

    def __init__(self, clean_dir, psf_dir, n_samples=100000):
        self.clean_paths = sorted([
            os.path.join(clean_dir, f)
            for f in os.listdir(clean_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".npy"))
        ])
        self.psf_paths = sorted([
            os.path.join(psf_dir, f)
            for f in os.listdir(psf_dir)
            if f.lower().endswith(".npy")
        ])
        self.n_samples = n_samples
        print(f"Loaded {len(self.clean_paths)} clean images and {len(self.psf_paths)} PSFs.")

    def __len__(self):
        """Return the total number of virtual samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Generate a single (dirty, clean) training sample.

        Randomly selects a clean image and a PSF, performs FFT convolution,
        and returns the resulting tensors.

        Parameters
        ----------
        idx : int
            Dataset index (unused; random selection each call).

        Returns
        -------
        dirty : torch.Tensor
            The simulated dirty image tensor of shape (1, 256, 256).
        clean : torch.Tensor
            The original clean image tensor of shape (1, 256, 256).
        """
        clean_path = random.choice(self.clean_paths)
        psf_path = random.choice(self.psf_paths)

        # Load clean image
        if clean_path.lower().endswith(".npy"):
            clean = np.load(clean_path).astype(np.float32)
        else:
            clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0

        # Load and normalize PSF
        psf = np.load(psf_path).astype(np.float32)
        psf /= psf.sum() + 1e-8

        # Resize clean image to 256×256
        if clean.shape != (256, 256):
            clean = resize(clean, (256, 256), anti_aliasing=True)

        # Perform FFT convolution
        dirty = fftconvolve(clean, psf, mode="same")
        dirty = dirty / dirty.max() if dirty.max() > 0 else dirty

        return (
            torch.tensor(dirty[None], dtype=torch.float32),
            torch.tensor(clean[None], dtype=torch.float32),
        )


# -------------------- MODEL --------------------
class UNet256(nn.Module):
    """
    U-Net architecture for 256×256 grayscale image reconstruction.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default: 1).
    out_channels : int, optional
        Number of output channels (default: 1).
    base_channels : int, optional
        Number of filters in the first convolution layer (default: 64).

    Notes
    -----
    - Uses standard encoder-decoder with skip connections.
    - Each block includes BatchNorm and ReLU activations.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        self.enc1 = self.double_conv(in_channels, base_channels)
        self.enc2 = self.double_conv(base_channels, base_channels * 2)
        self.enc3 = self.double_conv(base_channels * 2, base_channels * 4)
        self.enc4 = self.double_conv(base_channels * 4, base_channels * 8)
        self.bottleneck = self.double_conv(base_channels * 8, base_channels * 16)
        self.up4 = self.up_block(base_channels * 16, base_channels * 8)
        self.up3 = self.up_block(base_channels * 8, base_channels * 4)
        self.up2 = self.up_block(base_channels * 4, base_channels * 2)
        self.up1 = self.up_block(base_channels * 2, base_channels)
        self.final = nn.Conv2d(base_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def double_conv(self, in_c, out_c):
        """Two consecutive 3×3 convolutions with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_c, out_c):
        """Upsampling block combining transpose convolution and double conv."""
        return nn.ModuleList([
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            self.double_conv(out_c * 2, out_c)
        ])

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 1, 256, 256).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 1, 256, 256) with sigmoid activation.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4[0](b)
        d4 = self.up4[1](torch.cat([d4, e4], dim=1))
        d3 = self.up3[0](d4)
        d3 = self.up3[1](torch.cat([d3, e3], dim=1))
        d2 = self.up2[0](d3)
        d2 = self.up2[1](torch.cat([d2, e2], dim=1))
        d1 = self.up1[0](d2)
        d1 = self.up1[1](torch.cat([d1, e1], dim=1))
        out = torch.sigmoid(self.final(d1))
        return out


# -------------------- TRAINING --------------------
def train(model, loader, epochs=20, lr=1e-3, device="cpu"):
    """
    Train the U-Net model on dynamically generated data.

    Computes MSE loss and logs two image quality metrics:
    PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    loader : DataLoader
        PyTorch DataLoader providing (dirty, clean) pairs.
    epochs : int, optional
        Number of training epochs (default: 20).
    lr : float, optional
        Learning rate for the Adam optimizer (default: 1e-3).
    device : str, optional
        Training device, e.g. 'cpu' or 'cuda' (default: 'cpu').

    Saves
    -----
    - Model weights per epoch (`checkpoints/unet_epochXX.pth`)
    - Metric plots as `training_metrics.png`
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.to(device)

    loss_hist, psnr_hist, ssim_hist = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_psnr, running_ssim = 0, 0, 0

        for dirty, clean in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            dirty, clean = dirty.to(device), clean.to(device)
            pred = model(dirty)
            loss = loss_fn(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            pred_np = pred.detach().cpu().numpy()
            clean_np = clean.detach().cpu().numpy()
            for i in range(pred_np.shape[0]):
                p, c = pred_np[i, 0], clean_np[i, 0]
                running_psnr += psnr_metric(c, p, data_range=1.0)
                running_ssim += ssim_metric(c, p, data_range=1.0)

        avg_loss = running_loss / len(loader)
        avg_psnr = running_psnr / len(loader.dataset)
        avg_ssim = running_ssim / len(loader.dataset)
        loss_hist.append(avg_loss)
        psnr_hist.append(avg_psnr)
        ssim_hist.append(avg_ssim)

        print(f"Epoch {epoch}: MSE={avg_loss:.6f}, PSNR={avg_psnr:.3f}, SSIM={avg_ssim:.3f}")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch{epoch:02d}.pth"))

    # Plot metrics
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, loss_hist, "o-")
    plt.xlabel("Epoch"); plt.ylabel("MSE")

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, psnr_hist, "o-", color="green")
    plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)")

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, ssim_hist, "o-", color="red")
    plt.xlabel("Epoch"); plt.ylabel("SSIM")

    plt.suptitle("Training Metrics (On-the-Fly Convolution)")
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "training_metrics.png"))
    plt.close()
    print("Training complete. Model and metric plots saved.")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    dataset = OnTheFlyDeconvDataset(clean_dir, psf_dir, n_samples=100000)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)
    model = UNet256()
    print(f"Dataset (virtual): {len(dataset)} samples")
    train(model, loader, epochs=20, lr=1e-3, device="cpu")

