import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from skimage.transform import resize
from PIL import Image
from torch.utils.data import Dataset


# -------------------- DATASET --------------------
class OnTheFlyDeconvDataset(Dataset):
    """
    Dataset class that generates dirty images dynamically via convolution.

    Each sample consists of:
    - A clean image (from Galaxy Zoo or other source)
    - A PSF generated from an interferometer configuration
    - A dirty image obtained by convolving the clean image with the PSF

    This allows effectively unlimited data generation, since each call
    randomly pairs an image with a PSF.

    Parameters
    ----------
    clean_dir : str
        Directory containing clean images (.jpg, .png, or .npy).
    psf_dir : str
        Directory containing PSFs saved as `.npy` arrays.
    n_samples : int, optional
        Number of virtual samples to produce (default: 100000).

    Notes
    -----
    - All images are resized to 256×256 pixels.
    - PSFs are normalized and sign-corrected to ensure a positive main lobe.
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
        Generate one synthetic dirty–clean sample pair.

        Randomly selects a clean image and a PSF, performs convolution,
        and returns all three components.

        Parameters
        ----------
        idx : int
            Sample index (unused, random selection each call).

        Returns
        -------
        clean : np.ndarray
            The clean 2D image array, normalized to [0, 1].
        psf : np.ndarray
            The point spread function used for convolution.
        dirty : np.ndarray
            The resulting dirty image (normalized to [0, 1]).
        """
        clean_path = random.choice(self.clean_paths)
        psf_path = random.choice(self.psf_paths)

        # --- Load clean image ---
        if clean_path.lower().endswith(".npy"):
            clean = np.load(clean_path).astype(np.float32)
        else:
            clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0

        # --- Load and normalize PSF ---
        psf = np.load(psf_path).astype(np.float32)
        psf /= psf.sum() + 1e-8

        # --- Enforce positive peak convention ---
        if np.abs(psf.min()) > np.abs(psf.max()):
            psf = -psf

        # --- Resize clean image to 256x256 ---
        if clean.shape != (256, 256):
            clean = resize(clean, (256, 256), anti_aliasing=True)

        # --- Perform on-the-fly convolution ---
        dirty = fftconvolve(clean, psf, mode="same")
        dirty = dirty / dirty.max() if dirty.max() > 0 else dirty

        return clean, psf, dirty


# -------------------- VISUALIZATION --------------------
def inspect_sample(clean_dir, psf_dir):
    """
    Interactive visualization of generated (clean, PSF, dirty) triplets.

    Displays one randomly generated training sample at a time, updating
    continuously until the user closes the figure window.

    Parameters
    ----------
    clean_dir : str
        Directory containing clean input images.
    psf_dir : str
        Directory containing `.npy` PSF files.

    Notes
    -----
    - Each call generates a new random combination.
    - Closes and refreshes the window after each iteration.
    """
    dataset = OnTheFlyDeconvDataset(clean_dir, psf_dir, n_samples=1)

    while True:
        clean, psf, dirty = dataset[0]
        psf_display = psf / psf.max() if psf.max() > 0 else psf

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(clean, cmap="gray")
        axs[0].set_title("Clean Image")

        axs[1].imshow(psf_display, cmap="inferno")
        axs[1].set_title("PSF (normalized)")

        axs[2].imshow(dirty, cmap="gray")
        axs[2].set_title("Dirty Image (Clean * PSF)")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    clean_dir = "/home/haris/Documents/BlurProof/clean_images"
    psf_dir = "/home/haris/Documents/BlurProof/psfs"
    inspect_sample(clean_dir, psf_dir)

