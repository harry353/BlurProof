# Usage

This section explains how to use **BlurProof** to generate PSFs, inspect simulated data, and train the deconvolution network.

---

## 1. Generate PSFs

To generate telescope-specific point spread functions (PSFs):

```bash
python utilities/create_psf.py
```

This script reads array configuration files under `array_configurations/` (ALMA, VLA, ngVLA) and creates realistic PSFs in `/psfs`. Each PSF is labeled according to the array, declination, and observing duration.

Output example:

```
psfs/alma.cycle6.3_dec-20_t4h.npy
psfs/ngvla-core-revC_dec+10_t6h.npy
```

---

## 2. Inspect Random Samples

To visualize how a clean image is convolved with a PSF to form a dirty image:

```bash
python utilities/inspect_sample.py
```

This displays the clean image, PSF, and resulting dirty image side by side. A new random sample is shown each time you close the plot window.

---

## 3. Resize Input Dataset

To ensure consistent input sizes (256×256):

```bash
python utilities/scale_images.py
```

This resizes all images in `clean_images/` in place.

---

## 4. Train the Model

Once your data and PSFs are ready, start training:

```bash
python train.py
```

By default, training uses on-the-fly convolution of random clean images and PSFs, generating new dirty–clean pairs at every iteration. The model checkpoints and metric plots are stored under `/checkpoints`.

---

## 5. Outputs

After training, you will find:

```
checkpoints/
├── unet_epoch01.pth
├── unet_epoch02.pth
└── training_metrics.png
```

`training_metrics.png` includes loss, PSNR, and SSIM curves.

---

## 6. Notes

* The model has not been trained yet — all scripts are ready.
* You can replace the Galaxy Zoo dataset with other astronomical images.
* GPU acceleration is optional but recommended for faster training.

