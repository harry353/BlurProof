# Dataset

The **BlurProof** model uses real galaxy images as the basis for training synthetic dirty–clean pairs. The dataset provides clean astronomical images that are later convolved with simulated telescope PSFs to create realistic dirty images.

---

## 1. Source

The dataset used in this project is the **Galaxy Zoo 2** dataset from Kaggle:

> [https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images](https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)

It contains thousands of galaxy images sourced from the Sloan Digital Sky Survey (SDSS), each labeled by morphological type.

---

## 2. Download and Setup

1. Go to the Kaggle link above and download the dataset ZIP file.
2. Extract the images directly into the `clean_images/` directory:

```
BlurProof/
├── clean_images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── 000003.jpg
│   └── ...
```

3. (Optional) Resize all images to 256×256 pixels using:

```bash
python utilities/scale_images.py
```

This ensures uniform input size during training.

---

## 3. Data Preprocessing

During training, the dataset is not directly loaded into memory. Instead, the **OnTheFlyDeconvDataset** class:

* Randomly samples a clean image and a PSF at each iteration.
* Convolves them using FFT to generate a dirty image.
* Normalizes both to [0, 1].

This approach allows effectively unlimited data generation from a finite dataset.

---

## 4. Directory Structure

After setup, the project should include:

```
BlurProof/
├── clean_images/          # Galaxy Zoo images
├── psfs/                  # Generated PSFs
├── checkpoints/           # Model checkpoints
├── train.py               # Main training script
└── utilities/             # Data utilities
```

---

## 5. Alternative Datasets

While Galaxy Zoo 2 is used here, any astronomy image collection (e.g., HST, JWST, simulated sources) can be used — simply place them in the `clean_images/` directory.

All images are automatically normalized and resized by the dataset loader.

