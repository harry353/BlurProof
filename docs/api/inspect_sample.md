# API: inspect_sample.py

This module allows visualization of simulated **dirty images** produced by the convolution of clean astronomical images with telescope PSFs. It is mainly used for qualitative inspection and debugging.

---

## Classes

### `OnTheFlyDeconvDataset`

Dataset class that dynamically generates dirty–clean pairs during data loading.

**Parameters**

* `clean_dir` (*str*): Directory containing clean galaxy images.
* `psf_dir` (*str*): Directory containing PSF `.npy` files.
* `n_samples` (*int*, optional): Number of virtual samples to generate (default: 100000).

**Attributes**

* `clean_paths` (*list[str]*): Paths to clean images.
* `psf_paths` (*list[str]*): Paths to PSF files.
* `n_samples` (*int*): Total number of synthetic samples.

**Methods**

#### `__getitem__(idx)`

Randomly selects a clean image and PSF, performs FFT convolution, and returns the resulting triplet.

**Returns**

* `tuple[np.ndarray, np.ndarray, np.ndarray]`: (clean image, PSF, dirty image)

#### `__len__()`

Returns the total number of virtual samples (`n_samples`).

---

## Functions

### `inspect_sample(clean_dir, psf_dir)`

Display a random clean–PSF–dirty image triplet.

**Parameters**

* `clean_dir` (*str*): Path to the directory with clean images.
* `psf_dir` (*str*): Path to the directory with PSF `.npy` files.

**Behavior**

* Continuously generates and displays new samples in a `while True` loop.
* Each sample is displayed as a three-panel Matplotlib figure:

  * **Left:** Clean image
  * **Middle:** PSF (normalized)
  * **Right:** Dirty image (convolved)

**Usage Example**

```bash
python utilities/inspect_sample.py
```

or in Python:

```python
from utilities.inspect_sample import inspect_sample
inspect_sample('/path/to/clean_images', '/path/to/psfs')
```

---

## Notes

* PSFs are automatically normalized and flipped to ensure positive peaks.
* Useful for confirming dataset integrity and PSF diversity before training.
* Displays continue until manually closed by the user.

