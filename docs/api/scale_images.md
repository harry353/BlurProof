# API: scale_images.py

This module resizes all images in a specified directory to a target resolution (default: **256×256**). It ensures that all inputs are consistent for model training and convolution operations.

---

## Functions

### `resize_image(path)`

Resize a single image to the target size using high-quality interpolation.

**Parameters**

* `path` (*str*): Path to the image file.

**Returns**

* `bool` or `str`: Returns `True` if resizing succeeds, or an error message if it fails.

**Details**

* Converts the image to RGB.
* Uses **LANCZOS** interpolation for best visual quality.
* Overwrites the original file with the resized version.

---

### `main()`

Resizes all images in the specified directory in parallel using multiple CPU cores.

**Behavior**

1. Scans the target directory for valid image files (`.jpg`, `.jpeg`, `.png`).
2. Resizes each file using `resize_image()`.
3. Uses a **ProcessPoolExecutor** for parallel execution.
4. Displays progress with **tqdm**.

**Example**

```bash
python utilities/scale_images.py
```

**Output**

```
Resizing images: 100%|███████████████████████████| 250000/250000 [03:25<00:00]
All images resized to (256, 256) in /home/user/BlurProof/clean_images
```

---

## Notes

* Limits parallel workers to a maximum of 16 to avoid oversubscription.
* Overwrites files in place — keep a backup if needed.
* This script should be run once after downloading the Galaxy Zoo 2 dataset.

