import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

img_dir = "/home/haris/Documents/BlurProof/clean_images"
target_size = (256, 256)
valid_ext = (".jpg", ".jpeg", ".png")


def resize_image(path):
    """
    Resize a single image to the target resolution.

    Opens an image, converts it to RGB, resizes it to the specified
    target size using the LANCZOS filter, and overwrites it in place.

    Parameters
    ----------
    path : str
        Path to the input image file.

    Returns
    -------
    bool or str
        Returns True if resizing succeeded, otherwise returns an error message.

    Notes
    -----
    - The image is saved with quality 95 to preserve visual detail.
    - If the file cannot be opened or processed, an exception message is returned.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(path, quality=95)
        return True
    except Exception as e:
        return f"{path}: {e}"


def main():
    """
    Resize all images in the target directory to a uniform resolution.

    This function scans the directory for supported image formats,
    and resizes all matching files in parallel using multiple CPU cores.

    Parameters
    ----------
    None

    Notes
    -----
    - Input directory: `/home/haris/Documents/BlurProof/clean_images`
    - Target size: (256, 256)
    - Supported extensions: .jpg, .jpeg, .png
    - Uses a process pool for parallelization, limited to 16 workers maximum.
    """
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if f.lower().endswith(valid_ext)]
    n_workers = min(multiprocessing.cpu_count(), 16)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(resize_image, f): f for f in files}
        for _ in tqdm(as_completed(futures), total=len(files), desc="Resizing images"):
            pass

    print(f"All images resized to {target_size} in {img_dir}")


if __name__ == "__main__":
    main()

