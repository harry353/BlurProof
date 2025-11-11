# Changelog

All notable changes to **BlurProof** will be documented in this file.

---

## [0.1.0] – 2025-11-10
### Added
- Initial public release.
- `train.py` – model training script with on-the-fly convolution.
- `utilities/create_psf.py` – PSF generator using ALMA, VLA, and ngVLA cfgs.
- `utilities/inspect_sample.py` – sample visualizer for clean / PSF / dirty triplets.
- `utilities/scale_images.py` – image rescaling and normalization tools.
- Added antenna configuration sets for ALMA, ngVLA, and VLA.

### Dataset
- Integrated Galaxy Zoo 2 images from Kaggle for clean training inputs.

---

## [Unreleased]
- Actually train the network.
- Add dataset download automation.
- Implement larger U-Net and residual models.

