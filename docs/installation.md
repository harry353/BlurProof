# Installation

This guide explains how to set up **BlurProof** and install all required dependencies.

---

## 1. Clone the Repository

Clone the official repository from GitHub:

```bash
git clone https://github.com/harry353/BlurProof.git
cd BlurProof
```

---

## 2. Create a Virtual Environment

To keep dependencies isolated, it’s recommended to use a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# OR (Windows)
.venv\Scripts\activate
```

---

## 3. Install Dependencies

Install all required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The following key packages will be installed:

* **PyTorch** – Deep learning framework used for model training.
* **NumPy**, **SciPy** – Core scientific computation libraries.
* **Matplotlib**, **tqdm** – Visualization and progress bars.
* **scikit-image**, **Pillow** – Image manipulation and processing.

---

## 4. Verify Installation

To confirm everything works correctly, run:

```bash
python -c "import torch, numpy, matplotlib, PIL; print('All good!')"
```

If no errors appear, your environment is ready.

---

## 5. Directory Structure

After installation, your project directory should look like this:

```
BlurProof/
├── train.py
├── utilities/
│   ├── create_psf.py
│   ├── inspect_sample.py
│   └── scale_images.py
├── array_configurations/
│   ├── alma/
│   ├── ngvla/
│   └── vla/
└── checkpoints/
```

---

## 6. (Optional but recommended) GPU Setup

If you have a CUDA-compatible GPU, install the corresponding PyTorch build:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This is not required — the model can also run entirely on CPU.

---

## 7. Ready for Use

Once installed, you can start generating PSFs and inspecting sample data:

```bash
python utilities/create_psf.py
python utilities/inspect_sample.py
```

