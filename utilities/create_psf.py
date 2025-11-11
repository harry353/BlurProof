import os
import numpy as np
from tqdm import tqdm

LAT_ALMA = np.radians(-23.023)  # ALMA latitude [rad]


def load_cfg(filepath):
    """
    Load antenna positions from a CASA .cfg configuration file.

    Each .cfg file contains the XYZ coordinates of antennas in meters.
    Commented lines starting with '#' are ignored.

    Parameters
    ----------
    filepath : str
        Path to the CASA configuration file (.cfg).

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing antenna positions (X, Y, Z) in meters.
    """
    data = []
    with open(filepath) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                try:
                    x, y, z = map(float, line.split()[:3])
                    data.append([x, y, z])
                except ValueError:
                    continue
    return np.array(data)


def compute_uv_tracks(antpos, dec_deg, ha_hours, freq_hz=230e9):
    """
    Compute (u, v) tracks for all baselines and hour angles.

    Given antenna positions, declination, and observing hour angles,
    this function computes the sampled spatial frequency coordinates
    (u, v) in units of wavelengths for each baseline pair.

    Parameters
    ----------
    antpos : np.ndarray
        Array of antenna positions with shape (N, 3).
    dec_deg : float
        Source declination in degrees.
    ha_hours : array_like
        Hour angles in hours for which to compute uv coverage.
    freq_hz : float, optional
        Observing frequency in Hz (default is 230e9 for ALMA Band 6).

    Returns
    -------
    u_all, v_all : np.ndarray
        Arrays of sampled u and v coordinates in wavelengths.
        Both arrays have length equal to twice the number of computed baselines.
    """
    dec = np.radians(dec_deg)
    wavelength = 3e8 / freq_hz

    baselines = np.array([antpos[j] - antpos[i]
                          for i in range(len(antpos))
                          for j in range(i + 1, len(antpos))])

    u_all, v_all = [], []
    for ha_h in ha_hours:
        ha = np.radians(15 * ha_h)
        sinH, cosH = np.sin(ha), np.cos(ha)
        sinD, cosD = np.sin(dec), np.cos(dec)
        x, y, z = baselines[:, 0], baselines[:, 1], baselines[:, 2]
        u = sinH * x + cosH * y
        v = -sinD * cosH * x + sinD * sinH * y + cosD * z
        u_all.append(u / wavelength)
        v_all.append(v / wavelength)

    u_all = np.concatenate(u_all)
    v_all = np.concatenate(v_all)
    u_all = np.concatenate([u_all, -u_all])
    v_all = np.concatenate([v_all, -v_all])
    return u_all, v_all


def uv_to_psf(u, v, grid_size=512):
    """
    Generate a point spread function (PSF) from sampled uv-coordinates.

    This function constructs a binary uv-coverage map, then performs
    an inverse Fourier transform to obtain the corresponding dirty beam.

    Parameters
    ----------
    u, v : np.ndarray
        Arrays of u and v coordinates in wavelengths.
    grid_size : int, optional
        Size of the 2D uv-grid and resulting PSF image in pixels (default: 512).

    Returns
    -------
    np.ndarray
        2D array of shape (grid_size, grid_size) representing the normalized PSF.

    Notes
    -----
    - The PSF is normalized such that its maximum is +1.
    - If the negative lobe dominates, the PSF is sign-flipped to ensure
      the central peak is positive.
    """
    uv_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    center = grid_size // 2
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    scale = (grid_size / 2 - 1) / max(umax, vmax)
    ui = np.round(center + u * scale).astype(int)
    vi = np.round(center + v * scale).astype(int)
    mask = (ui >= 0) & (ui < grid_size) & (vi >= 0) & (vi < grid_size)
    uv_grid[vi[mask], ui[mask]] = 1.0

    psf = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))).real

    # Enforce positive peak convention
    if np.abs(psf.min()) > np.abs(psf.max()):
        psf = -psf

    psf /= np.max(psf)
    return psf


def main():
    """
    Generate a library of PSFs from ALMA configuration files.

    This script loops through all ALMA .cfg files and computes
    PSFs across a grid of declinations and observation durations.

    The resulting PSFs are stored as `.npy` arrays.

    Notes
    -----
    - Output directory: `/home/haris/Documents/BlurProof/psfs`
    - Input directory: `/home/haris/Documents/BlurProof/array_configurations/alma`
    - Default observing frequency: 230 GHz (ALMA Band 6)
    """
    alma_dir = "/home/haris/Documents/BlurProof/array_configurations/alma"
    output_dir = "/home/haris/Documents/BlurProof/psfs"
    os.makedirs(output_dir, exist_ok=True)

    cfg_files = sorted([f for f in os.listdir(alma_dir) if f.endswith(".cfg")])
    if len(cfg_files) == 0:
        raise FileNotFoundError("No ALMA cfg files found.")

    declinations = np.linspace(-60, 20, 10)   # degrees
    durations = np.linspace(2, 8, 7)         # hours
    step_min = 5.0
    freq_hz = 230e9

    total = len(cfg_files) * len(declinations) * len(durations)
    print(f"Generating ~{total} PSFs...")

    for cfg_file in tqdm(cfg_files, desc="Configurations"):
        antpos = load_cfg(os.path.join(alma_dir, cfg_file))
        for dec_deg in declinations:
            for obs_hours in durations:
                ha_hours = np.arange(-obs_hours / 2, obs_hours / 2, step_min / 60)
                u, v = compute_uv_tracks(antpos, dec_deg, ha_hours, freq_hz)
                psf = uv_to_psf(u, v, grid_size=256)
                outname = f"{os.path.splitext(cfg_file)[0]}_dec{dec_deg:+.0f}_t{obs_hours:.0f}h.npy"
                np.save(os.path.join(output_dir, outname), psf)

    print("All PSFs generated successfully.")


if __name__ == "__main__":
    main()

