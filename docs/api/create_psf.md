# API: create_psf.py

This module generates realistic radio interferometric **point spread functions (PSFs)** from telescope array configuration files. It reads antenna positions from CASA `.cfg` files and computes the corresponding (u, v) coverage and PSFs.

---

## Functions

### `load_cfg(filepath)`

Load antenna XYZ positions from a CASA `.cfg` configuration file.

**Parameters**

* `filepath` (*str*): Path to the configuration file.

**Returns**

* `np.ndarray`: Array of shape (N, 3) with antenna coordinates in meters.

**Notes**

* Lines beginning with `#` are ignored.
* Non-numeric lines are skipped automatically.

---

### `compute_uv_tracks(antpos, dec_deg, ha_hours, freq_hz=230e9)`

Compute **(u, v)** tracks for all baselines and hour angles.

**Parameters**

* `antpos` (*np.ndarray*): Antenna positions, shape (N, 3).
* `dec_deg` (*float*): Source declination in degrees.
* `ha_hours` (*array-like*): Hour angles in hours.
* `freq_hz` (*float*, optional): Observing frequency in Hz (default: 230 GHz).

**Returns**

* `(u, v)` (*tuple[np.ndarray, np.ndarray]*): Arrays of UV coordinates in wavelengths.

**Details**

* Computes baselines between all antenna pairs.
* Includes conjugate (negative) baselines for full symmetry.

---

### `uv_to_psf(u, v, grid_size=512)`

Generate a PSF from (u, v) coordinates.

**Parameters**

* `u, v` (*np.ndarray*): UV coordinates in wavelengths.
* `grid_size` (*int*, optional): Size of the PSF image grid (default: 512).

**Returns**

* `np.ndarray`: Normalized PSF array with a positive central peak.

**Notes**

* The UV grid is sampled and Fourier transformed to create the PSF.
* If the PSF has a stronger negative peak, it is inverted for consistency.

---

### `main()`

Generate PSFs for a set of telescope configurations.

**Process**

1. Loads ALMA array configurations from `/array_configurations/alma/`.
2. Iterates over declination and observing time bins.
3. Computes (u, v) coverage and PSF for each configuration.
4. Saves each PSF as a `.npy` file in `/psfs/`.

**Output Example**

```
psfs/alma.cycle6.3_dec-20_t4h.npy
psfs/alma.cycle7.5_dec+10_t6h.npy
```

---

## Example

```python
from utilities.create_psf import load_cfg, compute_uv_tracks, uv_to_psf

antpos = load_cfg('array_configurations/alma/alma.cycle6.3.cfg')
u, v = compute_uv_tracks(antpos, dec_deg=-20, ha_hours=range(-3, 4))
psf = uv_to_psf(u, v, grid_size=256)
```

---

## Notes

* The generated PSFs are used by the dataset loader to simulate dirty images.
* This process mimics real interferometric sampling in the UV plane.
* Supports ALMA, VLA, and ngVLA configuration files.

