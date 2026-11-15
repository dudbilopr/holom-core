# API Reference

Every top-level module can be imported from `holom`:

```python
import holom
from holom import cgh, cgh_methods, diffraction, filters, focus, gpu, legacy, simulate
```

## `holom.diffraction`
- `centered_coords(n, d)` — Centered spatial coordinate vector in meters (`n // 2` at `0`).
- `fresnel(u0, wavelength, dx, dy, z)` — Single-FFT Fresnel propagation returning `(field, dxi, deta)`.
- `fresnel_convolution(u0, wavelength, dx, dy, z, pad)` — Three-FFT convolution Fresnel propagation with optional zero padding.
- `angular_spectrum(u0, wavelength, dx, dy, z)` — Angular-spectrum propagation with evanescent frequency filtering.
- `critical_distance(n, dx, wavelength)` — Returns critical sampling distance $z_c = n \Delta x^2 / \lambda$.
- `fresnel_legacy(image, wavelength, dx, dy, z)` — Legacy audit reference matching `HOLO_GOM/Fresnel.m`.

## `holom.filters`
- `subtract_mean(hologram)` — Mean DC intensity subtraction.
- `highpass_fft(hologram, window)` — Rectangular Fourier high-pass filter.
- `laplacian(hologram)` — Discrete Laplacian high-pass filter.
- `phase_shifting_4step(i0, i1, i2, i3)` — Four-step phase extraction recovering $O R^*$.
- `sideband_filter(hologram, center, radius)` — Off-axis circular spectral bandpass filter.

## `holom.focus`
- `fresnel_stack(h, wavelength, dx, zs, block, z_ref)` — Block-averaged intensity stack resampled onto a common physical grid at `z_ref`.
- `tenengrad(I, smooth)` — Normalized gradient-energy focus metric.
- `tamura(I)` — Tamura contrast coefficient focus metric.
- `roi_around(center, half)` — Utility slice builder for rectangular ROIs.
- `autofocus(h, wavelength, dx, zs, roi, metric, block)` — ROI autofocus sweep over candidate distances `zs`.
- `local_focus_stack(field, wavelength, dx, zs, block, window, z_ref)` — Per-pixel local focus score volume across axial planes.
- `depth_from_focus(field, wavelength, dx, zs, block, window)` — Estimates per-pixel depth map, validity mask, and physical pixel pitch.

## `holom.gpu` *(requires `pip install "holom-core[gpu]"`)*
- `available()` — Returns `True` if `torch` is installed and CUDA is available.
- `FresnelPlan(shape, wavelength, dx, dy, z, device, dtype, subtract_mean)` — Precomputes input/output chirps on device (`__call__` returns $|U|^2$; `.field(hologram)` returns complex $U$).
- `angular_spectrum_torch(u0, wavelength, dx, dy, z, device, dtype)` — PyTorch angular-spectrum propagator.

## `holom.cgh` & `holom.cgh_methods`
- `holom.cgh`: `kinoform`, `burch`, `fourier_amplitude`, `fourier_real`, `fourier_imag`, `square_wave`, `crossed_square_wave`, `cuaternari`.
- `holom.cgh_methods`: `reconstruct_fourier`, `efficiency`, `gerchberg_saxton`, `weighted_gs_spots`, `burch`, `lee_delayed_sampling`, `lohmann_detour_phase`, `double_phase`, `fork_hologram`, `laguerre_gauss`, `point_cloud_cgh`, `layer_cgh`, `drpe_encrypt`, `drpe_decrypt`.

## `holom.simulate`
- `THESIS_2015` — Dictionary of the 2015 experimental setup parameters (`532 nm`, `2.2 um` pitch).
- `max_carrier_angle(wavelength, dx)` — Maximum Nyquist carrier angle $\arcsin(\lambda / 2\Delta x)$.
- `off_axis_hologram(...)`, `fresnel_off_axis_hologram(...)`, `phase_shifted_set(...)` — Ground-truth synthetic interferogram generators.

## `holom.legacy`
Archival 1:1 ports kept exclusively for reproducibility and defect verification: `holom.legacy.cgh2015`, `holom.legacy.fresnel_gom`, `holom.legacy.jtc`, `holom.legacy.matlab`, `holom.legacy.reconstruction`.
