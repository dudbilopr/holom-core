# holom-core (`v0.3.0`)

`holom-core` is an open-source Python and PyTorch library (`Apache-2.0`) for **scalar optical diffraction**, **computer-generated holography (CGH)**, **zero-order and twin-image suppression**, **multi-plane autofocus**, and **reproducible numerical auditing** of historical MATLAB digital holography codes developed in 2012–2015.

It pairs numerically with the browser-based 3-D virtual laboratory **[HoloLab10.easy](https://github.com/dudbilopr/hololab-10easy)**.

## Key Capabilities

- **Scalar Diffraction (`holom.diffraction`):** Single-FFT Fresnel propagation (`fresnel`), padded 3-FFT convolution (`fresnel_convolution`), and evanescent-filtered angular spectrum (`angular_spectrum`), verified against analytic Gaussian beam solutions and Parseval energy conservation.
- **Computer-Generated Holograms (`holom.cgh`, `holom.cgh_methods`):** Classical thesis encodings (`Burch`, `Kinoform`, `SquareWave`, `Fourier`, `FourierReal`, `FourierImag`, `Cuaternario`, `CrossedSquareWave`) alongside iterative Gerchberg–Saxton, weighted GS optical tweezers, Lee, Lohmann detour phase, double-phase, vortex/OAM fork CGH, 3-D point-cloud and layer CGH, and DRPE optical encryption.
- **Zero-Order Suppression & Autofocus (`holom.filters`, `holom.focus`):** Mean subtraction, FFT high-pass, discrete Laplacian, 4-step phase shifting, off-axis sideband spatial filtering, common-grid stack resampling (`fresnel_stack`), ROI Tenengrad/Tamura autofocus, and per-pixel `depth_from_focus`.
- **Optional PyTorch/CUDA Acceleration (`holom.gpu`):** `FresnelPlan` precomputes chirps in `float64` before executing `complex64` reconstructions at `61 fps` end-to-end on full `2592 × 1944` CMOS frames (NVIDIA GeForce GTX 1650 benchmark).
- **Isolated Legacy Audit (`holom.legacy`):** Faithful line-by-line ports of 2012–2015 MATLAB archives with defects pinned in unit tests for scientific traceability.

## Quickstart

```bash
pip install holom-core
# Optional GPU support:
pip install "holom-core[gpu]"
```

```python
import numpy as np
from holom import fresnel, filters
from holom.simulate import THESIS_2015 as P

hologram = np.asarray(...)
filtered = filters.subtract_mean(hologram)
field, dxi, deta = fresnel(filtered, P["wavelength"], P["dx"], P["dy"], -0.12)
intensity = np.abs(field) ** 2
```
