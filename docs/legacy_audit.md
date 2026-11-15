# Legacy Code Audit & GPU Benchmarks

`holom-core` separates corrected physical propagators from faithful audit ports (`holom.legacy`) so historical thesis results can be regenerated, measured, and compared quantitatively.

## 1. Legacy Defects Identified & Pinned in Tests

| Script / Module | Historical Defect | Quantitative Finding (`audit/run_audit.py` / `audit/validate_legacy.py`) |
| :--- | :--- | :--- |
| `HoloKinoform.m` (`cgh.kinoform`) | Reconstructed kinoform with `ifft2(H)` treating normalized phase $H \in [0, 1]$ as amplitude instead of pure-phase transmittance $\exp(i 2\pi H)$. | Object window efficiency drops from `0.917` (physical phase) to `0.096` (amplitude decoding), leaving `0.750` of energy in zero order. |
| `HOLO_GOM/Fresnel.m` (`diffraction.fresnel_legacy`) | Omitted the global $\exp(ikz)/(i\lambda z)$ factor and input pixel area $\Delta x \Delta y$, and applied the forward chirp sign regardless of reconstruction direction. | Intensity correlation with corrected `fresnel` is `0.99966`, while signal-region phase error has standard deviation **`1.81 rad`**. |
| `PhaseShift.m` (`legacy.reconstruction.phase_shift`) | `atan(num / den)` division-by-zero produces `NaN` propagating across global FFT arrays. | Legacy MATLAB output yields all-zero / NaN image; corrected `phase_shifting_4step` matches `< 1e-10` numerically and scores `1.0000` 8-bit agreement vs archival outputs where non-NaN. |
| `DH_Reconstruccion.m` (convolución `5460×5460`) | Replicated identical 2-D kernel across RGB channels inside `fftn`. | Python legacy port reproduces the saved archival image with correlation **`0.9996`**. |

## 2. Single-Shot Zero-Order Suppression (`audit/compare_zero_order_real.py`)

Evaluated on four phase-shifted 2015 aeroplane holograms ($\lambda = 532\text{ nm}$, $\Delta x = 2.2\ \mu\text{m}$, $z = -0.23\text{ m}$) using the 4-step wave $O R^*$ as ground-truth reference:

| Method | Correlation inside Object ROI | Out-of-object Energy Fraction | Excess vs. Reference |
| :--- | :---: | :---: | :---: |
| Unfiltered ($I_1$) | `0.201` | `0.006` | `-0.040` |
| 7.3.1 Mean subtraction | `0.599` | `0.058` | `+0.012` |
| 7.3.2 FFT high-pass (`64×64`) | `0.786` | `0.086` | `+0.040` |
| 7.3.3 Laplacian | `0.737` | `0.482` | `+0.435` |
| Off-axis sideband (`r = 157 px`) | `0.697` | `0.003` | `-0.043` |
| 4-step phase shifting (reference) | `1.000` | `0.047` | `+0.000` |

## 3. End-to-End CPU vs. CUDA GPU Benchmark (`audit/benchmark_gpu.py`)

Measured across full camera-to-intensity reconstruction (`uint8` $\rightarrow$ mean subtraction $\rightarrow$ input chirp $\rightarrow$ FFT $\rightarrow$ output phase $\rightarrow |U|^2$) on Python `3.14.3`, NumPy `2.5.4`, PyTorch `2.14.1+cu126`, NVIDIA GeForce GTX 1650:

| Frame Size | NumPy `complex128` | NumPy `complex64` Plan | GPU End-to-End (H2D+D2H) | GPU On-Device | End-to-End FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `512 × 512` | `61.2 ms` | `22.0 ms` | `2.0 ms` | `0.92 ms` | `499 fps` |
| `1024 × 1024` | `367.8 ms` | `94.3 ms` | `4.2 ms` | `1.87 ms` | `237 fps` |
| `2048 × 2048` | `1920.1 ms` | `752.2 ms` | `16.8 ms` | `7.71 ms` | `60 fps` |
| `2592 × 1944` | `1865.4 ms` | `425.5 ms` | `16.4 ms` | `9.16 ms` | `61 fps` |

- Standalone `2592 × 2592` `fft2`: `616.84 ms` (NumPy CPU) vs **`3.06 ms`** (PyTorch CUDA `complex64`).
