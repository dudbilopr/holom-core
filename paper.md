---
title: 'holom-core: An open-source Python and PyTorch framework for digital holography and numerical verification of legacy optical software'
tags:
  - Python
  - digital holography
  - Fresnel diffraction
  - computer-generated holography
  - PyTorch
  - GPU acceleration
  - scientific software audit
authors:
  - name: Dudbil Olvasada Pabón Riaño
    orcid: 0000-0001-7239-3497
    corresponding: true
    affiliation: 1
affiliations:
  - name: Universidad Autónoma de Bucaramanga (UNAB), Bucaramanga, Colombia
    index: 1
date: 15 November 2026
bibliography: paper.bib
---

# Summary

`holom-core` is an open-source Python and PyTorch library for scalar diffraction propagation, computer-generated hologram (CGH) synthesis, zero-order suppression, multi-plane autofocus, and reproducible numerical auditing of historical MATLAB digital holography codes. Written as a clean-room implementation under Apache License 2.0, the package provides modular CPU (`NumPy`/`SciPy`) and optional CUDA GPU (`PyTorch`) backends together with 85 unit tests validating physical invariants—including analytic Gaussian beam propagation, Parseval energy conservation, forward-backward invertibility, and CPU-to-GPU numerical agreement—alongside standalone audit scripts that benchmark end-to-end reconstruction pipelines.

# Statement of need

Academic optics groups routinely accumulate thesis-era MATLAB scripts whose equations, boundary conditions, and numerical conventions are rarely re-verified against analytic solutions years after graduation. At the same time, modern research and teaching workflows—particularly when coupled with spatial light modulators (SLMs), neural reconstruction pipelines, or real-time 3-D virtual laboratories—require open, scriptable, and hardware-accelerated Python tooling with transparent numerical guarantees.

`holom-core` addresses both needs in a single reproducible codebase:

1. It separates **verified physical implementations** (`holom.diffraction`, `holom.cgh_methods`, `holom.filters`, `holom.focus`, `holom.gpu`, `holom.simulate`) from **faithful audit ports** (`holom.legacy`) of legacy 2012–2015 thesis routines, pinning known historical defects (such as missing phase normalization in single-FFT Fresnel kernels, amplitude vs. phase misuse in kinoform reconstruction, or NaN collapse in multi-step phase shifting) in automated regression tests rather than hiding them.
2. It provides an end-to-end **GPU acceleration plan** (`holom.gpu.FresnelPlan`) that precomputes single-FFT quadratic phase chirps in double precision before executing repeated `complex64` reconstructions on device.

# Architecture and implemented methods

The package is structured into eight interoperable submodules:

- **`holom.diffraction`**: Single-FFT Fresnel transform (`fresnel`), three-FFT convolution Fresnel propagator with optional zero-padding (`fresnel_convolution`), evanescent-filtered angular spectrum (`angular_spectrum`), and `fresnel_legacy` kept explicitly for numerical comparison.
- **`holom.cgh` and `holom.cgh_methods`**: Classical and iterative CGH encodings from the literature—Gerchberg–Saxton [@Gerchberg1972], single-pass kinoform [@Lesem1969], Burch [@Burch1967], Lee [@Lee1970], Brown–Lohmann detour phase [@Brown1966], double-phase encoding [@Hsueh1978], forked vortex/OAM holograms [@Heckenberg1992], weighted Gerchberg–Saxton optical tweezers [@DiLeonardo2007], 3-D point-cloud and layer-based CGH, and double random phase encryption (DRPE) [@Refregier1995]—alongside the eight 2012 thesis encodings.
- **`holom.filters` and `holom.focus`**: Zero-order suppression via mean subtraction, FFT high-pass filtering, discrete Laplacian, four-step phase shifting, and off-axis sideband filtering, plus `fresnel_stack` multi-plane resampling onto a common physical pixel pitch $\Delta\xi = \lambda z / (N \Delta x)$ for ROI autofocus (`tenengrad`, `tamura`) and depth-from-focus [@Dubois2006].
- **`holom.gpu`**: Optional `PyTorch` backend implementing `FresnelPlan` and `angular_spectrum_torch` with graceful fallback when `torch` is absent.
- **`holom.legacy` and `holom.simulate`**: Synthetic off-axis Fresnel recording generator (`THESIS_2015` optical parameters: $\lambda = 532\text{ nm}$, $\Delta x = 2.2\ \mu\text{m}$) and documented audit ports covering CGH, Fresnel reconstruction, and Joint Transform Correlator (JTC) optical encryption routines originally formulated by Prof. Jorge E. Rueda-Parada.

# Verification and benchmarks

All 85 automated tests in `tests/` pass on both CPU and CUDA backends (`pytest`). Standalone verification scripts in `audit/` quantify legacy behavior and full-pipeline performance without requiring proprietary external datasets:

- **Legacy audit (`audit/run_audit.py`):** Compares legacy `Fresnel.m` against corrected `fresnel` at $1296 \times 972$ ($\lambda = 532\text{ nm}$, $\Delta x = 2.2\ \mu\text{m}$, $z = 0.12\text{ m}$), showing high intensity correlation ($0.99966$) alongside $1.81\text{ rad}$ standard deviation in signal phase error, and contrasts improper amplitude kinoform decoding ($0.096$ window efficiency) against phase-only exponentiation ($0.917$).
- **End-to-end GPU reconstruction (`audit/benchmark_gpu.py`):** Measured on an Intel x86_64 CPU and NVIDIA GeForce GTX 1650 (Python 3.14.3, NumPy 2.5.4, PyTorch 2.14.1+cu126) for the complete camera-to-intensity pipeline (`uint8` frame $\rightarrow$ mean subtraction $\rightarrow$ input chirp $\rightarrow$ FFT $\rightarrow$ output phase $\rightarrow$ $|U|^2$):
  - $512 \times 512$: `61.2 ms` (`numpy128`), `22.0 ms` (`numpy64_plan`), `2.0 ms` GPU end-to-end (`499 fps`), `0.92 ms` on-device.
  - $1024 \times 1024$: `367.8 ms` (`numpy128`), `94.3 ms` (`numpy64_plan`), `4.2 ms` GPU end-to-end (`237 fps`), `1.87 ms` on-device.
  - $2048 \times 2048$: `1920.1 ms` (`numpy128`), `752.2 ms` (`numpy64_plan`), `16.8 ms` GPU end-to-end (`60 fps`), `7.71 ms` on-device.
  - $2592 \times 1944$ (full 5 MP sensor): `1865.4 ms` (`numpy128`), `425.5 ms` (`numpy64_plan`), `16.4 ms` GPU end-to-end (`61 fps`), `9.16 ms` on-device.

# Acknowledgements

The author thanks Prof. Jorge E. Rueda-Parada for supervising the 2012 and 2015 theses whose historical MATLAB scripts are preserved in archival audit context under `holom.legacy.jtc`.

# References
