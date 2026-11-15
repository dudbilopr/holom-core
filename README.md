# holom-core

[![CI](https://github.com/dudbilopr/holom-core/actions/workflows/ci.yml/badge.svg)](https://github.com/dudbilopr/holom-core/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1459117818.svg)](https://doi.org/10.5281/zenodo.24255654)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](pyproject.toml)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7239--3497-A6CE39.svg)](https://orcid.org/0000-0001-7239-3497)

Clean-room Python and PyTorch framework for scalar diffraction, computer-generated holography (CGH), zero-order suppression, multi-plane autofocus, and reproducible numerical auditing of the legacy MATLAB digital holography codes developed in:

- **HGC_GOM_V2012** (BSc thesis, Universidad de Pamplona, 2012): eight computer-generated hologram encodings;
- **HOLO_GOM / HDFresnel_GOM_V1** (MSc thesis, Universidad de Pamplona, 2015): Fresnel reconstruction and zero-order suppression.

Original legacy MATLAB sources are archived separately and not redistributed in this repository.

## Modules

| Module | Contents |
|---|---|
| `holom.diffraction` | `fresnel` (single FFT), `fresnel_convolution` (3 FFTs, optional zero padding), `angular_spectrum`, `fresnel_legacy` (audit reference) |
| `holom.cgh` | `Burch`, `Kinoform`, `SquareWave`, `Fourier`, `FourierReal`, `FourierImag`, `Cuaternario`, `CrossedSquareWave` |
| `holom.cgh_methods` | Gerchberg-Saxton, weighted GS (optical tweezers), Burch, Lee, Lohmann, double phase, fork/LG (OAM), point-cloud and layer 3-D CGH, DRPE |
| `holom.filters` | Mean subtraction, FFT high-pass, Laplacian, 4-step phase shifting, off-axis sideband filter (Cuche et al.) |
| `holom.focus` | Multi-plane refocusing on a common physical grid (`fresnel_stack`), ROI autofocus (`Tenengrad`, `Tamura`), `depth_from_focus` |
| `holom.gpu` | `FresnelPlan` (precomputed chirps, PyTorch/CUDA), `angular_spectrum_torch` (optional `.[gpu]` extra) |
| `holom.legacy` | Faithful ports of the 2012–2015 MATLAB archive (`cgh2015`, `fresnel_gom`, `jtc`, `matlab`, `reconstruction`) with defects documented for auditing |
| `holom.simulate` | Off-axis Fresnel holograms with documented recording parameters (`THESIS_2015`) |

## Installation

```bash
git clone https://github.com/dudbilopr/holom-core.git
cd holom-core
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

To enable GPU acceleration with PyTorch/CUDA:

```bash
pip install -e ".[dev,gpu]"
```

## Verification & Standalone Numerical Audits

```bash
pytest -v
python audit/run_audit.py
python audit/compare_zero_order_real.py
python audit/depth_from_focus.py
python audit/benchmark_gpu.py
```

All audit scripts run out-of-the-box using deterministic synthetic fallbacks when the local 2012/2015 archival images are not mounted, and automatically score against the archival captures when present.

## Quick Example

```python
import numpy as np
from holom import fresnel, filters
from holom.simulate import THESIS_2015 as P

holo = np.asarray(...)                       # CMOS hologram, 2.2 um pixels
h = filters.subtract_mean(holo)
field, dxi, deta = fresnel(h, P["wavelength"], P["dx"], P["dy"], -0.12)
image = np.abs(field) ** 2                   # pixel pitch dxi = lambda z / (N dx)
```

## Versioning, Governance & Publication Readiness

- **Current Version:** `0.3.0`
- **License:** [Apache License 2.0](LICENSE) and [NOTICE](NOTICE)
- **Citation Metadata:** [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json) (ready for Zenodo DOI minting on GitHub Release)
- **CI / CD:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `pytest` across Linux, Windows, and macOS (Python `3.10`–`3.13`), static checks, standalone audit verification, and `sdist`/`wheel` validation (`twine check`); [`.github/workflows/publish.yml`](.github/workflows/publish.yml) supports OIDC Trusted Publishing to PyPI.
- **Interactive Companion:** [HoloLab10.easy](https://github.com/dudbilopr/hololab-10easy) renders live 3-D optical setups (`Mach-Zehnder`, `2F SLM`, `JTC`) aligned with the equations implemented in `holom-core`.

## Intellectual Property & Attribution

- `holom-core` is an original 2026 clean-room Python/PyTorch implementation written by Dudbil Olvasada Pabón Riaño (`ORCID: 0000-0001-7239-3497`). It is legally distinct from the historical MATLAB software **`HOLOGOM`** registered at the Colombian Copyright Office (DNDA No. `13-75-287`, 2019), whose proprietary files are not redistributed here.
- `holom.legacy.jtc` preserves faithful audit ports of the optical Joint Transform Correlator (JTC) encryption routines originally authored in MATLAB by Prof. Jorge E. Rueda-Parada (Universidad de Pamplona).

## How to Cite

Please use GitHub's **"Cite this repository"** sidebar or [`CITATION.cff`](CITATION.cff) when referencing `holom-core`.
