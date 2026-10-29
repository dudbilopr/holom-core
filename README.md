# holom-core

Python port and numerical audit of the digital holography codes developed in

- **HGC_GOM_V2012** (BSc thesis, Universidad de Pamplona, 2012): eight computer-generated hologram encodings;
- **HOLO_GOM / HDFresnel_GOM_V1** (MSc thesis, 2015): Fresnel reconstruction and zero-order suppression.

The MATLAB sources in `../HOLO_GOM` and `../Tesis Pregrado/.../HGC_GOM_V2012` are left untouched.

## Modules

| Module | Contents |
|---|---|
| `holom.diffraction` | `fresnel` (single FFT), `fresnel_convolution` (3 FFTs, optional zero padding), `angular_spectrum`, `fresnel_legacy` (verbatim port of `Fresnel.m`) |
| `holom.cgh` | Burch, Kinoform, SquareWave, Fourier, FourierReal, FourierImag, Cuaternario, CrossedSquareWave |
| `holom.filters` | mean subtraction, FFT high-pass, Laplacian, 4-step phase shifting, sideband filter |
| `holom.simulate` | off-axis Fresnel holograms with the 2015 setup parameters (`THESIS_2015`) |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe audit/run_audit.py
```

## Example

```python
import numpy as np
from holom import fresnel, filters
from holom.simulate import THESIS_2015 as P

holo = np.asarray(...)                       # CMOS hologram, 2.2 um pixels
h = filters.subtract_mean(holo)
field, dxi, deta = fresnel(h, P["wavelength"], P["dx"], P["dy"], -0.12)
image = np.abs(field) ** 2                   # pixel pitch dxi = lambda z / (N dx)
```

Audit findings: `../Vault_Tesis_Holografia/07_Auditoria_Tecnica_Integral/03_Auditoria_Numerica_holom_core.md`.

## Versioning & Scientific Traceability

- **Current Release:** `v0.1.0` (Git Tag: `v0.1.0`).
- **Commit Convention:** Follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `perf:`).
- **CI / CD:** `.github/workflows/ci.yml` defines a test matrix (Ubuntu, Windows, macOS; Python 3.10–3.12). It will run once the repository is pushed to GitHub; so far the suite has been run locally only (Windows, Python 3.14).
- **Publication Target:** *SoftwareX (Elsevier)* / *Journal of Open Source Software (JOSS)*.

