# Deployment Playbook & Munich 2027 Conference Submission

## 1. Web & Academic Certification Checklist

Follow this exact sequence once logged into GitHub (`dudbilopr`):

### Step A — Push `holom-core` (`v0.3.0`) & Enable Zenodo + PyPI
```bash
cd C:\Users\dudbi\Downloads\Tesis\holom-core
# Create empty repository `holom-core` on github.com/dudbilopr, then:
git remote add origin https://github.com/dudbilopr/holom-core.git
git push -u origin main --tags
```
1. **GitHub Actions CI & Docs:** Under *Settings → Pages*, set Source to **GitHub Actions** so `.github/workflows/docs.yml` publishes to `https://dudbilopr.github.io/holom-core/`.
2. **Zenodo DOI:** Log into [zenodo.org](https://zenodo.org) with GitHub, toggle **`dudbilopr/holom-core` ON**, then publish GitHub Release **`v0.3.0`** from tag `v0.3.0`. Zenodo will read `.zenodo.json` and `CITATION.cff` and mint a permanent DOI (`10.5281/zenodo.xxxxxxx`). Paste that DOI badge into `README.md`.
3. **PyPI Trusted Publishing:** Log into [pypi.org](https://pypi.org) → *Publishing* → *Add a new pending publisher*:
   - PyPI Project Name: `holom-core`
   - Owner: `dudbilopr`
   - Repository: `holom-core`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
   Publishing any GitHub Release will automatically upload verified `.whl` and `.tar.gz` artifacts via OIDC.

### Step B — Deploy `HoloLab10.easy` (`hololab-10easy`)
```bash
cd C:\Users\dudbi\Downloads\Tesis\hololab-10easy-web
# Create empty repository `hololab-10easy` on github.com/dudbilopr, then:
git remote add origin https://github.com/dudbilopr/hololab-10easy.git
git push -u origin main
```
Under *Settings → Pages*, set Source to **GitHub Actions** (`.github/workflows/pages.yml`). Live URL: `https://dudbilopr.github.io/hololab-10easy/`.

---

## 2. Ready-to-Submit Abstract Draft: SPIE Optical Metrology 2027 (Munich)

- **Target Event:** SPIE Optical Metrology 2027 (21–24 June 2027, ICM Messe München, Germany)
- **Primary Conference Track:** *Modeling Aspects in Optical Metrology XI* (OM104) · *(Alternate: Holography: Advances and Modern Trends X)*
- **Abstract Deadline:** **3 February 2027** *(Notification: 29 March 2027 · Full Manuscript: 2 June 2027)*

> **Proposed Title:**  
> *Clean-Room Numerical Auditing, GPU Acceleration, and Browser Digital Twinning for Legacy Fresnel Holography Software*
>
> **Authors:**  
> Dudbil Olvasada Pabón Riaño *(ORCID: 0000-0001-7239-3497; Universidad Autónoma de Bucaramanga, Colombia)*
>
> **Abstract (~250 words):**  
> Legacy MATLAB scripts developed across academic digital holography theses frequently remain in circulation years after completion without regression testing against analytical wave-optics invariants. We present `holom-core`, an open-source clean-room Python and PyTorch framework (`Apache-2.0`), coupled with `HoloLab10.easy`, a browser-based WebGL/Three.js optical digital twin, designed to verify, benchmark, and modernize archival Fresnel diffraction and zero-order suppression routines recorded at $\lambda = 532\text{ nm}$ and $\Delta x = 2.2\ \mu\text{m}$.
>
> By pairing faithful line-by-line audit ports with physically corrected single-FFT Fresnel, convolution Fresnel, and evanescent-filtered angular-spectrum propagators, 85 automated unit checks isolate subtle historical defects that visual inspection overlooks. On a $1296 \times 972$ test field ($z = 0.12\text{ m}$), legacy Fresnel propagation retains $0.99966$ intensity correlation while introducing $1.81\text{ rad}$ standard deviation in phase error due to omitted quadratic normalization and conjugate sign errors; similarly, amplitude-based kinoform decoding drops object window efficiency to $0.096$ versus $0.917$ for physical phase exponentiation, and raw division in four-step phase-shifting collapses archival frames to `NaN`. Across experimental 4-step phase-shifted aeroplane holograms ($z = -0.23\text{ m}$), we quantify single-shot zero-order suppression performance relative to the extracted object wave ($O R^*$).
>
> To bridge modeling and real-time laboratory operation, `FresnelPlan` precomputes chirps in double precision and executes full camera-to-intensity `complex64` reconstructions on an NVIDIA GeForce GTX 1650 in $16.4\text{ ms}$ end-to-end ($61\text{ fps}$, $9.16\text{ ms}$ on-device) for $2592 \times 1944$ sensor frames—compared against $1865.4\text{ ms}$ in NumPy `complex128`. All verification scripts execute out-of-the-box with synthetic fallbacks alongside the interactive 3-D optical twin.

---

## 3. Preprint Strategy (Optica Open / arXiv)

1. Publish the technical software note first on **Optica Open** (`opticaopen.org`) — zero *endorsement* barrier, immediate citable DOI, native compatibility with Optica Publishing Group journals, and accepted as non-prior publication by SPIE, Optica, and JOSS.
2. In parallel, if you request a `physics.optics` **endorsement** from an academic colleague already endorsed on arXiv, mirror the identical LaTeX manuscript to `arXiv` (`physics.optics`) citing the Zenodo DOI of `holom-core v0.3.0`.
