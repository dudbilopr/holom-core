# Contributing to `holom-core`

Contributions, bug reports, physical conservation checks, and documentation improvements are welcome.

## Development Setup

```bash
git clone https://github.com/dudbilopr/holom-core.git
cd holom-core
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -e ".[dev]"
```

If you have a CUDA-capable GPU and want to run GPU tests and benchmarks:

```bash
pip install -e ".[dev,gpu]"
```

## Running Tests & Audit Scripts

```bash
pytest -v
python audit/run_audit.py
python audit/benchmark_gpu.py
```

## Guidelines

1. **Physical rigor & numerical verification:** Every new propagator, CGH encoding, or filter must include unit tests checking analytic limits, Parseval energy conservation, or round-trip invertibility where physically applicable.
2. **Preserve `holom.legacy` fidelity:** Functions under `src/holom/legacy/` exist specifically as faithful ports of the 2012–2015 MATLAB routines (including known defects documented in docstrings and `tests/test_legacy.py`). Do not silently alter legacy behavior; implement corrected physics in `holom.diffraction`, `holom.filters`, or `holom.cgh_methods`.
3. **Licensing & SPDX headers:** All contributions are licensed under **Apache-2.0**. New `.py` files must include:
   ```python
   # SPDX-License-Identifier: Apache-2.0
   ```
4. **Commits:** Follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `perf:`).

## Opening Pull Requests

- Keep changes focused and run `pytest -v` before pushing.
- If proposing a literature algorithm, cite the primary peer-reviewed paper and DOI in the function docstring.
