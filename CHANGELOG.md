# Changelog

All notable changes to `holom-core` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-15

### Added
- Official `Apache-2.0` `LICENSE`, `NOTICE` (explicit clean-room differentiation from legacy `HOLOGOM` DNDA `13-75-287` and attribution for `holom.legacy.jtc`), and SPDX copyright headers across all Python modules, tests, and audit scripts.
- Scientific citation and archival metadata (`CITATION.cff` with ORCID `0000-0001-7239-3497`, `.zenodo.json`), repository governance (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`), and JOSS manuscript draft (`paper.md`, `paper.bib`).
- Top-level package exports for `cgh_methods`, `focus`, `gpu`, and `legacy` in `holom.__init__`.
- Standalone synthetic fallbacks in `audit/run_audit.py`, `audit/compare_zero_order_real.py`, `audit/depth_from_focus.py`, and graceful archive-absent handling in `audit/validate_legacy.py` so every script executes cleanly on fresh clones without local thesis archives.
- Expanded GitHub Actions CI matrix (Python `3.10`–`3.13`, `pytest --cov`, standalone audit smoke run, and `build`/`twine check`) plus OIDC PyPI Trusted Publishing workflow (`.github/workflows/publish.yml`).

### Changed
- Aligned package metadata version (`0.3.0`) across `pyproject.toml`, `holom/__init__.py`, `CITATION.cff`, and `README.md`, and completed `[project.urls]`, `classifiers`, `keywords`, and `[tool.ruff]` configuration.
- Updated `audit/run_audit.py` to dynamically report active PyTorch/CUDA GPU timing when available.

## [0.2.1] - 2026-10-29

### Added
- Real-hologram zero-order suppression comparison script (`audit/compare_zero_order_real.py`).
- Full reconstruction GPU vs. NumPy benchmark harness (`audit/benchmark_gpu.py`).

### Fixed
- Fixed edge resampling artifact in `holom.focus.depth_from_focus` using NaN-safe boundary padding and updated `tests/test_focus.py` (82 verified tests across CPU and CUDA).

## [0.2.0] - 2026-10-29

### Added
- Full `holom.legacy` migration (`cgh2015`, `fresnel_gom`, `jtc`, `matlab`, `reconstruction`).
- Literature CGH methods (`holom.cgh_methods`), multi-plane refocusing and ROI autofocus (`holom.focus`), and optional PyTorch/CUDA backend (`holom.gpu`).

## [0.1.0] - 2026-10-29

### Added
- Initial core propagators (`holom.diffraction`), 2012 CGH encodings (`holom.cgh`), zero-order suppression filters (`holom.filters`), synthetic off-axis hologram simulator (`holom.simulate`), initial audit (`audit/run_audit.py`), and GitHub Actions workflow.
