# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Package exports, version alignment, and optional-GPU import safety checks."""
import importlib
import re
import sys
from pathlib import Path

import holom


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"))
    assert match is not None
    assert holom.__version__ == match.group(1)


def test_top_level_exports_are_accessible():
    expected = (
        "cgh",
        "cgh_methods",
        "diffraction",
        "filters",
        "focus",
        "gpu",
        "legacy",
        "simulate",
        "angular_spectrum",
        "fresnel",
        "fresnel_convolution",
    )
    for name in expected:
        assert hasattr(holom, name)
        assert name in holom.__all__


def test_package_imports_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    reloaded_gpu = importlib.reload(holom.gpu)
    try:
        assert not reloaded_gpu.available()
    finally:
        monkeypatch.undo()
        importlib.reload(holom.gpu)
