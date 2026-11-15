# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""holom-core: Python and PyTorch framework for digital holography and legacy code audit."""
from . import cgh, cgh_methods, diffraction, filters, focus, gpu, legacy, simulate
from .diffraction import angular_spectrum, fresnel, fresnel_convolution

__version__ = "0.3.0"
__all__ = [
    "__version__",
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
]
