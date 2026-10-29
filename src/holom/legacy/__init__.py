"""Faithful Python ports of every MATLAB routine in the author's archive.

These functions reproduce the original code line by line, including its
defects, so that published figures can be regenerated and audited. Each
docstring names the source file and lists the known defects; corrected
implementations live in ``holom.diffraction``, ``holom.filters`` and
``holom.cgh``. Interactive MATLAB steps (imrect, impixel, roipoly,
uigetfile) become explicit parameters.
"""
from . import cgh2015, fresnel_gom, jtc, matlab, reconstruction

__all__ = ["cgh2015", "fresnel_gom", "jtc", "matlab", "reconstruction"]
