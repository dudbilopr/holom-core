# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Multi-plane refocusing on a common physical grid and region-of-interest autofocus.

The single-FFT Fresnel transform returns a pixel pitch d_xi = lambda z / (N dx) that grows with z,
so planes at different z cannot be compared pixel by pixel. `fresnel_stack` resamples every plane
to the grid of a reference distance z_ref (coordinate scale z_ref / z) before any metric is taken.

Autofocus needs a region of interest that contains the real image only. Automatic detection was
tried and is not reliable for these holograms (the |O|^2 term is comparable to the reference and
the twin is larger than the real image), so the ROI is an explicit input: pick it on a
reconstruction near the expected distance (`roi_around`).

The focus metric reaches an extremum at the image plane (Dubois et al., Opt. Express 14, 5895,
2006, doi:10.1364/OE.14.005895). For the normalised metrics below it is a minimum, for diffuse and
for amplitude objects alike; tests/test_focus.py recovers the recording distance of simulated
off-axis Fresnel holograms of the 2015 set-up to within one 5 mm step.
"""
import numpy as np
from scipy import ndimage

from .diffraction import fresnel


def fresnel_stack(h, wavelength, dx, zs, block=4, z_ref=None):
    """Intensities |U(z)|^2 for every z, block-averaged (speckle reduction) and resampled to the
    physical grid of z_ref (default: middle of zs). Returns (stack, pitch_ref_m).
    Use negative z to back-propagate a recorded hologram (holom.fresnel convention)."""
    h = np.asarray(h)
    ny, nx = h.shape
    my, mx = ny // block, nx // block
    zs = np.asarray(zs, dtype=float)
    z_ref = float(z_ref if z_ref is not None else zs[len(zs) // 2])
    stack = np.empty((len(zs), my, mx))
    cy, cx = (my - 1) / 2, (mx - 1) / 2
    for i, z in enumerate(zs):
        u, _, _ = fresnel(h, wavelength, dx, dx, z)
        I = (np.abs(u[:my * block, :mx * block]) ** 2).reshape(my, block, mx, block).mean(axis=(1, 3))
        k = z_ref / z                                          # output pixel of z_ref, in pixels of z
        stack[i] = ndimage.affine_transform(I, [k, k], offset=[cy - k * cy, cx - k * cx], order=1)
    pitch = wavelength * abs(z_ref) / (nx * dx) * block
    return stack, pitch


def tenengrad(I, smooth=2.0):
    """Normalised gradient energy of the speckle-smoothed intensity."""
    Is = ndimage.gaussian_filter(I, smooth)
    g = ndimage.sobel(Is, 0) ** 2 + ndimage.sobel(Is, 1) ** 2
    return float(g.mean() / (Is.mean() ** 2 + 1e-30))


def tamura(I):
    """Tamura coefficient sqrt(std/mean) of the amplitude."""
    a = np.sqrt(np.maximum(I, 0))
    return float(np.sqrt(a.std() / (a.mean() + 1e-30)))


METRICS = {'tenengrad': tenengrad, 'tamura': tamura}


def roi_around(center, half):
    """ROI (rows, cols) of side 2*half around center = (row, col) on the analysis grid."""
    r, c = center
    return slice(max(0, r - half), r + half), slice(max(0, c - half), c + half)


def _native_roi(roi, k, shape):
    """Map a ROI given on the z_ref grid to the native grid of a plane at z (k = z_ref / z)."""
    out = []
    for sl, n in zip(roi, shape):
        c = (n - 1) / 2
        a, b = c + (sl.start - c) * k, c + (sl.stop - c) * k
        out.append(slice(int(round(max(0, a))), int(round(min(n, b)))))
    return tuple(out)


def autofocus(h, wavelength, dx, zs, roi, metric='tenengrad', block=4):
    """Best focus distance over the candidate distances zs (metres, sign as in holom.fresnel),
    measured inside roi, given as (rows, cols) on the grid of fresnel_stack(..., block) at the
    middle distance of zs.

    The metric is evaluated on each plane's native grid, with the ROI mapped to the same
    physical region: there the speckle grain is one pixel at every z (both equal lambda z/(N dx)),
    whereas resampling the images to one grid changes the speckle smoothing with z and biases
    the metric. Returns the curve, z_best, the analysis pitch, whether the minimum sits on the
    scan edge (then widen the scan) and the relative depth of the dip."""
    h = np.asarray(h)
    zs = np.asarray(zs, dtype=float)
    z_ref = float(zs[len(zs) // 2])
    ny, nx = h.shape
    my, mx = ny // block, nx // block
    f = METRICS[metric]
    curve = []
    for z in zs:
        u, _, _ = fresnel(h, wavelength, dx, dx, z)
        I = (np.abs(u[:my * block, :mx * block]) ** 2).reshape(my, block, mx, block).mean(axis=(1, 3))
        curve.append(f(I[_native_roi(roi, z_ref / z, (my, mx))]))
    curve = np.array(curve)
    pitch = wavelength * abs(z_ref) / (nx * dx) * block
    i = int(curve.argmin())
    low_edge = min(curve[0], curve[-1])
    depth = float((low_edge - curve[i]) / (low_edge + 1e-30))  # 0: no dip, 1: metric vanishes at focus
    return {'z_best': float(np.asarray(zs)[i]), 'curve': curve, 'zs': np.asarray(zs), 'roi': roi,
            'pitch_m': pitch, 'peak_at_edge': i in (0, len(zs) - 1), 'dip_depth': depth}


def local_focus_stack(field, wavelength, dx, zs, block=4, window=9, z_ref=None):
    """Per-pixel focus measure for every z: local Tenengrad of the block-averaged intensity,
    computed on each plane's native grid and then resampled to the physical grid of z_ref.
    Pixels whose physical position falls outside a plane are NaN (filling them with 0 would
    create a false focus at the edge of the scan). Returns (stack, pitch_ref_m)."""
    h = np.asarray(field)
    ny, nx = h.shape
    my, mx = ny // block, nx // block
    zs = np.asarray(zs, dtype=float)
    z_ref = float(z_ref if z_ref is not None else zs[len(zs) // 2])
    cy, cx = (my - 1) / 2, (mx - 1) / 2
    out = np.empty((len(zs), my, mx))
    for i, z in enumerate(zs):
        u, _, _ = fresnel(h, wavelength, dx, dx, z)
        I = (np.abs(u[:my * block, :mx * block]) ** 2).reshape(my, block, mx, block).mean(axis=(1, 3))
        Is = ndimage.gaussian_filter(I, 1.0)
        g = ndimage.sobel(Is, 0) ** 2 + ndimage.sobel(Is, 1) ** 2
        F = ndimage.uniform_filter(g, window) / (ndimage.uniform_filter(Is, window) ** 2 + 1e-30)
        k = z_ref / z
        out[i] = ndimage.affine_transform(F, [k, k], offset=[cy - k * cy, cx - k * cx], order=1, cval=np.nan)
    return out, wavelength * abs(z_ref) / (nx * dx) * block


def depth_from_focus(field, wavelength, dx, zs, block=4, window=9):
    """Depth map z(x, y): at every pixel, the z of minimum local focus measure (the same extremum
    as the ROI autofocus; validated on two diffuse objects at 12 and 18 cm, tests/test_focus.py).
    Use a field free of zero order and twin when possible (phase-shifting O R*).
    Returns (depth_m on the z_ref grid, valid mask, pitch_m)."""
    st, pitch = local_focus_stack(field, wavelength, dx, zs, block=block, window=window)
    finite = np.isfinite(st)
    idx = np.argmin(np.where(finite, st, np.inf), axis=0)
    valid = finite.sum(axis=0) > 0.6 * len(zs)
    return np.asarray(zs)[idx], valid, pitch
