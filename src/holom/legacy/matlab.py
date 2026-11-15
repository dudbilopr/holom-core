# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Exact equivalents of the MATLAB built-ins used by the legacy code."""
import numpy as np
from scipy import ndimage


def grid(n, m):
    """[x, y] = meshgrid(-M/2:M/2-1, -N/2:N/2-1)."""
    return np.meshgrid(np.arange(-m / 2, m / 2 - 1 + 1e-9), np.arange(-n / 2, n / 2 - 1 + 1e-9))


def cextreme(a, fn):
    """MATLAB min/max of a complex array compare by modulus and return the complex element."""
    a = np.asarray(a)
    if not np.iscomplexobj(a):
        return fn(a)
    idx = (np.argmax if fn is np.max else np.argmin)(np.abs(a))
    return a.flat[idx]


def rgb2gray(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return rgb if rgb.ndim == 2 else 0.298936 * rgb[..., 0] + 0.587043 * rgb[..., 1] + 0.114021 * rgb[..., 2]


def medfilt2(a):
    """3x3 median filter with zero padding (MATLAB default)."""
    if np.iscomplexobj(a):
        return ndimage.median_filter(a.real, 3, mode='constant') + 1j * ndimage.median_filter(a.imag, 3, mode='constant')
    return ndimage.median_filter(a, 3, mode='constant')


def fspecial_unsharp(alpha=0.2):
    return np.array([[-alpha, alpha - 1, -alpha], [alpha - 1, alpha + 5, alpha - 1], [-alpha, alpha - 1, -alpha]]) / (alpha + 1)


def imfilter_replicate(a, h):
    return ndimage.correlate(a, h, mode='nearest')


def graythresh(a):
    """Otsu threshold on the [0, 1] range, 256 bins (MATLAB graythresh)."""
    v = np.clip(np.asarray(a, dtype=float).ravel(), 0, 1)
    hist, _ = np.histogram(v, bins=256, range=(0, 1))
    p = hist / hist.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    k = np.nanargmax(sigma_b)
    return k / 255


def im2bw(a, level):
    return (np.asarray(a, dtype=float) > level).astype(float)


def imnoise_speckle(img, v, rng):
    """J = I + n.*I with n uniform, zero mean, variance v."""
    n = np.sqrt(12 * v) * (rng.random(np.shape(img)) - 0.5)
    return img + n * img


def unwrap_columns(phase):
    """MATLAB unwrap on a matrix works down each column (axis 0)."""
    return np.unwrap(phase, axis=0)


def padarray(a, pad_rows, pad_cols):
    pw = [(int(pad_rows), int(pad_rows)), (int(pad_cols), int(pad_cols))] + [(0, 0)] * (np.ndim(a) - 2)
    return np.pad(a, pw)


def sind(deg):
    return np.sin(np.deg2rad(deg))
