# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Computer-generated holograms of HGC_GOM_V2012 (thesis 2012, RCTA 2013).

Faithful ports of the eight MATLAB routines. Every function returns
(embedded_object, hologram, reconstruction) like the originals; the only
deliberate changes are a seedable RNG and reconstructions normalised to max 1
(MATLAB scaled them by 800 or 2024 for display only).

Legacy details kept on purpose (see audit/AUDITORIA_CODIGO.md):
- the carrier frequency is alpha/127 cycles/pixel regardless of image size;
- CuaternariCGH returns a complex array, not a realisable transmittance;
- none of the routines is iterative (there is no Gerchberg-Saxton loop).
"""
import numpy as np

ALPHA = 0.5
LEGACY_PERIOD = 127


def _embed(image, d=0, s=0):
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:
        img = img[:, :, 0]
    n, m = img.shape
    A = np.zeros((2 * n, 2 * m))
    A[d:n + d, s:m + s] = img / 255
    return A


def _diffuse_spectrum(A, rng):
    R = rng.random(A.shape) - 0.5
    F = A * np.exp(2j * np.pi * R)
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(F)))


def _reconstruct(hologram):
    O = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(hologram))))
    return O / O.max()


def _carrier(shape, axis):
    n = shape[axis]
    phase = 2 * np.pi * ALPHA * np.arange(n) / LEGACY_PERIOD
    return np.broadcast_to(phase[None, :] if axis == 1 else phase[:, None], shape)


def _normalised_spectrum(A, rng):
    F = _diffuse_spectrum(A, rng)
    return F / np.abs(F).max()


def kinoform(image, d=0, s=0, rng=None):
    """HoloKinoform.m: phase of the diffuse Fourier spectrum mapped to [0, 1]."""
    rng = np.random.default_rng(rng)
    A = _embed(image, d, s)
    F = _diffuse_spectrum(A, rng)
    H = np.angle(F) / (2 * np.pi) + 0.5
    O = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(H))))
    return A, H, O / O.max()


def burch(image, d=0, s=0, rng=None):
    """HoloBurch.m: 0.5 + 0.5 Re{F exp(-i carrier)}."""
    rng = np.random.default_rng(rng)
    A = _embed(image, d, s)
    F = _normalised_spectrum(A, rng)
    c = _carrier(A.shape, 1)
    H = 0.5 + 0.5 * (F.real * np.cos(c) + F.imag * np.sin(c))
    H = H / H.max()
    return A, H, _reconstruct(H)


def fourier_amplitude(image, d=0, s=0, rng=None):
    """Fourier.m: |F/2 + 0.5|."""
    rng = np.random.default_rng(rng)
    A = _embed(image, d, s)
    F = _normalised_spectrum(A, rng)
    H = np.abs(F / 2 + 0.5)
    H = H / H.max()
    return A, H, _reconstruct(H)


def fourier_real(image, d=0, s=0, rng=None):
    """FourierReal.m: |Re F + 0.5|."""
    rng = np.random.default_rng(rng)
    A = _embed(image, d, s)
    F = _normalised_spectrum(A, rng)
    H = np.abs(F.real + 0.5)
    H = H / H.max()
    return A, H, _reconstruct(H)


def fourier_imag(image, d=0, s=0, rng=None):
    """FourierImag.m: |Im F / 2 + 0.5|."""
    rng = np.random.default_rng(rng)
    A = _embed(image, d, s)
    F = _normalised_spectrum(A, rng)
    H = np.abs(F.imag / 2 + 0.5)
    H = H / H.max()
    return A, H, _reconstruct(H)


def square_wave(image, d=0, s=0, rng=None, K=1, a=0.25, L=0.5, amp=1.0):
    """SquareWave.m: Fourier series of a phase-modulated square grating, |n| <= K."""
    rng = np.random.default_rng(rng)
    A = _embed(image, d, s)
    F = _normalised_spectrum(A, rng)
    C = np.angle(F)
    R = _carrier(A.shape, 1)
    Sq = 0
    for n in range(-K, K + 1):
        if n != 0:
            Sq = Sq + (amp * a / (np.pi * L)) * np.sinc(n * a / 2) * np.exp(1j * R * n) * np.exp(1j * C * n)
    H = np.abs(a / L - Sq)
    H = H / H.max()
    return A, H, _reconstruct(H)


def crossed_square_wave(image, d=0, s=0, rng=None):
    """CrossedSquareWave.m: product of x and y carriers applied to Re/Im of F."""
    rng = np.random.default_rng(rng)
    A = _embed(image, d, s)
    F = _normalised_spectrum(A, rng)
    a, b = F.real, F.imag
    cx, cy = _carrier(A.shape, 1), _carrier(A.shape, 0)
    A1, B1 = np.cos(cx), np.sin(cx)
    A2, B2 = np.cos(cy), np.sin(cy)
    MM = (A1 * A2 * a + 1j * B1 * A2 * a + 1j * A1 * A2 * b - B1 * A2 * b
          - 1j * A1 * B2 * a - B1 * B2 * a - A1 * B2 * b - 1j * B1 * B2 * b)
    H = np.abs(0.3 + MM)
    H = H / H.max()
    return A, H, _reconstruct(H)


def cuaternari(image, d=0, s=0, rng=None):
    """CuaternariCGH.m: 0.5 + exp(i(phase + carrier))/pi.

    Returns a complex hologram, divided by the complex element of largest
    modulus exactly as MATLAB's max() does for complex input.
    """
    rng = np.random.default_rng(rng)
    A = _embed(image, d, s)
    F = _diffuse_spectrum(A, rng)
    G = _carrier(A.shape, 1)
    H = 0.5 + np.exp(1j * (np.angle(F) + G)) / np.pi
    H = H / H.flat[np.argmax(np.abs(H))]
    return A, H, _reconstruct(H)


ALGORITHMS = {
    "Burch": burch,
    "Kinoform": kinoform,
    "SquareWave": square_wave,
    "Fourier": fourier_amplitude,
    "FourierReal": fourier_real,
    "FourierImag": fourier_imag,
    "Cuaternario": cuaternari,
    "CrossedSquareWave": crossed_square_wave,
}
