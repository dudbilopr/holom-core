# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Scalar diffraction propagators.

Conventions: arrays are indexed (row, col) = (y, x); spatial coordinates are
centred so that index n//2 sits on the optical axis. Lengths in metres.
"""
import numpy as np


def centered_coords(n, d):
    return (np.arange(n) - n // 2) * d


def fresnel(u0, wavelength, dx, dy, z):
    """Single-FFT Fresnel transform (thesis 2015, eq. 4.10).

    Returns (field, dxi, deta), where dxi = lambda*|z|/(Nx*dx) is the output
    pixel pitch. Negative z back-propagates (uses the inverse kernel).
    """
    u0 = np.asarray(u0)
    ny, nx = u0.shape
    X, Y = np.meshgrid(centered_coords(nx, dx), centered_coords(ny, dy))
    dxi = wavelength * abs(z) / (nx * dx)
    deta = wavelength * abs(z) / (ny * dy)
    XI, ETA = np.meshgrid(centered_coords(nx, dxi), centered_coords(ny, deta))

    k = 2 * np.pi / wavelength
    h = np.exp(1j * np.pi / (wavelength * z) * (X**2 + Y**2))
    p = (np.exp(1j * k * z) / (1j * wavelength * z)
         * np.exp(1j * np.pi / (wavelength * z) * (XI**2 + ETA**2)))

    g = np.fft.ifftshift(u0 * h)
    G = np.fft.fft2(g) if z > 0 else np.fft.ifft2(g) * (nx * ny)
    return p * np.fft.fftshift(G) * dx * dy, dxi, deta


def fresnel_legacy(image, wavelength, dx, dy, z):
    """Line-by-line port of HOLO_GOM/Fresnel.m (2015), kept for audit.

    The output chirp P is built on a centred grid but multiplied with the
    uncentred fft2 output, so the reconstructed phase is wrong while the
    intensity is (up to a sub-pixel shift) correct. Use `fresnel` instead.
    """
    m, n = image.shape
    x, y = np.meshgrid(np.arange(1 - n / 2, n / 2 + 1), np.arange(1 - m / 2, m / 2 + 1))
    r1 = (x / (n * dx)) ** 2 + (y / (m * dy)) ** 2
    P = (1j / (wavelength * z)) * np.exp(-2j * np.pi * z / wavelength) \
        * np.exp(1j * np.pi * wavelength * z * r1)
    r2 = (x * dx) ** 2 + (y * dy) ** 2
    H = np.exp((1j * np.pi / (z * wavelength)) * r2)
    return np.fft.ifftshift(P * np.fft.fft2(image * H))


def angular_spectrum(u0, wavelength, dx, dy, z):
    """Exact (non-paraxial) angular-spectrum propagation; output pitch = input pitch.

    Evanescent components are discarded.
    """
    u0 = np.asarray(u0)
    ny, nx = u0.shape
    FX, FY = np.meshgrid(np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dy))
    arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    kz = 2 * np.pi / wavelength * np.sqrt(np.maximum(arg, 0.0))
    H = np.where(arg > 0, np.exp(1j * kz * z), 0.0)
    return np.fft.ifft2(np.fft.fft2(u0) * H)


def fresnel_convolution(u0, wavelength, dx, dy, z, pad=False):
    """Fresnel convolution (impulse-response, 3 FFTs; thesis 2015, sec. 4.3).

    Output pitch equals the input pitch. With pad=True the field is
    zero-padded to 2x to suppress circular wrap-around.
    """
    u0 = np.asarray(u0)
    ny, nx = u0.shape
    if pad:
        big = np.zeros((2 * ny, 2 * nx), dtype=complex)
        big[ny // 2:ny // 2 + ny, nx // 2:nx // 2 + nx] = u0
        out = fresnel_convolution(big, wavelength, dx, dy, z)
        return out[ny // 2:ny // 2 + ny, nx // 2:nx // 2 + nx]

    X, Y = np.meshgrid(centered_coords(nx, dx), centered_coords(ny, dy))
    k = 2 * np.pi / wavelength
    g = (np.exp(1j * k * z) / (1j * wavelength * z)
         * np.exp(1j * np.pi / (wavelength * z) * (X**2 + Y**2)))
    G = np.fft.fft2(np.fft.ifftshift(g))
    return np.fft.ifft2(np.fft.fft2(u0) * G) * dx * dy


def critical_distance(n, dx, wavelength):
    """Nyquist distance z_c = N*dx^2/lambda for chirp sampling."""
    return n * dx**2 / wavelength
