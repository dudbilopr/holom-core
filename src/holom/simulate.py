# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Synthetic off-axis Fresnel holograms with known ground truth.

Replaces the missing raw CMOS captures of the 2015 thesis: same wavelength,
pixel pitch and sensor, so every reconstruction can be scored numerically.
"""
import numpy as np

from .diffraction import angular_spectrum, centered_coords, fresnel

# Mach-Zehnder setup of the 2015 thesis (chapter 7).
THESIS_2015 = dict(wavelength=532e-9, dx=2.2e-6, dy=2.2e-6, shape=(1944, 2592),
                   distances=(0.12, 0.24, 0.48))


def max_carrier_angle(wavelength, dx):
    """Largest reference tilt that keeps fringes below Nyquist: sin(t) = lambda/(2 dx)."""
    return np.arcsin(wavelength / (2 * dx))


def off_axis_hologram(obj, wavelength, dx, dy, z, angle=(0.0, 0.0),
                      ref_amplitude=1.0, bits=None, noise=0.0, rng=None):
    """Record |O + R|^2 at distance z.

    obj: complex field (or amplitude) in the object plane, same sampling as sensor.
    angle: reference tilt (theta_y, theta_x) in radians.
    Returns (hologram, object_field_at_sensor, reference).
    """
    ny, nx = np.shape(obj)
    o = angular_spectrum(np.asarray(obj, dtype=complex), wavelength, dx, dy, z)
    Y, X = np.meshgrid(np.arange(ny) * dy, np.arange(nx) * dx, indexing="ij")
    k = 2 * np.pi / wavelength
    r = ref_amplitude * np.exp(1j * k * (np.sin(angle[1]) * X + np.sin(angle[0]) * Y))
    I = np.abs(o + r) ** 2
    if noise:
        rng = np.random.default_rng(rng)
        I = I + noise * I.mean() * rng.standard_normal(I.shape)
    if bits:
        levels = 2**bits - 1
        I = np.round(np.clip(I / I.max(), 0, 1) * levels) / levels
    return I, o, r


def fresnel_off_axis_hologram(obj, wavelength, dx, dy, z, angle=(0.0, 0.0),
                              ref_amplitude=1.0, bits=None, noise=0.0, rng=None):
    """Lensless Fresnel recording of an object larger than the sensor (thesis 2015 setup).

    obj is sampled at the pitch the single-FFT Fresnel reconstruction returns,
    lambda*z/(N*dx), so `fresnel(holo, ..., -z)` maps back onto the same grid.
    Returns (hologram, object_pitch_xy).
    """
    ny, nx = np.shape(obj)
    dxi, deta = wavelength * z / (nx * dx), wavelength * z / (ny * dy)
    o, _, _ = fresnel(np.asarray(obj, dtype=complex), wavelength, dxi, deta, z)
    Y, X = np.meshgrid(centered_coords(ny, dy), centered_coords(nx, dx), indexing="ij")
    k = 2 * np.pi / wavelength
    r = ref_amplitude * np.abs(o).max() * np.exp(1j * k * (np.sin(angle[1]) * X + np.sin(angle[0]) * Y))
    I = np.abs(o + r) ** 2
    if noise:
        rng = np.random.default_rng(rng)
        I = I + noise * I.mean() * rng.standard_normal(I.shape)
    if bits:
        levels = 2**bits - 1
        I = np.round(np.clip(I / I.max(), 0, 1) * levels) / levels
    return I, (dxi, deta)


def phase_shifted_set(obj, wavelength, dx, dy, z, ref_amplitude=1.0):
    """Four in-line holograms with reference phases 0, pi/2, pi, 3pi/2."""
    o = angular_spectrum(np.asarray(obj, dtype=complex), wavelength, dx, dy, z)
    holos = [np.abs(o + ref_amplitude * np.exp(1j * d)) ** 2
             for d in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
    return holos, o
