import numpy as np
import pytest

from holom.diffraction import (angular_spectrum, centered_coords, critical_distance,
                               fresnel, fresnel_convolution, fresnel_legacy)

WL = 532e-9
DX = 2.2e-6
N = 512
W0 = 50e-6


def gaussian(n=N, dx=DX, w0=W0):
    X, Y = np.meshgrid(centered_coords(n, dx), centered_coords(n, dx))
    return np.exp(-(X**2 + Y**2) / w0**2), X, Y


def gaussian_amplitude(X, Y, z, wl=WL, w0=W0):
    zr = np.pi * w0**2 / wl
    w = w0 * np.sqrt(1 + (z / zr) ** 2)
    return (w0 / w) * np.exp(-(X**2 + Y**2) / w**2)


def max_rel_error(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("z", [5e-3, 20e-3])
def test_angular_spectrum_matches_gaussian_beam(z):
    u0, X, Y = gaussian()
    u = angular_spectrum(u0, WL, DX, DX, z)
    assert max_rel_error(np.abs(u), gaussian_amplitude(X, Y, z)) < 1e-3


@pytest.mark.parametrize("z", [5e-3, 20e-3, 50e-3])
def test_fresnel_convolution_matches_gaussian_beam(z):
    u0, X, Y = gaussian()
    u = fresnel_convolution(u0, WL, DX, DX, z, pad=True)
    assert max_rel_error(np.abs(u), gaussian_amplitude(X, Y, z)) < 1e-10


def test_fresnel_convolution_without_padding_wraps_around():
    """Circular convolution aliases the chirp tails back into the window."""
    u0, X, Y = gaussian()
    z = 50e-3
    u = fresnel_convolution(u0, WL, DX, DX, z)
    assert max_rel_error(np.abs(u), gaussian_amplitude(X, Y, z)) > 0.1


def test_fresnel_single_fft_matches_gaussian_beam_on_output_grid():
    u0, _, _ = gaussian()
    z = 2 * critical_distance(N, DX, WL)
    u, dxi, deta = fresnel(u0, WL, DX, DX, z)
    XI, ETA = np.meshgrid(centered_coords(N, dxi), centered_coords(N, deta))
    assert max_rel_error(np.abs(u), gaussian_amplitude(XI, ETA, z)) < 1e-3


def test_fresnel_conserves_energy():
    rng = np.random.default_rng(0)
    u0 = rng.random((256, 384)) * np.exp(2j * np.pi * rng.random((256, 384)))
    u, dxi, deta = fresnel(u0, WL, DX, DX, 0.12)
    e_in = np.sum(np.abs(u0) ** 2) * DX * DX
    e_out = np.sum(np.abs(u) ** 2) * dxi * deta
    assert e_out == pytest.approx(e_in, rel=1e-10)


def test_fresnel_forward_then_backward_recovers_field():
    u0, _, _ = gaussian(256)
    z = 0.05
    u, dxi, deta = fresnel(u0, WL, DX, DX, z)
    back, dx2, _ = fresnel(u, WL, dxi, deta, -z)
    assert dx2 == pytest.approx(DX)
    assert max_rel_error(back, u0) < 1e-10


def test_angular_spectrum_round_trip():
    u0, _, _ = gaussian()
    back = angular_spectrum(angular_spectrum(u0, WL, DX, DX, 0.03), WL, DX, DX, -0.03)
    assert max_rel_error(back, u0) < 1e-10


def test_legacy_fresnel_intensity_agrees_but_phase_does_not():
    """Audit of HOLO_GOM/Fresnel.m: the intensity it shows is right, the phase is not."""
    u0, _, _ = gaussian(256)
    z = 0.05
    new, _, _ = fresnel(u0, WL, DX, DX, z)
    old = fresnel_legacy(u0, WL, DX, DX, z)

    a, b = np.abs(new), np.abs(old)
    a, b = a / a.max(), b / b.max()
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.999

    mask = a > 0.1
    dphi = np.angle(new[mask] * np.conj(old[mask]))
    dphi -= np.angle(np.mean(np.exp(1j * dphi)))  # ignore a constant piston
    assert np.std(np.angle(np.exp(1j * dphi))) > 0.5
