# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Each published method reconstructs what it should; physical figures of merit are checked."""
import numpy as np
import pytest

from holom import cgh_methods as cm
from holom.diffraction import angular_spectrum

N = 128


def letter(n=N):
    """Off-centre 'F' in the far field (avoids overlap with the zero order and its twin)."""
    t = np.zeros((n, n))
    t[20:50, 20:26] = 1
    t[20:26, 20:44] = 1
    t[32:37, 20:38] = 1
    return t


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def test_gs_iterations_raise_efficiency_above_the_single_pass_kinoform():
    t = letter()
    _, hist = cm.gerchberg_saxton(t, iterations=40, rng=0, return_history=True)
    assert hist[-1] > hist[0] and hist[-1] > 0.9


def test_gs_reconstruction_matches_target():
    t = letter()
    phase = cm.gerchberg_saxton(t, iterations=40, rng=0)
    I = cm.reconstruct_fourier(np.exp(1j * phase))
    assert corr(I, t) > 0.8


def test_weighted_gs_makes_uniform_traps():
    spots = [(r, c) for r in (-30, 0, 30) for c in (-30, 0, 30) if (r, c) != (0, 0)]
    _, I_w = cm.weighted_gs_spots((N, N), spots, iterations=30, rng=0)
    _, I_1 = cm.weighted_gs_spots((N, N), spots, iterations=1, rng=0)
    unif = lambda I: 1 - (I.max() - I.min()) / (I.max() + I.min())
    assert unif(I_w) > 0.99 and unif(I_w) >= unif(I_1)


def _first_order(h, target, shift_cols):
    I = cm.reconstruct_fourier(h)
    return corr(np.roll(I, -shift_cols, axis=1), target)


def test_burch_is_a_valid_transmittance_and_reconstructs():
    t = letter()
    h = cm.burch(t, period=4, rng=0)
    assert 0 <= h.min() and h.max() <= 1
    I = cm.reconstruct_fourier(h)
    I[N // 2 - 3:N // 2 + 4, N // 2 - 3:N // 2 + 4] = 0      # exclude the zero order (DC term 1/2)
    best = max(corr(np.roll(I, s, axis=1), t) for s in (N // 4, -N // 4))
    assert best > 0.5


def test_lee_reconstructs_in_first_order():
    t = letter(64)
    h = cm.lee_delayed_sampling(t, rng=0)
    I = cm.reconstruct_fourier(h)
    best = max(corr(np.roll(I, s, axis=1)[:, 96:160], t) for s in (64, -64))
    assert h.min() >= 0 and best > 0.5


def test_lohmann_is_binary_and_reconstructs():
    t = letter(32)
    h = cm.lohmann_detour_phase(t, cell=8, rng=0)
    assert set(np.unique(h)) <= {0.0, 1.0}
    I = cm.reconstruct_fourier(h)
    n = 32 * 8
    win = lambda s: np.roll(I, s, axis=1)[n // 2 - 16:n // 2 + 16, n // 2 - 16:n // 2 + 16]
    assert max(corr(win(s), t) for s in (32, -32)) > 0.4


def test_double_phase_encodes_a_complex_field():
    u = cm.laguerre_gauss((N, N), l=2, p=0, w0=12)
    h = cm.double_phase(u)
    rec = 0.5 * (np.exp(1j * h))                       # average of the two phasors per pixel pair
    lp = np.fft.ifft2(np.fft.fft2(rec) * (np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N))) < 0.25))
    assert np.abs(lp - u * 0.5).max() < 0.1


def test_fork_hologram_carries_the_requested_charge():
    n = 256
    y, x = np.indices((n, n)) - n / 2
    gauss = np.exp(-(x ** 2 + y ** 2) / (n / 6) ** 2)    # Gaussian illumination, as on the bench
    for l in (1, 3):
        h = cm.fork_hologram((n, n), charge=l, period=8)
        far = cm.fft2c(gauss * np.exp(1j * h))            # centred FFT: no (-1)^(m+n) checkerboard phase
        c0 = n // 2 + n // 8                             # first order: N / period columns away
        a = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        ring = far[np.round(n // 2 + 3 * np.sin(a)).astype(int), np.round(c0 + 3 * np.cos(a)).astype(int)]
        winding = np.sum(np.angle(np.roll(ring, -1) / ring)) / (2 * np.pi)
        assert round(abs(winding)) == l


def test_point_cloud_refocuses_each_point_at_its_depth():
    lam, dx = 532e-9, 8e-6
    pts = [(0.0, 0.0, 0.05, 1.0), (2e-4, 0.0, 0.08, 1.0)]
    H = cm.point_cloud_cgh(pts, (N, N), dx, lam)
    a = np.abs(angular_spectrum(H, lam, dx, dx, -0.05))
    b = np.abs(angular_spectrum(H, lam, dx, dx, -0.08))
    assert np.unravel_index(a.argmax(), a.shape) == (N // 2, N // 2)
    assert np.unravel_index(b.argmax(), b.shape) == (N // 2, N // 2 + 25)


def test_layer_cgh_focuses_each_layer_at_its_own_depth():
    lam, dx = 532e-9, 8e-6
    A = np.zeros((N, N)); A[30:60, 30:40] = 1
    B = np.zeros((N, N)); B[70:100, 80:90] = 1
    H = cm.layer_cgh([(A, 0.02), (B, 0.05)], dx, lam, rng=0)
    ra = np.abs(angular_spectrum(H, lam, dx, dx, 0.02)) ** 2
    rb = np.abs(angular_spectrum(H, lam, dx, dx, 0.05)) ** 2
    assert corr(ra, A) > corr(ra, B) and corr(rb, B) > corr(rb, A)


def test_drpe_round_trip_and_white_ciphertext():
    img = letter()
    E, n, b = cm.drpe_encrypt(img, rng=0)
    dec = cm.drpe_decrypt(E, b)
    assert np.allclose(dec, img, atol=1e-10)
    assert abs(corr(np.abs(E), img)) < 0.1
    wrong = cm.drpe_decrypt(E, np.random.default_rng(1).random(img.shape))
    assert corr(wrong, img) < 0.1
