# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Hologram generation methods from the literature, implemented as they are published.

Every function documents its equation and the original reference (DOIs verified in OpenAlex,
2026-10-29). Fourier holograms are reconstructed by a lens (2F): the far field is fft2 of the
hologram. Fresnel / 3-D holograms use holom.diffraction.angular_spectrum.

References
----------
Gerchberg & Saxton, Optik 35, 237 (1972).
Lesem, Hirsch & Jordan, IBM J. Res. Dev. 13, 150 (1969), doi:10.1147/rd.132.0150 (kinoform).
Brown & Lohmann, Appl. Opt. 5, 967 (1966), doi:10.1364/AO.5.000967 (detour phase).
Burch, Proc. IEEE 55, 599 (1967), doi:10.1109/PROC.1967.5620.
Lee, Appl. Opt. 9, 639 (1970), doi:10.1364/AO.9.000639 (delayed sampling).
Hsueh & Sawchuk, Appl. Opt. 17, 3874 (1978), doi:10.1364/AO.17.003874 (double phase).
Heckenberg et al., Opt. Lett. 17, 221 (1992), doi:10.1364/OL.17.000221 (fork / vortex).
Di Leonardo, Ianni & Ruocco, Opt. Express 15, 1913 (2007), doi:10.1364/OE.15.001913 (weighted GS).
Waters, Appl. Phys. Lett. 9, 405 (1966), doi:10.1063/1.1754630 (point-source CGH).
Zhang et al., Appl. Opt. 56, F138 (2017), doi:10.1364/AO.56.00F138 (layer-based CGH).
Refregier & Javidi, Opt. Lett. 20, 767 (1995), doi:10.1364/OL.20.000767 (DRPE).
"""
import numpy as np

from .diffraction import angular_spectrum, centered_coords

fft2c = lambda a: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))
ifft2c = lambda a: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))


def reconstruct_fourier(hologram):
    """Far field of a Fourier hologram (lens reconstruction): |FFT|^2, normalised to 1."""
    I = np.abs(fft2c(hologram)) ** 2
    return I / I.max()


def efficiency(intensity, mask):
    """Fraction of the reconstructed power that lands inside `mask`."""
    return float(intensity[mask].sum() / intensity.sum())


# --- phase-only Fourier holograms ------------------------------------------------
def gerchberg_saxton(target, iterations=30, rng=None, return_history=False):
    """Phase-only Fourier hologram by error reduction (Gerchberg & Saxton 1972).

    Loop: hologram plane |h| = 1 (phase-only SLM) <-> far field |H| = sqrt(target).
    iterations = 1 is the classic single-pass kinoform with a random diffuser (Lesem 1969).
    Returns the hologram phase in (-pi, pi] (and, optionally, the efficiency per iteration).
    """
    rng = np.random.default_rng(rng)
    amp = np.sqrt(np.asarray(target, dtype=float))
    mask = amp > 0
    field = amp * np.exp(2j * np.pi * rng.random(amp.shape))
    history = []
    for it in range(iterations):
        h = ifft2c(field)
        phase = np.angle(h)
        far = fft2c(np.exp(1j * phase))
        if return_history:
            I = np.abs(far) ** 2
            history.append(efficiency(I, mask))
        field = amp * np.exp(1j * np.angle(far))
    return (phase, history) if return_history else phase


def weighted_gs_spots(shape, spots, iterations=30, rng=None):
    """Holographic optical tweezers: weighted Gerchberg-Saxton for an array of point traps
    (Di Leonardo, Ianni & Ruocco 2007). spots: list of (row, col) in the centred far field.

    Weights w_m <- w_m <|V|>/|V_m| equalise the trap intensities. Returns (phase, intensities)."""
    rng = np.random.default_rng(rng)
    ny, nx = shape
    rows = np.array([s[0] for s in spots]) + ny // 2
    cols = np.array([s[1] for s in spots]) + nx // 2
    w = np.ones(len(spots))
    ph_spot = 2 * np.pi * rng.random(len(spots))
    for _ in range(iterations):
        far = np.zeros(shape, dtype=complex)
        far[rows, cols] = w * np.exp(1j * ph_spot)
        phase = np.angle(ifft2c(far))
        V = fft2c(np.exp(1j * phase))[rows, cols]
        ph_spot = np.angle(V)
        w = w * np.abs(V).mean() / np.abs(V)
    I = np.abs(fft2c(np.exp(1j * phase))[rows, cols]) ** 2
    return phase, I


# --- amplitude (binary or grey) Fourier holograms with a carrier ------------------
def _diffuse_spectrum(target, rng):
    amp = np.sqrt(np.asarray(target, dtype=float))
    F = ifft2c(amp * np.exp(2j * np.pi * np.random.default_rng(rng).random(amp.shape)))
    return F / np.abs(F).max()


def burch(target, period=4, rng=None):
    """Burch (1967): h = 1/2 [1 + |F| cos(2 pi x / period - arg F)], 0 <= h <= 1.
    |F| is normalised to max 1 (the 2015 script did not). The image appears at the +1 order,
    shifted by N/period columns from the centre."""
    F = _diffuse_spectrum(target, rng)
    x = np.arange(F.shape[1])[None, :]
    return 0.5 * (1 + np.abs(F) * np.cos(2 * np.pi * x / period - np.angle(F)))


def lee_delayed_sampling(target, rng=None):
    """Lee (1970): each complex sample is split into four non-negative components along
    0, 90, 180, 270 degrees, written in four adjacent columns. The fourfold-wider hologram
    reconstructs the image in the first order at N columns (one quarter of the width)."""
    F = _diffuse_spectrum(target, rng)
    re, im = F.real, F.imag
    comps = [np.maximum(re, 0), np.maximum(im, 0), np.maximum(-re, 0), np.maximum(-im, 0)]
    ny, nx = F.shape
    h = np.zeros((ny, 4 * nx))
    for k in range(4):
        h[:, k::4] = comps[k]
    return h


def lohmann_detour_phase(target, cell=8, rng=None, aperture_width=None):
    """Brown & Lohmann (1966), type I detour phase: one cell of cell x cell pixels per
    spectral sample; a transparent slot of height |F| cell, shifted arg(F)/(2 pi) cell along x.
    Binary transmittance; the image is in the first diffraction order."""
    F = _diffuse_spectrum(target, rng)
    ny, nx = F.shape
    w = aperture_width or max(1, cell // 2)
    h = np.zeros((ny * cell, nx * cell))
    A = np.abs(F)
    shift = np.round(np.angle(F) / (2 * np.pi) * cell).astype(int)
    height = np.round(A * cell).astype(int)
    for i in range(ny):
        for j in range(nx):
            if height[i, j] == 0:
                continue
            c0 = j * cell + cell // 2 - w // 2 + shift[i, j]
            r0 = i * cell + (cell - height[i, j]) // 2
            cols = np.arange(c0, c0 + w) % (nx * cell)
            h[r0:r0 + height[i, j], cols] = 1
    return h


def double_phase(field, max_amp=None):
    """Hsueh & Sawchuk (1978): A e^{i phi} = (e^{i theta1} + e^{i theta2}) with
    theta_{1,2} = phi +/- arccos(A / A_max), interleaved on a checkerboard. Returns a phase-only
    hologram that encodes the complex field (low-pass part reproduced exactly up to a factor 1/2)."""
    field = np.asarray(field, dtype=complex)
    A = np.abs(field)
    A = A / (max_amp or A.max())
    phi = np.angle(field)
    t1, t2 = phi + np.arccos(A), phi - np.arccos(A)
    yy, xx = np.indices(field.shape)
    return np.where((yy + xx) % 2 == 0, t1, t2)


# --- structured light ------------------------------------------------------------------
def fork_hologram(shape, charge, period=8.0, blazed=True):
    """Heckenberg et al. (1992): phase l*theta + 2 pi x / period. Blazed (mod 2 pi, phase-only)
    sends the vortex of charge l into the first order; binary (cosine > 0) gives +/- l."""
    ny, nx = shape
    y, x = np.indices(shape)
    y, x = y - ny / 2, x - nx / 2
    phase = charge * np.arctan2(y, x) + 2 * np.pi * x / period
    if blazed:
        return np.mod(phase, 2 * np.pi)
    return (np.cos(phase) > 0).astype(float)


def laguerre_gauss(shape, l, p, w0, dx=1.0):
    """LG_p^l mode at its waist (complex amplitude), for use with double_phase."""
    from scipy.special import eval_genlaguerre
    ny, nx = shape
    y, x = np.meshgrid(centered_coords(ny, dx), centered_coords(nx, dx), indexing='ij')
    r2 = (x ** 2 + y ** 2) / w0 ** 2
    u = (np.sqrt(2 * r2)) ** abs(l) * eval_genlaguerre(p, abs(l), 2 * r2) * np.exp(-r2) * np.exp(1j * l * np.arctan2(y, x))
    return u / np.abs(u).max()


# --- 3-D Fresnel holograms ---------------------------------------------------------------
def point_cloud_cgh(points, shape, dx, wavelength):
    """Waters (1966): superposition of spherical waves from points (x, y, z, a) in metres,
    sampled on the hologram plane z = 0 (paraxial phase-exact form exp(i k r)/r)."""
    ny, nx = shape
    Y, X = np.meshgrid(centered_coords(ny, dx), centered_coords(nx, dx), indexing='ij')
    k = 2 * np.pi / wavelength
    H = np.zeros(shape, dtype=complex)
    for (px, py, pz, a) in points:
        r = np.sqrt((X - px) ** 2 + (Y - py) ** 2 + pz ** 2)
        H += a * np.exp(1j * k * r) / r
    return H


def layer_cgh(layers, dx, wavelength, rng=None):
    """Layer-based CGH (e.g. Zhang et al. 2017): each layer (amplitude image, z in metres) gets a
    random diffuser and is back-propagated to the hologram plane with the angular spectrum; the
    complex fields add. Returns the complex hologram (encode with double_phase or a carrier)."""
    rng = np.random.default_rng(rng)
    H = None
    for amp, z in layers:
        u = np.sqrt(np.asarray(amp, dtype=float)) * np.exp(2j * np.pi * rng.random(np.shape(amp)))
        h = angular_spectrum(u, wavelength, dx, dx, -z)
        H = h if H is None else H + h
    return H


# --- optical encryption ------------------------------------------------------------------
def drpe_encrypt(image, rng=None):
    """Refregier & Javidi (1995): E = IFT{ FT{ f exp(i 2 pi n) } exp(i 2 pi b) }, with independent
    uniform random phases n (input plane) and b (Fourier plane). Returns (E, key_n, key_b)."""
    rng = np.random.default_rng(rng)
    f = np.asarray(image, dtype=float)
    n, b = rng.random(f.shape), rng.random(f.shape)
    E = np.fft.ifft2(np.fft.fft2(f * np.exp(2j * np.pi * n)) * np.exp(2j * np.pi * b))
    return E, n, b


def drpe_decrypt(E, key_b):
    """|IFT{ FT{E} exp(-i 2 pi b) }| recovers the image (only the Fourier-plane key is needed)."""
    return np.abs(np.fft.ifft2(np.fft.fft2(E) * np.exp(-2j * np.pi * key_b)))
