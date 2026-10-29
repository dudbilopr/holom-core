"""Zero-order (DC) and twin-image suppression, thesis 2015 chapter 7.

Methods 7.3.1-7.3.4 are ports of the thesis; `sideband_filter` is the
off-axis spatial filtering of Cuche et al. (2000), added as a reference.
"""
import numpy as np


def subtract_mean(hologram):
    """7.3.1 Mean-intensity subtraction: I - <I>."""
    h = np.asarray(hologram, dtype=float)
    return h - h.mean()


def highpass_fft(hologram, window):
    """7.3.2 Rectangular high-pass filter: zero a (rows, cols) window around DC."""
    h = np.asarray(hologram, dtype=float)
    ny, nx = h.shape
    wy, wx = window
    S = np.fft.fftshift(np.fft.fft2(h))
    cy, cx = ny // 2, nx // 2
    S[cy - wy // 2:cy - wy // 2 + wy, cx - wx // 2:cx - wx // 2 + wx] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(S)))


def laplacian(hologram):
    """7.3.3 Five-point discrete Laplacian (periodic boundaries)."""
    h = np.asarray(hologram, dtype=float)
    return (np.roll(h, 1, 0) + np.roll(h, -1, 0)
            + np.roll(h, 1, 1) + np.roll(h, -1, 1) - 4 * h)


def phase_shifting_4step(i0, i1, i2, i3):
    """7.3.4 Four-step phase shifting (0, pi/2, pi, 3pi/2).

    With I_k = |O + R exp(i d_k)|^2 returns O R*, free of DC and twin terms.
    """
    i0, i1, i2, i3 = (np.asarray(i, dtype=float) for i in (i0, i1, i2, i3))
    return ((i0 - i2) + 1j * (i1 - i3)) / 4


def sideband_filter(hologram, center, radius):
    """Keep a circular region of the spectrum centred at `center` (row, col offsets
    from DC) and move it to the origin, removing DC and the twin image."""
    h = np.asarray(hologram, dtype=float)
    ny, nx = h.shape
    S = np.fft.fftshift(np.fft.fft2(h))
    Y, X = np.ogrid[:ny, :nx]
    cy, cx = ny // 2 + center[0], nx // 2 + center[1]
    S = np.where((Y - cy) ** 2 + (X - cx) ** 2 <= radius**2, S, 0)
    S = np.roll(S, (-center[0], -center[1]), axis=(0, 1))
    return np.fft.ifft2(np.fft.ifftshift(S))
