"""Propagators and reference wave of the 2015 MATLAB scripts (Algunos Resultados/).

All functions accept 2-D arrays or (N, M, P) stacks, like the originals.
"""
import numpy as np

from .matlab import cextreme, grid, medfilt2, sind


def _stack(a):
    a = np.asarray(a)
    return (a[..., None], True) if a.ndim == 2 else (a, False)


def onda_ref(tfres, lam, dx0, dy0, angulo):
    """Onda_Ref.m (identical in CGHFourier, CGH_Fresnel_Final_Excelente, DH_REconstruccion_Final).

    R = A exp(i pi/lambda sind(angulo) (x dx0 + y dy0)), A = (min + max)/2 of each plane.

    Defects: the phase lacks a factor 2 (a tilted plane wave is exp(i 2 pi sin(theta) x / lambda)),
    so the effective tilt is arcsin(sin(theta)/2); the tilt is along the x = y diagonal; for a
    complex input A is complex (MATLAB min/max compare moduli).
    """
    t, squeeze = _stack(tfres)
    n, m, p = t.shape
    x, y = grid(n, m)
    r = x * dx0 + y * dy0
    out = np.empty((n, m, p), dtype=complex)
    for j in range(p):
        a = 0.5 * (cextreme(t[:, :, j], np.min) + cextreme(t[:, :, j], np.max))
        out[:, :, j] = a * np.exp((1j * np.pi / lam) * sind(angulo) * r)
    return out[:, :, 0] if squeeze else out


def _pr(n, m, lam, dx0, dy0, z0):
    x, y = grid(n, m)
    r1 = (x * (lam * z0) / (m * dx0)) ** 2 + (y * (lam * z0) / (n * dy0)) ** 2
    r2 = (x * dx0) ** 2 + (y * dy0) ** 2
    return r1, r2


def t_fresnel(imagen, onda_ref_, lam, dx0, dy0, z0):
    """T_Fresnel.m (CGH_Fresnel_Final_Excelente, DH_REconstruccion_Final).

    U = P .* fftshift(fft2(I .* R .* H)), sign convention exp(-i ...): |T_Fresnel(z)| equals
    |holom.fresnel(-z)| exactly, which is why the scripts pass negative Z0 (-0.13, -0.23 m).
    Defect: the input is not ifftshift-ed, which adds a linear phase to the output (intensity unaffected).
    """
    im, squeeze = _stack(imagen)
    n, m, planes = im.shape
    r1, r2 = _pr(n, m, lam, dx0, dy0, z0)
    P = (1j / (lam * z0)) * np.exp(-1j * 2 * np.pi * z0 / lam) * np.exp(-(1j * np.pi / (lam * z0)) * r1)
    H = np.exp(-(1j * np.pi / (lam * z0)) * r2)
    out = np.empty(im.shape, dtype=complex)
    for i in range(planes):
        Ri = onda_ref_[:, :, i] if np.ndim(onda_ref_) == 3 else onda_ref_
        out[:, :, i] = P * np.fft.fftshift(np.fft.fft2(im[:, :, i] * Ri * H))
    return out[:, :, 0] if squeeze else out


def t_fresnel_medfilt(imagen, onda_ref_, lam, dx0, dy0, z0):
    """T_Fresnel.m of DHFresnel_GOM/ and Pactrica_Reconstruccion/: same transform, then
    |U| median-filtered twice (3x3). Returns a real magnitude image."""
    U, squeeze = _stack(t_fresnel(imagen, onda_ref_, lam, dx0, dy0, z0))
    out = np.empty(U.shape)
    for i in range(U.shape[2]):
        out[:, :, i] = medfilt2(medfilt2(np.abs(U[:, :, i])))
    return out[:, :, 0] if squeeze else out


def ti_fresnel(imagen, onda_ref_, lam, dx0, dy0, z0):
    """TI_Fresnel.m: inverse transform, P .* ifftshift(ifft2(R .* I .* H)) with P and H roles swapped."""
    im, squeeze = _stack(imagen)
    n, m, planes = im.shape
    r1, r2 = _pr(n, m, lam, dx0, dy0, z0)
    P = (1j / (lam * z0)) * np.exp(-1j * 2 * np.pi * z0 / lam) * np.exp(-(1j * np.pi / (lam * z0)) * r2)
    H = np.exp(-(1j * np.pi / (lam * z0)) * r1)
    out = np.empty(im.shape, dtype=complex)
    for i in range(planes):
        Ri = onda_ref_[:, :, i] if np.ndim(onda_ref_) == 3 else onda_ref_
        out[:, :, i] = P * np.fft.ifftshift(np.fft.ifft2(Ri * im[:, :, i] * H))
    return out[:, :, 0] if squeeze else out


def t_fresnel_conv(imagen, onda_ref_, lam, dx0, dy0, z0):
    """T_FresnelConv.m: P .* fftshift(fft2(fft2(I .* R) .* fft2(H))).
    Defect: a second forward FFT where an inverse FFT belongs (output is point-reflected)."""
    im, squeeze = _stack(imagen)
    n, m, planes = im.shape
    r1, r2 = _pr(n, m, lam, dx0, dy0, z0)
    P = (1j / (lam * z0)) * np.exp(-1j * 2 * np.pi * z0 / lam) * np.exp(-(1j * np.pi / (lam * z0)) * r2)
    H = np.exp(-(1j * np.pi / (lam * z0)) * r1)
    out = np.empty(im.shape, dtype=complex)
    for i in range(planes):
        Ri = onda_ref_[:, :, i] if np.ndim(onda_ref_) == 3 else onda_ref_
        out[:, :, i] = P * np.fft.fftshift(np.fft.fft2(np.fft.fft2(im[:, :, i] * Ri) * np.fft.fft2(H)))
    return out[:, :, 0] if squeeze else out


def t_fresnel_h(imagen, onda_ref_, lam, dx0, dy0, z0):
    """T_FresnelH.m: P .* fftshift(fft2(fft2(I .* R) .* H)) (chirp applied in the spectrum)."""
    im, squeeze = _stack(imagen)
    n, m, planes = im.shape
    r1, r2 = _pr(n, m, lam, dx0, dy0, z0)
    P = (1j / (lam * z0)) * np.exp(-1j * 2 * np.pi * z0 / lam) * np.exp(-(1j * np.pi / (lam * z0)) * r1)
    H = np.exp(-(1j * np.pi / (lam * z0)) * r2)
    out = np.empty(im.shape, dtype=complex)
    for i in range(planes):
        Ri = onda_ref_[:, :, i] if np.ndim(onda_ref_) == 3 else onda_ref_
        out[:, :, i] = P * np.fft.fftshift(np.fft.fft2(np.fft.fft2(im[:, :, i] * Ri) * H))
    return out[:, :, 0] if squeeze else out


def t_fresnel_convdir(imagen, onda_ref_, lam, dx0, dy0, z0):
    """T_Fresnel.m of Difraccion_Por_Convolucion/: exp(+i ...) convention and ifftn.
    Defect: for RGB stacks ifftn also transforms along the colour axis, mixing channels."""
    im = np.asarray(imagen)
    n, m = im.shape[:2]
    r1, r2 = _pr(n, m, lam, dx0, dy0, z0)
    P = (1 / (1j * lam * z0)) * np.exp(1j * 2 * np.pi * z0 / lam) * np.exp((1j * np.pi / (lam * z0)) * r1)
    H = np.exp((1j * np.pi / (lam * z0)) * r2)
    if im.ndim == 3:
        P, H = np.dstack([P] * im.shape[2]), np.dstack([H] * im.shape[2])
    return P * np.fft.fftshift(np.fft.ifftn(im * onda_ref_ * H))


def difraccion_convolucion(imagen, onda_ref_, lam, dx0, dy0, z0, zp=None):
    """Difraccion_Convolucion.m: Rayleigh-Sommerfeld-like impulse response g = e^{ikr}/(i lambda r).

    Defects: the exact RS-I kernel is z e^{ikr}/(i lambda r^2) (missing obliquity z/r); no
    zero padding inside (the caller pads); fftn/ifftn over RGB stacks mix channels; the
    parameter zp is unused."""
    im = np.asarray(imagen)
    n, m = im.shape[:2]
    x, y = grid(n, m)
    if im.ndim == 3:
        x, y = np.dstack([x] * im.shape[2]), np.dstack([y] * im.shape[2])
    r = ((x * dx0) ** 2 + (y * dy0) ** 2 + z0 ** 2) ** 0.5
    g = (1 / (1j * lam)) * (np.exp((2j * np.pi / lam) * r) / r)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftn(im * 1 * onda_ref_) * np.fft.fftn(g)))


def difraccion_convolucion_tf(imagen, onda_ref_, lam, dx0, dy0, z0):
    """Difraccion_ConvolucionTF.m, kept as written.

    Defects: sqrt(1 + ...) instead of sqrt(1 - (lambda fx)^2 - (lambda fy)^2); (x-1)/(N dx0) uses
    the row count for the x axis; (1/1i*Lambda) evaluates to -1i*Lambda (multiplies by lambda);
    the transfer function is centred while the FFT is not. Use holom.angular_spectrum instead."""
    im = np.asarray(imagen)
    n, m = im.shape[:2]
    x, y = grid(n, m)
    if im.ndim == 3:
        x, y = np.dstack([x] * im.shape[2]), np.dstack([y] * im.shape[2])
    r = ((lam * (x - 1) / (n * dx0)) ** 2 + (lam * (y - 1) / (m * dy0)) ** 2 + 1) ** 0.5
    H = (1 / 1j * lam) * (np.exp((2j * np.pi * z0 / lam) * r) / r)
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftn(im) * onda_ref_ * H))


def circulo(n, m, r, xo, yo, p=1):
    """circulo.m: binary disc of radius r centred at (xo, yo), 1-based pixel coordinates."""
    X, Y = np.meshgrid(np.arange(1, m + 1), np.arange(1, n + 1))
    disc = (np.abs((X - xo) + 1j * (Y - yo)) < r).astype(float)
    return disc if p == 1 else np.dstack([disc] * p)
