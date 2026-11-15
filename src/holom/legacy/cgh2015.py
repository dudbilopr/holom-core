# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Computer-generated holograms of the 2015 archive (Algunos Resultados/): Fourier and Fresnel
CGH with a reference wave, Burch, Lee, Lohmann detour phase, kinoforms and SLM phase masks.
Each function returns the hologram and the reconstruction the script displayed."""
import numpy as np

from . import fresnel_gom as fg
from .matlab import rgb2gray


def _rng(rng):
    return np.random.default_rng(rng)


def cgh_fourier(imagen, rng=None, lam=532e-9, l0x=7.3e-3, l0y=5.55e-3, angulo_o=0.1, angulo_r=30):
    """CGHFourier.m: object x exp(-i phi), phi uniform in [-pi/2, pi/2); FFT; interference with a
    reference of 30 deg (Onda_Ref, so the real tilt is arcsin(sin 30 deg / 2)); reconstruction ifft2."""
    im = np.asarray(imagen, dtype=float)
    n, m = im.shape[:2]
    phi = (_rng(rng).random(im.shape) - 0.5) * 1 * np.pi
    im = im * np.exp(-1j * phi)
    dx0, dy0 = l0x / m, l0y / n
    tf = np.fft.fftshift(np.fft.fftn(fg.onda_ref(im, lam, dx0, dy0, angulo_o) * im, axes=(0, 1)), axes=(0, 1))
    R = fg.onda_ref(tf, lam, dx0, dy0, angulo_r)
    holo = ((tf + R) * np.conj(tf + R)).real
    return holo, np.fft.ifftn(holo, axes=(0, 1))


def _cgh_fresnel_core(objeto, rng, phi_scale, l0x, l0y, z_factor, ang_ref, normalise):
    o = np.asarray(objeto, dtype=float)
    n, m = o.shape[:2]
    if normalise:
        o = o / o.max()
    phi = (_rng(rng).random(o.shape) - 0.5) * phi_scale
    o = o * np.exp(1j * phi) if phi_scale else o
    lam = 532e-9
    dx0, dy0 = l0x / m, l0y / n
    zmin = (m * dx0 ** 2 + n * dy0 ** 2) / (2 * lam)
    angulo = (lam / (2 * dx0) + lam / (2 * dy0)) * 180 / np.pi / 2
    z0 = z_factor * zmin
    ang_ref = ang_ref(angulo)
    tfres = fg.t_fresnel(o, fg.onda_ref(o, lam, dx0, dy0, 0), lam, dx0, dy0, z0)
    R = fg.onda_ref(tfres, lam, dx0, dy0, ang_ref)
    holo = ((tfres + R) * np.conj(tfres + R)).real
    Rrec = fg.onda_ref(holo, lam, dx0, dy0, 180 - ang_ref)
    return dict(objeto=o, tfres=tfres, holograma=holo, espectro=np.fft.fftshift(np.fft.fft2(holo, axes=(0, 1)), axes=(0, 1)),
                lam=lam, dx0=dx0, dy0=dy0, z0=z0, Rrec=Rrec, zmin=zmin, angulo_max=angulo)


def cgh_fresnel(objeto, rng=None):
    """CGH_Fresnel.m (folder CGH_Fresnel_Final_Excelente): object with random phase in
    [-pi/4, pi/4), window 7.3 x 5.55 mm, z0 = 1.5*2.5*Zmin, reference 0.1 deg; reconstructions
    with TI_Fresnel and T_FresnelH of (H - mean) at -z0."""
    d = _cgh_fresnel_core(objeto, rng, np.pi / 2, 7.3e-3, 5.55e-3, 1.5 * 2.5, lambda a: 0.1, False)
    h0 = d['holograma'] - d['holograma'].mean()
    d['rec_ti'] = np.fft.fftshift(fg.ti_fresnel(h0, np.conj(d['Rrec']), d['lam'], d['dx0'], d['dy0'], -d['z0']))
    d['rec_h'] = np.fft.fftshift(fg.t_fresnel_h(h0, np.conj(d['Rrec']), d['lam'], d['dx0'], d['dy0'], -d['z0']))
    return d


def extraccion_fase(objeto):
    """Extraccion_Fase.m: object without random phase, window 9.03 x 6.77 mm, z0 = 1.3 Zmin,
    reference 0.6 theta_max; second reconstruction divides (H - mean) by 2|O||U_z|."""
    d = _cgh_fresnel_core(objeto, None, 0, 9.03e-3, 6.77e-3, 1.3, lambda a: 0.6 * a, True)
    d['rec_ti'] = np.fft.fftshift(fg.ti_fresnel(d['holograma'], np.conj(d['Rrec']), d['lam'], d['dx0'], d['dy0'], -d['z0']))
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = np.nan_to_num((d['holograma'] - d['holograma'].mean()) / (2 * np.abs(d['objeto']) * np.abs(d['tfres'])))
    d['rec_h'] = np.fft.fftshift(fg.t_fresnel_h(norm, np.conj(d['Rrec']), d['lam'], d['dx0'], d['dy0'], -d['z0']))
    return d


def _diffuse_spectrum(img, rng):
    F = img * np.exp(1j * _rng(rng).random(img.shape) * 2 * np.pi)
    return np.fft.fftshift(np.fft.fftn(F))


def holo_burch(imagen, rng=None, alphax_deg=-0.0, alphay_deg=-90.0):
    """Holo_Burch.m: B = 0.5 (1 - |F| cos(ax x + ay y - arg F)) with ax, ay in rad/pixel
    (-90 deg -> period of 4 pixels along y).
    Defect: |F| is not normalised, so B takes values far outside [0, 1] (not a transmittance)."""
    img = np.asarray(imagen, dtype=float)
    img = img[..., 0] if img.ndim == 3 else img
    F = _diffuse_spectrum(img, rng)
    n, m = img.shape
    x, y = np.meshgrid(np.arange(-m / 2, m / 2), np.arange(-n / 2, n / 2))
    B = 0.5 * (1 - np.abs(F) * np.cos(np.deg2rad(alphax_deg) * x + np.deg2rad(alphay_deg) * y - np.angle(F)))
    rec = np.abs(np.fft.ifftn(B))
    return B, (rec / rec.max()) ** 0.25


def holo_lee(imagen, rng=None, alphax_deg=0.0, alphay_deg=0.0):
    """Holo_Lee.m: L = |F| (cos(.) + |cos(.)|), a half-wave rectified carrier.
    Defect: the carrier angles are 0, so there is no off-axis separation of the orders."""
    img = np.asarray(imagen, dtype=float)
    img = img[..., 0] if img.ndim == 3 else img
    F = _diffuse_spectrum(img, rng)
    n, m = img.shape
    x, y = np.meshgrid(np.arange(-m / 2, m / 2), np.arange(-n / 2, n / 2))
    c = np.cos(np.deg2rad(alphax_deg) * x + np.deg2rad(alphay_deg) * y - np.angle(F))
    L = np.abs(F) * (c + np.abs(c))
    rec = np.abs(np.fft.ifftn(L))
    return L, (rec / rec.max()) ** 0.25


def lohmann(imagen, rng=None, dx=5, dy=5, w=2, M=0.5, gamma=0.15):
    """Lohmann.m (variants _a: w=1, _b: aperture height 5): binary detour-phase hologram.
    Each spectral sample becomes a cell of dx x dy pixels with an opaque slot of width w,
    shifted c = arg(F) dx/(2 pi) M along the rows and with height h = |F|/max|F| * dy."""
    img = np.asarray(imagen, dtype=float)
    img = img[..., None] if img.ndim == 2 else img
    n, m, p = img.shape
    F = np.fft.fftshift(np.fft.fftn(img * np.exp(1j * _rng(rng).random(img.shape) * 2 * np.pi)))
    fase, amp = np.angle(F), np.abs(F / np.abs(F).max())
    holo = np.ones((n * dx + dx + 2, m * dy + dy + 2, p))
    rmax, cmax = n * dx, m * dy                                  # MATLAB grows the array past the edge
    for q in range(p):
        for ii in range(1, n + 1):
            for jj in range(1, m + 1):
                cnm = fase[ii - 1, jj - 1, q] * dx / (2 * np.pi) * M
                hnm = amp[ii - 1, jj - 1, q] * dy
                r0, r1 = round(ii * dx + cnm - w / 2), round(ii * dx + cnm + w / 2)
                c0, c1 = round(jj * dy - hnm / 2), round(jj * dy + hnm / 2)
                holo[r0 - 1:r1, c0 - 1:c1, q] = 0
                rmax, cmax = max(rmax, r1), max(cmax, c1)
    holo = holo[:rmax, :cmax]
    rec = np.abs(np.fft.ifftn(holo))
    return np.squeeze(holo), np.squeeze((rec / rec.max()) ** gamma)


def kinoform_2015(imagen, rng=None):
    """KinoForm.m: F = exp(i 2 pi I R); kinoform = arg(FFT(F)) + pi; reconstruction from exp(i kinoform).

    The image is encoded as a phase object (random phase where I > 0, constant phase where I = 0),
    not as I exp(i 2 pi R). The reconstruction still shows the image (corr ~ 0.9): the constant-phase
    background concentrates in the zero order and the kinoform, by discarding the spectral amplitude,
    whitens the spectrum, so the random-phase region appears bright. Grey levels are not reproduced."""
    img = np.asarray(imagen, dtype=float)
    img = img[..., 0] if img.ndim == 3 else img
    F = np.fft.fftshift(np.fft.fftn(np.exp(img * 1j * _rng(rng).random(img.shape) * 2 * np.pi)))
    k1 = np.angle(np.exp(1j * np.angle(F))) + np.pi
    rec = np.abs(np.fft.ifftn(np.exp(1j * k1)))
    return np.uint8(255 * k1 / k1.max()), rec / rec.max()


def holo_kinoform_mask(field):
    """Mascara_de_Fase/HoloKinoform.m: kinoform = arg(F) + 1/2 (radians); correct phase-only
    reconstruction |fftshift(ifft2(fftshift(exp(i kinoform))))|."""
    F = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
    kino = np.angle(F) + 0.5
    obj = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.exp(1j * kino)))))
    return kino, obj


def mascara_de_fase(logo, rng=None, shape=(768, 1024)):
    """Mascara_de_Fase.m: binarised logo resized to the LC-R 2500 (768 x 1024), random diffuser,
    kinoform for the SLM. `logo` must already be resized to `shape` (imresize is not reproduced)."""
    a = (rgb2gray(logo) > 0.5).astype(float)          # im2bw on a 0-255 double image: threshold 0.5 absolute
    assert a.shape == shape
    R = _rng(rng).random(shape) - 0.5
    F = a * np.cos(R * 2 * np.pi) + 1j * a * np.sin(R * 2 * np.pi)
    kino, obj = holo_kinoform_mask(F)
    return kino, obj
