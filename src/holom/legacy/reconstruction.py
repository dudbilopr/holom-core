"""Reconstruction scripts of the 2015 work (DH_REconstruccion_Final/, Difraccion_Por_Convolucion/,
DHFresnel_GOM/ GUIs). Interactive selections become explicit arguments."""
import numpy as np

from . import fresnel_gom as fg
from .matlab import fspecial_unsharp, graythresh, im2bw, imfilter_replicate, medfilt2, padarray, rgb2gray, unwrap_columns

LAMBDA_SCRIPTS = 532.8e-9      # value typed in DH_Reconstruccion*.m and PhaseShift.m (the thesis uses 532 nm)
PIXEL = 2.2e-6


def phase_shift(i1, i2, i3, i4, lam=LAMBDA_SCRIPTS, dx0=PIXEL, dy0=PIXEL, z0=-0.23, channel=1):
    """PhaseShift.m: 4-step phase shifting of the aeroplane holograms (I1..I4, z0 = -0.23 m).

    Fase = unwrap(atan((I4 - I2)./(I1 - I3))), Amplitud = 0.25 sqrt((I1-I3)^2 + (I4-I2)^2),
    then T_Fresnel with Onda_Ref = 1.
    Defects: atan instead of atan2 keeps the phase in (-pi/2, pi/2) and loses the quadrant;
    unwrap acts down the columns only; where I1 = I3 and I2 = I4 the quotient is 0/0 = NaN,
    unwrap carries it down the column and the FFT spreads it over the whole plane, so the
    reconstruction the script saved is entirely black. Use holom.filters.phase_shifting_4step.
    Returns (field_at_sensor, reconstruction)."""
    I = [np.asarray(a, dtype=float)[..., channel] if np.ndim(a) == 3 else np.asarray(a, dtype=float) for a in (i1, i2, i3, i4)]
    with np.errstate(divide='ignore', invalid='ignore'):
        fase = unwrap_columns(np.arctan((I[3] - I[1]) / (I[0] - I[2])))   # 0/0 -> NaN, as in MATLAB
    amplitud = 0.25 * ((I[0] - I[2]) ** 2 + (I[3] - I[1]) ** 2) ** 0.5
    H = amplitud * np.exp(1j * fase)
    return H, fg.t_fresnel(H, 1, lam, dx0, dy0, z0)


def dh_reconstruccion_ultimo(holograma, rect, lam=LAMBDA_SCRIPTS, dx0=PIXEL, dy0=PIXEL, z0=-0.13):
    """DH_Reconstruccion_Ultimo.m: grey hologram / max, minus its mean, rectangular spectral
    window rect = (x, y, w, h) in 1-based pixel units (what imrect returned), Fresnel at z0.
    Returns (spectrum, filtered_hologram, reconstruction)."""
    h = rgb2gray(holograma)
    h = h / h.max()
    h = h - h.mean()
    S = np.fft.fftshift(np.fft.fft2(h))
    x, y, w, hh = [int(round(v)) for v in rect]
    F = np.zeros_like(h)
    F[y - 1:y + hh, x - 1:x + w] = 1
    hf = np.fft.ifft2(S * F)
    return S, hf, np.fft.fftshift(fg.t_fresnel(hf, 1, lam, dx0, dy0, z0))


def fcircular_tfre(holograma, centre, edge, lam=532e-9, dx0=PIXEL, dy0=PIXEL, z0=0.140):
    """FCircularTfre.m: circular spectral filter picked with two clicks (centre, a point on the edge),
    Fresnel at z0 = 140 mm, then a 3x3 median filter.

    Defect in the original folder: it calls T_Fresnel with 5 arguments while the local T_Fresnel.m
    takes 6, so the script fails as distributed; here the reference wave is 1 and the
    median-filtered T_Fresnel of that folder is used."""
    h = np.asarray(holograma, dtype=float)
    h = h / h.max()
    h = rgb2gray(h)
    n, m = h.shape
    radio = round(np.hypot(centre[0] - edge[0], centre[1] - edge[1]))
    S = np.fft.fftshift(np.fft.fft2(h))
    filtro = fg.circulo(n, m, radio, centre[0], centre[1])
    hf = np.fft.ifft2(S * filtro)
    rec = np.fft.fftshift(fg.t_fresnel_medfilt(hf, 1, lam, dx0, dy0, z0))
    return S, filtro, hf, medfilt2(rec)


def dhr_reconstruccion(holo, lam=LAMBDA_SCRIPTS, dx0=PIXEL, dy0=PIXEL, z0=-0.21, crop=500, m=1 / 5):
    """DHR_REconstruccion.m: 500x500 crop, zero padding of N/m on every side, then the
    convolution propagator. Returns the normalised |U|^0.25 image the script saved."""
    h = np.asarray(holo, dtype=float)
    h = h / h.max()
    h = h[:crop, :crop]
    n, mm = h.shape[:2]
    h = padarray(h, mm / m, n / m)
    U = fg.difraccion_convolucion(h, 1, lam, dx0, dy0, z0, 0)
    return np.abs(U / np.abs(U).max()) ** 0.25


def hdfresnel_filter(holo, spectral_mask, lam=532e-9, dx0=PIXEL, dy0=PIXEL, z_cm=12.0):
    """Filtro_Callback of HDFresnel_GOM_V1.m (2015 GUI).

    medfilt2(H - mean) -> unsharp mask (fspecial('unsharp'), replicate) -> Otsu threshold and
    binarisation (im2bw) -> spectral mask (drawn with roipoly) -> T_Fresnel at z.
    Defect: binarising the hologram is non-linear and creates spurious harmonics in the spectrum."""
    h = medfilt2(np.asarray(holo, dtype=float) - np.mean(holo))
    h = imfilter_replicate(h, fspecial_unsharp())
    h = im2bw(h, graythresh(h))
    S = np.fft.fftshift(np.fft.fft2(h))
    hf = np.fft.ifft2(S * spectral_mask)
    return h, np.fft.fftshift(fg.t_fresnel_medfilt(hf, 1, lam, dx0, dy0, z_cm * 1e-2))


def dhfresnel_reconstruct(holo, lam=532e-9, dx0=PIXEL, dy0=PIXEL, z_cm=12.0):
    """Fresnel_Callback of HDFresnel_GOM_V1.m / DHFresnel_GOM_V1.m: T_Fresnel (median-filtered
    magnitude) of the captured grey hologram; dx, dy in um and z in cm as typed in the GUI."""
    return fg.t_fresnel_medfilt(np.asarray(holo, dtype=float), 1, lam, dx0, dy0, z_cm * 1e-2)


def simulate_dh_reconstruccion(objeto, lam=632.8e-9, dx0=6.8e-6, dy0=6.8e-6, z0=1.054):
    """DH_REconstruccion_Final/DH_Reconstruccion.m: Fresnel transform of (hologram - mean) with a
    reference of angle 0. The parameters (He-Ne 632.8 nm, 6.8 um pixel, z0 = 1.054 m) and the
    file Holograma_U_Shnars_2005.png in that folder indicate the Schnars & Jueptner example
    hologram, not the author's 2015 set-up."""
    o = np.asarray(objeto, dtype=float)
    R = fg.onda_ref(o, lam, dx0, dy0, 0)
    return fg.t_fresnel(o - o.mean(), R, lam, dx0, dy0, z0)


def dh_reconstruccion_convolucion(objeto, lam=632.8e-9, dx0=6.8e-6, dy0=6.8e-6, z0=1.054, crop=780, m=1 / 3.5):
    """Difraccion_Por_Convolucion/DH_Reconstruccion.m: 780x780 crop of the Schnars example
    hologram, zero padding of N/m = 2730 px per side (5460 x 5460... plus the crop), then the
    convolution propagator at z0 = 1.054 m. Returns the complex field."""
    o = np.asarray(objeto, dtype=float)[:crop, :crop]
    n, mm = o.shape[:2]
    o = padarray(o, n / m, mm / m)
    return fg.difraccion_convolucion(o, 1, lam, dx0, dy0, z0, m * 1.054)
