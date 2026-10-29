"""Joint transform correlator encryption (JTC/, JTC_Jorge_Rueda/).

Code by Prof. Jorge E. Rueda-Parada, ported for reproducibility and cited accordingly.
Layout of every script: 768 x 1024 input plane, two 101 x 101 windows at rows 334:434,
key at columns 544+t:644+t, object at 280-t:380-t, t = 12.
"""
import numpy as np

from .matlab import imnoise_speckle

ROWS = slice(333, 434)


def _plane(key, obj, t=12, shape=(768, 1024)):
    f = np.zeros(shape, dtype=complex)
    f[ROWS, 543 + t:644 + t] = key
    kk1 = f.copy()
    if obj is not None:
        f[ROWS, 279 - t:380 - t] = obj
    return f, kk1


def speckle_phase_key(rng, size=101, var=10):
    """K = exp(i angle(fftshift(fft2(imnoise(ones, 'speckle', 10)))))."""
    noisy = imnoise_speckle(np.ones((size, size)), var, rng)
    return np.exp(1j * np.angle(np.fft.fftshift(np.fft.fft2(noisy))))


def jtc_speckle(objeto, rng=None, t=12):
    """JTC/JTC.m: speckle-derived phase key K and object mask M; I_E = |fft2(f)|^2;
    decryption |ifft2(I_E .* fft2(KK1))|^0.3 with the key alone in the input plane."""
    rng = np.random.default_rng(rng)
    K, M = speckle_phase_key(rng), speckle_phase_key(rng)
    o = np.asarray(objeto, dtype=float)
    o = o[..., 0] if o.ndim == 3 else o
    f, kk1 = _plane(K, o * M, t)
    IE = np.abs(np.fft.fft2(f)) ** 2
    dec = np.abs(np.fft.ifft2(IE * np.fft.fft2(kk1))) ** 0.3
    return f, IE, dec


def jtc_rueda_script(objeto, rng=None, t=12):
    """JTC_Jorge_Rueda.m: key and object mask drawn as phases in [-pi, pi).
    Defect: they are used directly as real amplitudes (K = fase1), not as exp(i fase1)."""
    rng = np.random.default_rng(rng)
    K = (rng.random((101, 101)) - 0.5) * 2 * np.pi
    M = (rng.random((101, 101)) - 0.5) * 2 * np.pi
    o = np.asarray(objeto, dtype=float)
    o = o[..., 0] if o.ndim == 3 else o
    f, kk1 = _plane(K, o * M, t)
    IE = np.fft.fftshift(np.fft.fft2(f))
    IE = (IE * np.conj(IE)).real
    dec = np.abs(np.fft.ifft2(IE * np.fft.fftshift(np.fft.fft2(kk1)))) ** 0.3
    return f, IE, dec


def jtc_gui_encrypt(objeto, llave, intensity=False):
    """JTC.m (GUI): I_E = abs(fftshift(fft2(PE))).
    Defect: a camera records |F|^2; the GUI keeps |F| (intensity=True gives the physical record)."""
    F = np.fft.fftshift(np.fft.fft2(objeto + llave))
    return np.abs(F) ** 2 if intensity else np.abs(F)


def jtc_gui_decrypt(IE, llave_plane):
    """JTC.m (GUI): I = abs(ifft2(I_E .* fftshift(fft2(key plane))))."""
    return np.abs(np.fft.ifft2(IE * np.fft.fftshift(np.fft.fft2(llave_plane))))


def decriptar(espectro_conjunto, llave):
    """Decriptar.m / Decrip.m: Desc = fft2(EC .* fftshift(fft2(key)))."""
    return np.fft.fft2(espectro_conjunto * np.fft.fftshift(np.fft.fft2(llave)))
