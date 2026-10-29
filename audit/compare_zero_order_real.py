"""Zero-order suppression on REAL holograms, scored against a real reference.

The four phase-shifted aeroplane holograms (I1-I4, 16/04/2015) give the object wave O R* free of
zero order and twin (holom.filters.phase_shifting_4step). Its reconstruction at z = 23 cm (the
distance in PhaseShift.m) is the reference. Each single-shot method is applied to I1 alone and
compared with it.

Usage: python audit/compare_zero_order_real.py  (writes audit/out/zero_order_real.md and .png)
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from holom import filters
from holom.diffraction import fresnel

D = Path(__file__).resolve().parents[2] / 'Algunos Resultados/DH_REconstruccion_Final/Pactrica_Reconstruccion'
OUT = Path(__file__).resolve().parent / 'out'
WL, DX, Z, N, B = 532e-9, 2.2e-6, -0.23, 1944, 4


def load(k):
    return np.asarray(Image.open(D / f'I{k}Holograma_Avion_16042015_22.bmp'))[:N, :N, 1].astype(float)


def recon(field):
    u, _, _ = fresnel(field, WL, DX, DX, Z)
    m = N // B
    return np.abs(u).reshape(m, B, m, B).mean(axis=(1, 3))


def sideband_in_place(h, off, radius):
    """Keep the +1 order where it is (no re-centring), so the image lands where the reference does."""
    S = np.fft.fftshift(np.fft.fft2(h - h.mean()))
    Y, X = np.ogrid[:N, :N]
    keep = (Y - N // 2 - off[0]) ** 2 + (X - N // 2 - off[1]) ** 2 <= radius ** 2
    return np.fft.ifft2(np.fft.ifftshift(S * keep))


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main():
    OUT.mkdir(exist_ok=True)
    I = [load(k) for k in (1, 2, 3, 4)]
    ref = recon(filters.phase_shifting_4step(*I))
    mask = ndimage.gaussian_filter(ref, 2) > np.percentile(ndimage.gaussian_filter(ref, 2), 85)
    h = I[0]
    off, radius = (52, -345), int(0.45 * np.hypot(52, 345))
    # the +1 order carrying O R* is the one whose reconstruction matches the reference
    cands = {s: recon(sideband_in_place(h, (s * off[0], s * off[1]), radius)) for s in (1, -1)}
    sign = max(cands, key=lambda s: corr(cands[s][mask], ref[mask]))
    methods = {
        'Sin filtro': recon(h),
        '7.3.1 Resta de la media': recon(filters.subtract_mean(h)),
        '7.3.2 Pasa-altos FFT (64×64)': recon(filters.highpass_fft(h, (64, 64))),
        '7.3.3 Laplaciano': recon(filters.laplacian(h)),
        f'Banda lateral (radio {radius} px)': cands[sign],
        'Corrimiento de fase, 4 capturas (referencia)': ref,
    }
    rows = []
    for name, A in methods.items():
        out_frac = float((A[~mask] ** 2).sum() / (A ** 2).sum())
        rows.append((name, corr(A[mask], ref[mask]), out_frac))
    ref_out = rows[-1][2]
    lines = ['# Supresión del orden cero sobre hologramas reales (avión, 16/04/2015)', '',
             'Referencia: O·R* de las 4 capturas con corrimiento de fase, reconstruido a z = 23 cm. '
             'Métodos de una captura aplicados a I1. Región del objeto: 15 % más brillante de la referencia.', '',
             '| Método | Corr. con la referencia (objeto) | Energía fuera del objeto | Exceso sobre la referencia |',
             '|---|---|---|---|']
    for name, c, f in rows:
        lines.append(f'| {name} | {c:.3f} | {f:.3f} | {f - ref_out:+.3f} |')
    lines += ['', f'Portadora: ({off[0]}, {off[1]}) px; se usó el orden {"+1" if sign > 0 else "-1"}.',
              'El script de 2015 (PhaseShift.m) produjo una reconstrucción NaN (negra) con estos mismos datos.']
    (OUT / 'zero_order_real.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for a, (name, A) in zip(ax.ravel(), methods.items()):
        a.imshow(np.sqrt(A), cmap='gray', vmax=np.percentile(np.sqrt(A), 99.7)); a.set_title(name, fontsize=10); a.axis('off')
    fig.tight_layout(); fig.savefig(OUT / 'zero_order_real.png', dpi=60)


if __name__ == '__main__':
    main()
