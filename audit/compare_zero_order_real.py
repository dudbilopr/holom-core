# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Zero-order suppression on real 2015 holograms (with standalone synthetic fallback).

The four phase-shifted aeroplane holograms (I1-I4, 16/04/2015) give the object wave O R* free of
zero order and twin (holom.filters.phase_shifting_4step). Its reconstruction at z = 23 cm (the
distance in PhaseShift.m) is the reference. Each single-shot method is applied to I1 alone and
compared with it. When the local 2015 thesis archive is absent, a deterministic synthetic 4-step
off-axis sequence matching the optical setup is generated instead.

Usage: python audit/compare_zero_order_real.py [--holo-dir PATH] [--require-real]
"""
import argparse
import sys
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


def load_real(holo_dir):
    return [np.asarray(Image.open(holo_dir / f'I{k}Holograma_Avion_16042015_22.bmp'))[:N, :N, 1].astype(float)
            for k in (1, 2, 3, 4)]


def simulate_4step_offaxis():
    """Standalone 4-step phase-shifted off-axis Fresnel sequence matching the 2015 geometry."""
    rng = np.random.default_rng(0)
    obj = np.zeros((N, N), dtype=float)
    yy, xx = np.ogrid[:N, :N]
    # compact asymmetric target off-center inside the reconstruction field
    obj[((yy - 760) ** 2 + (xx - 1180) ** 2) < 95 ** 2] = 1.0
    obj[710:810, 990:1150] = 1.0
    obj[660:725, 1140:1190] = 1.0
    u_obj = obj * np.exp(1j * 2 * np.pi * rng.random((N, N)))
    O, _, _ = fresnel(u_obj, WL, DX, DX, abs(Z))
    O = O / np.abs(O).max()
    off = (52, -345)
    ky, kx = 2 * np.pi * off[0] / N, 2 * np.pi * off[1] / N
    Y, X = np.ogrid[:N, :N]
    R = np.exp(1j * (ky * Y + kx * X))
    I = [np.abs(O + R * np.exp(1j * k * np.pi / 2)) ** 2 for k in range(4)]
    return I, off


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holo-dir', type=Path, default=D, help='Directory containing I1..I4 aeroplane BMP files')
    parser.add_argument('--require-real', action='store_true', help='Fail if real 2015 BMP holograms are missing')
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    real_files_exist = all((args.holo_dir / f'I{k}Holograma_Avion_16042015_22.bmp').exists() for k in (1, 2, 3, 4))
    if real_files_exist:
        I = load_real(args.holo_dir)
        off = (52, -345)
        source_note = 'Hologramas reales del avión (16/04/2015, archivo de tesis).'
    else:
        if args.require_real:
            print(f'Error: missing real 2015 holograms in {args.holo_dir}', file=sys.stderr)
            sys.exit(1)
        I, off = simulate_4step_offaxis()
        source_note = 'SIMULATED FALLBACK (archivo local de tesis 2015 no montado; secuencia sintética equivalente).'

    ref = recon(filters.phase_shifting_4step(*I))
    mask = ndimage.gaussian_filter(ref, 2) > np.percentile(ndimage.gaussian_filter(ref, 2), 85)
    h = I[0]
    radius = int(0.45 * np.hypot(*off))
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
    lines = ['# Supresión del orden cero sobre hologramas (avión, z = 23 cm)', '',
             f'Fuente de datos: {source_note}',
             'Referencia: O·R* de las 4 capturas con corrimiento de fase, reconstruido a z = 23 cm. '
             'Métodos de una captura aplicados a I1. Región del objeto: 15 % más brillante de la referencia.', '',
             '| Método | Corr. con la referencia (objeto) | Energía fuera del objeto | Exceso sobre la referencia |',
             '|---|---|---|---|']
    for name, c, f in rows:
        lines.append(f'| {name} | {c:.3f} | {f:.3f} | {f - ref_out:+.3f} |')
    lines += ['', f'Portadora: ({off[0]}, {off[1]}) px; se usó el orden {"+1" if sign > 0 else "-1"}.',
              'El script de 2015 (PhaseShift.m) produjo una reconstrucción NaN (negra) con los datos reales.']
    (OUT / 'zero_order_real.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for a, (name, A) in zip(ax.ravel(), methods.items()):
        a.imshow(np.sqrt(A), cmap='gray', vmax=np.percentile(np.sqrt(A), 99.7)); a.set_title(name, fontsize=10); a.axis('off')
    fig.tight_layout(); fig.savefig(OUT / 'zero_order_real.png', dpi=60)
    plt.close(fig)


if __name__ == '__main__':
    main()
