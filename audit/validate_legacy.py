# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Check the Python ports against images the original MATLAB scripts saved in Algunos Resultados/.

Usage: python audit/validate_legacy.py [--archive-dir PATH] [--strict]
Writes audit/out/validation_legacy.md and PNGs when the external MATLAB outputs archive is present.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from holom.legacy import reconstruction as rec

AR = Path(__file__).resolve().parents[2] / 'Algunos Resultados'
OUT = Path(__file__).resolve().parent / 'out'
rows = []


def load(p, gray=True):
    im = Image.open(p)
    return np.asarray(im.convert('L') if gray else im, dtype=float)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def figure(name, ref, ours, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(ref, cmap='gray'); ax[0].set_title('Salida guardada por MATLAB')
    ax[1].imshow(ours, cmap='gray'); ax[1].set_title('Port en Python (holom.legacy)')
    for a in ax: a.axis('off')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(OUT / name, dpi=70); plt.close(fig)


def check_phase_shift():
    d = AR / 'DH_REconstruccion_Final/Pactrica_Reconstruccion'
    I = [np.asarray(Image.open(d / f'I{k}Holograma_Avion_16042015_22.bmp'), dtype=float) for k in (1, 2, 3, 4)]
    t0 = time.perf_counter()
    _, U = rec.phase_shift(*I)
    ours = np.abs(U / np.abs(U).max())
    ref = load(d / 'Transformada_de_Fresnel.jpg') / 255
    # the saved MATLAB image is all black: imwrite of |U|/max at 8 bits, where the zero order dominates
    ours8 = np.round(255 * ours)
    agree = float(np.mean((ours8 > 0) == (ref > 0)))
    figure('val_phase_shift.png', ref, np.log1p(ours * 1e6), f'PhaseShift.m: MATLAB guardó negro; port 8 bits coincide en {agree:.4%} de píxeles')
    rows.append(('PhaseShift.m (avión, 4 pasos, z = -0.23 m): coincidencia píxel a píxel a 8 bits', ref.shape, agree, time.perf_counter() - t0))


def check_convolution():
    d = AR / 'Difraccion_Por _Convolucion'
    holo = load(d / 'Holograma_U_Shnars_2005.png', gray=False)[:780, :780]
    t0 = time.perf_counter()
    # The saved 5460 px JPEG corresponds to m = 1/3 (780 + 2*2340). The original fftn/ifftn over an
    # RGB stack with a kernel identical in every channel equals a 2-D convolution of the channel sum
    # (tests/test_legacy.py::test_rgb_fftn_equals_channel_sum), which avoids a 3 GB array.
    from holom.legacy import fresnel_gom as fg
    from holom.legacy.matlab import padarray
    s = padarray(holo.sum(axis=2), 780 * 3, 780 * 3)
    U = fg.difraccion_convolucion(s, 1, 632.8e-9, 6.8e-6, 6.8e-6, 1.054)
    ours = np.abs(U / np.abs(U).max())          # that JPEG was saved without the ^0.25 of the current script
    from PIL import Image as _I
    _I.MAX_IMAGE_PIXELS = None
    ref = np.asarray(_I.open(d / 'Transformada_de_Fresnel.jpg').convert('L'), dtype=float) / 255
    c = corr(ours, ref)
    figure('val_convolucion.png', ref, np.clip(ours, 0, 1), f'DH_Reconstruccion.m (convolución, 5460 px, sin gamma), corr = {c:.4f}')
    rows.append(('DH_Reconstruccion.m convolución (holograma U de Schnars, 5460 px)', ref.shape, c, time.perf_counter() - t0))
    black = np.asarray(Image.open(d / 'Transformada_de_Fresnel.bmp'), dtype=float)
    rows.append(('Transformada_de_Fresnel.bmp (versión m = 2) guardada en negro por MATLAB', black.shape, float(black.std()), 0.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, default=AR, help='Path to Algunos Resultados/')
    parser.add_argument('--strict', action='store_true', help='Exit non-zero if external MATLAB archive is missing')
    args = parser.parse_args()
    AR = args.archive_dir

    OUT.mkdir(parents=True, exist_ok=True)
    if not (AR / 'DH_REconstruccion_Final/Pactrica_Reconstruccion/Transformada_de_Fresnel.jpg').exists():
        msg = f'External MATLAB output archive not found at {AR}; skipping reference image check.'
        print(msg)
        if args.strict:
            sys.exit(1)
        sys.exit(0)
    check_phase_shift()
    check_convolution()
    lines = ['# Validación de los ports contra salidas guardadas por MATLAB', '',
             '| Script | Tamaño | Correlación | Tiempo (s) |', '|---|---|---|---|']
    lines += [f'| {n} | {s} | {c:.4f} | {t:.1f} |' for n, s, c, t in rows]
    (OUT / 'validation_legacy.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
