"""3-D shape of real objects by depth from focus (holom.focus.depth_from_focus).

Usage: python audit/depth_from_focus.py   (writes audit/out/depth_*.png, *.ply, depth_real.json)

- Aeroplane: object wave O R* from the four phase-shifted holograms (no zero order, no twin).
- Die: single hologram minus its mean; the analysis is limited to the window of its real image.
Depth is the z of minimum local focus measure (validated on two simulated depths). An earlier
version of this script filled pixels outside each plane with 0, which created false foci at
the edge of the scan; its die result (median 12.0 cm) is superseded.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from holom import filters, focus
from holom.diffraction import fresnel

R = Path(__file__).resolve().parents[2] / 'Algunos Resultados'
OUT = Path(__file__).resolve().parent / 'out'
WL, DX, N, B = 532e-9, 2.2e-6, 1944, 4


def green(p):
    return np.asarray(Image.open(p))[:N, :N, 1].astype(float)


def intensity(field, z):
    u, _, _ = fresnel(field, WL, DX, DX, z)
    m = N // B
    return (np.abs(u) ** 2).reshape(m, B, m, B).mean(axis=(1, 3))


def analyse(name, field, zs, window=None):
    depth, valid, pitch = focus.depth_from_focus(field, WL, DX, zs, block=B)
    z_ref = float(zs[len(zs) // 2])
    I = intensity(field, z_ref)
    Is = ndimage.gaussian_filter(I, 2)
    bright = Is > np.percentile(Is if window is None else Is[window], 70)
    mask = valid & bright
    if window is not None:
        w = np.zeros_like(mask); w[window] = True; mask &= w
    zcm = -depth * 100
    n_obj = int(mask.sum())
    interior = (depth != zs[0]) & (depth != zs[-1])    # a minimum on the scan edge is not a focus
    mask &= interior
    med = ndimage.median_filter(np.where(mask, zcm, np.nan), 7)
    keep = mask & (np.abs(zcm - med) < 1.0)          # drop isolated outliers
    vals = zcm[keep]
    summary = {'name': name, 'z_scan_cm': [float(-zs[0] * 100), float(-zs[-1] * 100)], 'z_ref_cm': -z_ref * 100,
               'pixel_mm': pitch * 1e3, 'object_pixels': n_obj, 'interior_minima': int(mask.sum()),
               'points': int(keep.sum()), 'z_cm_iqr': float(np.subtract(*np.percentile(vals, [75, 25]))) if vals.size else None,
               'z_cm_p5_p50_p95': [float(np.percentile(vals, q)) for q in (5, 50, 95)] if vals.size else None}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    ax[0].imshow(np.sqrt(I), cmap='gray', vmax=np.percentile(np.sqrt(I), 99.7)); ax[0].set_title(f'{name}: intensidad a z = {-z_ref*100:.1f} cm')
    im = ax[1].imshow(np.where(keep, zcm, np.nan), cmap='viridis'); ax[1].set_title('Mapa de profundidad z (cm)')
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].hist(vals, bins=40, color='#b3246b'); ax[2].set_xlabel('z (cm)'); ax[2].set_title('Distribución de z en el objeto')
    for a in ax[:2]:
        a.axis('off')
    fig.tight_layout(); fig.savefig(OUT / f'depth_{name}.png', dpi=60); plt.close(fig)
    yy, xx = np.nonzero(keep)
    m = I.shape[0]
    imax = np.sqrt(I[keep]).max() if len(yy) else 1
    with open(OUT / f'depth_{name}.ply', 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(yy)}\nproperty float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for y, x in zip(yy, xx):
            v = int(255 * min(1, np.sqrt(I[y, x]) / imax))
            f.write(f'{(x - m / 2) * pitch * 1e3:.3f} {(y - m / 2) * pitch * 1e3:.3f} {zcm[y, x] * 10:.2f} {v} {v} {v}\n')
    print(summary)
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    d = R / 'DH_REconstruccion_Final/Pactrica_Reconstruccion'
    I = [green(d / f'I{k}Holograma_Avion_16042015_22.bmp') for k in (1, 2, 3, 4)]
    res = [analyse('avion_psi', filters.phase_shifting_4step(*I), -np.arange(0.15, 0.4001, 0.005))]
    dado = filters.subtract_mean(green(R / 'DHFresnel_GOM/2Holograma_dado_16042015_14.bmp'))
    # real-image window chosen on a reconstruction at 13 cm (rows 245-330, cols 290-395), rescaled to z_ref = 14 cm
    k, c = 0.14 / 0.13, (N // B - 1) / 2
    win = (slice(int(c + (245 - c) * k), int(c + (330 - c) * k)), slice(int(c + (290 - c) * k), int(c + (395 - c) * k)))
    res.append(analyse('dado', dado, -np.arange(0.08, 0.2001, 0.0025), window=win))
    (OUT / 'depth_real.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
