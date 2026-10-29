"""3D shape of a real object from a single off-axis hologram (depth from focus).

Usage: python audit/depth_from_focus.py <hologram.bmp> <z_min_cm> <z_max_cm> <step_cm> <out_prefix>

1. Isolate the +1 order (O R*) with a sideband filter: no zero order, no twin.
2. Refocus with the single-FFT Fresnel transform at every z of the scan.
3. Resample each plane to a common physical grid (Δξ grows with z, λz/(NΔx)).
4. Per-pixel focus measure: local Tenengrad of the speckle-averaged intensity.
5. Depth = z of maximum focus, kept only where the object is bright and the
   focus curve has a clear peak. Writes a depth map, an all-in-focus image,
   a PLY point cloud and a JSON summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from holom import filters
from holom.diffraction import fresnel

WL, DX = 532e-9, 2.2e-6


def sideband(h):
    S = np.abs(np.fft.fftshift(np.fft.fft2(h)))
    ny, nx = h.shape
    cy, cx = ny // 2, nx // 2
    S[cy - 60:cy + 61, cx - 60:cx + 61] = 0
    S[:, :cx - 2] = S[:, :cx - 2] if True else 0
    py, px = np.unravel_index(S.argmax(), S.shape)
    dy, dx = py - cy, px - cx
    r = int(0.45 * np.hypot(dy, dx))
    return filters.sideband_filter(h, (dy, dx), r), (dy, dx, r)


def main():
    path, z0, z1, dz, prefix = sys.argv[1], *map(float, sys.argv[2:5]), sys.argv[5]
    h = np.asarray(Image.open(path))[..., 1].astype(float)
    n = min(h.shape)
    h = h[:n, :n] - h[:n, :n].mean()                    # square crop of the sensor
    U0, sb = sideband(h)
    zs = np.arange(z0, z1 + 1e-9, dz) / 100
    zref = zs[len(zs) // 2]
    ds = 4                                              # analysis grid: n/ds pixels
    m = n // ds
    stack, inten = [], []
    for z in zs:
        u, dxi, _ = fresnel(U0, WL, DX, DX, -z)
        I = np.abs(u) ** 2
        I = I.reshape(m, ds, m, ds).mean(axis=(1, 3))     # block average: speckle reduction
        # resample to the physical grid of zref: coordinate scale z/zref
        k = zref / z
        c = (m - 1) / 2
        I = ndimage.affine_transform(I, [k, k], offset=[c - k * c, c - k * c], order=1)
        Is = ndimage.gaussian_filter(I, 2.0)
        gx, gy = ndimage.sobel(Is, 1), ndimage.sobel(Is, 0)
        F = ndimage.uniform_filter(gx * gx + gy * gy, 15) / (ndimage.uniform_filter(Is, 15) ** 2 + 1e-12)
        stack.append(F); inten.append(I)
    stack, inten = np.array(stack), np.array(inten)
    k = stack.argmax(0)
    peak = stack.max(0) / (np.median(stack, 0) + 1e-12)   # focus-curve contrast
    depth = zs[k] * 100
    allfocus = np.take_along_axis(inten, k[None], 0)[0]
    bright = ndimage.gaussian_filter(allfocus, 3)
    mask = (bright > np.percentile(bright, 75)) & (peak > 1.5)
    dmed = ndimage.median_filter(np.where(mask, depth, np.nan), 9)

    out = Path(prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(allfocus ** 0.5, cmap='gray'); ax[0].set_title('Imagen todo-enfocado')
    im = ax[1].imshow(np.where(mask, dmed, np.nan), cmap='viridis'); ax[1].set_title('Mapa de profundidad z (cm)')
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    yy, xx = np.nonzero(mask)
    ax[2].hist(depth[mask], bins=40, color='#b3246b'); ax[2].set_title('Distribución de z en el objeto'); ax[2].set_xlabel('z (cm)')
    for a in ax[:2]: a.axis('off')
    fig.tight_layout(); fig.savefig(str(out) + '_depth.png', dpi=90); plt.close(fig)

    dxi_ref = WL * zref / (n * DX) * ds * 1e3          # mm per analysis pixel
    pts = [(float((x - m / 2) * dxi_ref), float((y - m / 2) * dxi_ref), float(depth[y, x] * 10), float(allfocus[y, x]))
           for y, x in zip(yy, xx)]
    imax = max(p[3] for p in pts) or 1
    with open(str(out) + '.ply', 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for x, y, z, i in pts:
            v = int(255 * min(1, (i / imax) ** 0.5))
            f.write(f'{x:.3f} {y:.3f} {z:.3f} {v} {v} {v}\n')
    summary = {'hologram': path, 'sideband_dy_dx_r': sb, 'z_scan_cm': [z0, z1, dz], 'z_ref_cm': zref * 100,
               'pixel_mm': dxi_ref, 'points': len(pts),
               'depth_cm_percentiles_5_50_95': [float(np.percentile(depth[mask], q)) for q in (5, 50, 95)] if mask.any() else None,
               'depth_of_field_mm_lambda_over_NA2': float(WL / (n * DX / (2 * zref)) ** 2 * 1e3)}
    Path(str(out) + '.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
