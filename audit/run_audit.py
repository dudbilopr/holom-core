"""Quantitative audit of the legacy MATLAB algorithms through their Python ports.

Usage:  python audit/run_audit.py
Writes audit/out/results.md and PNG figures. All inputs are synthetic with
known ground truth, using the optical parameters reported in the theses.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from holom import cgh, filters
from holom.diffraction import angular_spectrum, fresnel, fresnel_legacy
from holom.simulate import THESIS_2015, fresnel_off_axis_hologram, max_carrier_angle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OBJ_2012 = ROOT / "Tesis Pregrado/FUENTES_Trabajo_de_GradoHGC-GOM2012/Archivos_Programa/Objetos/GOM512X384.bmp"
lines = []


def log(s=""):
    lines.append(s)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def audit_cgh():
    log("## 1. Hologramas generados por computador (HGC_GOM_V2012)")
    log()
    img = np.asarray(Image.open(OBJ_2012).convert("L"))
    log(f"Objeto: `{OBJ_2012.name}` {img.shape[1]}x{img.shape[0]} px, semilla RNG = 0. "
        "Reconstrucción digital tal como la calcula cada rutina MATLAB.")
    log()
    n, m = img.shape
    log("| Algoritmo | Corr. copia invertida | Corr. copia directa | Energía en ventanas objeto | Energía orden cero |")
    log("|---|---|---|---|---|")
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for k, (name, fn) in enumerate(cgh.ALGORITHMS.items()):
        A, H, O = fn(img, rng=0)
        I = O**2
        obj = A[:n, :m]
        # the fft-based reconstruction maps r -> -r about the centre: one copy lands
        # where A was, the point-reflected one in the opposite quadrant
        win_up = I[:n, :m]
        win_inv = np.flip(np.roll(I, (-1, -1), axis=(0, 1)), axis=(0, 1))[:n, :m]
        c = I.shape[0] // 2, I.shape[1] // 2
        dc = I[c[0] - 3:c[0] + 4, c[1] - 3:c[1] + 4].sum() / I.sum()
        log(f"| {name} | {corr(win_inv, obj):.3f} | {corr(win_up, obj):.3f} | "
            f"{(win_up.sum() + win_inv.sum()) / I.sum():.3f} | {dc:.3f} |")
        axes[0, k].imshow(np.real(H), cmap="gray"); axes[0, k].set_title(name); axes[0, k].axis("off")
        axes[1, k].imshow(np.sqrt(O), cmap="gray"); axes[1, k].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "cgh_2012.png", dpi=80); plt.close(fig)
    shift_px = cgh.ALPHA / cgh.LEGACY_PERIOD * 2 * m
    log()
    log(f"Portadora heredada: alpha/127 = {cgh.ALPHA / cgh.LEGACY_PERIOD:.4f} ciclos/px, "
        f"desplazamiento espectral de {shift_px:.1f} px en una ventana de {2 * m} px. "
        "La separación de órdenes la produce el desplazamiento (d, s) del objeto en el plano 2Nx2M, no la portadora.")
    log()


def audit_kinoform():
    log("### 1b. Reconstrucción del kinoform")
    log()
    img = np.asarray(Image.open(OBJ_2012).convert("L"))
    n, m = img.shape
    A, H, _ = cgh.kinoform(img, rng=0)
    log("| Reconstrucción | Corr. con el objeto | Eficiencia (energía en ventana) | Orden cero |")
    log("|---|---|---|---|")
    for label, t in (("HoloKinoform.m: ifft2(H), fase tratada como amplitud", H),
                     ("Física: ifft2(exp(i 2 pi H))", np.exp(2j * np.pi * H))):
        I = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(t)))) ** 2
        c = I.shape[0] // 2, I.shape[1] // 2
        log(f"| {label} | {corr(I[:n, :m], A[:n, :m]):.3f} | {I[:n, :m].sum() / I.sum():.3f} | "
            f"{I[c[0] - 3:c[0] + 4, c[1] - 3:c[1] + 4].sum() / I.sum():.3f} |")
    log()


def audit_fresnel_legacy():
    log("## 2. Transformada de Fresnel heredada (HOLO_GOM/Fresnel.m)")
    log()
    p = THESIS_2015
    ny, nx = 972, 1296
    rng = np.random.default_rng(0)
    obj = np.zeros((ny, nx)); obj[400:560, 500:800] = 1
    u0 = angular_spectrum(obj * np.exp(1j * 0.5 * rng.random((ny, nx))), p["wavelength"], p["dx"], p["dy"], 0.12)
    # HOLO_GOM always applies the forward kernel with the z typed by the user (> 0)
    new, _, _ = fresnel(u0, p["wavelength"], p["dx"], p["dy"], 0.12)
    old = fresnel_legacy(u0, p["wavelength"], p["dx"], p["dy"], 0.12)
    a, b = np.abs(new) / np.abs(new).max(), np.abs(old) / np.abs(old).max()
    mask = a > 0.2
    dphi = np.angle(new[mask] * np.conj(old[mask]))
    dphi = np.angle(np.exp(1j * (dphi - np.angle(np.mean(np.exp(1j * dphi))))))
    log(f"Campo de prueba {nx}x{ny}, lambda = 532 nm, dx = 2.2 um, z = 12 cm.")
    log()
    log(f"- Correlación de intensidades (heredada vs corregida): **{corr(a**2, b**2):.5f}**")
    log(f"- Desviación estándar de la diferencia de fase donde hay señal: **{np.std(dphi):.2f} rad** "
        "(0 si ambas fases coincidieran)")
    log()


def audit_filters():
    log("## 3. Supresión del orden cero (tesis 2015, cap. 7) con parámetros reales")
    log()
    p = THESIS_2015
    wl, dx, z, N = p["wavelength"], p["dx"], 0.12, 1024
    obj = np.zeros((N, N))
    yy, xx = np.ogrid[:N, :N]
    obj[((yy - 512) ** 2 + (xx - 512) ** 2) < 60**2] = 1
    obj[440:455, 430:600] = 1
    w = (slice(380, 640), slice(380, 640))
    log(f"Registro de Fresnel sin lente simulado, {N}x{N} px de 2.2 um, z = 12 cm, 8 bits, ruido 2 %. "
        "Reconstrucción con Fresnel de 1 FFT como en HDFresnel_GOM_V1. "
        "Métricas en una ventana alrededor de la imagen real.")
    log()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for row, frac in enumerate((0.5, 0.15)):
        theta = frac * max_carrier_angle(wl, dx)
        holo, _ = fresnel_off_axis_hologram(obj, wl, dx, dx, z, angle=(0, theta), bits=8, noise=0.02, rng=0)
        shift = int(round(np.sin(theta) / wl * N * dx))
        overlap = ", órdenes superpuestos" if frac < 0.3 else ""
        log(f"**Ángulo de referencia {np.degrees(theta):.2f}° ({frac:.0%} del máximo de Nyquist, "
            f"portadora a {shift} px{overlap})**")
        log()
        log("| Método | Corr. imagen real vs objeto | Energía en la imagen real |")
        log("|---|---|---|")
        methods = {
            "Sin filtro": (holo, -shift),
            "7.3.1 Intensidad promedio": (filters.subtract_mean(holo), -shift),
            "7.3.2 Pasa-altos FFT (32x32)": (filters.highpass_fft(holo, (32, 32)), -shift),
            "7.3.3 Laplaciano": (filters.laplacian(holo), -shift),
            "Banda lateral (Cuche 2000)": (filters.sideband_filter(holo, (0, -shift), max(shift // 2, 8)), 0),
        }
        for ax, (name, (h, s)) in zip(axes[row], methods.items()):
            rec, _, _ = fresnel(h, wl, dx, dx, -z)
            I = np.abs(np.roll(rec, (0, s), (0, 1))) ** 2
            log(f"| {name} | {corr(np.sqrt(I[w]), obj[w]):.3f} | {I[w].sum() / I.sum():.3f} |")
            ax.imshow(np.abs(rec) ** 0.5, cmap="gray")
            ax.set_title(f"{name} ({frac:.0%} Nyquist)", fontsize=9); ax.axis("off")
        log()
    fig.tight_layout(); fig.savefig(OUT / "zero_order_2015.png", dpi=70); plt.close(fig)
    log("El corrimiento de fase de 4 pasos (7.3.4) recupera O R* con error < 1e-10 en la prueba "
        "unitaria `test_phase_shifting_recovers_object_field_exactly`, pero exige 4 capturas.")
    log()


def audit_benchmark():
    log("## 4. Benchmark FFT (réplica de la figura 6.3 de la tesis)")
    log()
    a = np.random.default_rng(0).random((2592, 2592))
    np.fft.fft2(a)
    t = []
    for _ in range(5):
        t0 = time.perf_counter(); np.fft.fft2(a); t.append(time.perf_counter() - t0)
    log(f"- Tesis 2015 (i7-4510U, MATLAB): CPU 0.6561 s, GPU GeForce 840M 0.0387 s, razón **{0.6561 / 0.0387:.1f}x**")
    log(f"- Esta máquina, NumPy CPU: **{np.median(t):.3f} s** (mediana de 5)")
    log("- GPU (GTX 1650) pendiente: requiere instalar PyTorch con CUDA en el entorno.")
    log()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    log("# Resultados de auditoría holom-core")
    log()
    audit_cgh()
    audit_kinoform()
    audit_fresnel_legacy()
    audit_filters()
    audit_benchmark()
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Escrito {OUT / 'results.md'}")
