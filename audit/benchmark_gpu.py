"""Benchmark of the complete reconstruction (not only the FFT) on this machine.

Usage: python audit/benchmark_gpu.py   (writes audit/out/benchmark_gpu.md and .json)

Pipeline timed: hologram (uint8, as the camera delivers it) -> float -> mean subtraction ->
input chirp -> FFT -> output phase -> |U|^2. Variants:
  numpy128      holom.diffraction.fresnel (chirps rebuilt every call), complex128
  numpy64_plan  NumPy with precomputed chirps, complex64
  torch_gpu_e2e PyTorch on the GPU, including host->device upload and device->host download
  torch_gpu     PyTorch on the GPU with the frame already on the device
Also the FFT alone of a 2592x2592 random matrix, the measurement of the 2015 thesis (fig. 6.3).
Medians over repeated runs after warm-up.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np

from holom.diffraction import centered_coords, fresnel

OUT = Path(__file__).resolve().parent / 'out'
WL, DX, Z = 532e-9, 2.2e-6, -0.12
SIZES = [(512, 512), (1024, 1024), (2048, 2048), (1944, 2592)]


def timeit(fn, reps, sync=None):
    for _ in range(2):
        fn()
    if sync:
        sync()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


class NumpyPlan:
    def __init__(self, shape):
        ny, nx = shape
        Y, X = np.meshgrid(centered_coords(ny, DX), centered_coords(nx, DX), indexing='ij')
        dxi, deta = WL * abs(Z) / (nx * DX), WL * abs(Z) / (ny * DX)
        ETA, XI = np.meshgrid(centered_coords(ny, deta), centered_coords(nx, dxi), indexing='ij')
        a = np.pi / (WL * Z)
        self.h = np.exp(1j * a * (X ** 2 + Y ** 2)).astype(np.complex64)
        self.p = np.exp(1j * a * (XI ** 2 + ETA ** 2)).astype(np.complex64)

    def __call__(self, holo):
        g = holo.astype(np.float32)
        g -= g.mean()
        G = np.fft.ifft2(np.fft.ifftshift(g * self.h))
        U = self.p * np.fft.fftshift(G)
        return U.real ** 2 + U.imag ** 2


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    info = {'cpu': platform.processor(), 'python': platform.python_version(), 'numpy': np.__version__}
    rows = []
    try:
        import torch
        from holom.gpu import FresnelPlan
        has_gpu = torch.cuda.is_available()
        info |= {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0) if has_gpu else None,
                 'cuda': torch.version.cuda}
    except ImportError:
        has_gpu = False
    for shape in SIZES:
        holo = rng.integers(0, 256, shape, dtype=np.uint8)
        r = {'shape': f'{shape[1]}x{shape[0]}'}
        reps = 5 if shape[0] * shape[1] > 3e6 else 10
        r['numpy128'] = timeit(lambda: np.abs(fresnel(holo - holo.mean(), WL, DX, DX, Z)[0]) ** 2, reps)
        npl = NumpyPlan(shape)
        r['numpy64_plan'] = timeit(lambda: npl(holo), reps)
        if has_gpu:
            plan = FresnelPlan(shape, WL, DX, DX, Z, device='cuda')
            sync = torch.cuda.synchronize
            r['torch_gpu_e2e'] = timeit(lambda: plan(torch.from_numpy(holo).to('cuda', non_blocking=False)).cpu(), 30, sync)
            dev_holo = torch.from_numpy(holo).to('cuda')
            r['torch_gpu'] = timeit(lambda: plan(dev_holo), 50, sync)
        rows.append(r)
        print(r)
    a = rng.random((2592, 2592))
    fft = {'numpy_fft2_2592': timeit(lambda: np.fft.fft2(a), 5)}
    if has_gpu:
        ta = torch.from_numpy(a.astype(np.float32)).to('cuda')
        fft['torch_gpu_fft2_2592_complex64'] = timeit(lambda: torch.fft.fft2(ta), 50, torch.cuda.synchronize)
    print(fft)
    res = {'info': info, 'pipeline_s': rows, 'fft_only_s': fft,
           'thesis_2015_fft_2592_s': {'cpu_i7_4510U_matlab': 0.6561, 'gpu_geforce_840M': 0.0387}}
    (OUT / 'benchmark_gpu.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    lines = ['# Benchmark de la reconstrucción completa', '',
             f"CPU: {info.get('cpu')} · GPU: {info.get('gpu')} · Python {info['python']} · NumPy {info['numpy']} · PyTorch {info.get('torch')} (CUDA {info.get('cuda')})", '',
             '| Tamaño | NumPy complex128 | NumPy complex64 (plan) | GPU con transferencias | GPU (dato en GPU) | fps GPU e2e |',
             '|---|---|---|---|---|---|']
    for r in rows:
        g, ge = r.get('torch_gpu'), r.get('torch_gpu_e2e')
        lines.append(f"| {r['shape']} | {r['numpy128']*1e3:.1f} ms | {r['numpy64_plan']*1e3:.1f} ms | "
                     f"{ge*1e3:.1f} ms | {g*1e3:.2f} ms | {1/ge:.0f} |" if g else
                     f"| {r['shape']} | {r['numpy128']*1e3:.1f} ms | {r['numpy64_plan']*1e3:.1f} ms | — | — | — |")
    lines += ['', '| FFT sola 2592×2592 | Tiempo |', '|---|---|']
    lines += [f'| {k} | {v*1e3:.2f} ms |' for k, v in fft.items()]
    lines += ['| Tesis 2015, MATLAB CPU i7-4510U | 656.1 ms |', '| Tesis 2015, MATLAB GPU GeForce 840M | 38.7 ms |']
    (OUT / 'benchmark_gpu.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
