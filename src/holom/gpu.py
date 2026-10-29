"""GPU reconstruction with PyTorch (optional dependency).

`FresnelPlan` precomputes the input and output chirps of the single-FFT Fresnel transform
(holom.diffraction.fresnel) on the device for a fixed shape, wavelength, pixel and distance:
the situation of holographic video at constant z. Each call then does
mean subtraction -> chirp -> FFT -> output phase -> |U|^2 entirely on the device.
complex64 is used by default (consumer GPUs run double precision at a small fraction of single).

`angular_spectrum_torch` mirrors holom.diffraction.angular_spectrum.
"""
import numpy as np

try:
    import torch
except ImportError:                                     # pragma: no cover
    torch = None


def available():
    return torch is not None and torch.cuda.is_available()


def _device(device):
    if torch is None:
        raise ImportError('holom.gpu needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu126')
    return torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))


def _centered(n, d, dev, dtype):
    return (torch.arange(n, device=dev, dtype=dtype) - n // 2) * d


class FresnelPlan:
    """Single-FFT Fresnel reconstruction with precomputed chirps.

    plan = FresnelPlan((ny, nx), wavelength, dx, dy, z)
    I = plan(hologram)              # |U|^2 on the device (torch tensor)
    field = plan.field(hologram)    # complex U on the device
    """

    def __init__(self, shape, wavelength, dx, dy, z, device=None, dtype=None, subtract_mean=True):
        self.dev = _device(device)
        cdt = dtype or torch.complex64
        rdt = torch.float32 if cdt == torch.complex64 else torch.float64
        ny, nx = shape
        f64 = torch.float64                              # chirp phases reach ~1e4 rad: build them in double
        y, x = _centered(ny, dy, self.dev, f64), _centered(nx, dx, self.dev, f64)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        dxi, deta = wavelength * abs(z) / (nx * dx), wavelength * abs(z) / (ny * dy)
        ETA, XI = torch.meshgrid(_centered(ny, deta, self.dev, f64), _centered(nx, dxi, self.dev, f64), indexing='ij')
        a = np.pi / (wavelength * z)
        k = 2 * np.pi / wavelength
        self.h = torch.polar(torch.ones_like(X), a * (X ** 2 + Y ** 2)).to(cdt)
        phase0 = (k * z) % (2 * np.pi)
        pre = np.exp(1j * phase0) / (1j * wavelength * z) * dx * dy
        self.p = (torch.polar(torch.ones_like(XI), a * (XI ** 2 + ETA ** 2)) * complex(pre)).to(cdt)
        self.forward = z > 0
        self.n = nx * ny
        self.subtract_mean = subtract_mean
        self.pitch = (dxi, deta)
        self.cdt, self.rdt = cdt, rdt

    def field(self, hologram):
        g = torch.as_tensor(hologram, device=self.dev)
        g = g.to(self.rdt) if not torch.is_complex(g) else g.to(self.cdt)
        if self.subtract_mean:
            g = g - g.mean()
        g = torch.fft.ifftshift(g * self.h)
        G = torch.fft.fft2(g) if self.forward else torch.fft.ifft2(g) * self.n
        return self.p * torch.fft.fftshift(G)

    def __call__(self, hologram):
        U = self.field(hologram)
        return U.real ** 2 + U.imag ** 2


def angular_spectrum_torch(u0, wavelength, dx, dy, z, device=None, dtype=None):
    """Angular-spectrum propagation on the device (evanescent components removed)."""
    dev = _device(device)
    cdt = dtype or torch.complex64
    rdt = torch.float32 if cdt == torch.complex64 else torch.float64
    u = torch.as_tensor(u0, device=dev).to(cdt)
    ny, nx = u.shape
    f64 = torch.float64                                  # kz z reaches ~1e5 rad: transfer function in double
    fy = torch.fft.fftfreq(ny, dy, device=dev, dtype=f64)
    fx = torch.fft.fftfreq(nx, dx, device=dev, dtype=f64)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')
    arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    kz = 2 * np.pi / wavelength * torch.sqrt(torch.clamp(arg, min=0))
    H = torch.where(arg > 0, torch.polar(torch.ones_like(kz), kz * z), torch.zeros_like(kz)).to(cdt)
    return torch.fft.ifft2(torch.fft.fft2(u) * H)
