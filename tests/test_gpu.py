"""The PyTorch backend reproduces the NumPy reference (complex64 on the device)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from holom import gpu
from holom.diffraction import angular_spectrum, fresnel

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
WL, DX = 532e-9, 2.2e-6


def rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('z', [0.12, -0.12])
def test_fresnel_plan_matches_numpy(device, z):
    rng = np.random.default_rng(0)
    h = rng.random((256, 384)) * 255
    ref, _, _ = fresnel(h - h.mean(), WL, DX, DX, z)
    plan = gpu.FresnelPlan(h.shape, WL, DX, DX, z, device=device)
    U = plan.field(h).cpu().numpy()
    assert rel(U, ref) < 1e-4
    assert rel(plan(h).cpu().numpy(), np.abs(ref) ** 2) < 1e-4


@pytest.mark.parametrize('device', DEVICES)
def test_angular_spectrum_matches_numpy(device):
    rng = np.random.default_rng(1)
    u = rng.random((128, 128)) * np.exp(2j * np.pi * rng.random((128, 128)))
    ref = angular_spectrum(u, WL, DX, DX, 0.01)
    out = gpu.angular_spectrum_torch(u, WL, DX, DX, 0.01, device=device).cpu().numpy()
    assert rel(out, ref) < 1e-4


def test_double_precision_is_exact():
    rng = np.random.default_rng(2)
    h = rng.random((128, 128))
    ref, _, _ = fresnel(h - h.mean(), WL, DX, DX, 0.2)
    plan = gpu.FresnelPlan(h.shape, WL, DX, DX, 0.2, device='cpu', dtype=torch.complex128)
    assert rel(plan.field(h).numpy(), ref) < 1e-10
