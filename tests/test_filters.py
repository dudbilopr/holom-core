import numpy as np
import pytest

from holom import filters
from holom.diffraction import angular_spectrum
from holom.simulate import max_carrier_angle, off_axis_hologram, phase_shifted_set

WL = 532e-9
DX = 2.2e-6
N = 512
Z = 0.02


def make_object():
    obj = np.zeros((N, N))
    obj[200:312, 150:180] = 1.0
    obj[200:230, 150:260] = 1.0
    obj[270:300, 150:240] = 1.0
    return obj


@pytest.fixture(scope="module")
def recording():
    obj = make_object()
    theta = 0.5 * max_carrier_angle(WL, DX)
    holo, o, r = off_axis_hologram(obj, WL, DX, DX, Z, angle=(0.0, theta), ref_amplitude=3.0)
    # carrier position in the centred spectrum, in pixels
    fx = np.sin(theta) / WL
    shift = int(round(fx * N * DX))
    return obj, holo, o, r, shift


def dc_fraction(h):
    S = np.abs(np.fft.fftshift(np.fft.fft2(h))) ** 2
    c = N // 2
    return S[c - 8:c + 8, c - 8:c + 8].sum() / S.sum()


def correlation(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


@pytest.mark.parametrize("method", ["mean", "highpass", "laplacian"])
def test_thesis_filters_reduce_zero_order(recording, method):
    _, holo, *_ = recording
    out = {"mean": lambda h: filters.subtract_mean(h),
           "highpass": lambda h: filters.highpass_fft(h, (32, 32)),
           "laplacian": lambda h: filters.laplacian(h)}[method](holo)
    assert dc_fraction(out) < 0.1 * dc_fraction(holo)


def test_phase_shifting_recovers_object_field_exactly():
    obj = make_object()
    holos, o = phase_shifted_set(obj, WL, DX, DX, Z, ref_amplitude=2.0)
    o_rec = filters.phase_shifting_4step(*holos) / 2.0
    assert np.abs(o_rec - o).max() / np.abs(o).max() < 1e-10
    back = angular_spectrum(o_rec, WL, DX, DX, -Z)
    assert correlation(np.abs(back), obj) > 0.99


def test_sideband_filter_reconstructs_object(recording):
    obj, holo, o, r, shift = recording
    # O R* sits at -f_carrier; the +f_carrier lobe is the twin image O* R
    field = filters.sideband_filter(holo, (0, -shift), radius=shift // 2)
    back = angular_spectrum(field, WL, DX, DX, -Z)
    assert correlation(np.abs(back), obj) > 0.9
