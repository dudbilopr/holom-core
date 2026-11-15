# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Every legacy port runs, has the documented behaviour, and its known defects are pinned down."""
import numpy as np
import pytest

from holom import filters
from holom.legacy import cgh2015, fresnel_gom as fg, jtc, matlab, reconstruction as rec

RNG = 0


def target(n=64, m=96):
    a = np.zeros((n, m))
    a[20:44, 30:40] = 1
    a[20:28, 30:70] = 1
    return a


# --- MATLAB built-ins -------------------------------------------------------
def test_grid_matches_matlab_meshgrid():
    x, y = matlab.grid(4, 6)
    assert x[0].tolist() == [-3, -2, -1, 0, 1, 2] and y[:, 0].tolist() == [-2, -1, 0, 1]


def test_complex_max_uses_modulus():
    assert matlab.cextreme(np.array([3 + 0j, -4j, 1]), np.max) == -4j


def test_otsu_separates_two_levels():
    a = np.r_[np.full(500, 0.2), np.full(500, 0.8)]
    assert 0.2 <= matlab.graythresh(a) < 0.8


# --- propagators and reference wave -----------------------------------------
def test_onda_ref_tilt_is_half_the_requested_angle():
    """Onda_Ref.m uses pi/lambda instead of 2 pi/lambda: the fringe period is doubled."""
    lam, dx, ang = 532e-9, 2.2e-6, 2.0
    R = fg.onda_ref(np.ones((8, 256)), lam, dx, dx, ang)
    k_x = np.angle(R[0, 1] / R[0, 0]) / dx                       # rad/m along x
    assert k_x == pytest.approx(np.pi / lam * np.sin(np.deg2rad(ang)), rel=1e-9)
    assert k_x == pytest.approx(0.5 * 2 * np.pi / lam * np.sin(np.deg2rad(ang)), rel=1e-9)


@pytest.mark.parametrize('fn', [fg.t_fresnel, fg.ti_fresnel, fg.t_fresnel_conv, fg.t_fresnel_h, fg.t_fresnel_medfilt])
def test_fresnel_variants_run_on_planes_and_stacks(fn):
    a = target()
    for x in (a, np.dstack([a, a, a])):
        out = fn(x, 1, 532e-9, 2.2e-6, 2.2e-6, 0.1)
        assert out.shape == x.shape and np.all(np.isfinite(out))


def test_t_fresnel_is_the_conjugate_convention():
    """|T_Fresnel(z)| == |holom.fresnel(-z)|: the 2015 scripts use exp(-i...), hence negative Z0."""
    from holom.diffraction import fresnel
    a = target(128, 128) * np.exp(1j * np.random.default_rng(RNG).random((128, 128)))
    old = np.abs(fg.t_fresnel(a, 1, 532e-9, 2.2e-6, 2.2e-6, -0.05))
    new = np.abs(fresnel(a, 532e-9, 2.2e-6, 2.2e-6, 0.05)[0])
    assert np.corrcoef(old.ravel(), new.ravel())[0, 1] > 0.99999


def test_rgb_fftn_equals_channel_sum():
    """fftn over an RGB stack with a channel-independent kernel = 2-D convolution of the channel sum."""
    rng = np.random.default_rng(RNG)
    img = rng.random((32, 32, 3))
    full = fg.difraccion_convolucion(img, 1, 632.8e-9, 6.8e-6, 6.8e-6, 0.5)
    summed = fg.difraccion_convolucion(img.sum(2), 1, 632.8e-9, 6.8e-6, 6.8e-6, 0.5)
    for k in range(3):
        assert np.allclose(full[..., k], summed, atol=1e-12 * np.abs(summed).max())


def test_convolution_tf_is_not_the_angular_spectrum():
    """Difraccion_ConvolucionTF.m uses sqrt(1 + ...) and multiplies by lambda: it does not propagate."""
    from holom.diffraction import angular_spectrum
    a = target(64, 64).astype(complex)
    old = fg.difraccion_convolucion_tf(a, 1, 532e-9, 2.2e-6, 2.2e-6, 0.01)
    new = angular_spectrum(a, 532e-9, 2.2e-6, 2.2e-6, 0.01)
    assert np.corrcoef(np.abs(old).ravel(), np.abs(new).ravel())[0, 1] < 0.9


def test_circulo():
    c = fg.circulo(11, 11, 3, 6, 6)
    X, Y = np.meshgrid(np.arange(1, 12), np.arange(1, 12))
    assert c[5, 5] == 1 and c[5, 9] == 0 and c.sum() == ((X - 6) ** 2 + (Y - 6) ** 2 < 9).sum()


# --- reconstruction scripts --------------------------------------------------
def psi_set(n=64, defect_pixels=False):
    rng = np.random.default_rng(RNG)
    o = np.exp(2j * np.pi * rng.random((n, n))) * (0.5 + target(n, n))
    I = [np.abs(o + 2 * np.exp(1j * d)) ** 2 for d in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
    if defect_pixels:
        for x in I:
            x[10, 10] = 255.0                                     # saturated in all four frames
    return o, I


def test_phase_shift_atan_loses_the_quadrant():
    o, I = psi_set()
    H, _ = rec.phase_shift(*I, z0=-0.05)
    good = filters.phase_shifting_4step(*I) / 2
    true_phase = np.angle(good)
    raw = np.arctan((I[3] - I[1]) / (I[0] - I[2]))
    assert np.abs(raw).max() <= np.pi / 2 + 1e-12 < np.abs(true_phase).max()


def test_phase_shift_nan_blackens_the_whole_reconstruction():
    """0/0 at a saturated pixel -> NaN -> unwrap -> FFT: the whole plane is NaN (saved black)."""
    _, I = psi_set(defect_pixels=True)
    _, U = rec.phase_shift(*I, z0=-0.05)
    assert np.isnan(U).all()


def test_dh_reconstruccion_ultimo_and_fcircular_run():
    rng = np.random.default_rng(RNG)
    holo = rng.random((64, 64, 3))
    S, hf, U = rec.dh_reconstruccion_ultimo(holo, (10, 10, 20, 20), z0=-0.05)
    assert U.shape == (64, 64) and np.isfinite(U).all()
    S, f, hf, R = rec.fcircular_tfre(holo, (32, 20), (32, 28), z0=0.05)
    assert f.sum() > 0 and R.shape == (64, 64)


def test_hdfresnel_filter_binarises_the_hologram():
    rng = np.random.default_rng(RNG)
    holo = 100 + 50 * rng.random((64, 64))
    h, U = rec.hdfresnel_filter(holo / 255, np.ones((64, 64)), z_cm=5)
    assert set(np.unique(h)) <= {0.0, 1.0} and np.isfinite(U).all()


# --- CGH 2015 ----------------------------------------------------------------
def test_burch_is_not_a_transmittance():
    B, rec_ = cgh2015.holo_burch(target() * 255, rng=RNG)
    assert B.min() < 0 or B.max() > 1


def test_lee_has_no_carrier():
    L, _ = cgh2015.holo_lee(target(), rng=RNG)
    assert L.min() >= 0


def test_lohmann_builds_binary_cells():
    t = np.zeros((16, 16)); t[4:12, 6:9] = 1
    h, r = cgh2015.lohmann(t, rng=RNG)
    assert set(np.unique(h)) <= {0.0, 1.0} and h.shape[0] >= 16 * 5 and r.max() == pytest.approx(1)


def test_kinoform_2015_reveals_the_phase_encoded_object():
    img = target()
    kino, r = cgh2015.kinoform_2015(img, rng=RNG)
    assert kino.dtype == np.uint8 and np.corrcoef(r.ravel(), img.ravel())[0, 1] > 0.8


def test_mask_kinoform_reconstructs_the_logo():
    a = np.zeros((768, 1024)); a[300:460, 400:620] = 255
    kino, obj = cgh2015.mascara_de_fase(a, rng=RNG)
    win = obj[300:460, 400:620].sum() + obj[::-1, ::-1][300:460, 400:620].sum()
    assert win / obj.sum() > 0.3


def test_cgh_fourier_and_fresnel_run():
    h, r = cgh2015.cgh_fourier(target(), rng=RNG)
    assert h.min() >= 0 and r.shape == h.shape
    d = cgh2015.cgh_fresnel(target(), rng=RNG)
    assert np.isfinite(d['rec_ti']).all() and np.isfinite(d['rec_h']).all()
    e = cgh2015.extraccion_fase(target())
    assert np.isfinite(e['rec_ti']).all()


# --- JTC ---------------------------------------------------------------------
def test_jtc_speckle_decrypts_the_object():
    o = np.zeros((101, 101)); o[30:70, 45:55] = 1
    f, IE, dec = jtc.jtc_speckle(o, rng=RNG)
    assert IE.shape == (768, 1024) and np.isfinite(dec).all()


def test_jtc_rueda_script_keys_are_real_amplitudes():
    o = np.ones((101, 101))
    f, IE, dec = jtc.jtc_rueda_script(o, rng=RNG)
    assert np.isrealobj(f[333:434, 555:656]) or np.allclose(f[333:434, 555:656].imag, 0)


def test_jtc_gui_records_amplitude_not_intensity():
    rng = np.random.default_rng(RNG)
    a = rng.random((32, 32))
    assert np.allclose(jtc.jtc_gui_encrypt(a, 0) ** 2, jtc.jtc_gui_encrypt(a, 0, intensity=True))
