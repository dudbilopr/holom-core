import numpy as np
import pytest

from holom import filters, focus
from holom.simulate import fresnel_off_axis_hologram, max_carrier_angle

WL, DX, N = 532e-9, 2.2e-6, 512


def die_object(n=N):
    """Diffuse square with dark pips, placed off-centre so its real image avoids the zero order."""
    o = np.zeros((n, n))
    y0, x0, s = 60, 60, 90                     # real image lands at rows 60-150, cols 188-278
    o[y0:y0 + s, x0:x0 + s] = 1
    yy, xx = np.ogrid[:n, :n]
    for dy, dx_ in ((0.25, 0.25), (0.5, 0.5), (0.75, 0.75)):
        o[(yy - (y0 + dy * s)) ** 2 + (xx - (x0 + dx_ * s)) ** 2 < 8 ** 2] = 0
    return o * np.exp(2j * np.pi * np.random.default_rng(0).random((n, n)))


def test_stack_planes_share_one_physical_grid():
    h = np.random.default_rng(1).random((128, 128))
    stack, pitch = focus.fresnel_stack(h, WL, DX, [-0.10, -0.12, -0.14], block=2)
    assert stack.shape == (3, 64, 64)
    assert pitch == pytest.approx(WL * 0.12 / (128 * DX) * 2)


@pytest.mark.parametrize('z_true', [0.12, 0.18])
def test_autofocus_recovers_the_recording_distance(z_true):
    """Object at rows 60-150, cols 60-150; its real image sits +N/4 columns away (carrier at half
    the Nyquist angle): rows 15-38, cols 47-70 on the block-4 analysis grid."""
    holo, _ = fresnel_off_axis_hologram(die_object(), WL, DX, DX, z_true,
                                        angle=(0, 0.5 * max_carrier_angle(WL, DX)), bits=8, noise=0.02, rng=0)
    h = filters.subtract_mean(holo)
    zs = -np.arange(z_true - 0.04, z_true + 0.0401, 0.005)
    r = focus.autofocus(h, WL, DX, zs, roi=(slice(15, 38), slice(47, 70)), block=4)
    assert not r['peak_at_edge']
    assert abs(abs(r['z_best']) - z_true) <= 0.0051
    assert r['dip_depth'] > 0.5


def test_metrics_are_scale_invariant():
    I = np.random.default_rng(2).random((64, 64)) + 1
    for f in focus.METRICS.values():
        assert f(I) == pytest.approx(f(7.0 * I), rel=1e-9)


def test_depth_from_focus_separates_two_depths():
    rng = np.random.default_rng(0)

    def obj_field(z, rows, cols):
        d = WL * z / (N * DX)
        o = np.zeros((N, N)); o[rows, cols] = 1
        o[rows.start + 15:rows.start + 30, cols.start + 15:cols.start + 30] = 0
        from holom.diffraction import fresnel
        return fresnel(o * np.exp(2j * np.pi * rng.random((N, N))), WL, d, d, z)[0]

    O = obj_field(0.12, slice(100, 200), slice(80, 180)) + obj_field(0.18, slice(300, 400), slice(320, 420))
    zs = -np.arange(0.08, 0.2201, 0.005)
    depth, valid, _ = focus.depth_from_focus(O, WL, DX, zs)
    c, zref = 63.5, 0.15

    def region(z, rows, cols):
        k = z / zref
        return (slice(int(c + (rows.start / 4 - c) * k), int(c + (rows.stop / 4 - c) * k)),
                slice(int(c + (cols.start / 4 - c) * k), int(c + (cols.stop / 4 - c) * k)))

    ra, rb = region(0.12, slice(100, 200), slice(80, 180)), region(0.18, slice(300, 400), slice(320, 420))
    assert valid[ra].all() and valid[rb].all()
    assert abs(-np.median(depth[ra]) - 0.12) < 0.0076
    assert abs(-np.median(depth[rb]) - 0.18) < 0.0076
