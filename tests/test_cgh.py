# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

import numpy as np
import pytest

from holom import cgh


def letter_image(n=96, m=128):
    img = np.zeros((n, m), dtype=np.uint8)
    img[20:76, 30:40] = 255
    img[20:30, 30:90] = 255
    img[44:54, 30:80] = 255
    return img


@pytest.mark.parametrize("name", list(cgh.ALGORITHMS))
def test_shapes_and_determinism(name):
    fn = cgh.ALGORITHMS[name]
    img = letter_image()
    A, H, O = fn(img, rng=1)
    assert A.shape == H.shape == O.shape == (192, 256)
    assert O.max() == pytest.approx(1.0)
    _, H2, _ = fn(img, rng=1)
    assert np.array_equal(H, H2)


@pytest.mark.parametrize("name", [n for n in cgh.ALGORITHMS if n not in ("Cuaternario", "Kinoform")])
def test_real_holograms_are_valid_transmittances(name):
    _, H, _ = cgh.ALGORITHMS[name](letter_image(), rng=1)
    assert np.isrealobj(H)
    assert H.min() >= 0 and H.max() == pytest.approx(1.0)


def test_cuaternario_is_not_a_realisable_transmittance():
    """Audit: CuaternariCGH.m produces a complex array, not 4 quantised levels."""
    _, H, _ = cgh.cuaternari(letter_image(), rng=1)
    assert np.iscomplexobj(H)
    assert np.abs(H.imag).max() > 0.1


def test_kinoform_is_phase_only_in_unit_range():
    _, H, _ = cgh.kinoform(letter_image(), rng=1)
    assert H.min() >= 0 and H.max() <= 1


def test_embedding_matches_matlab_offsets():
    A = cgh._embed(letter_image(), d=5, s=7)
    assert A[5 + 20, 7 + 30] == 1.0 and A[:5].sum() == 0 and A[:, :7].sum() == 0
