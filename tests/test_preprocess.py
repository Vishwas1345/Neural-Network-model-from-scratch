"""Tests for the demo's image preprocessing.

These guard the part that "silently breaks" demos: the transform from a raw
sketch to the (784, 1) MNIST-style vector the network expects.
"""

import numpy as np

from app.app import _center_of_mass, preprocess


def test_blank_canvas_returns_none():
    # A white canvas (no ink) must not produce a bogus prediction vector.
    white = np.full((280, 280), 255, dtype=np.uint8)
    assert preprocess(white) is None


def test_output_shape_and_range():
    canvas = np.zeros((280, 280), dtype=np.uint8)
    canvas[60:220, 120:160] = 255  # a bright vertical bar (white-on-black)
    x = preprocess(canvas)
    assert x.shape == (784, 1)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_digit_is_recentred():
    # Ink drawn in a corner should be re-centred near the middle of the 28x28.
    canvas = np.zeros((280, 280), dtype=np.uint8)
    canvas[10:90, 10:60] = 255  # top-left blob
    x = preprocess(canvas).reshape(28, 28)
    cy, cx = _center_of_mass(x)
    assert abs(cy - 13.5) < 3 and abs(cx - 13.5) < 3


def test_polarity_invariance():
    # Black-ink-on-white and white-ink-on-black of the same shape should yield
    # (near-)identical vectors, because preprocess normalises polarity.
    shape = np.zeros((280, 280), dtype=np.uint8)
    shape[60:220, 120:160] = 255
    white_on_black = preprocess(shape)
    black_on_white = preprocess(255 - shape)
    assert np.allclose(white_on_black, black_on_white, atol=1e-6)
