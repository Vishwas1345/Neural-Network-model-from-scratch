"""Tests for the from-scratch neural network.

Run with:  pytest -q

Covered:
* forward-pass output shape
* numerical gradient check of the analytical back-prop (both losses)
* loss actually decreases over a few training steps on dummy data
* softmax is a valid probability distribution and numerically stable
* save/load round-trips weights exactly
"""

import numpy as np
import pytest

from src.layers import Dense
from src.losses import MSE, SoftmaxCrossEntropy, softmax
from src.network import NeuralNetwork


def _dummy_data(n_features=12, n_classes=4, m=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_features, m))
    Y = rng.integers(0, n_classes, size=m)
    return X, Y, n_features, n_classes, m


# --------------------------------------------------------------- shape check

def test_forward_output_shape():
    X, Y, n_features, n_classes, m = _dummy_data()
    net = NeuralNetwork([n_features, 16, n_classes], seed=1)
    logits = net.forward(X)
    assert logits.shape == (n_classes, m)
    proba = net.predict_proba(X)
    assert proba.shape == (n_classes, m)
    # Columns of the probability matrix must sum to 1.
    np.testing.assert_allclose(proba.sum(axis=0), np.ones(m), atol=1e-10)
    assert net.predict(X).shape == (m,)


# ---------------------------------------------------- numerical gradient check

@pytest.mark.parametrize("loss_name", ["cross_entropy", "mse"])
def test_gradient_check(loss_name):
    """Compare analytical dW/db to a central finite-difference estimate."""
    X, Y, n_features, n_classes, m = _dummy_data(m=5, seed=7)
    net = NeuralNetwork([n_features, 10, n_classes], loss=loss_name, seed=3)
    Y_onehot = net.one_hot(Y, n_classes)

    # Analytical gradients.
    net.loss.forward(net.forward(X), Y_onehot)
    net.backward(Y_onehot)

    def loss_value():
        return net.loss.forward(net.forward(X), Y_onehot)

    eps = 1e-5
    dense_layers = [l for l in net.layers if isinstance(l, Dense)]
    for layer in dense_layers:
        for param, grad in ((layer.W, layer.dW), (layer.b, layer.db)):
            # Check a handful of random entries (full check is O(params)).
            rng = np.random.default_rng(0)
            flat_idx = rng.choice(param.size, size=min(8, param.size), replace=False)
            for fi in flat_idx:
                idx = np.unravel_index(fi, param.shape)
                original = param[idx]
                param[idx] = original + eps
                plus = loss_value()
                param[idx] = original - eps
                minus = loss_value()
                param[idx] = original
                numerical = (plus - minus) / (2 * eps)
                assert abs(numerical - grad[idx]) < 1e-6, (
                    f"{loss_name}: analytic {grad[idx]:.3e} vs numeric {numerical:.3e}")


# ------------------------------------------------------------- loss decreases

@pytest.mark.parametrize("loss_name", ["cross_entropy", "mse"])
def test_loss_decreases(loss_name):
    X, Y, n_features, n_classes, m = _dummy_data(m=32, seed=11)
    net = NeuralNetwork([n_features, 16, n_classes], loss=loss_name, seed=5)
    history = net.train(X, Y, epochs=100, lr=0.5, log_every=1, verbose=False)
    assert history["loss"][-1] < history["loss"][0], "loss did not decrease"
    # A tiny separable-ish problem should reach high train accuracy.
    assert history["train_acc"][-1] > history["train_acc"][0]


# ----------------------------------------------------------- softmax numerics

def test_softmax_stability():
    # Huge logits would overflow a naive exp(); the stable softmax must not.
    logits = np.array([[1000.0], [1000.0], [1000.0]])
    probs = softmax(logits)
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(axis=0), [1.0], atol=1e-12)


# ------------------------------------------------------------- save/load

def test_save_load_roundtrip(tmp_path):
    X, Y, n_features, n_classes, m = _dummy_data()
    net = NeuralNetwork([n_features, 16, n_classes], seed=2)
    path = tmp_path / "w.npz"
    net.save(str(path))
    loaded = NeuralNetwork.load(str(path))
    np.testing.assert_array_equal(net.forward(X), loaded.forward(X))
    assert loaded.layer_sizes == net.layer_sizes
    assert loaded.loss_name == net.loss_name
