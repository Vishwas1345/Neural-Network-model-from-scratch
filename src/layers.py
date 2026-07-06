"""Building-block layers: a fully-connected (Dense) layer and a ReLU activation.

Every layer implements the same tiny interface so the network can treat them
uniformly:

    forward(A)  -> output          (and caches whatever backward() needs)
    backward(dOut) -> dInput       (and caches parameter gradients)
    update(lr)                     (applies a gradient-descent step; no-op if
                                    the layer has no parameters)

All arrays are column-major: shape ``(features, batch_size)``.
"""

import numpy as np


class Dense:
    """A fully-connected layer computing ``Z = W @ A + b``.

    Weights use He initialization (``std = sqrt(2 / fan_in)``), which keeps the
    variance of activations stable through ReLU layers — a big practical
    improvement over the original notebook's ``rand() - 0.5`` scheme.
    """

    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((out_features, in_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((out_features, 1))
        # Filled in during backward().
        self.dW = None
        self.db = None
        self._A = None  # input cache

    def forward(self, A):
        self._A = A
        return self.W @ A + self.b

    def backward(self, dZ):
        m = self._A.shape[1]
        self.dW = (dZ @ self._A.T) / m
        # Per-neuron bias gradient. The original notebook used np.sum(dZ),
        # which collapses this to a single scalar — a real bug we fix here.
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = self.W.T @ dZ
        return dA

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    """Rectified Linear Unit: ``max(0, Z)`` element-wise."""

    def __init__(self):
        self._mask = None

    def forward(self, Z):
        self._mask = Z > 0
        return Z * self._mask

    def backward(self, dA):
        # Gradient flows only where the input was positive.
        return dA * self._mask

    def update(self, lr):
        # No trainable parameters.
        pass
