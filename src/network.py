"""The NeuralNetwork class — a plain feed-forward classifier assembled from the
Dense / ReLU layers in :mod:`src.layers` and trained against a loss from
:mod:`src.losses`.

The architecture is described by ``layer_sizes``, e.g. ``[784, 128, 64, 10]``
builds::

    Dense(784->128) -> ReLU -> Dense(128->64) -> ReLU -> Dense(64->10)  -> logits

The final Dense layer outputs raw logits; the softmax lives inside the loss.
All data is column-major, shape ``(features, batch)``.
"""

import numpy as np

from .layers import Dense, ReLU
from .losses import build_loss, softmax


class NeuralNetwork:
    def __init__(self, layer_sizes, loss="cross_entropy", seed=42):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least an input and output size")
        self.layer_sizes = list(layer_sizes)
        self.loss_name = loss
        self.loss = build_loss(loss)
        self.seed = seed

        # Build the Dense/ReLU stack. Give each Dense its own derived seed so a
        # fixed `seed` makes the whole network reproducible.
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer_seed = None if seed is None else seed + i
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], seed=layer_seed))
            # ReLU on every hidden layer, but not after the output layer.
            if i < len(layer_sizes) - 2:
                self.layers.append(ReLU())

    # ------------------------------------------------------------------ core

    def forward(self, X):
        """Run the input through every layer and return the output **logits**."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, Y_onehot):
        """Back-propagate from the loss through every layer, filling gradients."""
        dA = self.loss.backward(Y_onehot)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return dA

    def _update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    # ------------------------------------------------------------- inference

    def predict_proba(self, X):
        """Return class probabilities, shape ``(n_classes, batch)``."""
        return softmax(self.forward(X))

    def predict(self, X):
        """Return predicted class indices, shape ``(batch,)``."""
        return np.argmax(self.forward(X), axis=0)

    @staticmethod
    def one_hot(Y, n_classes):
        """Turn an integer label vector into a ``(n_classes, batch)`` matrix."""
        one_hot = np.zeros((n_classes, Y.size))
        one_hot[Y, np.arange(Y.size)] = 1
        return one_hot

    @staticmethod
    def accuracy(predictions, Y):
        return float(np.mean(predictions == Y))

    # -------------------------------------------------------------- training

    def train(self, X, Y, epochs, lr, batch_size=None,
              X_val=None, Y_val=None, log_every=10, verbose=True):
        """Fit the network with (mini-batch) gradient descent.

        Parameters
        ----------
        X, Y        : training inputs ``(features, m)`` and integer labels ``(m,)``.
        epochs      : number of passes over the data.
        lr          : learning rate.
        batch_size  : mini-batch size. ``None`` means full-batch (all m at once).
        X_val, Y_val: optional held-out set for accuracy tracking.
        log_every   : print/record metrics every N epochs.

        Returns
        -------
        history : dict with per-recorded-epoch lists: ``epoch``, ``loss``,
                  ``train_acc`` and (if validation given) ``val_acc``.
        """
        n_classes = self.layer_sizes[-1]
        m = X.shape[1]
        Y_onehot_full = self.one_hot(Y, n_classes)
        rng = np.random.default_rng(self.seed)

        history = {"epoch": [], "loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(epochs):
            if batch_size is None or batch_size >= m:
                batches = [np.arange(m)]
            else:
                order = rng.permutation(m)
                batches = [order[k:k + batch_size] for k in range(0, m, batch_size)]

            for idx in batches:
                Xb, Yb = X[:, idx], Y_onehot_full[:, idx]
                logits = self.forward(Xb)
                self.loss.forward(logits, Yb)   # caches probs for backward()
                self.backward(Yb)
                self._update(lr)

            if epoch % log_every == 0 or epoch == epochs - 1:
                # Metrics on the full training set for a stable curve.
                logits = self.forward(X)
                epoch_loss = self.loss.forward(logits, Y_onehot_full)
                train_acc = self.accuracy(np.argmax(logits, axis=0), Y)
                history["epoch"].append(epoch)
                history["loss"].append(epoch_loss)
                history["train_acc"].append(train_acc)
                msg = f"epoch {epoch:4d} | loss {epoch_loss:.4f} | train acc {train_acc*100:5.2f}%"
                if X_val is not None:
                    val_acc = self.accuracy(self.predict(X_val), Y_val)
                    history["val_acc"].append(val_acc)
                    msg += f" | val acc {val_acc*100:5.2f}%"
                if verbose:
                    print(msg)

        return history

    # ----------------------------------------------------------- persistence

    def save(self, path):
        """Save architecture + weights to a single ``.npz`` file."""
        arrays = {"layer_sizes": np.array(self.layer_sizes),
                  "loss_name": np.array(self.loss_name)}
        dense_i = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                arrays[f"W{dense_i}"] = layer.W
                arrays[f"b{dense_i}"] = layer.b
                dense_i += 1
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Reconstruct a network previously written by :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        layer_sizes = data["layer_sizes"].tolist()
        loss_name = str(data["loss_name"])
        net = cls(layer_sizes, loss=loss_name, seed=None)
        dense_i = 0
        for layer in net.layers:
            if isinstance(layer, Dense):
                layer.W = data[f"W{dense_i}"]
                layer.b = data[f"b{dense_i}"]
                dense_i += 1
        return net
