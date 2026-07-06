"""Loss functions.

The network's final Dense layer emits **logits** (raw scores). Each loss here
takes those logits plus one-hot targets, turns them into probabilities with a
numerically stable softmax, and reports both the scalar loss and the gradient
of the loss w.r.t. the logits (``dLogits``) so back-prop can start.

Two losses are provided:

* ``SoftmaxCrossEntropy`` — the classification-appropriate choice. Its
  logit-gradient collapses to the famously clean ``probs - targets``.
* ``MSE`` — mean-squared error on the softmax outputs. Included so we can
  *measure* how much worse an inappropriate loss actually performs, rather
  than just asserting it.

Both expose an identical interface:

    forward(logits, Y_onehot) -> float      # also caches probs
    backward(Y_onehot)        -> dLogits    # shape == logits.shape
    probs                                    # softmax outputs from last forward
"""

import numpy as np


def softmax(logits):
    """Column-wise softmax, stabilised by subtracting the per-column max."""
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=0, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax followed by cross-entropy, fused for stability and a clean grad."""

    name = "cross_entropy"

    def __init__(self):
        self.probs = None

    def forward(self, logits, Y_onehot):
        self.probs = softmax(logits)
        m = Y_onehot.shape[1]
        # + eps guards against log(0).
        loss = -np.sum(Y_onehot * np.log(self.probs + 1e-9)) / m
        return loss

    def backward(self, Y_onehot):
        # d(CE)/d(logits) for a softmax head is simply (probs - targets). This
        # is the gradient of the *summed* loss; the Dense layer divides param
        # gradients by the batch size, so we deliberately do not divide here.
        return self.probs - Y_onehot


class MSE:
    """Mean-squared error computed on softmax probabilities.

    Keeping the softmax means both losses see the same output space, so the
    comparison isolates the loss function itself. The logit-gradient needs the
    softmax Jacobian, which vectorises to::

        dLogits = probs * (g - sum(g * probs, axis=0))
        where g = dL/dprobs
    """

    name = "mse"

    def __init__(self):
        self.probs = None

    def forward(self, logits, Y_onehot):
        self.probs = softmax(logits)
        m = Y_onehot.shape[1]
        diff = self.probs - Y_onehot
        return np.sum(diff * diff) / m

    def backward(self, Y_onehot):
        # Summed-loss gradient (no /m — see SoftmaxCrossEntropy.backward).
        g = 2.0 * (self.probs - Y_onehot)              # dL/dprobs
        dot = np.sum(g * self.probs, axis=0, keepdims=True)
        return self.probs * (g - dot)                  # dL/dlogits via softmax Jacobian


def build_loss(name):
    """Factory used by the CLI / network: map a string to a loss instance."""
    losses = {
        "cross_entropy": SoftmaxCrossEntropy,
        "mse": MSE,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss {name!r}; choose from {sorted(losses)}")
    return losses[name]()
