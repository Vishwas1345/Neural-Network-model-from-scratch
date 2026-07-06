"""A neural network built from scratch with NumPy — no autograd frameworks.

Convention used throughout this package: data is stored **column-major**, i.e.
arrays are shaped ``(features, batch_size)``. This mirrors the math in the
original notebook (``W @ X + b``) and keeps the linear algebra readable.
"""
