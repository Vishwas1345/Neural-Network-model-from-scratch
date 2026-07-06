"""Load and normalize MNIST — no torchvision / keras dependency.

Strategy (in order):

1. If the raw IDX ubyte(.gz) files are present in ``data/``, parse them.
2. Otherwise download them from a public mirror into ``data/`` first.
3. A committed CSV (``label,pixel...`` with 784 pixel columns) can also be read
   via :func:`load_csv` — handy for the original notebook's ``mnist_test.csv``.

Everything is returned column-major and normalized to ``[0, 1]``:

    X : float array, shape (784, n_samples), pixels / 255.0
    Y : int array,   shape (n_samples,)
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# data/ lives next to the repo root (one level up from this file's package).
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _download(filename, data_dir):
    dest = os.path.join(data_dir, filename)
    if not os.path.exists(dest):
        os.makedirs(data_dir, exist_ok=True)
        url = f"{MIRROR}/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _read_idx(path):
    """Parse an IDX (ubyte) file — the native MNIST format — into a NumPy array."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic == 2051:  # images: two more dims (rows, cols) follow
            rows, cols = struct.unpack(">II", f.read(8))
            buf = f.read(rows * cols * num)
            return np.frombuffer(buf, dtype=np.uint8).reshape(num, rows * cols)
        elif magic == 2049:  # labels: flat vector
            buf = f.read(num)
            return np.frombuffer(buf, dtype=np.uint8)
        raise ValueError(f"Unexpected IDX magic number {magic} in {path}")


def load_mnist(data_dir=DATA_DIR, flatten=True, normalize=True):
    """Return ``(X_train, Y_train, X_test, Y_test)`` column-major & normalized."""
    paths = {k: _download(v, data_dir) for k, v in FILES.items()}

    X_train = _read_idx(paths["train_images"]).astype(np.float64)  # (60000, 784)
    Y_train = _read_idx(paths["train_labels"]).astype(np.int64)
    X_test = _read_idx(paths["test_images"]).astype(np.float64)    # (10000, 784)
    Y_test = _read_idx(paths["test_labels"]).astype(np.int64)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    if flatten:
        # Column-major: (784, n_samples).
        X_train, X_test = X_train.T, X_test.T

    return X_train, Y_train, X_test, Y_test


def load_csv(path, normalize=True):
    """Load a ``label,pixel0,...,pixel783`` CSV into column-major (X, Y)."""
    import pandas as pd

    df = pd.read_csv(path)
    Y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if normalize:
        X /= 255.0
    return X.T, Y


if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = load_mnist()
    print("train:", Xtr.shape, Ytr.shape, "test:", Xte.shape, Yte.shape)
    print("pixel range:", Xtr.min(), "..", Xtr.max())
