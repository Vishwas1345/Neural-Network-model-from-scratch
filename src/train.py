"""Command-line training entrypoint.

Examples
--------
    python -m src.train --epochs 980 --lr 0.1 --loss cross_entropy
    python -m src.train --epochs 50 --lr 0.5 --loss cross_entropy --batch-size 64
    python -m src.train --epochs 200 --lr 0.1 --loss mse

Trains the from-scratch network on MNIST, reports final train/test accuracy,
saves the weights to ``models/weights.npz`` and (unless ``--no-plots``) writes a
loss curve and confusion matrix to ``results/``.
"""

import argparse
import os

import numpy as np

from .data_loader import load_mnist
from .network import NeuralNetwork

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Train the from-scratch MNIST network.")
    p.add_argument("--epochs", type=int, default=980)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--loss", choices=["cross_entropy", "mse"], default="cross_entropy")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Mini-batch size. Omit for full-batch gradient descent.")
    p.add_argument("--hidden", type=int, nargs="*", default=[128, 64],
                   help="Hidden layer sizes, e.g. --hidden 128 64.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", default=os.path.join(ROOT, "models", "weights.npz"))
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--no-plots", action="store_true", help="Skip PNG generation.")
    return p.parse_args()


def confusion_matrix(y_true, y_pred, n_classes=10):
    """Hand-rolled confusion matrix (rows = true, cols = predicted)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_loss_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(history["epoch"], history["loss"], color="tab:red", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], [a * 100 for a in history["train_acc"]],
             color="tab:blue", label="train acc")
    if history["val_acc"]:
        ax2.plot(history["epoch"], [a * 100 for a in history["val_acc"]],
                 color="tab:green", linestyle="--", label="test acc")
    ax2.set_ylabel("accuracy (%)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Training loss & accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(cm, path, accuracy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — test accuracy {accuracy*100:.2f}%")
    # Annotate each cell.
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    print("Loading MNIST ...")
    X_train, Y_train, X_test, Y_test = load_mnist()
    print(f"train {X_train.shape}, test {X_test.shape}")

    layer_sizes = [X_train.shape[0], *args.hidden, 10]
    print(f"Architecture: {layer_sizes} | loss={args.loss} | "
          f"batch_size={args.batch_size} | lr={args.lr} | epochs={args.epochs}")

    net = NeuralNetwork(layer_sizes, loss=args.loss, seed=args.seed)
    history = net.train(
        X_train, Y_train, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, X_val=X_test, Y_val=Y_test,
        log_every=args.log_every,
    )

    train_acc = net.accuracy(net.predict(X_train), Y_train)
    test_pred = net.predict(X_test)
    test_acc = net.accuracy(test_pred, Y_test)
    print("\n==== Results ====")
    print(f"final train accuracy: {train_acc*100:.2f}%")
    print(f"final test  accuracy: {test_acc*100:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    net.save(args.out)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"saved weights -> {args.out} ({size_kb:.0f} KB)")

    if not args.no_plots:
        os.makedirs(args.results_dir, exist_ok=True)
        loss_png = os.path.join(args.results_dir, "loss_curve.png")
        cm_png = os.path.join(args.results_dir, "confusion_matrix.png")
        save_loss_curve(history, loss_png)
        cm = confusion_matrix(Y_test, test_pred)
        save_confusion_matrix(cm, cm_png, test_acc)
        print(f"saved plots  -> {loss_png}, {cm_png}")

    return test_acc


if __name__ == "__main__":
    main()
