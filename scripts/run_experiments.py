"""Reproduce the Step-3 experiment matrix and print a comparison table.

Runs several configurations on the *same* MNIST split so the numbers are
comparable, then saves the best model to models/weights.npz and the loss curve
+ confusion matrix to results/.

    python -m scripts.run_experiments
"""

import json
import os
import time

import numpy as np

from src.data_loader import load_mnist
from src.network import NeuralNetwork
from src.train import confusion_matrix, save_confusion_matrix, save_loss_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(name, X_train, Y_train, X_test, Y_test, *, hidden, loss, lr,
        batch_size, epochs, seed=42):
    layer_sizes = [784, *hidden, 10]
    net = NeuralNetwork(layer_sizes, loss=loss, seed=seed)
    t0 = time.time()
    history = net.train(X_train, Y_train, epochs=epochs, lr=lr,
                        batch_size=batch_size, X_val=X_test, Y_val=Y_test,
                        log_every=max(1, epochs // 10), verbose=False)
    elapsed = time.time() - t0
    train_acc = net.accuracy(net.predict(X_train), Y_train)
    test_pred = net.predict(X_test)
    test_acc = net.accuracy(test_pred, Y_test)
    print(f"[{name}] arch={layer_sizes} loss={loss} bs={batch_size} lr={lr} "
          f"epochs={epochs} -> train {train_acc*100:.2f}% test {test_acc*100:.2f}% "
          f"({elapsed:.0f}s)")
    return {
        "name": name, "arch": layer_sizes, "loss": loss, "batch_size": batch_size,
        "lr": lr, "epochs": epochs, "train_acc": train_acc, "test_acc": test_acc,
        "seconds": elapsed,
        "net": net, "history": history, "test_pred": test_pred,
    }


def main():
    print("Loading MNIST ...")
    X_train, Y_train, X_test, Y_test = load_mnist()

    results = []
    # 1. Faithful reproduction of the original tiny network (CE, full-batch).
    results.append(run("orig-arch (784-10-10)", X_train, Y_train, X_test, Y_test,
                       hidden=[10], loss="cross_entropy", lr=0.1,
                       batch_size=None, epochs=980))
    # 2. Expanded net, full-batch, cross-entropy vs MSE (equal epochs).
    results.append(run("expanded CE full-batch", X_train, Y_train, X_test, Y_test,
                       hidden=[128, 64], loss="cross_entropy", lr=0.1,
                       batch_size=None, epochs=200))
    results.append(run("expanded MSE full-batch", X_train, Y_train, X_test, Y_test,
                       hidden=[128, 64], loss="mse", lr=0.1,
                       batch_size=None, epochs=200))
    # 3. Expanded net, mini-batch, cross-entropy vs MSE.
    results.append(run("expanded CE mini-batch", X_train, Y_train, X_test, Y_test,
                       hidden=[128, 64], loss="cross_entropy", lr=0.5,
                       batch_size=64, epochs=30))
    results.append(run("expanded MSE mini-batch", X_train, Y_train, X_test, Y_test,
                       hidden=[128, 64], loss="mse", lr=0.5,
                       batch_size=64, epochs=30))

    # Best model by test accuracy -> save weights + plots.
    best = max(results, key=lambda r: r["test_acc"])
    print(f"\nBest: {best['name']} @ {best['test_acc']*100:.2f}% test")

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    best["net"].save(os.path.join(ROOT, "models", "weights.npz"))
    save_loss_curve(best["history"], os.path.join(ROOT, "results", "loss_curve.png"))
    cm = confusion_matrix(Y_test, best["test_pred"])
    save_confusion_matrix(cm, os.path.join(ROOT, "results", "confusion_matrix.png"),
                          best["test_acc"])

    # Dump a compact table (no net/history objects) for the README.
    table = [{k: r[k] for k in ("name", "arch", "loss", "batch_size", "lr",
                                "epochs", "train_acc", "test_acc", "seconds")}
             for r in results]
    with open(os.path.join(ROOT, "results", "experiments.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("wrote results/experiments.json, models/weights.npz, results/*.png")


if __name__ == "__main__":
    main()
