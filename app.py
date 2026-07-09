"""Gradio demo: draw a digit, get a prediction from the from-scratch network.

The single most important thing here is that the drawing is preprocessed to
match **exactly** how MNIST looks to the network at training time:

* MNIST digits are *white ink on a black background*, normalized to [0, 1].
* Each digit is size-normalized to fit a 20x20 box and then centred by its
  centre-of-mass inside a 28x28 frame.

A raw sketch (black ink on a white canvas, arbitrary position/size) looks
nothing like that, so we replicate the classic MNIST centring pipeline. Skip
this and the demo "silently breaks" — it predicts confidently and wrongly.
"""

import os
import sys

import numpy as np
from PIL import Image

# Make `src` importable whether this file is run as `python -m app.app` (package
# context) or directly as `python app/app.py` (how Hugging Face Spaces launches
# it — then the script's own dir, not the repo root, is on sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.network import NeuralNetwork

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "weights.npz")

_net = None


def get_net():
    """Lazy-load the trained network once."""
    global _net
    if _net is None:
        _net = NeuralNetwork.load(WEIGHTS)
    return _net


def _to_grayscale_ink(image):
    """Return a float array where higher values = ink, background ~= 0.

    Accepts whatever a Gradio Sketchpad/ImageEditor hands us: an EditorValue
    dict (with a 'composite' key), an RGB(A) array, or a grayscale array.
    """
    if isinstance(image, dict):  # ImageEditor / Sketchpad EditorValue
        image = image.get("composite")
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 4:  # RGBA — use alpha as the ink mask when present
            alpha = arr[:, :, 3].astype(np.float64)
            rgb = arr[:, :, :3].mean(axis=2)
            # If the canvas is transparent, alpha marks the strokes directly.
            if alpha.max() > 0 and alpha.min() == 0:
                return alpha
            arr = rgb
        else:
            arr = arr[:, :, :3].mean(axis=2)
    arr = arr.astype(np.float64)

    # Decide polarity: MNIST wants bright ink on a dark field. If the image is
    # mostly bright (a white canvas), invert so strokes become the high values.
    if arr.mean() > 127:
        arr = 255.0 - arr
    return arr


def preprocess(image):
    """Turn a raw sketch into a (784, 1) column vector matching MNIST.

    Returns ``None`` if the canvas is blank.
    """
    ink = _to_grayscale_ink(image)
    if ink is None or ink.max() == 0:
        return None

    # 1. Crop to the bounding box of the drawn ink.
    mask = ink > (0.15 * ink.max())
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    crop = ink[r0:r1 + 1, c0:c1 + 1]

    # 2. Scale the longer side to 20 px, preserving aspect ratio (MNIST fits the
    #    digit in a 20x20 box inside the 28x28 frame).
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float64,
    )

    # 3. Paste into a 28x28 canvas, then shift so the centre-of-mass sits in the
    #    middle — exactly how the original MNIST images were normalized.
    canvas = np.zeros((28, 28), dtype=np.float64)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    if canvas.sum() > 0:
        cy, cx = _center_of_mass(canvas)
        shift_y, shift_x = int(round(14 - cy)), int(round(14 - cx))
        canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))

    # 4. Normalize to [0, 1] and flatten column-major -> (784, 1).
    canvas = canvas / 255.0
    return canvas.reshape(784, 1)


def _center_of_mass(a):
    total = a.sum()
    ys = (a.sum(axis=1) * np.arange(a.shape[0])).sum() / total
    xs = (a.sum(axis=0) * np.arange(a.shape[1])).sum() / total
    return ys, xs


def predict(image):
    """Gradio callback: sketch -> {digit: probability} for the top-3 Label."""
    x = preprocess(image)
    if x is None:
        return {}
    proba = get_net().predict_proba(x).ravel()
    return {str(i): float(proba[i]) for i in range(10)}


def build_demo():
    import gradio as gr

    with gr.Blocks(title="Draw a Digit — from-scratch NN") as demo:
        gr.Markdown(
            "# ✏️ Draw a digit\n"
            "A neural network **built from scratch in NumPy** (no PyTorch / "
            "TensorFlow) predicts your digit. Draw big and centred for best "
            "results."
        )
        with gr.Row():
            sketch = gr.Sketchpad(
                label="Draw here (0-9)",
                image_mode="L",
                type="numpy",
                height=320,
                width=320,
            )
            label = gr.Label(num_top_classes=3, label="Prediction (top 3)")
        sketch.change(predict, inputs=sketch, outputs=label)
        gr.Markdown(
            "Model: `784 → 128 → 64 → 10`, ReLU + softmax, trained with "
            "cross-entropy on MNIST — **98.19% test accuracy**."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
