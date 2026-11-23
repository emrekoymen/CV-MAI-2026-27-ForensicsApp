"""Set 8.1/8.2: show a trained CNN's model passport and classify the loaded image with it.

The models are trained in notebooks/set8_train_models.ipynb. Each ``<name>.keras`` file has a
``<name>.passport.json`` next to it with the dataset, class names, preprocessing and training history.
"""

import io
import json
import tkinter as tk
from functools import lru_cache
from pathlib import Path
from tkinter import filedialog

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from .base import ForensicsTool, ToolResult

# models/ is ignored except the passports; unzip set8_models.zip from the notebook into models/
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "set8_models"

# The passport states the scaling as a formula; these are the ones the notebook writes
SCALINGS = {"x / 255.0": lambda x: x / 255.0, "x / 127.5 - 1.0": lambda x: x / 127.5 - 1.0}


def choose_file(parent, title, pattern):
    """One button per matching file in models/set8_models, plus Browse...; None if closed."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    choice = None

    def select(path):
        nonlocal choice
        choice = path
        dialog.destroy()

    def browse():
        path = filedialog.askopenfilename(title=title, parent=dialog, filetypes=[(title, pattern)])
        if path:
            select(Path(path))

    for path in sorted(MODELS_DIR.glob(pattern)):
        tk.Button(dialog, text=path.name, command=lambda p=path: select(p)).pack(fill="x", padx=12, pady=4)
    tk.Button(dialog, text="Browse...", command=browse).pack(fill="x", padx=12, pady=(4, 12))
    dialog.wait_window()
    return choice


def load_passport(path):
    with open(path) as f:
        return {**json.load(f), "_path": str(path)}


def preprocess(image, passport):
    """Follow the passport: colour mode, bilinear resize, float32, scaling; add a batch axis."""
    steps = passport["preprocessing"]
    if steps["formula"] not in SCALINGS:
        raise ValueError(f"Unknown scaling in passport: {steps['formula']}")
    resized = image.convert(steps["color_mode"]).resize(tuple(steps["input_size"]), Image.BILINEAR)
    return SCALINGS[steps["formula"]](np.asarray(resized, dtype=np.float32))[None]


@lru_cache(maxsize=2)
def load_model(path):
    import keras  # slow to import, so only when a model is needed
    return keras.models.load_model(path, compile=False)


def classify(image, passport):
    """Class probabilities of the image, in the order of the passport's class names."""
    model = load_model(str(Path(passport["_path"]).parent / passport["model_file"]))
    return model.predict(preprocess(image, passport), verbose=0)[0].astype(float)


def passport_summary(passport):
    """The passport as rows for the details panel."""
    samples, steps = passport["dataset"]["samples"], passport["preprocessing"]
    return {
        "Model": passport["name"],
        "Dataset": passport["dataset"]["name"],
        "Samples (train/val/test)": f"{samples['train']} / {samples['validation']} / {samples['test']}",
        "Classes": ", ".join(passport["class_names"]),
        "Preprocessing": f"{steps['color_mode']} {steps['input_size'][0]}×{steps['input_size'][1]}, {steps['formula']}",
        "Parameters": f"{passport['architecture']['parameters']:,}",
        "Epochs": passport["training"]["epochs_run"],
        "Test accuracy": f"{passport['evaluation']['test_accuracy']:.3f}",
    }


def figure_to_image(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def passport_figure(passport):
    """Training curves (train vs validation) and the per-class test accuracy."""
    history = passport["training"]["history"]
    fig = Figure(figsize=(15, 4.5), layout="constrained")
    axes = fig.subplots(1, 3)
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        phases = history.get("phase", [])
        for epoch in np.flatnonzero(np.diff(phases)):  # e.g. head training -> fine-tuning
            ax.axvline(epoch + 0.5, color="grey", linestyle="--")
        ax.set(title=key.capitalize(), xlabel="epoch")
        ax.legend()
    per_class = passport["evaluation"]["per_class_accuracy"]
    axes[2].barh(list(per_class), list(per_class.values()), color="tab:green")
    axes[2].invert_yaxis()
    axes[2].set(xlim=(0, 1), title="Test accuracy per class")
    fig.suptitle(f"Model passport: {passport['name']}")
    return figure_to_image(fig)


def prediction_figure(image, probabilities, passport):
    """The input image next to the probability of every class."""
    names = passport["class_names"]
    best = int(np.argmax(probabilities))
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    ax_image, ax_bars = fig.subplots(1, 2)
    ax_image.imshow(image.convert("RGB"))
    ax_image.axis("off")
    ax_bars.barh(names, probabilities, color=["tab:green" if i == best else "tab:blue" for i in range(len(names))])
    ax_bars.invert_yaxis()
    ax_bars.set(xlim=(0, 1), title="Class probabilities")
    fig.suptitle(f"{passport['name']}: {names[best]} ({probabilities[best]:.0%})")
    return figure_to_image(fig)


class ModelPassportTool(ForensicsTool):
    tool_id = "model_passport"
    title = "Model passport"
    category = "Deep learning"
    description = "Show a trained CNN's passport: dataset, classes, preprocessing, training curves and test accuracy."
    requires_image = False

    def run(self, parent, document):
        path = choose_file(parent, "Model passport", "*.passport.json")
        if path is None:
            return None
        passport = load_passport(path)
        return ToolResult(image=passport_figure(passport), message=f"Passport of {passport['name']}",
                          details={"Operation": "Model passport", **passport_summary(passport)})


class CnnClassificationTool(ForensicsTool):
    tool_id = "cnn_classification"
    title = "CNN classification"
    category = "Deep learning"
    description = "Classify the image with a trained CNN, preprocessed as its model passport says."

    def run(self, parent, document):
        path = choose_file(parent, "Model passport", "*.passport.json")
        if path is None:
            return None
        passport = load_passport(path)
        probabilities = classify(document.current, passport)
        names = passport["class_names"]
        top = np.argsort(probabilities)[::-1][:3]
        return ToolResult(
            image=prediction_figure(document.current, probabilities, passport),
            message=f"Classified as {names[top[0]]} with probability {probabilities[top[0]]:.0%}",
            details={"Operation": "CNN classification", "Model": passport["name"],
                     **{f"Top {rank}": f"{names[i]} ({probabilities[i]:.2f})" for rank, i in enumerate(top, 1)},
                     "Model test accuracy": f"{passport['evaluation']['test_accuracy']:.3f}"},
        )
