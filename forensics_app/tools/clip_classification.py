"""Set 10.1: CLIP zero-shot classification with any number of classes, each described by free text.

CLIP has an image encoder and a text encoder trained so that an image and its caption land close together
in a shared embedding space. Zero-shot classification needs no training: each class name is written into a
caption template ("a photo of a {}."), the image and all captions are encoded, and the softmax of the scaled
cosine similarities gives one probability per class. The classifier is just the list of class texts.

The pretrained model (openai/clip-vit-base-patch32, ~600 MB) is downloaded by Hugging Face on first use and
cached in ~/.cache/huggingface. To work offline, save it once with ``save_pretrained`` into
models/clip-vit-base-patch32/; that folder is used when it exists.
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from matplotlib.figure import Figure

from .base import ForensicsTool, ToolResult
from .cnn_classification import figure_to_image

MODEL_NAME = "openai/clip-vit-base-patch32"
LOCAL_MODEL = Path(__file__).resolve().parents[2] / "models" / "clip-vit-base-patch32"   # optional, not committed

# "other" gives softmax somewhere to go when none of the real classes fits
DEFAULT_CLASSES = ["cat", "dog", "tiger", "rabbit", "fox", "owl", "other"]
DEFAULT_TEMPLATE = "a photo of a {}."
# Showcase cases offered by the "Defaults:" buttons of the dialog: button text -> (classes, template)
PRESETS = {
    "Animals": (DEFAULT_CLASSES, DEFAULT_TEMPLATE),
    "# Ducks": (["one duck", "two ducks", "three ducks", "four ducks", "no ducks"], "a photo of {}."),
    "Man/Woman": (["man", "woman", "other"], DEFAULT_TEMPLATE),
    "# vs animal": (["one fish", "two cats", "two birds", "two foxes", "other"], "a photo of {}."),
}
# Prompt ensembling as in the CLIP paper and the Set 10 notebook: average the caption embeddings of a class
ENSEMBLE_TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a good photo of a {}.",
                      "a cropped photo of a {}.", "a picture of a {}."]

_clips = {}   # loaded models by source, so running the tool again does not reload the weights


def _normalise(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


class Clip:
    """A CLIP model and its processor; embeddings come back as L2-normalised numpy arrays."""

    def __init__(self, source):
        import torch   # imported here so the rest of the app works without torch/transformers
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.name = str(source)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(source).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(source)
        self.logit_scale = self.model.logit_scale.exp().item()   # learned temperature, 100 for this model

    def _to_numpy(self, output):
        # newer transformers versions may return an output object instead of a tensor
        tensor = output if self.torch.is_tensor(output) else output.pooler_output
        return _normalise(tensor.float().cpu().numpy())

    def embed_images(self, images):
        inputs = self.processor(images=[image.convert("RGB") for image in images], return_tensors="pt")
        with self.torch.no_grad():
            return self._to_numpy(self.model.get_image_features(**inputs.to(self.device)))

    def embed_texts(self, texts):
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        with self.torch.no_grad():
            return self._to_numpy(self.model.get_text_features(**inputs.to(self.device)))


def load_clip():
    """The CLIP model, from models/clip-vit-base-patch32 if present, otherwise from the Hugging Face cache."""
    source = str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else MODEL_NAME
    if source not in _clips:
        try:
            _clips[source] = Clip(source)
        except ImportError as error:
            raise ImportError("CLIP needs transformers and torch: python -m pip install -r requirements.txt") from error
    return _clips[source]


def parse_classes(text):
    """One class per line; blank lines are ignored. Needs at least two different classes."""
    classes = [line.strip() for line in text.splitlines() if line.strip()]
    if len(classes) < 2:
        raise ValueError("Write at least two classes, one per line.")
    repeated = sorted({c for c in classes if classes.count(c) > 1})
    if repeated:
        raise ValueError(f"Each class must appear once; repeated: {', '.join(repeated)}.")
    return classes


def check_template(template):
    """The template must say where the class goes with {}."""
    template = template.strip()
    if "{}" not in template:
        raise ValueError('The template must contain {} where the class name goes, e.g. "a photo of a {}."')
    return template


def caption(template, label):
    return template.replace("{}", label)   # not str.format, so other braces in the template are kept


def zero_shot(clip, image, labels, templates):
    """Probability and cosine similarity of the image with every class.

    Each class embedding is the normalised mean of its captions over the templates; with one template this
    is exactly the notebook's zero_shot, with several it is zero_shot_ensemble.
    """
    image_embedding = clip.embed_images([image])[0]                                    # (D,)
    texts = [caption(t, label) for label in labels for t in templates]
    text_embeddings = clip.embed_texts(texts).reshape(len(labels), len(templates), -1)  # (C, T, D)
    class_embeddings = _normalise(text_embeddings.mean(axis=1))                         # (C, D)
    similarities = class_embeddings @ image_embedding                                   # cosine, (C,)
    logits = clip.logit_scale * similarities
    probabilities = np.exp(logits - logits.max())
    return probabilities / probabilities.sum(), similarities


def prediction_figure(image, labels, probabilities, title):
    """The image next to the class probabilities, highest at the top, as in the Set 10 slides."""
    order = np.argsort(probabilities)
    fig = Figure(figsize=(11, max(3.2, 0.4 * len(labels) + 1.5)), layout="constrained")
    ax_image, ax_bars = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
    ax_image.imshow(image.convert("RGB"))
    ax_image.axis("off")
    ax_bars.barh(np.array(labels)[order], probabilities[order], color="#3b6ea8")
    for row, value in enumerate(probabilities[order]):
        ax_bars.text(min(value + 0.02, 0.9), row, f"{value:.2f}", va="center")
    ax_bars.set(xlim=(0, 1), xlabel="probability", title=title)
    return figure_to_image(fig)


class ClipZeroShotTool(ForensicsTool):
    tool_id = "clip_zero_shot"
    title = "CLIP zero-shot classification"
    category = "Deep learning"
    description = ("Classify the image into classes you write (one per line) with CLIP, without training. "
                   "Softmax always picks one of them, so add an 'other'-style class if none may fit.")

    # The last choice is kept for the next run
    classes = DEFAULT_CLASSES
    template = DEFAULT_TEMPLATE
    ensemble = False

    def ask_classes(self, parent):
        """Dialog for the classes, the template and prompt ensembling; None if cancelled."""
        dialog = tk.Toplevel(parent)
        dialog.title("CLIP zero-shot classification")
        dialog.transient(parent)
        dialog.grab_set()
        chosen = None

        tk.Label(dialog, text="Classes (one per line, any number ≥ 2):").pack(anchor="w", padx=12, pady=(12, 2))
        inputs = tk.Frame(dialog)   # classes box on the left, "Defaults:" buttons on the right
        inputs.pack(fill="both", expand=True, padx=12)
        classes_box = tk.Text(inputs, width=40, height=10)
        classes_box.insert("1.0", "\n".join(self.classes))
        classes_box.pack(side="left", fill="both", expand=True)
        presets = tk.Frame(inputs)
        presets.pack(side="left", anchor="n", padx=(10, 0))
        tk.Label(presets, text="Defaults:").pack(anchor="w")

        tk.Label(dialog, text="Caption template ({} is replaced by the class):").pack(anchor="w", padx=12, pady=(10, 2))
        template_text = tk.StringVar(value=self.template)
        template_entry = tk.Entry(dialog, textvariable=template_text, width=40)
        template_entry.pack(fill="x", padx=12)

        ensemble = tk.BooleanVar(value=self.ensemble)

        def update_template_entry():
            template_entry.config(state="disabled" if ensemble.get() else "normal")

        tk.Checkbutton(dialog, variable=ensemble, command=update_template_entry,
                       text="Prompt ensembling: average 5 photo templates (ignores the template above)",
                       ).pack(anchor="w", padx=12, pady=(8, 0))
        update_template_entry()

        def use_preset(name):
            """Put a showcase case in the boxes; its own template is used, so ensembling is switched off."""
            classes, template = PRESETS[name]
            classes_box.delete("1.0", "end")
            classes_box.insert("1.0", "\n".join(classes))
            template_text.set(template)
            ensemble.set(False)
            update_template_entry()

        for name in PRESETS:
            tk.Button(presets, text=name, command=lambda n=name: use_preset(n)).pack(fill="x", pady=2)

        def on_ok():
            nonlocal chosen
            try:
                classes = parse_classes(classes_box.get("1.0", "end"))
                template = DEFAULT_TEMPLATE if ensemble.get() else check_template(template_text.get())
            except ValueError as error:
                messagebox.showerror("CLIP zero-shot classification", str(error), parent=dialog)
                return   # keep the dialog open to fix the input
            chosen = classes, template, ensemble.get()
            dialog.destroy()

        buttons = tk.Frame(dialog)
        buttons.pack(pady=12)
        tk.Button(buttons, text="Classify", command=on_ok).pack(side="left", padx=4)
        tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
        dialog.wait_window()
        return chosen

    def run(self, parent, document):
        chosen = self.ask_classes(parent)
        if chosen is None:
            return None
        labels, template, ensemble = chosen
        type(self).classes, type(self).template, type(self).ensemble = labels, template, ensemble
        templates = ENSEMBLE_TEMPLATES if ensemble else [template]

        clip = load_clip()
        start = time.perf_counter()
        probabilities, similarities = zero_shot(clip, document.current, labels, templates)
        seconds = time.perf_counter() - start

        title = f"{len(templates)} templates averaged" if ensemble else f'template: "{template}"'
        ranking = np.argsort(probabilities)[::-1]
        best = ranking[0]
        return ToolResult(
            image=prediction_figure(document.current, labels, probabilities, title),
            message=f"CLIP: {labels[best]} with probability {probabilities[best]:.0%} among {len(labels)} classes",
            details={
                "Operation": "CLIP zero-shot classification",
                "Model": f"{clip.name} ({clip.device})",
                "Templates": " | ".join(templates),
                "Classes": len(labels),
                **{f"{rank}. {labels[i]}": f"p = {probabilities[i]:.3f}, cos = {similarities[i]:.3f}"
                   for rank, i in enumerate(ranking, 1)},
                "Inference time": f"{1000 * seconds:.0f} ms",
            },
        )
