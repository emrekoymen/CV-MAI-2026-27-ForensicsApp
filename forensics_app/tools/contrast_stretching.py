import tkinter as tk
from tkinter import messagebox
from typing import NamedTuple

import numpy as np
from PIL import Image
from skimage import exposure, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


DEFAULT_PERCENTILES = (2.0, 98.0)


class StretchResult(NamedTuple):
    pixels: np.ndarray                          # uint8, alpha (if any) as the last channel
    input_range: tuple[float, float] | None     # stretched range on a 0–255 scale; None if nothing to stretch


def check_percentiles(low: float, high: float) -> None:
    if not 0 <= low < high <= 100:
        raise ValueError(f"Percentiles must satisfy 0 ≤ low < high ≤ 100 (got {low:g} / {high:g}).")


def to_unit_float(pixels: np.ndarray) -> np.ndarray:
    """Return ``pixels`` as floats 0.0–1.0."""
    if pixels.dtype == np.uint8:
        return pixels / 255
    if pixels.dtype == bool:  # 1-bit arrays: True is white
        return pixels.astype(float)
    if np.issubdtype(pixels.dtype, np.floating):  # float images use 0.0 (black) to 1.0 (white)
        return np.clip(pixels, 0, 1).astype(float) # Force 0.0-1.0
    # Wider integers are 16-bit images (Pillow opens some 16-bit PNG/TIFF files as mode "I")
    return np.clip(pixels, 0, 65535) / 65535


def stretch_contrast(
    image: Image.Image | np.ndarray,
    low: float = DEFAULT_PERCENTILES[0],
    high: float = DEFAULT_PERCENTILES[1],
) -> StretchResult:
    """Stretch the ``low``–``high`` percentile range of ``image`` to the full 0–255 range.

    All colour channels are stretched together, so colours keep their balance.
    Transparency is kept unchanged. Images with a single value (or no pixels)
    have nothing to stretch and are returned unchanged.
    """
    check_percentiles(low, high)
    if isinstance(image, Image.Image):
        if image.mode == "1":
            image = image.convert("L")
        elif image.mode in ("P", "PA"):  # palette -> real colours, keeping transparency
            has_alpha = image.mode == "PA" or "transparency" in image.info
            image = image.convert("RGBA" if has_alpha else "RGB")
        elif not (image.mode in ("L", "LA", "RGB", "RGBA") or image.mode.startswith("I") or image.mode == "F"):
            image = image.convert("RGB")  # e.g. CMYK or YCbCr
    pixels = np.asarray(image)

    # Take transparency out, so it neither affects the percentiles nor gets stretched
    alpha = None
    if pixels.ndim == 3 and pixels.shape[2] in (2, 4):
        alpha = pixels[:, :, -1]
        pixels = pixels[:, :, 0] if pixels.shape[2] == 2 else pixels[:, :, :3]
    colour = to_unit_float(pixels)

    input_range = None
    if colour.size == 0:  # no pixels: np.percentile and img_as_ubyte both fail on empty arrays
        result = colour.astype(np.uint8)
    else:
        p_low, p_high = np.percentile(colour, (low, high))  # over all channels together
        if p_high > p_low:
            colour = exposure.rescale_intensity(colour, in_range=(p_low, p_high), out_range=(0, 1))
            input_range = (p_low * 255, p_high * 255)
        result = img_as_ubyte(colour)
    if alpha is not None:
        result = np.dstack([result, alpha.astype(np.uint8)])
    return StretchResult(result, input_range)


class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"       # unique, stable identifier
    title = "Contrast Stretching"         # text shown in the sidebar
    category = "Filtering"
    description = "Stretch the contrast of the image."

    def choose_percentiles(self, parent: tk.Misc) -> tuple[float, float]:
        """Ask for the low/high percentiles; closing the dialog keeps the defaults."""
        default_low, default_high = DEFAULT_PERCENTILES
        dialog = tk.Toplevel(parent)
        dialog.title("Contrast stretching")
        dialog.transient(parent)
        dialog.grab_set()
        chosen = DEFAULT_PERCENTILES

        tk.Label(
            dialog,
            text="Image percentiles for stretching\n"
                 f"(defaults: {default_low:g} / {default_high:g})",
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6))
        low_text = tk.StringVar(value=f"{default_low:g}")
        high_text = tk.StringVar(value=f"{default_high:g}")
        tk.Label(dialog, text="Low (%):").grid(row=1, column=0, sticky="e", padx=(12, 4))
        tk.Entry(dialog, textvariable=low_text, width=8).grid(row=1, column=1, sticky="w", padx=(0, 12))
        tk.Label(dialog, text="High (%):").grid(row=2, column=0, sticky="e", padx=(12, 4))
        tk.Entry(dialog, textvariable=high_text, width=8).grid(row=2, column=1, sticky="w", padx=(0, 12))

        def on_ok():
            nonlocal chosen
            try:
                values = (float(low_text.get()), float(high_text.get()))
                check_percentiles(*values)
            except ValueError:
                messagebox.showerror(
                    "Invalid percentiles", "Enter two numbers with 0 ≤ low < high ≤ 100.", parent=dialog)
                return  # keep the dialog open to fix the values
            chosen = values
            dialog.destroy()

        tk.Button(dialog, text="OK", command=on_ok).grid(row=3, column=0, columnspan=2, pady=12)
        dialog.bind("<Return>", lambda event: on_ok())
        dialog.wait_window()  # closing the window keeps the defaults
        return chosen

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        low, high = self.choose_percentiles(parent)
        result = stretch_contrast(document.current, low, high)

        details = {"Operation": "Contrast Stretch", "Percentiles": f"{low:g} / {high:g}"}
        if result.input_range is None:
            reason = "no pixels" if result.pixels.size == 0 else "image has a single value"
            details["Stretched range"] = f"nothing to stretch ({reason})"
            message = "Nothing to stretch; image unchanged."
        else:
            range_low, range_high = (round(value, 1) for value in result.input_range)
            details["Stretched range"] = f"{range_low:g} – {range_high:g} → 0 – 255"
            message = "Contrast stretched."

        return ToolResult(
            image=Image.fromarray(result.pixels),
            message=message,
            details=details,
        )
