import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import io

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


CHANNEL_COLOURS = {"Red": "red", "Green": "green", "Blue": "blue", "Gray": "black"}


def to_8bit(pixels: np.ndarray) -> np.ndarray:
    """Return ``pixels`` as uint8 values 0–255."""
    if pixels.dtype == np.uint8:
        return pixels
    if pixels.dtype == bool:  # 1-bit arrays: True is white
        return pixels.astype(np.uint8) * 255
    if np.issubdtype(pixels.dtype, np.floating):  # float images use 0.0 (black) to 1.0 (white)
        return np.round(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    # Wider integers are 16-bit images (Pillow opens some 16-bit PNG/TIFF files as mode "I")
    return np.round(np.clip(pixels, 0, 65535) / 65535 * 255).astype(np.uint8)


def compute_histogram(image: Image.Image | np.ndarray) -> dict[str, np.ndarray]:
    """Count how many pixels have each value 0–255, per channel.

    Returns ``{"Red": counts, "Green": counts, "Blue": counts}`` for colour images
    or ``{"Gray": counts}`` for grayscale ones, where each ``counts`` has 256 entries.
    """
    if isinstance(image, Image.Image):
        if image.mode in ("1", "L", "LA", "La"):
            image = image.convert("L")      # 1-bit white -> 255, alpha dropped
        elif not (image.mode.startswith("I") or image.mode == "F"):
            image = image.convert("RGB")    # palette -> real colours, alpha dropped
    pixels = to_8bit(np.asarray(image))

    if pixels.ndim == 3 and pixels.shape[2] >= 3:
        channels = {"Red": pixels[:, :, 0], "Green": pixels[:, :, 1], "Blue": pixels[:, :, 2]}
    elif pixels.ndim == 3:
        channels = {"Gray": pixels[:, :, 0]}  # e.g. a grayscale + alpha array
    else:
        channels = {"Gray": pixels}
    counts = {name: np.bincount(channel.ravel(), minlength=256) for name, channel in channels.items()}
    return counts


def histogram_stats(counts: np.ndarray) -> dict[str, object]:
    """Summarise one channel's histogram; empty if the image has no pixels."""
    total = int(counts.sum())
    if total == 0:
        return {}
    used = np.nonzero(counts)[0]
    low, high = int(used[0]), int(used[-1])
    return {
        "min": low,
        "max": high,
        "mean": round(float((np.arange(256) * counts).sum() / total), 1),
        "empty bins": int((counts[low:high + 1] == 0).sum()),  # gaps between min and max
        "range bins": high - low + 1,
    }


class HistogramTool(ForensicsTool):
    tool_id = "histogram"       # unique, stable identifier
    title = "Histogram"         # text shown in the sidebar
    category = "Image Plots"
    description = "Generate the histogram of the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        counts = compute_histogram(document.current)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        edges = np.arange(257)
        if len(counts) == 1:
            ax.stairs(counts["Gray"], edges, fill=True, color='black', alpha=0.7)
        else:
            for name, channel_counts in counts.items():
                # Outlines only, so all three channels stay visible
                ax.stairs(channel_counts, edges, color=CHANNEL_COLOURS[name], label=name,
                          alpha=0.8, linewidth=1.5)
            ax.legend(loc='upper right')
        ax.set_xlim([0, 256])

        ax.set_title("Intensity Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        # Render Figure to In-Memory Buffer and decode into PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        output_img = Image.open(buf).convert("RGB")

        # Cleanup Matplotlib resources
        plt.close(fig)

        details = {"Operation": "Histogram"}
        for name, channel_counts in counts.items():
            stats = histogram_stats(channel_counts)
            if not stats:
                details[name] = "no pixels"
                continue
            details[f"{name} min/max/mean"] = f"{stats['min']}/{stats['max']}/{stats['mean']}"
            details[f"{name} empty bins"] = f"{stats['empty bins']} of {stats['range bins']}"

        return ToolResult(
            image=output_img,
            message="Generated histogram.",
            details=details,
        )
