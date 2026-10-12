import numpy as np
from PIL import Image
from skimage.feature import canny
from tkinter import simpledialog

from .base import ForensicsTool, ToolResult


def compute_edges(image, sigma, low, high):
    image = np.array(image, dtype=float) / 255
    return canny(image, sigma=sigma, low_threshold=low, high_threshold=high)


class CannyTool(ForensicsTool):
    tool_id = "canny"
    title = "Canny edges"
    category = "Filtering"
    description = "Detect edges with adjustable sigma and thresholds."

    def choose_settings(self, parent):
        sigma = simpledialog.askfloat(
            "Canny edges", "Sigma:", parent=parent, minvalue=0.0, initialvalue=1.0)
        if sigma is None:
            return None
        low = simpledialog.askfloat(
            "Canny edges", "Low threshold (0-1):", parent=parent,
            minvalue=0.0, maxvalue=1.0, initialvalue=0.1)
        if low is None:
            return None
        high = simpledialog.askfloat(
            "Canny edges", f"High threshold ({low:g}-1):", parent=parent,
            minvalue=low, maxvalue=1.0, initialvalue=max(low, 0.2))
        if high is None:
            return None
        return sigma, low, high

    def run(self, parent, document):
        settings = self.choose_settings(parent)
        if settings is None:
            return None
        sigma, low, high = settings

        assert document.current is not None
        edges = compute_edges(document.current.convert("L"), sigma, low, high)
        return ToolResult(
            image=Image.fromarray(edges.astype(np.uint8) * 255),
            message="Detected Canny edges.",
            details={"Sigma": sigma, "Low threshold": low, "High threshold": high},
        )
