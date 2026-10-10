import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import filters, img_as_ubyte

from .base import ForensicsTool, ToolResult

FILTER_NAMES = ("Gaussian", "Sobel", "Prewitt")


def compute_filter(image, name, sigma=1.0):
    image = np.asarray(image.convert("L"), dtype=float) / 255
    if name == "Gaussian":
        output = filters.gaussian(image, sigma=sigma)
    elif name == "Sobel":
        output = filters.sobel(image)
    else:
        output = filters.prewitt(image)
    return Image.fromarray(img_as_ubyte(output))


class FiltersTool(ForensicsTool):
    tool_id = "skimage_filters"
    title = "Image filters"
    category = "Filtering"
    description = "Apply Gaussian, Sobel or Prewitt to a grayscale image."

    def choose_filter(self, parent):
        dialog = tk.Toplevel(parent)
        dialog.title("Select a filter")
        dialog.transient(parent)
        dialog.grab_set()
        choice = None

        def select(name):
            nonlocal choice
            choice = name
            dialog.destroy()

        for name in FILTER_NAMES:
            tk.Button(dialog, text=name, command=lambda v=name: select(v)).pack(fill="x", padx=12, pady=5)
        dialog.wait_window()  # closing the window cancels
        return choice

    def run(self, parent, document):
        name = self.choose_filter(parent)
        if name is None:
            return None
        sigma = 1.0
        details = {"Filter": name}
        if name == "Gaussian":
            sigma = simpledialog.askfloat(
                "Gaussian", "Sigma:", parent=parent, minvalue=0.0, initialvalue=1.0)
            if sigma is None:
                return None
            details["Sigma"] = sigma
        return ToolResult(
            image=compute_filter(document.current, name, sigma),
            message=f"Applied {name} filter.",
            details=details,
        )
