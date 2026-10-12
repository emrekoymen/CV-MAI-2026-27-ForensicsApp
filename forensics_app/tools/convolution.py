import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from .base import ForensicsTool, ToolResult

DIRECTIONS = ("Horizontal", "Vertical", "Both")


def build_kernel(size, direction):
    shape = {"Horizontal": (1, size), "Vertical": (size, 1), "Both": (size, size)}[direction]
    return np.ones(shape) / (shape[0] * shape[1])


def compute_convolution(image, kernel):
    return convolve(np.array(image, dtype=float), kernel)


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Filtering"
    description = "Average a grayscale image with a chosen kernel size and direction."

    def choose_direction(self, parent):
        dialog = tk.Toplevel(parent)
        dialog.title("Kernel direction")
        dialog.transient(parent)
        dialog.grab_set()
        choice = None

        def select(name):
            nonlocal choice
            choice = name
            dialog.destroy()

        for name in DIRECTIONS:
            tk.Button(dialog, text=name, command=lambda v=name: select(v)).pack(fill="x", padx=12, pady=5)
        dialog.wait_window()  # closing the window cancels
        return choice

    def run(self, parent, document):
        direction = self.choose_direction(parent)
        if direction is None:
            return None
        size = simpledialog.askinteger(
            "Convolution", "Kernel size:", parent=parent, minvalue=1, maxvalue=99, initialvalue=15)
        if size is None:
            return None

        assert document.current is not None
        response = compute_convolution(document.current.convert("L"), build_kernel(size, direction))
        return ToolResult(
            image=Image.fromarray(response.round().astype(np.uint8)),
            message=f"Applied {direction.lower()} convolution (size {size}).",
            details={"Direction": direction, "Kernel size": size},
        )
