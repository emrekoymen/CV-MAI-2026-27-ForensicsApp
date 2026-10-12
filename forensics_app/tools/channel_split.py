# Channel Split

import tkinter as tk
from tkinter import simpledialog
import numpy as np

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .panels import make_grid


def split_channels(image: Image.Image | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the red, green and blue channels of an RGB(A) image as 2-D arrays."""
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        if isinstance(image, Image.Image):      # "is this a PIL image?"
            mode = image.mode                   # e.g. "L"
        else:                                   # otherwise it's a NumPy array
            mode = f"array {pixels.shape}"      # e.g. "array (4, 4)"
        # The main window shows this message in an error dialog
        raise ValueError(
            f"Channel split needs a colour image, but the current image is "
            f"'{mode}' and has no RGB channels to split."
        )
    return pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"       # unique, stable identifier
    title = "Channel split"         # text shown in the sidebar
    category = "Filtering"
    description = "Split the image into individual RGB channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        assert document.current is not None

        r, g, b = split_channels(document.current)

        # 2x2 grid at full resolution: Original, Red / Green, Blue
        output = make_grid(
            [("Original", np.asarray(document.current)), ("Red channel", r),
             ("Green channel", g), ("Blue channel", b)],
            columns=2,
        )

        return ToolResult(
            image=output,
            message="Image split into individual RGB channels.",
            details={"Operation": "Channel Split"},
        )
