import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image

from skimage import filters, img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


class SharpeningTool(ForensicsTool):
    tool_id = "sharpening"       # unique, stable identifier
    title = "Image sharpening"         # text shown in the sidebar
    category = "Filtering"
    description = "Sharpen an image with a specified radius."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        assert document.current is not None


        radius = simpledialog.askfloat(
            "Image sharpening",
            "Choose radius (pixels):",
            parent=parent,
            minvalue=0.0,
            initialvalue=2.0,
        )
        if radius is None:
            return None

        im = np.array(document.current)
        alpha = None
        if im.ndim == 3 and im.shape[2] == 4:  # keep transparency untouched
            alpha = im[:, :, 3:]
            im = im[:, :, :3]
        im = img_as_float(im)

        sharpened_skimage = filters.unsharp_mask(
            im,
            radius = radius,
            amount = 1.5,
            channel_axis = 2 if im.ndim == 3 else None,  # per colour channel (negative axis is buggy in skimage 0.26)
        )


        sharpened_clipped = np.clip(sharpened_skimage, 0.0, 1.0)
        sharpened = img_as_ubyte(sharpened_clipped)
        if alpha is not None:
            sharpened = np.concatenate([sharpened, alpha], axis=2)
        output = Image.fromarray(sharpened)


        return ToolResult(
            image=output,
            message=f"Applied image sharpening (radius {radius:g}).",
            details={"Operation": "Sharpening", "Radius": radius},
        )
