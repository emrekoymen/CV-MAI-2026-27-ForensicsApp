import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image

from skimage import feature, img_as_float, img_as_ubyte
from skimage.color import rgba2rgb, rgb2gray, gray2rgb

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


class EdgeTool(ForensicsTool):
    tool_id = "edges"       # unique, stable identifier
    title = "Edge visualization"         # text shown in the sidebar
    category = "Filtering"
    description = "Visualize the edges of the image, superimposed on the original."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        assert document.current is not None


        sigma = simpledialog.askfloat(
            "Edge detection",
            "Choose sigma:",
            parent=parent,
            minvalue=0.0,
            initialvalue=2.0,
        )
        if sigma is None:
            return None

        im = np.array(document.current)
        if im.ndim > 2:
            if im.shape[2] > 3:
                    im = rgba2rgb(im)
            im = rgb2gray(im)
        im = img_as_float(im)
        edges = feature.canny(im, sigma=sigma)
        rgb = gray2rgb(im)
        red_edges = rgb.copy()
        red_edges[edges] = [1, 0, 0]


        output = Image.fromarray(img_as_ubyte(red_edges))


        return ToolResult(
            image=output,
            message=f"Applied edge visualization (sigma {sigma:g}).",
            details={"Operation": "Edge Visualization", "Sigma": sigma},
        )
