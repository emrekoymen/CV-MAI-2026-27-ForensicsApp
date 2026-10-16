from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import exposure, feature, img_as_ubyte

from .base import ForensicsTool, ToolResult


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG visualization"
    category = "Detection and matching"
    description = "Extract and draw Histogram of Oriented Gradients features."

    def run(self, parent, document):
        orientations = simpledialog.askinteger("HOG", "Orientations:", parent=parent, minvalue=1, initialvalue=9)
        if orientations is None:
            return None
        cell = simpledialog.askinteger("HOG", "Pixels per cell:", parent=parent, minvalue=1, initialvalue=8)
        if cell is None:
            return None
        block = simpledialog.askinteger("HOG", "Cells per block:", parent=parent, minvalue=1, initialvalue=2)
        if block is None:
            return None
        features, hog_image = feature.hog(
            np.array(document.current.convert("RGB")), orientations=orientations, pixels_per_cell=(cell, cell),
            cells_per_block=(block, block), visualize=True, channel_axis=-1)
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max() / 2 or 1), out_range=(0, 1))
        return ToolResult(image=Image.fromarray(img_as_ubyte(hog_image)), message=f"Extracted {features.size} HOG features.")
