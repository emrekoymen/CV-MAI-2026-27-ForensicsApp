from tkinter import simpledialog

import numpy as np
from PIL import ImageDraw
from skimage import feature

from .base import ForensicsTool, ToolResult
from .mask import open_image
from .panels import make_grid


def detect_person(image, template, step):
    """Return the top-left (row, col) of the best window by Euclidean distance and by dot product."""
    hog = lambda pixels: feature.hog(pixels, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    height, width = template.shape
    template_fd = hog(template)
    coords, distances, products = [], [], []
    for i in range(0, image.shape[0] - height + 1, step):
        for j in range(0, image.shape[1] - width + 1, step):
            window_fd = hog(image[i:i + height, j:j + width])
            coords.append((i, j))
            distances.append(np.linalg.norm(template_fd - window_fd))
            products.append(np.dot(template_fd, window_fd))
    return coords[np.argmin(distances)], coords[np.argmax(products)]


class PersonDetectionTool(ForensicsTool):
    tool_id = "hog_person_detection"
    title = "Person detection (HOG)"
    category = "Detection and matching"
    description = "Find the window whose HOG features best match a person template."

    def run(self, parent, document):
        template = open_image(parent, "Choose person template image")
        if template is None:
            return None
        step = simpledialog.askinteger("Person detection", "Window step (pixels):", parent=parent, minvalue=1, initialvalue=4)
        if step is None:
            return None
        template = np.array(template.convert("L"))
        height, width = template.shape
        best = detect_person(np.array(document.current.convert("L")), template, step)
        panels = []
        for label, (i, j) in zip(("Euclidean distance", "Dot product"), best):
            output = document.current.convert("RGB")
            ImageDraw.Draw(output).rectangle((j, i, j + width - 1, i + height - 1), outline="red", width=2)
            panels.append((label, np.array(output)))
        return ToolResult(
            image=make_grid(panels, columns=2),
            message=f"Person detection (step {step}).",
            details={"Operation": "Person detection", "Step": step, "Coordinates": "(row, column)", "Euclidean": best[0],
            "Dot product": best[1]},
        )
