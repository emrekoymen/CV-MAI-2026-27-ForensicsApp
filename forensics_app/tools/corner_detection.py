from tkinter import simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

from .base import ForensicsTool, ToolResult
from .panels import make_grid

class CornerDectectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner Detection"
    category = "Detection and matching"
    description = "Obtain the corners from an image using FAST and Harris"

    def run(self, parent, document):
        img = np.array(document.current.convert("L"), dtype=float) / 255

        rel_threshold = simpledialog.askfloat("Corner Detection", "Relative threshold (0-1):", parent=parent, minvalue=0, maxvalue=1, initialvalue=0.15)
        if rel_threshold is None:
            return None

        fast = feature.corner_fast(img)
        fast_points = feature.corner_peaks(fast, threshold_rel=rel_threshold)

        harris_corners = feature.corner_harris(img)
        harris_points = feature.corner_peaks(harris_corners, threshold_rel=rel_threshold,exclude_border=5)
        im1 = document.current.convert("L").convert("RGB")
        im2 = im1.copy()

        draw1 = ImageDraw.Draw(im1)
        draw2 = ImageDraw.Draw(im2)
        for row,col in fast_points:
            draw1.circle((col,row), radius = 4, outline="red", width=2)

        for row, col in harris_points:
            draw2.circle((col, row), radius = 4, outline="red", width=2)

        panels = [("FAST", np.array(im1)), ("Harris", np.array(im2))]


        return ToolResult(
            image=make_grid(panels, columns=2),
            message=f"Found {len(fast_points)} FAST and {len(harris_points)} Harris corners.",
            details={"Operation": "Corner detection", "FAST corners": len(fast_points), "Harris corners": len(harris_points)},
        )
