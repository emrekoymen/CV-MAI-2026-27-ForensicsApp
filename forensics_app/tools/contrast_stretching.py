import tkinter as tk
import numpy as np
import skimage

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"       # unique, stable identifier
    title = "Contrast Stretching"         # text shown in the sidebar
    category = "Filtering"
    description = "Stretch the contrast of the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        # im = document.current
        
        image = skimage.img_as_float(document.current)


        #Create low contrast

        low_contrast = skimage.exposure.rescale_intensity(
            image,
            in_range=(0,1),
            out_range=(0.3,0.7)
            )

        # Contrast stretching

        p2, p98 = np.percentile(low_contrast, (2,98))

        stretched = skimage.exposure.rescale_intensity(
            low_contrast,
            in_range=(p2,p98),
            out_range=(0,1)
            )

        output = skimage.img_as_ubyte(stretched)

        output = Image.fromarray(output)
        
        return ToolResult(
            image=output,
            message="Contrast stretched.",
            details={"Operation": "Contrast Stretch"},
        )