from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import exposure, feature, img_as_ubyte, transform
from skimage.color import gray2rgb, rgb2gray

from .base import ForensicsTool, ToolResult
from .mask import open_image


def compute_ncc(image, template, angle=0.0):
    image = rgb2gray(np.array(image.convert("RGB")))
    template = transform.rotate(rgb2gray(np.array(template.convert("RGB"))), angle)
    return image, feature.match_template(image, template, pad_input=True)


class CrossCorrelationTool(ForensicsTool):
    tool_id = "ncc_image"
    title = "Cross-correlation image"
    category = "Detection and matching"
    description = "Normalized cross-correlation of the image with a template."

    def run(self, parent, document):
        template = open_image(parent, "Choose template image")
        if template is None:
            return None
        _, ncc = compute_ncc(document.current, template)
        return ToolResult(
            image=Image.fromarray(img_as_ubyte(exposure.rescale_intensity(ncc, out_range=(0, 1)))),
            message=f"Normalized cross-correlation, best score {ncc.max():.3f}.",
            details={"Operation": "Cross-correlation", "Best score": f"{ncc.max():.3f}"},
        )


class TemplateMatchingTool(ForensicsTool):
    tool_id = "template_matching"
    title = "Template matching"
    category = "Detection and matching"
    description = "Match a rotated template and mark scores above a threshold in red."

    def run(self, parent, document):
        template = open_image(parent, "Choose template image")
        if template is None:
            return None
        angle = simpledialog.askfloat("Template matching", "Template angle (degrees):", parent=parent, initialvalue=0.0)
        if angle is None:
            return None
        threshold = simpledialog.askfloat(
            "Template matching", "Threshold:", parent=parent, minvalue=-1.0, maxvalue=1.0, initialvalue=0.8)
        if threshold is None:
            return None
        image, ncc = compute_ncc(document.current, template, angle)
        output = gray2rgb(image) * 0.4
        output[ncc >= threshold] = [1, 0, 0]
        return ToolResult(
            image=Image.fromarray(img_as_ubyte(output)),
            message=f"{(ncc >= threshold).sum()} pixel(s) above {threshold:g} at angle {angle:g}.",
            details={"Operation": "Template matching", "Angle": angle, "Threshold": threshold, "Pixels above th.": f"{(ncc >= threshold).sum()}"},
        )
