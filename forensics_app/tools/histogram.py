import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import io

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class HistogramTool(ForensicsTool):
    tool_id = "histogram"       # unique, stable identifier
    title = "Histogram"         # text shown in the sidebar
    category = "Image Plots"
    description = "Generate the histogram of the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        im = document.current
        im2 = np.array(im)
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        # Check if RGB or Single-channel
        if im2.ndim == 3 and im2.shape[2] >= 3:
            colors = ('red', 'green', 'blue')
            channels = [im2[:, :, i].ravel() for i in range(3)]
            
            ax.hist(
                channels, 
                bins=256, 
                range=(0, 256), 
                color=colors, 
                label=['Red', 'Green', 'Blue'],
                histtype='step',  # 'step' or 'stepfilled' keeps all channel outlines visible
                alpha=0.8,
                linewidth=1.5
            )
            ax.set_xlim([0, 256])
            ax.legend(loc='upper right')
        else:
            ax.hist(im2.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
            ax.set_xlim([0, 256])

        ax.set_title("Intensity Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        # Render Figure to In-Memory Buffer and decode into PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        output_img = Image.open(buf).convert("RGB")

        # Cleanup Matplotlib resources
        plt.close(fig)
        
        return ToolResult(
            image=output_img,
            message="Generated histogram.",
            details={"Operation": "Histogram"},
        )