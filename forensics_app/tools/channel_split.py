# Channel Split

#  WARNING! CHECK HOW TO AVOID DOUBLE SPLITS
# Error message when trying to split after grayscale?

import tkinter as tk
from tkinter import simpledialog
import numpy as np
import matplotlib.pyplot as plt
import io

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"       # unique, stable identifier
    title = "Channel split"         # text shown in the sidebar
    category = "Filtering"
    description = "Split the image into individual RGB channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        assert document.current is not None

        im = np.array(document.current)
        r = im[:,:,0]
        g = im[:,:,1]
        b = im[:,:,2]


        # output = im2
        fig, ax = plt.subplots(1,4,figsize=(12, 4), dpi=100)

        im = document.current

        # r, g, b = im.split()

        ax[0].imshow(im)
        ax[0].set_title('Original')
        ax[0].axis('off')
        ax[1].imshow(r, cmap='gray')
        ax[1].set_title('Red Channel')
        ax[1].axis('off')
        ax[2].imshow(g, cmap='gray')
        ax[2].set_title('Green Channel')
        ax[2].axis('off')
        ax[3].imshow(b, cmap='gray')
        ax[3].set_title('Blue Channel')
        ax[3].axis('off')
        fig.tight_layout()

        # output = im2

        # Render Figure to In-Memory Buffer and decode into PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        output = Image.open(buf).convert("RGB")

        # Cleanup Matplotlib resources
        plt.close(fig)

        return ToolResult(
            image=output,
            message="Image split into individual RGB channels.",
            details={"Operation": "Channel Split"},
        )
