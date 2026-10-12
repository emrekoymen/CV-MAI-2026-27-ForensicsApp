# Channel Split

#  WARNING! CHECK HOW TO AVOID DOUBLE SPLITS
# Error message when trying to split after grayscale?

import tkinter as tk
from tkinter import simpledialog

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
        # plt.figure(figsize=(16, 4))
        # fig.add_subplot(1, 4, 1)
        # plt.imshow(document.current)
        # plt.title('Default image')
        # fig.add_subplot(1, 4, 2)
        # plt.imshow(document.current[:,:,0])
        # fig.add_subplot(1, 4, 3)
        # plt.imshow(document.current[:,:,1])
        # fig.add_subplot(1, 4, 4)
        # plt.imshow(document.current[:,:,2])

        # im = np.array(document.current)
        # r = im[:,:,0]
        # g = im[:,:,1]
        # b = im[:,:,2]
        # w = im.size[0]+10
        # w2 = im.size[0]*4
        # h = im.size[1]
        # im2 = Image.new("RGB",(w2,h))
        # im = Image.fromarray(im)
        # r = Image.fromarray(r)
        # g = Image.fromarray(g)
        # b = Image.fromarray(b)
        # im2.paste(im)
        # im2.paste(r,(w, 0))
        # im2.paste(g,(w*2, 0))
        # im2.paste(b,(w*3, 0))
        
        # output = im2

        

        im = document.current
        
        r, g, b = im.split()
        w = im.size[0]+10
        w2 = im.size[0]*4
        h = im.size[1]
        im2 = Image.new("RGB",(w2,h))
        im2.paste(im)
        im2.paste(r,(w, 0))
        im2.paste(g,(w*2, 0))
        im2.paste(b,(w*3, 0))
        
        output = im2
        return ToolResult(
            image=output,
            message="Image split into individual RGB channels.",
            details={"Operation": "Channel Split."},
        )
