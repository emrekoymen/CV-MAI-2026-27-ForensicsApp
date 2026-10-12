import tkinter as tk
import numpy as np

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"       # unique, stable identifier
    title = "Channel swap"         # text shown in the sidebar
    category = "Filtering"
    description = "Swap RGB channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        def open_button_dialog():
            dialog = tk.Toplevel(parent)
            dialog.title("Type of channel swap")
            dialog.geometry('500x200')

            # Force to interact:
            # dialog.transient(parent)
            dialog.grab_set()
            swap_type=0

            def on_select(value):
                nonlocal swap_type
                swap_type = value
                dialog.destroy()

            frame1 = tk.Frame(dialog)
            frame2 = tk.Frame(dialog)

            label1 = tk.Label(frame1, text='Select the type of channel swap:')
            label1.pack(padx=10, pady=10)
            frame1.pack()
            # label2 = tk.Label(frame2, text='Select the type of channel swap:')
            b1 = tk.Button(frame2, text='Red-Green', command=lambda: on_select(1))
            b2 = tk.Button(frame2,text='Red-Blue',command=lambda: on_select(2))
            b3 = tk.Button(frame2,text='Green-Blue',command=lambda: on_select(3))


            b1.pack(side='left',padx=5,pady=10)
            b2.pack(side='left',padx=5,pady=10)
            b3.pack(side='left',padx=5,pady=10)

            frame2.pack()

            # Block until the dialog is destroyed
            dialog.wait_window()
            return swap_type

        swap_type = open_button_dialog()


        assert document.current is not None

        # im = document.current
        im = np.array(document.current)

        im2 = im.copy()


        if swap_type == 1:
            im2[:,:,0]=im[:,:,1]
            im2[:,:,1]=im[:,:,0]
        elif swap_type == 2:
            im2[:,:,0]=im[:,:,2]
            im2[:,:,2]=im[:,:,0]
        else:
            im2[:,:,1]=im[:,:,2]
            im2[:,:,2]=im[:,:,1]

        im = Image.fromarray(im)
        im2 = Image.fromarray(im2)
        w = im.size[0]
        w2 = im.size[0]*2 + 10
        h = im.size[1]
        im3 = Image.new("RGB",(w2,h))
        im3.paste(im)
        im3.paste(im2,(w+10, 0))



        # r, g, b = im.split()

        # im3 = Image.merge("RGB",(g,r,b))


        output = im3


        return ToolResult(
            image=output,
            message="Channel swap between RGB channels.",
            details={"Operation": "Channel Swap."},
        )
