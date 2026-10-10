import tkinter as tk
import numpy as np
import io

from PIL import Image
import matplotlib.pyplot as plt

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
            b1 = tk.Button(frame2,text='Red-Green', command=lambda: on_select(1))
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
        if swap_type == 0:
            return None  # dialog closed without choosing


        assert document.current is not None

        im = np.array(document.current)

        im2 = im.copy()


        if swap_type == 1:
            im2[:,:,0]=im[:,:,1]
            im2[:,:,1]=im[:,:,0]
            outtext = 'Red-Green'
        elif swap_type == 2:
            im2[:,:,0]=im[:,:,2]
            im2[:,:,2]=im[:,:,0]
            outtext = 'Red-Blue'
        else:
            im2[:,:,1]=im[:,:,2]
            im2[:,:,2]=im[:,:,1]
            outtext = 'Green-Blue'

        fig, ax = plt.subplots(1,2, figsize=(6, 4), dpi=100)
        ax[0].imshow(im)
        ax[0].set_title("Original image")
        ax[0].axis('off')

        ax[1].imshow(im2)
        ax[1].set_title(outtext+' swapped')
        ax[1].axis('off')
        fig.tight_layout()

        # Render Figure to In-Memory Buffer and decode into PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        output = Image.open(buf).convert("RGB")

        # Cleanup Matplotlib resources
        plt.close(fig)


        # r, g, b = im.split()

        # im3 = Image.merge("RGB",(g,r,b))


        return ToolResult(
            image=output,
            message="Channel swap between RGB channels.",
            details={"Operation": "Channel Swap", "Type": outtext},
        )
