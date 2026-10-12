import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.color import rgba2rgb
from skimage import img_as_ubyte
from skimage.transform import resize

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


class MaskTool(ForensicsTool):
    tool_id = "apply_mask"       # unique, stable identifier
    title = "Apply mask"         # text shown in the sidebar
    category = "Filtering"
    description = "Apply a mask to an image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        maskname = filedialog.askopenfilename(title='Add Mask File')

        if not maskname:
            return
        try:
            maskfile = Image.open(maskname)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=parent)
            return

        def open_button_dialog():
            dialog = tk.Toplevel(parent)
            dialog.title("Texture")
            dialog.geometry('500x200')

            # Force to interact:
            # dialog.transient(parent)
            dialog.grab_set()

            texturefile = None
            def on_yes():
                nonlocal texturefile
                texturename = filedialog.askopenfilename(title='Add Texture File')
                if not texturename:
                    return
                try:
                    texturefile = Image.open(texturename)
                except (OSError, UnidentifiedImageError) as error:
                    messagebox.showerror("Could not open image", str(error), parent=parent)
                    return
                dialog.destroy()


            frame1 = tk.Frame(dialog)
            frame2 = tk.Frame(dialog)

            label1 = tk.Label(frame1, text='Do you wish to apply a texture to the mask?')
            label1.pack(padx=10, pady=10)
            frame1.pack()
            # label2 = tk.Label(frame2, text='Select the type of channel swap:')
            b1 = tk.Button(frame2, text='Yes!', command=lambda: on_yes())
            b2 = tk.Button(frame2,text='No!',command=lambda: dialog.destroy())

            b1.pack(side='left',padx=5,pady=10)
            b2.pack(side='left',padx=5,pady=10)

            frame2.pack()

            # Block until the dialog is destroyed
            dialog.wait_window()
            return texturefile

        texturefile = open_button_dialog()


        assert document.current is not None

        im = np.array(document.current)
        if im.shape[2] > 3:
            im = img_as_ubyte(rgba2rgb(im))


        im_m = np.array(maskfile)
        if im_m.shape[2] > 3:
            im_m = img_as_ubyte(rgba2rgb(im_m))
        im_m = resize(im_m,im.shape,preserve_range=True)
        mask = im_m>1

        if texturefile is not None:
            im_t = np.array(texturefile)
            if im_t.shape[2] > 3:
                im_t = img_as_ubyte(rgba2rgb(im_t))
            im_t = resize(im_t,im_m.shape,preserve_range=True)
            im_m[mask] = im_t[mask]

        im[mask] = im_m[mask]

        im = Image.fromarray(im)

        output = im


        return ToolResult(
            image=output,
            message="Apply a mask to an image.",
            details={"Operation": "Masking."},
        )
