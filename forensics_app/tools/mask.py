import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


MASK_THRESHOLD = 0  # a mask pixel brighter than this in any channel counts as "masked"


def to_rgb_array(
    image: Image.Image | np.ndarray,
    background: str,
    size: tuple[int, int] | None = None,
    resample: int = Image.BILINEAR,
) -> np.ndarray:
    """Return ``image`` as an RGB uint8 array, optionally resized to ``size`` (width, height).

    Grayscale, 1-bit and palette images are converted to RGB, and transparent
    areas are filled with ``background``.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    rgba = image.convert("RGBA")
    if size is not None and rgba.size != size:
        rgba = rgba.resize(size, resample)
    flat = Image.new("RGBA", rgba.size, background)
    flat.alpha_composite(rgba)
    return np.array(flat.convert("RGB"))


def apply_mask(
    image: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    texture: Image.Image | np.ndarray | None = None,
) -> np.ndarray:
    """Paste the mask, or the texture if given, onto ``image`` wherever the mask is not black."""
    # Transparent parts of the image and texture become white (as rgba2rgb did before),
    # but transparent parts of the mask become black, so they count as "not masked".
    pixels = to_rgb_array(image, background="white")
    size = (pixels.shape[1], pixels.shape[0])  # PIL sizes are (width, height)
    # Nearest neighbour keeps the mask edges sharp instead of blending them
    mask_pixels = to_rgb_array(mask, background="black", size=size, resample=Image.NEAREST)
    selected = (mask_pixels > MASK_THRESHOLD).any(axis=2)  # one True/False per pixel

    if texture is None:
        source = mask_pixels
    else:
        source = to_rgb_array(texture, background="white", size=size)

    pixels[selected] = source[selected]
    return pixels


def open_image(parent: tk.Misc, title: str) -> Image.Image | None:
    """Ask for an image file; return ``None`` if cancelled or if it cannot be opened."""
    filename = filedialog.askopenfilename(title=title)
    if not filename:
        return None
    try:
        image = Image.open(filename)
        image.load()  # read it now, so damaged files fail here and not later
    except (OSError, UnidentifiedImageError) as error:
        messagebox.showerror("Could not open image", str(error), parent=parent)
        return None
    return image


class MaskTool(ForensicsTool):
    tool_id = "apply_mask"       # unique, stable identifier
    title = "Apply mask"         # text shown in the sidebar
    category = "Filtering"
    description = "Apply a mask to an image."

    def choose_mask(self, parent: tk.Misc) -> Image.Image | None:
        return open_image(parent, 'Add Mask File')

    def choose_texture(self, parent: tk.Misc) -> Image.Image | None:
        """Ask whether to use a texture; return it, or ``None`` for no texture."""
        dialog = tk.Toplevel(parent)
        dialog.title("Texture")
        dialog.geometry('500x200')

        # Force to interact:
        # dialog.transient(parent)
        dialog.grab_set()

        texturefile = None
        def on_yes():
            nonlocal texturefile
            texturefile = open_image(parent, 'Add Texture File')
            if texturefile is not None:
                dialog.destroy()

        frame1 = tk.Frame(dialog)
        frame2 = tk.Frame(dialog)

        label1 = tk.Label(frame1, text='Do you wish to apply a texture to the mask?')
        label1.pack(padx=10, pady=10)
        frame1.pack()
        b1 = tk.Button(frame2, text='Yes!', command=on_yes)
        b2 = tk.Button(frame2,text='No!',command=dialog.destroy)

        b1.pack(side='left',padx=5,pady=10)
        b2.pack(side='left',padx=5,pady=10)

        frame2.pack()

        # Block until the dialog is destroyed
        dialog.wait_window()
        return texturefile

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        assert document.current is not None

        mask = self.choose_mask(parent)
        if mask is None:
            return None  # file dialog cancelled, or the file could not be opened

        texture = self.choose_texture(parent)

        output = Image.fromarray(apply_mask(document.current, mask, texture))

        target = document.current
        details = {"Operation": "Masking", "Texture": "Yes" if texture is not None else "No"}
        for name, source in (("Mask", mask), ("Texture", texture)):
            if source is not None and source.size != target.size:
                details[f"{name} resized"] = (
                    f"{source.width} × {source.height} → {target.width} × {target.height} px"
                )

        return ToolResult(
            image=output,
            message="Apply a mask to an image.",
            details=details,
        )
