import tkinter as tk
import numpy as np

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .panels import make_grid


# swap_type chosen in the dialog -> (label, first channel, second channel)
SWAPS = {
    1: ("Red-Green", 0, 1),
    2: ("Red-Blue", 0, 2),
    3: ("Green-Blue", 1, 2),
}


def check_colour_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return the image as an array, or raise a clear error if it has no RGB channels."""
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        if isinstance(image, Image.Image):      # "is this a PIL image?"
            mode = image.mode                   # e.g. "L"
        else:                                   # otherwise it's a NumPy array
            mode = f"array {pixels.shape}"      # e.g. "array (4, 4)"
        # The main window shows this message in an error dialog
        raise ValueError(
            f"Channel swap needs a colour image, but the current image is "
            f"'{mode}' and has no RGB channels to swap."
        )
    return pixels


def swap_channels(image: Image.Image | np.ndarray, swap_type: int) -> np.ndarray:
    """Return a copy of an RGB(A) image with the two channels of ``swap_type`` exchanged."""
    if swap_type not in SWAPS:
        raise ValueError(f"Unknown swap type {swap_type}; expected one of {sorted(SWAPS)}.")
    pixels = check_colour_image(image)
    _, first, second = SWAPS[swap_type]
    swapped = pixels.copy()
    swapped[:, :, first] = pixels[:, :, second]
    swapped[:, :, second] = pixels[:, :, first]
    return swapped


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"       # unique, stable identifier
    title = "Channel swap"         # text shown in the sidebar
    category = "Filtering"
    description = "Swap RGB channels."

    def choose_swap_type(self, parent: tk.Misc) -> int:
        """Ask which channels to swap; return 0 if the dialog is closed."""
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

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        assert document.current is not None

        # Check before the dialog so the user isn't asked to choose a swap that can't run
        im = check_colour_image(document.current)

        swap_type = self.choose_swap_type(parent)
        if swap_type == 0:
            return None  # dialog closed without choosing

        im2 = swap_channels(document.current, swap_type)  # PIL image, so errors name its mode
        outtext = SWAPS[swap_type][0]

        # Side by side at full resolution
        output = make_grid([("Original", im), (outtext + " swapped", im2)], columns=2)

        return ToolResult(
            image=output,
            message="Channel swap between RGB channels.",
            details={"Operation": "Channel Swap", "Type": outtext},
        )
