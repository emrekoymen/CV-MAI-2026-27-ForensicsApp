import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

from .base import ForensicsTool, ToolResult
from .panels import make_grid

EXTRACTOR_NAMES = ("ORB","SIFT")

def extract_orb(image,n_keypoints=200):
    orb = feature.ORB(n_keypoints=n_keypoints)
    orb.detect_and_extract(image)
    return orb.keypoints

def extract_sift(image,n_octaves=8,n_scales=3,sigma=1.6):
    sift = feature.SIFT(n_octaves=n_octaves,n_scales=n_scales,sigma_min=sigma)
    sift.detect_and_extract(image)
    return sift.keypoints

class DescriptorsTool(ForensicsTool):
    tool_id = "descriptor_extraction"
    title = "Descriptor Extraction"
    category = "Detection and matching"
    description = "Extract the descriptors from and image and highlight its keypoints."

    def choose_extractor(self, parent):
        dialog = tk.Toplevel(parent)
        dialog.title("Select an extractor")
        dialog.transient(parent)
        dialog.grab_set()
        choice = None

        def select(name):
            nonlocal choice
            choice = name
            dialog.destroy()

        for name in EXTRACTOR_NAMES:
            tk.Button(dialog, text=name, command=lambda v=name: select(v)).pack(fill="x", padx=12, pady=5)
        dialog.wait_window()  # closing the window cancels
        return choice

    def choose_siftvalues(self, parent):
        dialog = tk.Toplevel(parent)
        dialog.title("SIFT values")
        dialog.transient(parent)
        dialog.grab_set()
        tk.Label(
            dialog,
            text="Select values for SIFT",
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6))
        octaves_text = tk.StringVar(value="8")
        scales_text = tk.StringVar(value="3")
        sigma_text = tk.StringVar(value="1.6")

        tk.Label(dialog, text="N. octaves:").grid(row=1, column=0, sticky="e", padx=(12, 4))
        tk.Entry(dialog, textvariable=octaves_text, width=8).grid(row=1, column=1, sticky="w", padx=(0, 12))
        tk.Label(dialog, text="N. scales:").grid(row=2, column=0, sticky="e", padx=(12, 4))
        tk.Entry(dialog, textvariable=scales_text, width=8).grid(row=2, column=1, sticky="w", padx=(0, 12))
        tk.Label(dialog, text="Sigma:").grid(row=3, column=0, sticky="e", padx=(12, 4))
        tk.Entry(dialog, textvariable=sigma_text, width=8).grid(row=3, column=1, sticky="w", padx=(0, 12))

        chosen = None
        def on_ok():
            nonlocal chosen
            try:
                values = (int(octaves_text.get()), int(scales_text.get()), float(sigma_text.get()))
                if min(values) <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid values", "Octaves and scales must be positive integers; sigma a positive number.",
                    parent=dialog)
                return
            chosen = values
            dialog.destroy()

        tk.Button(dialog, text="OK", command=on_ok).grid(row=4, column=0, columnspan=2, pady=12)
        dialog.bind("<Return>", lambda event: on_ok())
        dialog.wait_window()
        return chosen

    def run(self, parent, document):
        name = self.choose_extractor(parent)
        if name is None:
            return None
        img = np.array(document.current.convert("L"), dtype=float) / 255

        if name == "ORB":
            n_keypoints = simpledialog.askinteger("ORB ", "Maximum number of keypoints:", parent=parent, minvalue=1, initialvalue=200)
            if n_keypoints is None:
                return None
            keypoints = extract_orb(img,n_keypoints)
            settings = {"Max keypoints": n_keypoints}
        elif name == "SIFT":
            vals = self.choose_siftvalues(parent)
            if vals is None:
                return None
            n_octaves, n_scales, sigma = vals
            keypoints = extract_sift(img, n_octaves, n_scales, sigma)
            settings = {"Octaves": n_octaves, "Scales": n_scales, "Sigma": sigma}


        im1 = document.current.convert("L").convert("RGB")

        draw1 = ImageDraw.Draw(im1)
        for row,col in keypoints:
            draw1.circle((col,row), radius = 4, outline="red", width=2)


        panels = [(name, np.array(im1))]


        return ToolResult(
            image=make_grid(panels, columns=1),
            message=f"Found {len(keypoints)} keypoints using {name}",
            details={"Operation": "Descriptor extraction", "Extractor": name, **settings,
                "Keypoints": len(keypoints)},
        )
