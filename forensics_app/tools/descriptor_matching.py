import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import feature

from .base import ForensicsTool, ToolResult
from .mask import open_image

EXTRACTOR_NAMES = ("ORB","SIFT")
GAP = 10                   # pixels between the two images
BACKGROUND = (40, 40, 40)

def descriptor_orb(image,n_keypoints=200):
    orb = feature.ORB(n_keypoints=n_keypoints)
    orb.detect_and_extract(image)
    return orb.keypoints, orb.descriptors

def descriptor_sift(image,n_octaves=8,n_scales=3,sigma=1.6):
    sift = feature.SIFT(n_octaves=n_octaves,n_scales=n_scales,sigma_min=sigma)
    sift.detect_and_extract(image)
    return sift.keypoints, sift.descriptors

def descriptor_match(descriptors1,descriptors2,cross_check=True,max_ratio=1.0):
    matches = feature.match_descriptors(descriptors1,descriptors2,cross_check=cross_check,
        max_ratio=max_ratio)
    return matches

def draw_matches(im1, im2, keypoints1, keypoints2, matches, labels=None):
    """Place both images side by side, circle every keypoint and join matched pairs."""

    # Scale both images to the taller one's height, so a small image isn't dwarfed by a large one.
    # Keypoints were found on the original sizes, so they are scaled by the same factor.
    height = max(im1.height, im2.height)
    scale1, scale2 = height / im1.height, height / im2.height
    im1 = im1.resize((round(im1.width * scale1), height))
    im2 = im2.resize((round(im2.width * scale2), height))

    thickness = max(1, height // 400)
    radius = 4 * thickness

    offset = im1.width + GAP
    font_size = max(10, max(im1.width, im2.width) // 20)
    strip = int(font_size * 1.6) if labels else 0
    canvas = Image.new("RGB", (offset + im2.width, strip + max(im1.height, im2.height)), BACKGROUND)
    canvas.paste(im1, (0, strip))
    canvas.paste(im2, (offset, strip))
    draw = ImageDraw.Draw(canvas)



    if labels:
        font = ImageFont.load_default(size=font_size)
        draw.text((im1.width / 2, strip / 2), labels[0], font=font, anchor="mm")
        draw.text((offset + im2.width / 2, strip / 2), labels[1], font=font, anchor="mm")

    # Keypoints are  in each image. We shift them to where that image sits on the canvas
    to_left = lambda col, row: (col * scale1, row * scale1 + strip)
    to_right = lambda col, row: (col * scale2 + offset, row * scale2 + strip)

    matched1 = keypoints1[matches[:, 0]]
    matched2 = keypoints2[matches[:, 1]]

    unmatched1 = np.ones(len(keypoints1), dtype=bool)
    unmatched1[matches[:, 0]] = False
    unmatched2 = np.ones(len(keypoints2), dtype=bool)
    unmatched2[matches[:, 1]] = False

    for row, col in keypoints1[unmatched1]:
        draw.circle(to_left(col, row), radius=radius, outline="red", width=2 * thickness)
    for row, col in keypoints2[unmatched2]:
        draw.circle(to_right(col, row), radius=radius, outline="red", width=2 * thickness)

    # A different color per match, so crossing lines can be told apart
    colors = np.random.default_rng(0).integers(64, 256, size=(len(matches), 3))
    for (r1, c1), (r2, c2), color in zip(matched1, matched2, colors):
        color = tuple(int(v) for v in color)
        draw.circle(to_left(c1, r1), radius=radius, outline=color, width=2 * thickness)
        draw.circle(to_right(c2, r2), radius=radius, outline=color, width=2 * thickness)
        draw.line([to_left(c1, r1), to_right(c2, r2)], fill=color, width=thickness)
    return canvas

class DescriptorMatchingTool(ForensicsTool):
    tool_id = "descriptor_matching"
    title = "Descriptor Matching"
    category = "Detection and matching"
    description = "Extract the descriptors from two images and match them."

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
        image2 = open_image(parent, 'Add second image')
        if image2 is None:
            return None
        name = self.choose_extractor(parent)
        if name is None:
            return None
        img = np.array(document.current.convert("L"), dtype=float) / 255
        img2 = np.array(image2.convert("L"), dtype=float) / 255

        if name == "ORB":
            max_ratio = 1.0
            n_keypoints = simpledialog.askinteger("ORB ", "Maximum number of keypoints:", parent=parent, minvalue=1, initialvalue=200)
            if n_keypoints is None:
                return None
            keypoints1, descriptors1 = descriptor_orb(img,n_keypoints)
            keypoints2, descriptors2 = descriptor_orb(img2,n_keypoints)
            settings = {"Max keypoints": n_keypoints}
        elif name == "SIFT":
            max_ratio = 0.8
            # 0.8 ratio recommended for SIFT, remove irrelevant matches (D.G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints",
            # International Journal of Computer Vision, 2004.)
            vals = self.choose_siftvalues(parent)
            if vals is None:
                return None
            n_octaves, n_scales, sigma = vals
            keypoints1, descriptors1 = descriptor_sift(img, n_octaves, n_scales, sigma)
            keypoints2, descriptors2 = descriptor_sift(img2, n_octaves, n_scales, sigma)
            settings = {"Octaves": n_octaves, "Scales": n_scales, "Sigma": sigma}

        matches = descriptor_match(descriptors1, descriptors2, cross_check=True, max_ratio = max_ratio)

        im1 = document.current.convert("L").convert("RGB")
        im2 = image2.convert("L").convert("RGB")
        result = draw_matches(im1, im2, keypoints1, keypoints2, matches)

        return ToolResult(
            image=result,
            message=f"Found {len(matches)} matches using {name}",
            details={"Operation": "Descriptor matching", "Extractor": name, **settings,
                        "Keypoints (image 1)": len(keypoints1),"Keypoints (image 2)": len(keypoints2),
                        "Matches": len(matches)},
                )
