"""Arrange labelled image panels in a grid, at full resolution (used by channel split and swap)."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (40, 40, 40)   # colour of the label strips and the gaps between panels
TEXT_COLOUR = (255, 255, 255)


def label_font_size(panel_width: int) -> int:
    """Scale the label with the panel, so it stays readable when the grid is zoomed out."""
    return max(10, panel_width // 20)


def grid_layout(
    panel_width: int, panel_height: int, count: int, columns: int
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Return the canvas size and the top-left corner of each panel's pixels.

    Every panel has a label strip above it, and panels are separated by a small gap.
    """
    strip = int(label_font_size(panel_width) * 1.6)
    gap = max(2, panel_width // 200)
    rows = math.ceil(count / columns)
    size = (
        columns * panel_width + (columns - 1) * gap,
        rows * (strip + panel_height) + (rows - 1) * gap,
    )
    origins = []
    for index in range(count):
        row, column = divmod(index, columns)
        origins.append((column * (panel_width + gap), row * (strip + panel_height + gap) + strip))
    return size, origins


def to_rgb_panel(pixels: np.ndarray) -> np.ndarray:
    """Show a grayscale array as grey RGB, and drop alpha from RGBA, so all panels match."""
    if pixels.ndim == 2:
        return np.dstack([pixels] * 3)
    return pixels[:, :, :3]


def make_grid(panels: list[tuple[str, np.ndarray]], columns: int) -> Image.Image:
    """Place equally sized ``(label, pixels)`` panels in a grid with a label above each one."""
    height, width = panels[0][1].shape[:2]
    size, origins = grid_layout(width, height, len(panels), columns)
    canvas = Image.new("RGB", size, BACKGROUND)
    strip = origins[0][1]
    font = ImageFont.load_default(size=label_font_size(width))

    for (label, pixels), (x, y) in zip(panels, origins):
        canvas.paste(Image.fromarray(to_rgb_panel(pixels)), (x, y))
        # Draw the label on its own strip, so long labels are cut off at the panel edge
        # instead of running into the next panel
        label_strip = Image.new("RGB", (width, strip), BACKGROUND)
        ImageDraw.Draw(label_strip).text(
            (width / 2, strip / 2), label, font=font, fill=TEXT_COLOUR, anchor="mm")
        canvas.paste(label_strip, (x, y - strip))
    return canvas
