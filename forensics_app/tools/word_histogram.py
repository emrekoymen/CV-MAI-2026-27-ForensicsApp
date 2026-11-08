from tkinter import filedialog, messagebox
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import ForensicsTool, ToolResult
from .descriptor_matching import descriptor_sift
from .k_means import CODEBOOK_FILE, DEFAULT_FOLDER, file_hash

IMAGE_HEIGHT = 600   # the image is scaled to this height, and the histogram drawn just as tall
BAR_WIDTH = 20
GAP = 80             # space between the image and the histogram, where the lines run
LABEL_STRIP = 20     # space under the bars for the word numbers
AXIS_WIDTH = 40      # space left of the bars for the frequency labels
TOP_MARGIN = 25      # space above the bars for the axis title
BACKGROUND = (40, 40, 40)
TEXT_COLOUR = (255, 255, 255)
GRID_COLOUR = (80, 80, 80)
MAX_LINES_PER_WORD = 5   # lines are drawn only for the keypoints closest to their centroid

def word_histogram(kmeans, descriptors):
    """Assign every descriptor to its closest centroid and return the words and their relative frequencies."""
    words = kmeans.predict(descriptors.astype(float))
    counts = np.bincount(words, minlength=kmeans.n_clusters)
    return words, counts / counts.sum()

def lines_to_draw(kmeans, descriptors, words, max_per_word=MAX_LINES_PER_WORD):
    """Indices of the keypoints closest to the centroid of their word, at most ``max_per_word`` per word."""
    distances = kmeans.transform(descriptors.astype(float))[np.arange(len(words)), words]
    chosen = []
    for word in np.unique(words):
        indices = np.nonzero(words == word)[0]
        chosen += list(indices[np.argsort(distances[indices])[:max_per_word]])
    return chosen

def word_colours(n):
    cmap = plt.get_cmap("hsv")
    return [tuple(int(255 * v) for v in cmap(i / n)[:3]) for i in range(n)]

def draw_histogram(image, keypoints, words, frequencies, drawn):
    """Image on the left, histogram on the right, and a line from the keypoints in ``drawn`` to the bar of their word."""
    scale = IMAGE_HEIGHT / image.height
    image = image.convert("L").convert("RGB").resize((round(image.width * scale), IMAGE_HEIGHT))
    left = image.width + GAP + AXIS_WIDTH
    canvas = Image.new("RGB", (left + len(frequencies) * BAR_WIDTH, IMAGE_HEIGHT + LABEL_STRIP), BACKGROUND)
    canvas.paste(image, (0, 0))

    # Pick a round tick step that gives at most 6 ticks, and end the axis on the first tick above the tallest bar
    # (rounded first, so that e.g. 0.4 / 0.1 = 4.0000000001 doesn't add an extra tick)
    step = next(s for s in (0.01, 0.02, 0.05, 0.1, 0.2) if frequencies.max() / s <= 6)
    axis_max = np.ceil(round(frequencies.max() / step, 6)) * step
    to_y = lambda frequency: IMAGE_HEIGHT - frequency / axis_max * (IMAGE_HEIGHT - TOP_MARGIN)

    colours = word_colours(len(frequencies))
    tops = [(left + (word + 0.5) * BAR_WIDTH, to_y(frequency)) for word, frequency in enumerate(frequencies)]
    points = [(col * scale, row * scale) for row, col in keypoints]

    # Lines go on a transparent layer, so they don't hide the image completely
    lines = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(lines)
    for index in drawn:
        draw_lines.line([points[index], tops[words[index]]], fill=colours[words[index]] + (160,), width=1)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), lines).convert("RGB")

    # The axis is drawn after the lines, so its labels stay readable
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=10)
    draw.line([(left, TOP_MARGIN), (left, IMAGE_HEIGHT)], fill=TEXT_COLOUR)
    for tick in np.arange(0, axis_max + step / 2, step):
        y = to_y(tick)
        draw.line([(left, y), (canvas.width, y)], fill=GRID_COLOUR)
        draw.line([(left - 4, y), (left, y)], fill=TEXT_COLOUR)
        draw.text((left - 6, y), f"{tick:.2f}", font=font, fill=TEXT_COLOUR, anchor="rm")
    draw.text((left, TOP_MARGIN / 2), "Frequency", font=font, fill=TEXT_COLOUR, anchor="mm")

    for point, word in zip(points, words):
        draw.circle(point, radius=3, outline=colours[word], width=1)
    for word, (x, top) in enumerate(tops):
        draw.rectangle([x - BAR_WIDTH / 2 + 1, top, x + BAR_WIDTH / 2 - 1, IMAGE_HEIGHT], fill=colours[word])
        draw.text((x, IMAGE_HEIGHT + LABEL_STRIP / 2), str(word + 1), font=font, fill=TEXT_COLOUR, anchor="mm")
    return canvas


def histogram_result(parent, document):
    """Ask for the dataset folder and build the histogram of the loaded image.

    Returns the folder, the saved codebook, the frequencies and the ToolResult (so the classifier can reuse them),
    or None if the user cancels or the image can't be used.
    """
    folder = filedialog.askdirectory(title="Select dataset folder with the codebook", parent=parent,
        initialdir=DEFAULT_FOLDER)
    if not folder:
        return None
    codebook_path = Path(folder) / CODEBOOK_FILE
    if not codebook_path.exists():
        messagebox.showinfo("No codebook", "Run K-means clustering on this folder first.", parent=parent)
        return None
    saved = joblib.load(codebook_path)

    used_in = saved["split"].get(file_hash(document.path), "Not in dataset")
    if used_in == "Train":
        messagebox.showinfo("Training image",
            "This image was used to build the codebook, so its histogram would not be a fair test.", parent=parent)
        return None

    # The original is used, since the image on screen may be a previous tool's result (e.g. the codebook)
    image = document.original
    try:
        keypoints, descriptors = descriptor_sift(np.array(image.convert("L"), dtype=float) / 255)
    except RuntimeError:
        messagebox.showinfo("No features", "Not enough SIFT features were found in this image.", parent=parent)
        return None
    words, frequencies = word_histogram(saved["kmeans"], descriptors)
    drawn = lines_to_draw(saved["kmeans"], descriptors, words)
    top = int(frequencies.argmax())

    result = ToolResult(
        image=draw_histogram(image, keypoints, words, frequencies, drawn),
        message=f"Histogram of {len(keypoints)} keypoints over {len(frequencies)} visual words",
        details={"Operation": "Visual word histogram", "Image in dataset": used_in, "Keypoints": len(keypoints),
            "Lines drawn": f"{len(drawn)} (up to {MAX_LINES_PER_WORD} per word)",
            "Words used": int((frequencies > 0).sum()), "Most frequent word": top + 1,
            "Its frequency": round(float(frequencies[top]), 3)},
    )
    return folder, saved, frequencies, result


class WordHistogramTool(ForensicsTool):
    tool_id = "word_histogram"
    title = "Visual word histogram"
    category = "Bag-of-words"
    description = "Assign the SIFT descriptors of the image to the codebook words and show their frequency histogram."

    def run(self, parent, document):
        found = histogram_result(parent, document)
        return None if found is None else found[3]
