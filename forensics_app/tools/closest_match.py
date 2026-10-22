from tkinter import messagebox, simpledialog, filedialog
from pathlib import Path

import numpy as np
from PIL import Image

from .base import ForensicsTool, ToolResult
from .descriptor_matching import descriptor_sift, descriptor_orb, descriptor_match, draw_matches

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MATCH_WIDTH = 800   # every image is resized to this width, so the same object appears at similar scales

def prepare(image):
    """Resize to MATCH_WIDTH (keeping proportions) and return it with its grayscale array."""
    image = image.convert("RGB")
    image = image.resize((MATCH_WIDTH, round(image.height * MATCH_WIDTH / image.width)))
    return image, np.array(image.convert("L"), dtype=float) / 255

def best_match_in_folder(image,query_path,folder_path=None):
    """Match the image at ``query_path`` against every other image in ``folder``."""
    query_path = Path(query_path).resolve()
    if folder_path is None:
        folder_path = query_path.parent

    candidates = [p for p in Path(folder_path).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.resolve() != query_path]

    query, query_array = prepare(image)
    keypoints_q, descriptors_q = descriptor_sift(query_array)
    # keypoints_q, descriptors_q = descriptor_orb(query_array, n_keypoints = 1000)

    best_score = -1
    for path in candidates:
        candidate, candidate_array = prepare(Image.open(path))
        try:
            keypoints_c, descriptors_c = descriptor_sift(candidate_array)
            # keypoints_c, descriptors_c = descriptor_orb(candidate_array)
        except RuntimeError:  # too few features in this image; skip it
            continue
        matches = descriptor_match(descriptors_q, descriptors_c,max_ratio=0.8)
        score = len(matches)
        if score > best_score:
            best_score, best_path, best_candidate = score, path, candidate
            best_keypoints, best_matches = keypoints_c, matches
    if best_score < 0:
        return None
    result = draw_matches(query.convert("L"), best_candidate.convert("L"),
        keypoints_q, best_keypoints, best_matches, labels=(query_path.name, best_path.name))
    return result, best_path, best_score



class ClosestMatchTool(ForensicsTool):
    tool_id = "closest_match"
    title = "Closest Match"
    category = "Detection and matching"
    description = "Determine the figure in a folder with the highest number of matched keypoints to the loaded image."


    def run(self, parent, document):

        folder = filedialog.askdirectory(title="Select folder with images to compare", parent=parent,
            initialdir=document.path.parent)
        if not folder:
            return None
        found = best_match_in_folder(document.current,document.path,folder_path=folder)
        if found is None:
            messagebox.showinfo("No match", "No other image in that folder could be matched.", parent=parent)
            return None
        result, path, score = found


        return ToolResult(
            image=result,
            message=f"Closest match: {path.name} with {score} matches",
            details={"Operation": "Closest Match", "Best match": path.name, "Matches": score},
        )
