from tkinter import filedialog, messagebox
from pathlib import Path
import hashlib

import joblib
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from .base import ForensicsTool, ToolResult
from .closest_match import IMAGE_EXTENSIONS
from .descriptor_matching import descriptor_sift
from .panels import make_grid

N_CLUSTERS = 30
TRAIN_FRACTION = 0.7
PATCH_SIZE = 32       # side of the square patch cropped around each visual word
PANEL_SIZE = 128      # patches are upscaled to this size so they are visible in the grid
DEFAULT_FOLDER = Path(__file__).resolve().parents[2] / "Caltech_101_subset"
CODEBOOK_FILE = "codebook.joblib"   # saved inside the dataset folder

def split_dataset(folder, train_fraction=TRAIN_FRACTION, seed=0):
    """Shuffle the images of every class subfolder and split them into train and test lists of (path, class).

    The first image of each class (by name) always goes to test, so there is a known image to showcase the app.
    """
    rng = np.random.default_rng(seed)
    train, test = [], []
    for class_folder in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        paths = sorted(p for p in class_folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            continue
        first, rest = paths[0], paths[1:]
        rest = [rest[i] for i in rng.permutation(len(rest))]
        n_train = round(len(paths) * train_fraction)
        train += [(p, class_folder.name) for p in rest[:n_train]]
        test += [(p, class_folder.name) for p in [first] + rest[n_train:]]
    return train, test

def file_hash(path):
    """Hash of the file's bytes, so a training image is recognised even if renamed or copied elsewhere."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def extract_all(paths):
    """SIFT descriptors of every image, plus which image and keypoint each descriptor came from."""
    descriptors, sources = [], []
    for path in paths:
        image = np.array(Image.open(path).convert("L"), dtype=float) / 255
        try:
            keypoints, image_descriptors = descriptor_sift(image)
        except RuntimeError:  # too few features in this image; skip it
            continue
        descriptors.append(image_descriptors)
        sources += [(path, row, col) for row, col in keypoints]
    return np.vstack(descriptors).astype(float), sources

def build_codebook(descriptors, n_clusters=N_CLUSTERS):
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=3).fit(descriptors)

def crop_patch(path, row, col, size=PATCH_SIZE):
    """Square patch centred on a keypoint; the image is padded so patches near the border keep their size."""
    image = np.array(Image.open(path).convert("L"))
    half = size // 2
    padded = np.pad(image, half, mode="reflect")
    row, col = int(round(row)) + half, int(round(col)) + half
    return padded[row - half:row + half, col - half:col + half]

def visual_words(kmeans, descriptors, sources):
    """For every centroid, the patch of the descriptor closest to it."""
    distances = kmeans.transform(descriptors)   # distance from every descriptor to every centroid
    closest = distances.argmin(axis=0)
    panels = []
    for word, index in enumerate(closest):
        patch = Image.fromarray(crop_patch(*sources[index])).resize((PANEL_SIZE, PANEL_SIZE), Image.NEAREST)
        panels.append((f"Word {word + 1}", np.array(patch)))
    return panels


class KMeansTool(ForensicsTool):
    tool_id = "k-means"
    title = "K-means clustering"
    category = "Bag-of-words"
    description = "Cluster the SIFT descriptors of the training images with k-means and show the visual codebook."

    def run(self, parent, document):
        folder = filedialog.askdirectory(title="Select dataset folder (one subfolder per class)", parent=parent,
            initialdir=DEFAULT_FOLDER)
        if not folder:
            return None
        codebook_path = Path(folder) / CODEBOOK_FILE

        if codebook_path.exists():
            saved = joblib.load(codebook_path)
            message = f"Loaded the saved codebook of {N_CLUSTERS} visual words from {codebook_path.name}"
        else:
            train, test = split_dataset(folder)
            if not train:
                messagebox.showinfo("No images", "No class subfolders with images were found.", parent=parent)
                return None
            descriptors, sources = extract_all([path for path, _ in train])
            kmeans = build_codebook(descriptors)
            # Hashes of the train and test files, so the histogram tool can refuse training images
            split = {file_hash(path): "Train" for path, _ in train}
            split.update({file_hash(path): "Test" for path, _ in test})
            # The patches are saved too, since finding them again would need all the descriptors
            saved = {"kmeans": kmeans, "split": split, "panels": visual_words(kmeans, descriptors, sources),
                "details": {"Operation": "K-means clustering", "Descriptor": "SIFT", "Clusters": N_CLUSTERS,
                    "Train images": len(train), "Test images": len(test), "Descriptors": len(descriptors)}}
            joblib.dump(saved, codebook_path)
            message = f"Built a codebook of {N_CLUSTERS} visual words from {len(train)} training images"

        return ToolResult(
            image=make_grid(saved["panels"], columns=6),
            message=message,
            details={**saved["details"], "Codebook file": str(codebook_path)},
        )
