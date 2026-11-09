"""Set 6.2: save LFW to train/test folders, build an eigenface (PCA) model and show the first K eigenfaces."""

from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image

from .base import ForensicsTool, ToolResult
from .panels import make_grid

DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "lfw"   # not committed (datasets/ is ignored)
MODEL_FILE = "eigenfaces.npz"
N_COMPONENTS = 100
PANEL_SCALE = 4   # LFW faces are tiny (50 x 37), so enlarge them for display


def export_lfw(folder, min_faces_per_person=70, resize=0.4, test_size=0.25, seed=42):
    """Download LFW and save it as ``folder/{train,test}/<Person_Name>/<index>.png``.

    The person folder is the label, so every image keeps its name.
    """
    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split

    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    images = lfw.images if lfw.images.max() > 1 else lfw.images * 255
    train, test = train_test_split(np.arange(len(images)), test_size=test_size, random_state=seed)
    for split, indices in (("train", train), ("test", test)):
        for i in indices:
            person = Path(folder) / split / lfw.target_names[lfw.target[i]].replace(" ", "_")
            person.mkdir(parents=True, exist_ok=True)
            Image.fromarray(images[i].astype(np.uint8)).save(person / f"{i:04d}.png")


def load_faces(folder):
    """Return the flattened images (one per row), their names, relative paths and the image shape."""
    paths = sorted(Path(folder).glob("*/*.png"))
    images = [np.array(Image.open(path).convert("L"), dtype=float) for path in paths]
    names = [path.parent.name.replace("_", " ") for path in paths]
    return np.array([image.ravel() for image in images]), names, paths, images[0].shape


def build_model(X, n_components=N_COMPONENTS):
    """PCA via SVD: mean, eigenfaces (rows of U_K^T) and the coefficients w_i = U_K^T (x_i - mu)."""
    mean = X.mean(axis=0)
    _, singular, vt = np.linalg.svd(X - mean, full_matrices=False)
    eigenfaces = vt[:n_components]
    variance = singular ** 2
    return {
        "mean": mean,
        "eigenfaces": eigenfaces,
        "coefficients": (X - mean) @ eigenfaces.T,
        "explained_variance": variance[:n_components].sum() / variance.sum(),
    }


def to_panel(vector, shape):
    """Stretch a vector to 0-255 and show it as an enlarged grayscale image."""
    pixels = vector.reshape(shape)
    pixels = 255 * (pixels - pixels.min()) / (np.ptp(pixels) or 1)
    image = Image.fromarray(pixels.astype(np.uint8))
    return np.array(image.resize((shape[1] * PANEL_SCALE, shape[0] * PANEL_SCALE), Image.NEAREST))


class EigenfaceBuilderTool(ForensicsTool):
    tool_id = "eigenface_builder"
    title = "Eigenface builder"
    category = "Faces"
    description = "Build an eigenface model from a folder of train/test faces and show the first K eigenfaces."
    requires_image = False

    def run(self, parent, document):
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        folder = filedialog.askdirectory(title="Folder with train/ and test/ faces", parent=parent,
                                         initialdir=DATASET_DIR)
        if not folder:
            return None
        folder = Path(folder)
        if not (folder / "train").is_dir():
            if not messagebox.askyesno("No train/ folder", f"Download LFW and save it to {folder}?", parent=parent):
                return None
            export_lfw(folder)
        k = simpledialog.askinteger("Eigenfaces", "Number of eigenfaces to show (K):", parent=parent,
                                    minvalue=1, maxvalue=N_COMPONENTS, initialvalue=12)
        if k is None:
            return None

        X, names, paths, shape = load_faces(folder / "train")
        model = build_model(X)
        np.savez(folder / MODEL_FILE, **model, shape=shape, names=names,
                 paths=[str(path.relative_to(folder)) for path in paths])

        panels = [(f"eigenface {i}", to_panel(face, shape)) for i, face in enumerate(model["eigenfaces"][:k])]
        return ToolResult(
            image=make_grid(panels, columns=min(k, 4)),
            message=f"Saved eigenface model to {folder / MODEL_FILE}.",
            details={"Operation": "Eigenface builder", "Train images": len(names), "People": len(set(names)),
                     "Components": len(model["eigenfaces"]),
                     "Explained variance": f"{100 * model['explained_variance']:.1f}%", "Shown": k},
        )
