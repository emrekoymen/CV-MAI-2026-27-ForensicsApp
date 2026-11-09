"""Set 6.3: recognise the loaded (test) face by its nearest neighbours in eigenface space."""

from pathlib import Path
from tkinter import filedialog

import numpy as np
from PIL import Image

from .base import ForensicsTool, ToolResult
from .eigenfaces import MODEL_FILE, to_panel
from .panels import make_grid

N_CLOSEST = 3


def closest_faces(x, model, n=N_CLOSEST):
    """w_test = U_K^T (x - mu); return the indices and distances d_i = ||w_test - w_i|| of the n closest."""
    w_test = model["eigenfaces"] @ (x - model["mean"])
    distances = np.linalg.norm(model["coefficients"] - w_test, axis=1)
    closest = np.argsort(distances)[:n]
    return closest, distances[closest]


def find_model(image_path):
    """The model is saved next to train/ and test/, so look in the folders above the image."""
    for folder in Path(image_path).parents:
        if (folder / MODEL_FILE).exists():
            return folder / MODEL_FILE
    return None


class FaceRecognitionTool(ForensicsTool):
    tool_id = "face_recognition"
    title = "Face recognition (eigenfaces)"
    category = "Faces"
    description = "Show the training faces closest to the loaded face in eigenface space."

    def run(self, parent, document):
        model_path = find_model(document.path) if document.path else None
        model_path = model_path or filedialog.askopenfilename(
            title="Choose eigenface model", parent=parent, filetypes=[("Eigenface model", "*.npz")])
        if not model_path:
            return None
        model = np.load(model_path)
        height, width = model["shape"]

        query = document.current.convert("L").resize((int(width), int(height)))
        x = np.array(query, dtype=float).ravel()
        closest, distances = closest_faces(x, model)
        names = model["names"].tolist()

        true_name = document.path.parent.name.replace("_", " ") if document.path else ""
        label = f"true: {true_name}" if true_name in names else "query"
        panels = [(label, to_panel(x, (height, width)))]
        for rank, (i, distance) in enumerate(zip(closest, distances), start=1):
            train_image = Image.open(Path(model_path).parent / model["paths"][i]).convert("L")
            panels.append((f"{rank}: {names[i]} ({distance:.0f})", to_panel(np.array(train_image, dtype=float), (height, width))))

        return ToolResult(
            image=make_grid(panels, columns=len(panels)),
            message=f"Predicted: {names[closest[0]]}",
            details={"Operation": "Face recognition (eigenfaces)", "Predicted": names[closest[0]],
                     "True": true_name if true_name in names else "—",
                     **{f"Match {r}": f"{names[i]} (d={d:.0f})" for r, (i, d) in enumerate(zip(closest, distances), 1)}},
        )
