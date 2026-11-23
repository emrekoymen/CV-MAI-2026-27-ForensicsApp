"""Set 8.3: project saved CNN feature vectors onto 2D with t-SNE and UMAP and plot them by class.

The features of the CIFAR-10 test set are extracted in notebooks/set8_train_models.ipynb and saved as
``<model>.test_features.npz`` (features, labels, predictions, class_names, model_name, layer_name).
"""

import numpy as np
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from tkinter import simpledialog

from .base import ForensicsTool, ToolResult
from .cnn_classification import choose_file, figure_to_image

SEED = 42


def subsample(n_total, n, seed=SEED):
    """Sorted indices of n random points (all of them if n >= n_total)."""
    if n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def project(features, seed=SEED):
    import umap  # slow to import (numba), so only when needed
    return {
        "t-SNE": TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(features),
        "UMAP": umap.UMAP(n_components=2, n_neighbors=15, random_state=seed).fit_transform(features),
    }


def projection_figure(projections, labels, class_names, title):
    """One scatter plot per method, side by side, coloured by class."""
    fig = Figure(figsize=(16, 7), layout="constrained")
    for ax, (method, xy) in zip(fig.subplots(1, len(projections)), projections.items()):
        for cl, name in enumerate(class_names):
            ax.scatter(*xy[labels == cl].T, s=4, label=name)
        ax.set(title=method, xlabel=f"{method} 1", ylabel=f"{method} 2", xticks=[], yticks=[])
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper", markerscale=4, frameon=False)
    fig.suptitle(title)
    return figure_to_image(fig)


class FeatureProjectionTool(ForensicsTool):
    tool_id = "feature_projection"
    title = "Feature projection (t-SNE, UMAP)"
    category = "Deep learning"
    description = "Project saved CNN feature vectors of the test set to 2D with t-SNE and UMAP, coloured by class."
    requires_image = False

    def run(self, parent, document):
        path = choose_file(parent, "Feature vectors", "*.test_features.npz")
        if path is None:
            return None
        n = simpledialog.askinteger("Feature projection", "Points to project (10,000 takes about a minute):",
                                    initialvalue=3000, minvalue=100, parent=parent)
        if n is None:
            return None
        data = np.load(path)
        keep = subsample(len(data["labels"]), n)
        features, labels = data["features"][keep], data["labels"][keep]
        title = f"{data['model_name']}: '{data['layer_name']}' layer ({features.shape[1]}-d), {len(keep)} test images"
        return ToolResult(
            image=projection_figure(project(features), labels, list(data["class_names"]), title),
            message=f"Projected {len(keep)} feature vectors with t-SNE and UMAP",
            details={"Operation": "Feature projection", "Model": str(data["model_name"]),
                     "Layer": f"{data['layer_name']} ({features.shape[1]}-d)", "Points": f"{len(keep)} of {len(data['labels'])}"},
        )
