"""Set 9.3: detect blood cells with the YOLO11n model trained in the Set 9 notebook, next to the ground truth."""

from collections import Counter
from pathlib import Path

import numpy as np

from .base import ForensicsTool, ToolResult
from .panels import make_grid
from .yolo_detection import MODELS_DIR, detect, draw_boxes, load_model

# Not committed (models/ is ignored): the notebook's runs/detect/<run>/weights/best.pt, renamed
MODEL_PATH = MODELS_DIR / "blood_cells_yolo11n.pt"
CLASSES = ["PC", "RBC", "WBC"]   # same order as names in data.yaml
COLOURS = {"PC": (220, 30, 30), "RBC": (30, 160, 30), "WBC": (30, 60, 220)}   # as in the notebook plots
IOU_THRESHOLD = 0.5


def label_path(image_path):
    """YOLO layout: .../images/x.jpg has its boxes in .../labels/x.txt."""
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_labels(path, width, height):
    """Read 'class x_center y_center w h' lines (normalised to 0-1) as (class name, (x1, y1, x2, y2)) in pixels."""
    boxes = []
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        class_id, xc, yc, w, h = (float(v) for v in line.split())
        boxes.append((CLASSES[int(class_id)], ((xc - w / 2) * width, (yc - h / 2) * height,
                                               (xc + w / 2) * width, (yc + h / 2) * height)))
    return boxes


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(predictions, ground_truth, threshold=IOU_THRESHOLD):
    """Greedy matching, most confident prediction first: it takes the unused ground-truth box of the same class
    with the highest IoU, if that IoU reaches the threshold. Returns the number of true positives."""
    used = np.zeros(len(ground_truth), dtype=bool)
    true_positives = 0
    for name, _, box in sorted(predictions, key=lambda p: -p[1]):
        candidates = [(iou(box, gt_box), j) for j, (gt_name, gt_box) in enumerate(ground_truth)
                      if gt_name == name and not used[j]]
        best, j = max(candidates, default=(0.0, -1))
        if best >= threshold:
            used[j] = True
            true_positives += 1
    return true_positives


def counts(names):
    counter = Counter(names)
    return ", ".join(f"{counter[name]} {name}" for name in CLASSES)


class BloodCellDetectionTool(ForensicsTool):
    tool_id = "blood_cell_detection"
    title = "Blood cell detection (YOLO11)"
    category = "Object detection"
    description = "Detect platelets, red and white blood cells with the trained YOLO11n model, next to the ground truth."

    def run(self, parent, document):
        model = load_model(MODEL_PATH, "copy best.pt from the Set 9 notebook run into models/ as blood_cells_yolo11n.pt")
        image = document.current
        predictions, seconds = detect(model, image)
        predicted = draw_boxes(image, [(f"{name} {p:.2f}", name, box) for name, p, box in predictions], COLOURS)

        labels = label_path(document.path) if document.path else None
        details = {"Operation": "Blood cell detection (YOLO11n)", "Predicted": counts(n for n, _, _ in predictions),
                   "Inference time": f"{1000 * seconds:.0f} ms"}
        if labels is None or not labels.exists():
            details["Ground truth"] = "no labels/ file for this image"
            return ToolResult(image=predicted, message=f"Detected {len(predictions)} cell(s); no ground truth found.",
                              details=details)

        ground_truth = read_labels(labels, image.width, image.height)
        true_positives = match(predictions, ground_truth)
        precision = true_positives / len(predictions) if predictions else 0.0
        recall = true_positives / len(ground_truth) if ground_truth else 0.0
        truth = draw_boxes(image, [(name, name, box) for name, box in ground_truth], COLOURS)
        details.update({"Ground truth": counts(n for n, _ in ground_truth),
                        f"True positives (IoU >= {IOU_THRESHOLD})": true_positives,
                        "Precision": round(precision, 2), "Recall": round(recall, 2)})
        return ToolResult(
            image=make_grid([("Ground truth", np.array(truth)), ("Prediction", np.array(predicted))], columns=2),
            message=f"Detected {len(predictions)} cell(s), {len(ground_truth)} in the ground truth "
                    f"(precision {precision:.2f}, recall {recall:.2f}).",
            details=details,
        )
