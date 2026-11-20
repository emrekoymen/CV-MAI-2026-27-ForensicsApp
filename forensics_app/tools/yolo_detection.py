"""Set 9.1: object detection on the loaded image with a COCO-pretrained Ultralytics YOLO11 model."""

from collections import Counter
from pathlib import Path
import time

import matplotlib
from PIL import ImageDraw, ImageFont

from .base import ForensicsTool, ToolResult

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"   # not committed (models/ is ignored)
COCO_MODEL = MODELS_DIR / "yolo11n.pt"
COLOURS = matplotlib.colormaps.get_cmap("tab20")
_models = {}   # loaded models by path, so running a tool again does not reload the weights


def load_model(path, hint="copy it from the Set 9 materials"):
    """Load YOLO weights once; ultralytics is imported here so the rest of the app works without it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO model not found: {path}\nTo fix it, {hint}.")
    if path not in _models:
        from ultralytics import YOLO
        _models[path] = YOLO(str(path))
    return _models[path]


def detect(model, image, conf=0.25):
    """Return one (class name, confidence, (x1, y1, x2, y2)) per box, and the inference time in seconds.

    A PIL image is passed, which Ultralytics reads as RGB (a numpy array would be taken as BGR).
    """
    start = time.perf_counter()
    result = model.predict(image.convert("RGB"), conf=conf, verbose=False)[0]
    seconds = time.perf_counter() - start
    detections = [(result.names[int(c)], float(p), tuple(float(v) for v in box))
                  for c, p, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)]
    return detections, seconds


def class_colour(class_id):
    """RGB colour of a class from the tab20 colormap (as in the slides)."""
    return tuple(int(255 * c) for c in COLOURS(class_id % 20)[:3])


def draw_boxes(image, boxes, colours, width=None):
    """Draw (label, colour key, box) on a copy of the image, with the label on a filled tag above the box.

    Drawing in RGB avoids the swapped colours of plt.imshow(results[0].plot()), whose image is BGR.
    """
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    width = width or max(2, output.width // 300)
    font = ImageFont.load_default(size=max(12, output.width // 50))
    for label, key, (x1, y1, x2, y2) in boxes:
        colour = colours[key]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=width)
        if label:
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            top_of_tag = max(0, y1 - (bottom - top) - 4)
            draw.rectangle((x1, top_of_tag, x1 + right - left + 4, top_of_tag + bottom - top + 4), fill=colour)
            draw.text((x1 + 2, top_of_tag + 2 - (top - y1)), label, fill="white", font=font)
    return output


def count_classes(detections):
    """'2 bus, 4 person' style summary of the detections."""
    counts = Counter(name for name, _, _ in detections)
    return ", ".join(f"{n} {name}" for name, n in sorted(counts.items())) or "none"


class YoloDetectionTool(ForensicsTool):
    tool_id = "yolo_detection"
    title = "Object detection (YOLO11)"
    category = "Object detection"
    description = "Detect the 80 COCO object classes with a pretrained Ultralytics YOLO11n model."

    def run(self, parent, document):
        model = load_model(COCO_MODEL, "copy yolo11n.pt from the Set 9 materials into models/")
        detections, seconds = detect(model, document.current)
        class_ids = {name: i for i, name in model.names.items()}
        colours = {name: class_colour(class_ids[name]) for name, _, _ in detections}
        image = draw_boxes(document.current, [(f"{name} {p:.2f}", name, box) for name, p, box in detections], colours)
        return ToolResult(
            image=image,
            message=f"Detected {len(detections)} object(s): {count_classes(detections)}.",
            details={"Operation": "Object detection (YOLO11n, COCO)", "Objects": len(detections),
                     "Classes": count_classes(detections),
                     "Confidences": ", ".join(f"{p:.2f}" for _, p, _ in detections) or "—",
                     "Inference time": f"{1000 * seconds:.0f} ms"},
        )
