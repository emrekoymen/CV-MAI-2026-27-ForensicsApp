import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools.blood_cell_detection import (CLASSES, COLOURS, MODEL_PATH, BloodCellDetectionTool, iou,
    label_path, match, read_labels)
from forensics_app.tools.yolo_detection import COCO_MODEL, detect, draw_boxes, load_model

HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None
TEST_IMAGES = Path(__file__).resolve().parents[1] / "datasets" / "blood_cells" / "test" / "images"


class LabelTests(unittest.TestCase):
    def test_label_path_swaps_images_for_labels(self):
        path = Path("data") / "test" / "images" / "cell.jpg"
        self.assertEqual(label_path(path), Path("data") / "test" / "labels" / "cell.txt")

    def test_normalised_labels_become_pixel_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cell.txt"
            path.write_text("2 0.5 0.25 0.2 0.1\n0 0.1 0.1 0.2 0.2\n")
            boxes = read_labels(path, width=200, height=400)
        self.assertEqual([name for name, _ in boxes], ["WBC", "PC"])
        for value, expected in zip(boxes[0][1], (80, 80, 120, 120)):
            self.assertAlmostEqual(value, expected)


class MatchingTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)

    def test_overlapping_box_of_the_same_class_matches(self):
        self.assertEqual(match([("RBC", 0.9, (1, 1, 11, 11))], [("RBC", (0, 0, 10, 10))]), 1)

    def test_box_of_another_class_does_not_match(self):
        self.assertEqual(match([("WBC", 0.9, (0, 0, 10, 10))], [("RBC", (0, 0, 10, 10))]), 0)

    def test_two_predictions_cannot_match_the_same_cell(self):
        predictions = [("RBC", 0.9, (0, 0, 10, 10)), ("RBC", 0.8, (1, 1, 11, 11))]
        self.assertEqual(match(predictions, [("RBC", (0, 0, 10, 10))]), 1)


class DrawingTests(unittest.TestCase):
    def test_boxes_are_drawn_in_their_colour_without_resizing(self):
        image = Image.new("RGB", (200, 100), "black")
        output = draw_boxes(image, [("", "RBC", (10, 10, 50, 50))], COLOURS)
        self.assertEqual(output.size, image.size)
        self.assertEqual(output.getpixel((10, 30)), COLOURS["RBC"])
        self.assertEqual(image.getpixel((10, 30)), (0, 0, 0))


@unittest.skipUnless(HAS_ULTRALYTICS and COCO_MODEL.exists(), "ultralytics or models/yolo11n.pt missing")
class CocoModelTests(unittest.TestCase):
    def test_finds_the_astronaut(self):
        detections, _ = detect(load_model(COCO_MODEL), Image.fromarray(data.astronaut()))
        self.assertIn("person", [name for name, _, _ in detections])


@unittest.skipUnless(HAS_ULTRALYTICS and MODEL_PATH.exists() and TEST_IMAGES.is_dir(),
                     "ultralytics, models/blood_cells_yolo11n.pt or datasets/blood_cells/test missing")
class BloodCellModelTests(unittest.TestCase):
    def test_model_has_the_dataset_classes(self):
        self.assertEqual(list(load_model(MODEL_PATH).names.values()), CLASSES)

    def test_tool_compares_prediction_with_ground_truth(self):
        document = ImageDocument()
        document.load(sorted(TEST_IMAGES.glob("*.jpg"))[0])
        result = BloodCellDetectionTool().run(None, document)
        self.assertGreater(result.image.width, document.current.width)   # two panels side by side
        self.assertGreater(result.details["Precision"], 0.8)
        self.assertGreater(result.details["Recall"], 0.8)


class MissingModelTests(unittest.TestCase):
    def test_missing_weights_name_the_expected_path(self):
        missing = Path(tempfile.gettempdir()) / "no_such_model.pt"
        with self.assertRaises(FileNotFoundError) as raised:
            load_model(missing, "copy it into models/")
        self.assertIn(str(missing), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
