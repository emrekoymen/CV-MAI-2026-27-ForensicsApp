import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools.corner_detection import CornerDectectionTool


def square_image():
    """Black 64x64 image with a white square: its four corners are the only real corners."""
    image = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill=255)
    return image


def triangle_image():
    """Black 64x64 image with a white triangle. FAST (n=12) needs corners sharper than 90 degrees."""
    image = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(image).polygon([(32, 10), (10, 50), (54, 50)], fill=255)
    return image


def run_tool(image, threshold=0.15):
    document = ImageDocument()
    document.current = image
    with patch("tkinter.simpledialog.askfloat", return_value=threshold):
        return CornerDectectionTool().run(None, document), document


class CornerDetectionTests(unittest.TestCase):
    def test_constant_image_has_no_corners(self):
        result, _ = run_tool(Image.new("L", (32, 32), 120))
        self.assertEqual(result.details["FAST corners"], 0)
        self.assertEqual(result.details["Harris corners"], 0)

    def test_harris_finds_the_four_corners_of_a_square(self):
        result, _ = run_tool(square_image())
        self.assertEqual(result.details["Harris corners"], 4)

    def test_fast_finds_the_three_corners_of_a_triangle(self):
        result, _ = run_tool(triangle_image())
        self.assertEqual(result.details["FAST corners"], 3)

    def test_higher_threshold_finds_fewer_corners(self):
        image = Image.fromarray(data.camera())
        low, _ = run_tool(image, threshold=0.05)
        high, _ = run_tool(image, threshold=0.5)
        self.assertLess(high.details["Harris corners"], low.details["Harris corners"])
        self.assertLess(high.details["FAST corners"], low.details["FAST corners"])

    def test_result_shows_two_panels_and_keeps_document(self):
        image = square_image()
        result, document = run_tool(image)
        # Two panels side by side, so the result is at least twice as wide as the image
        self.assertGreaterEqual(result.image.width, 2 * image.width)
        np.testing.assert_array_equal(np.array(document.current), np.array(image))

    def test_cancel_returns_none(self):
        self.assertIsNone(run_tool(square_image(), threshold=None)[0])


if __name__ == "__main__":
    unittest.main()
