import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.canny import CannyTool, compute_edges


class CannyTests(unittest.TestCase):
    def test_constant_image_has_no_edges(self):
        edges = compute_edges(Image.new("L", (32, 32), 80), 1, 0.1, 0.2)
        self.assertFalse(edges.any())

    def test_square_edges_stay_near_its_boundary(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        image[8:24, 8:24] = 255
        edges = compute_edges(image, 1, 0.1, 0.2)
        self.assertTrue(edges[7:10, 12:20].any())
        self.assertTrue(edges[22:25, 12:20].any())
        self.assertFalse(edges[12:20, 12:20].any())

    def test_lower_thresholds_detect_faint_edges(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        image[8:24, 8:24] = 10
        self.assertFalse(compute_edges(image, 1, 0.1, 0.2).any())
        self.assertTrue(compute_edges(image, 1, 0.005, 0.01).any())

    def test_tool_returns_binary_image_without_changing_document(self):
        document = ImageDocument()
        document.current = Image.new("RGB", (32, 32), "red")
        original = np.array(document.current)
        with patch.object(CannyTool, "choose_settings", return_value=(1, 0.1, 0.2)):
            result = CannyTool().run(None, document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, document.current.size)
        self.assertTrue(set(np.unique(result.image)).issubset({0, 255}))
        np.testing.assert_array_equal(np.array(document.current), original)

    def test_cancel_returns_without_processing(self):
        with patch.object(CannyTool, "choose_settings", return_value=None):
            self.assertIsNone(CannyTool().run(None, ImageDocument()))


if __name__ == "__main__":
    unittest.main()
