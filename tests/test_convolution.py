import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.convolution import ConvolutionTool, build_kernel, compute_convolution


class ConvolutionTests(unittest.TestCase):
    def test_direction_controls_where_an_impulse_spreads(self):
        image = np.zeros((5, 5))
        image[2, 2] = 9
        for direction in ("Horizontal", "Vertical", "Both"):
            with self.subTest(direction=direction):
                result = compute_convolution(image, build_kernel(3, direction))
                expected = np.zeros((5, 5))
                if direction == "Horizontal":
                    expected[2, 1:4] = 3
                elif direction == "Vertical":
                    expected[1:4, 2] = 3
                else:
                    expected[1:4, 1:4] = 1
                np.testing.assert_allclose(result, expected)

    def test_average_keeps_constant_brightness_including_borders(self):
        image = np.full((2, 2), 80)
        result = compute_convolution(image, build_kernel(5, "Both"))
        np.testing.assert_allclose(result, image)

    def test_size_one_leaves_image_unchanged(self):
        image = np.array([[10, 20], [30, 40]])
        kernel = build_kernel(1, "Both")
        np.testing.assert_allclose(compute_convolution(image, kernel), image)

    def test_even_sizes_are_allowed(self):
        kernel = build_kernel(4, "Both")
        self.assertEqual(kernel.shape, (4, 4))
        self.assertAlmostEqual(kernel.sum(), 1.0)

    def test_tool_returns_grayscale_without_changing_document(self):
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        original = np.array(document.current)
        with patch.object(ConvolutionTool, "choose_direction", return_value="Both"), \
                patch("forensics_app.tools.convolution.simpledialog.askinteger", return_value=3):
            result = ConvolutionTool().run(None, document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, document.current.size)
        np.testing.assert_array_equal(np.array(document.current), original)
        self.assertFalse(document.can_undo)

    def test_closing_a_dialog_cancels(self):
        with patch.object(ConvolutionTool, "choose_direction", return_value=None):
            self.assertIsNone(ConvolutionTool().run(None, ImageDocument()))
        with patch.object(ConvolutionTool, "choose_direction", return_value="Both"), \
                patch("forensics_app.tools.convolution.simpledialog.askinteger", return_value=None):
            self.assertIsNone(ConvolutionTool().run(None, ImageDocument()))


if __name__ == "__main__":
    unittest.main()
