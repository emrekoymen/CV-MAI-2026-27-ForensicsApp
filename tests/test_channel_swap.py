import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_swap import SWAPS, ChannelSwapTool, swap_channels
from forensics_app.tools.panels import grid_layout


class ChannelSwapTests(unittest.TestCase):
    def setUp(self):
        # 2x2 image with a different value in every channel of every pixel,
        # so moving any value to the wrong channel is detected.
        self.pixels = np.array(
            [[[10, 20, 30], [40, 50, 60]],
             [[70, 80, 90], [100, 110, 120]]],
            dtype=np.uint8,
        )

    def test_each_swap_exchanges_the_right_channels(self):
        # Expected new channel order: e.g. Red-Green swap -> (G, R, B)
        expected_order = {1: (1, 0, 2), 2: (2, 1, 0), 3: (0, 2, 1)}
        for swap_type, order in expected_order.items():
            with self.subTest(swap=SWAPS[swap_type][0]):
                result = swap_channels(self.pixels, swap_type)
                np.testing.assert_array_equal(result, self.pixels[:, :, list(order)])

    def test_pure_red_becomes_the_swapped_colour(self):
        # A readable version of the test above: red moves to green or blue.
        red = Image.new("RGB", (1, 1), "red")
        self.assertEqual(tuple(swap_channels(red, 1)[0, 0]), (0, 255, 0))
        self.assertEqual(tuple(swap_channels(red, 2)[0, 0]), (0, 0, 255))
        self.assertEqual(tuple(swap_channels(red, 3)[0, 0]), (255, 0, 0))  # red untouched

    def test_swapping_twice_gives_back_the_original(self):
        for swap_type in SWAPS:
            with self.subTest(swap=SWAPS[swap_type][0]):
                twice = swap_channels(swap_channels(self.pixels, swap_type), swap_type)
                np.testing.assert_array_equal(twice, self.pixels)

    def test_input_is_not_modified(self):
        original = self.pixels.copy()
        swap_channels(self.pixels, 1)
        np.testing.assert_array_equal(self.pixels, original)

    def test_alpha_channel_is_kept(self):
        alpha = np.full((2, 2), 128, dtype=np.uint8)
        rgba = np.dstack([self.pixels, alpha])
        result = swap_channels(Image.fromarray(rgba, "RGBA"), 2)
        np.testing.assert_array_equal(result[:, :, 3], alpha)
        np.testing.assert_array_equal(result[:, :, :3], self.pixels[:, :, [2, 1, 0]])

    def test_grayscale_image_gives_clear_error(self):
        with self.assertRaisesRegex(ValueError, "'L' and has no RGB channels"):
            swap_channels(Image.new("L", (4, 4), 100), 1)

    def test_grayscale_array_gives_clear_error(self):
        with self.assertRaisesRegex(ValueError, r"'array \(4, 4\)' and has no RGB channels"):
            swap_channels(np.zeros((4, 4), dtype=np.uint8), 1)

    def test_unknown_swap_type_gives_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown swap type 4"):
            swap_channels(self.pixels, 4)

    def test_tool_returns_side_by_side_panels_without_changing_document(self):
        document = ImageDocument()
        document.current = Image.fromarray(self.pixels, "RGB")
        with patch.object(ChannelSwapTool, "choose_swap_type", return_value=2):
            result = ChannelSwapTool().run(None, document)
        size, origins = grid_layout(2, 2, count=2, columns=2)
        grid = np.array(result.image)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, size)
        # Full resolution, no resampling: original on the left, swapped on the right
        (x0, y0), (x1, y1) = origins
        np.testing.assert_array_equal(grid[y0:y0 + 2, x0:x0 + 2], self.pixels)
        np.testing.assert_array_equal(grid[y1:y1 + 2, x1:x1 + 2], self.pixels[:, :, [2, 1, 0]])
        self.assertEqual(result.details["Type"], "Red-Blue")
        np.testing.assert_array_equal(np.array(document.current), self.pixels)
        self.assertFalse(document.can_undo)

    def test_tool_on_grayscale_fails_before_opening_the_dialog(self):
        document = ImageDocument()
        document.current = Image.new("L", (4, 4), 100)
        with patch.object(ChannelSwapTool, "choose_swap_type", return_value=1) as dialog, \
                self.assertRaisesRegex(ValueError, "'L'"):
            ChannelSwapTool().run(None, document)
        dialog.assert_not_called()
        self.assertEqual(document.current.mode, "L")

    def test_closing_the_dialog_cancels(self):
        document = ImageDocument()
        document.current = Image.fromarray(self.pixels, "RGB")
        with patch.object(ChannelSwapTool, "choose_swap_type", return_value=0):
            self.assertIsNone(ChannelSwapTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
