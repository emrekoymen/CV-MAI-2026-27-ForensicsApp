import unittest

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
from forensics_app.tools.panels import grid_layout


class ChannelSplitTests(unittest.TestCase):
    def setUp(self):
        # 2x2 image with a different value in every channel of every pixel,
        # so swapping or mixing up any two channels is detected.
        self.pixels = np.array(
            [[[10, 20, 30], [40, 50, 60]],
             [[70, 80, 90], [100, 110, 120]]],
            dtype=np.uint8,
        )

    def test_each_channel_is_extracted(self):
        r, g, b = split_channels(self.pixels)
        for index, channel in enumerate((r, g, b)):
            with self.subTest(channel="RGB"[index]):
                np.testing.assert_array_equal(channel, self.pixels[:, :, index])

    def test_pil_image_gives_same_channels_as_array(self):
        from_image = split_channels(Image.fromarray(self.pixels, "RGB"))
        from_array = split_channels(self.pixels)
        for image_channel, array_channel in zip(from_image, from_array):
            np.testing.assert_array_equal(image_channel, array_channel)

    def test_pure_colours_light_up_only_their_own_channel(self):
        for index, colour in enumerate(("red", "lime", "blue")):
            with self.subTest(colour=colour):
                channels = split_channels(Image.new("RGB", (3, 3), colour))
                for channel_index, channel in enumerate(channels):
                    expected = 255 if channel_index == index else 0
                    self.assertTrue((channel == expected).all())

    def test_alpha_channel_is_ignored(self):
        rgba = np.dstack([self.pixels, np.full((2, 2), 128, dtype=np.uint8)])
        r, g, b = split_channels(Image.fromarray(rgba, "RGBA"))
        for index, channel in enumerate((r, g, b)):
            np.testing.assert_array_equal(channel, self.pixels[:, :, index])

    def test_single_pixel_image(self):
        r, g, b = split_channels(Image.new("RGB", (1, 1), (1, 2, 3)))
        self.assertEqual([r.item(), g.item(), b.item()], [1, 2, 3])

    def test_grayscale_image_gives_clear_error(self):
        with self.assertRaisesRegex(ValueError, "'L' and has no RGB channels"):
            split_channels(Image.new("L", (4, 4), 100))

    def test_grayscale_array_gives_clear_error(self):
        with self.assertRaisesRegex(ValueError, r"'array \(4, 4\)' and has no RGB channels"):
            split_channels(np.zeros((4, 4), dtype=np.uint8))

    def test_tool_returns_2x2_grid_without_changing_document(self):
        document = ImageDocument()
        document.current = Image.fromarray(self.pixels, "RGB")
        result = ChannelSplitTool().run(None, document)
        size, origins = grid_layout(2, 2, count=4, columns=2)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, size)
        np.testing.assert_array_equal(np.array(document.current), self.pixels)
        self.assertFalse(document.can_undo)

    def test_grid_panels_hold_the_exact_pixels(self):
        # Full resolution, no resampling: Original, Red / Green, Blue, each pixel for pixel
        document = ImageDocument()
        document.current = Image.fromarray(self.pixels, "RGB")
        grid = np.array(ChannelSplitTool().run(None, document).image)
        _, origins = grid_layout(2, 2, count=4, columns=2)
        expected = [self.pixels] + [np.dstack([self.pixels[:, :, i]] * 3) for i in range(3)]
        for (x, y), panel, name in zip(origins, expected, ("Original", "Red", "Green", "Blue")):
            with self.subTest(panel=name):
                np.testing.assert_array_equal(grid[y:y + 2, x:x + 2], panel)

    def test_rgba_original_panel_drops_alpha(self):
        rgba = np.dstack([self.pixels, np.zeros((2, 2), dtype=np.uint8)])  # fully transparent
        document = ImageDocument()
        document.current = Image.fromarray(rgba, "RGBA")
        grid = np.array(ChannelSplitTool().run(None, document).image)
        (x, y), *_ = grid_layout(2, 2, count=4, columns=2)[1]
        np.testing.assert_array_equal(grid[y:y + 2, x:x + 2], self.pixels)

    def test_tool_on_grayscale_leaves_document_unchanged(self):
        document = ImageDocument()
        document.current = Image.new("L", (4, 4), 100)
        with self.assertRaises(ValueError):
            ChannelSplitTool().run(None, document)
        self.assertEqual(document.current.mode, "L")


if __name__ == "__main__":
    unittest.main()
