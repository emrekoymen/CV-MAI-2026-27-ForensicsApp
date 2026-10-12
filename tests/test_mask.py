import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.mask import MaskTool, apply_mask


class ApplyMaskTests(unittest.TestCase):
    def setUp(self):
        # 4x4 image with a different colour in every pixel, so any pixel that
        # is changed by mistake (or left alone by mistake) is detected.
        self.pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3) * 5
        self.image = Image.fromarray(self.pixels, "RGB")
        # Grayscale mask: right half white (masked), left half black
        self.half_mask = np.zeros((4, 4), dtype=np.uint8)
        self.half_mask[:, 2:] = 255

    def test_black_mask_changes_nothing(self):
        result = apply_mask(self.image, Image.new("L", (4, 4), 0))
        np.testing.assert_array_equal(result, self.pixels)

    def test_white_mask_replaces_everything(self):
        result = apply_mask(self.image, Image.new("L", (4, 4), 255))
        self.assertTrue((result == 255).all())

    def test_half_mask_only_changes_the_masked_half(self):
        result = apply_mask(self.image, self.half_mask)
        np.testing.assert_array_equal(result[:, :2], self.pixels[:, :2])
        self.assertTrue((result[:, 2:] == 255).all())

    def test_coloured_mask_replaces_whole_pixel(self):
        # Per-pixel masking: a pure red mask pixel replaces all three channels,
        # it does not leave the image's green and blue showing through.
        result = apply_mask(Image.new("RGB", (1, 1), (10, 50, 90)), Image.new("RGB", (1, 1), "red"))
        self.assertEqual(tuple(result[0, 0]), (255, 0, 0))

    def test_any_non_black_mask_pixel_counts_as_masked(self):
        # Only pure black (0) is ignored; even the darkest grey (1) counts as masked.
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:, 2:] = 1
        result = apply_mask(self.image, mask)
        np.testing.assert_array_equal(result[:, :2], self.pixels[:, :2])
        self.assertTrue((result[:, 2:] == 1).all())

    def test_texture_is_pasted_where_masked(self):
        texture = Image.new("RGB", (4, 4), (0, 200, 0))
        result = apply_mask(self.image, self.half_mask, texture)
        np.testing.assert_array_equal(result[:, :2], self.pixels[:, :2])
        self.assertTrue((result[:, 2:] == (0, 200, 0)).all())

    def test_smaller_mask_and_texture_are_resized(self):
        small_mask = np.zeros((2, 2), dtype=np.uint8)
        small_mask[:, 1] = 255  # right half
        texture = Image.new("RGB", (8, 8), (0, 0, 200))
        result = apply_mask(self.image, small_mask, texture)
        self.assertEqual(result.shape, (4, 4, 3))
        np.testing.assert_array_equal(result[:, :2], self.pixels[:, :2])
        self.assertTrue((result[:, 2:] == (0, 0, 200)).all())  # sharp edge, nothing blended

    def test_one_bit_mask_works(self):
        result = apply_mask(self.image, Image.new("1", (4, 4), 1))
        self.assertTrue((result == 255).all())

    def test_transparent_mask_areas_are_not_masked(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[..., :3] = 255         # white everywhere...
        mask[:, 2:, 3] = 255        # ...but only the right half is visible
        result = apply_mask(self.image, Image.fromarray(mask, "RGBA"))
        np.testing.assert_array_equal(result[:, :2], self.pixels[:, :2])
        self.assertTrue((result[:, 2:] == 255).all())

    def test_grayscale_image_with_coloured_texture_gives_colour_result(self):
        image = Image.new("L", (4, 4), 100)
        texture = Image.new("RGB", (4, 4), (255, 0, 0))
        result = apply_mask(image, self.half_mask, texture)
        self.assertEqual(result.shape, (4, 4, 3))
        self.assertTrue((result[:, :2] == 100).all())
        self.assertTrue((result[:, 2:] == (255, 0, 0)).all())

    def test_input_image_is_not_modified(self):
        original = self.pixels.copy()
        apply_mask(self.pixels, Image.new("L", (4, 4), 255))
        np.testing.assert_array_equal(self.pixels, original)


class MaskToolTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (4, 3), (10, 20, 30))

    def run_tool(self, mask, texture=None):
        with patch.object(MaskTool, "choose_mask", return_value=mask), \
                patch.object(MaskTool, "choose_texture", return_value=texture):
            return MaskTool().run(None, self.document)

    def test_tool_returns_rgb_image_without_changing_document(self):
        result = self.run_tool(Image.new("L", (4, 3), 255))
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, (4, 3))
        self.assertEqual(self.document.current.getpixel((0, 0)), (10, 20, 30))
        self.assertFalse(self.document.can_undo)

    def test_cancelling_the_mask_file_cancels(self):
        with patch.object(MaskTool, "choose_mask", return_value=None), \
                patch.object(MaskTool, "choose_texture") as texture_dialog:
            self.assertIsNone(MaskTool().run(None, self.document))
        texture_dialog.assert_not_called()

    def test_answering_no_to_texture_uses_plain_mask(self):
        result = self.run_tool(Image.new("L", (4, 3), 255), texture=None)
        self.assertEqual(result.details["Texture"], "No")
        self.assertEqual(result.image.getpixel((0, 0)), (255, 255, 255))

    def test_same_size_inputs_report_no_resizing(self):
        result = self.run_tool(Image.new("L", (4, 3), 255), Image.new("RGB", (4, 3), "red"))
        self.assertEqual(result.details["Texture"], "Yes")
        self.assertNotIn("Mask resized", result.details)
        self.assertNotIn("Texture resized", result.details)

    def test_resized_mask_and_texture_are_reported(self):
        result = self.run_tool(Image.new("L", (2, 2), 255), Image.new("RGB", (8, 6), "red"))
        self.assertEqual(result.details["Mask resized"], "2 × 2 → 4 × 3 px")
        self.assertEqual(result.details["Texture resized"], "8 × 6 → 4 × 3 px")


if __name__ == "__main__":
    unittest.main()
