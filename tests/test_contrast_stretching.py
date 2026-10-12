import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.contrast_stretching import (
    DEFAULT_PERCENTILES,
    ContrastStretchingTool,
    check_percentiles,
    stretch_contrast,
)
from forensics_app.tools.histogram import compute_histogram, histogram_stats


class StretchContrastTests(unittest.TestCase):
    def setUp(self):
        # Low-contrast grayscale ramp: values 100–150 only
        self.ramp = np.linspace(100, 150, 51).round().astype(np.uint8).reshape(3, 17)

    def test_low_contrast_image_is_stretched_to_full_range(self):
        result = stretch_contrast(self.ramp)
        self.assertEqual(result.pixels.min(), 0)
        self.assertEqual(result.pixels.max(), 255)
        self.assertIsNotNone(result.input_range)

    def test_0_and_100_percentiles_map_min_and_max_exactly(self):
        result = stretch_contrast(self.ramp, 0, 100)
        self.assertEqual(result.input_range, (100.0, 150.0))
        expected = (self.ramp.astype(float) - 100) / 50 * 255
        # Values exactly halfway between two levels (e.g. 25.5) may round either way
        np.testing.assert_allclose(result.pixels, expected, atol=1)
        self.assertEqual((result.pixels[0, 0], result.pixels[-1, -1]), (0, 255))

    def test_darker_pixels_stay_darker(self):
        result = stretch_contrast(self.ramp).pixels.ravel()
        self.assertTrue((np.diff(result.astype(int)) >= 0).all())

    def test_values_outside_the_percentiles_are_clipped(self):
        pixels = np.full((10, 10), 120, dtype=np.uint8)
        pixels[0, :5] = 110          # 5% dark...
        pixels[-1, -5:] = 130        # ...and 5% bright pixels
        pixels[0, 0], pixels[-1, -1] = 0, 255  # one extreme outlier each
        result = stretch_contrast(pixels, 2, 98).pixels
        self.assertEqual(result[0, 1], 0)       # 110 is at the 2nd percentile -> black
        self.assertEqual(result[-1, -2], 255)   # 130 is at the 98th percentile -> white
        self.assertEqual(result[0, 0], 0)       # the outliers are clipped, not wrapped around
        self.assertEqual(result[-1, -1], 255)

    def test_full_range_image_is_unchanged_with_0_and_100(self):
        pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
        np.testing.assert_array_equal(stretch_contrast(pixels, 0, 100).pixels, pixels)

    def test_colour_channels_are_stretched_together(self):
        # Decision C: one stretch for all channels, so grey pixels stay grey
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        pixels[0, 0] = (100, 100, 100)  # grey
        pixels[0, 1] = (120, 120, 120)  # grey
        pixels[1, 0] = (100, 110, 140)  # bluish
        pixels[1, 1] = (110, 100, 130)
        result = stretch_contrast(pixels, 0, 100).pixels
        for column in range(2):
            self.assertEqual(len(set(result[0, column])), 1)  # R == G == B
        # Red and blue would both reach 255 if each channel were stretched alone
        self.assertLess(result[:, :, 0].max(), 255)


class PercentileCheckTests(unittest.TestCase):
    def test_valid_percentiles_are_accepted(self):
        for low, high in [(0, 100), (2, 98), (49.5, 50.5)]:
            with self.subTest(low=low, high=high):
                check_percentiles(low, high)

    def test_invalid_percentiles_give_clear_error(self):
        for low, high in [(50, 50), (90, 10), (-1, 50), (10, 101)]:
            with self.subTest(low=low, high=high), \
                    self.assertRaisesRegex(ValueError, "0 ≤ low < high ≤ 100"):
                stretch_contrast(np.zeros((2, 2), dtype=np.uint8), low, high)


class NothingToStretchTests(unittest.TestCase):
    """Edge cases that changed the image or crashed before."""

    def test_constant_image_is_unchanged(self):
        result = stretch_contrast(Image.new("L", (4, 4), 100))
        self.assertTrue((result.pixels == 100).all())
        self.assertIsNone(result.input_range)

    def test_one_bit_white_stays_white(self):
        result = stretch_contrast(Image.new("1", (4, 4), 1))
        self.assertTrue((result.pixels == 255).all())

    def test_empty_image_does_not_crash(self):
        for mode in ("L", "RGBA"):
            with self.subTest(mode=mode):
                result = stretch_contrast(Image.new(mode, (0, 0)))
                self.assertEqual(result.pixels.size, 0)
                self.assertIsNone(result.input_range)


class ImageModeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.photo = rng.integers(60, 180, size=(20, 20, 3), dtype=np.uint8)

    def test_rgba_alpha_is_kept_and_ignored(self):
        # Decision B: alpha is unchanged and does not affect the stretch
        rgba = np.dstack([self.photo, np.full((20, 20), 200, dtype=np.uint8)])
        result = stretch_contrast(Image.fromarray(rgba, "RGBA")).pixels
        self.assertEqual(result.shape, (20, 20, 4))
        self.assertTrue((result[:, :, 3] == 200).all())
        np.testing.assert_array_equal(result[:, :, :3], stretch_contrast(self.photo).pixels)

    def test_grayscale_alpha_is_kept(self):
        pixels = np.dstack([np.array([[50, 100]], dtype=np.uint8), np.array([[7, 9]], dtype=np.uint8)])
        result = stretch_contrast(Image.fromarray(pixels, "LA"), 0, 100).pixels
        np.testing.assert_array_equal(result[:, :, 0], [[0, 255]])
        np.testing.assert_array_equal(result[:, :, 1], [[7, 9]])

    def test_palette_image_stays_in_colour(self):
        palette = Image.fromarray(self.photo).convert("P", palette=Image.ADAPTIVE)
        result = stretch_contrast(palette).pixels
        self.assertEqual(result.shape, (20, 20, 3))
        self.assertEqual((result.min(), result.max()), (0, 255))

    def test_narrow_16_bit_image_keeps_its_detail(self):
        # Values 1060–1179 out of 65535 would all round to the same 8-bit level (4)
        image = Image.fromarray(1000 + self.photo[:, :, 0].astype(np.uint16))
        result = stretch_contrast(image)
        self.assertGreater(len(np.unique(result.pixels)), 100)

    def test_float_image_is_scaled(self):
        image = Image.fromarray((self.photo[:, :, 0] / 255).astype(np.float32))
        result = stretch_contrast(image)
        self.assertEqual((result.pixels.min(), result.pixels.max()), (0, 255))

    def test_stretched_image_shows_comb_in_histogram(self):
        # Cross-check with the histogram tool: stretching spreads 120 levels over 256,
        # leaving empty bins between them, which is what the histogram details look for.
        every_level = np.repeat(np.arange(60, 180, dtype=np.uint8), 4).reshape(16, 30)  # no gaps
        before = histogram_stats(compute_histogram(every_level)["Gray"])
        stretched = stretch_contrast(every_level).pixels
        after = histogram_stats(compute_histogram(stretched)["Gray"])
        self.assertEqual(before["empty bins"], 0)
        self.assertGreater(after["empty bins"], 50)


class ContrastStretchingToolTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.fromarray(
            np.linspace(100, 150, 12).round().astype(np.uint8).reshape(3, 4))

    def run_tool(self, percentiles=DEFAULT_PERCENTILES):
        with patch.object(ContrastStretchingTool, "choose_percentiles", return_value=percentiles):
            return ContrastStretchingTool().run(None, self.document)

    def test_tool_returns_stretched_image_without_changing_document(self):
        original = np.array(self.document.current)
        result = self.run_tool()
        self.assertEqual(result.image.size, (4, 3))
        self.assertEqual(np.array(result.image).max(), 255)
        np.testing.assert_array_equal(np.array(self.document.current), original)
        self.assertFalse(self.document.can_undo)

    def test_details_show_percentiles_and_range(self):
        result = self.run_tool((0, 100))
        self.assertEqual(result.message, "Contrast stretched.")
        self.assertEqual(result.details["Percentiles"], "0 / 100")
        self.assertEqual(result.details["Stretched range"], "100 – 150 → 0 – 255")

    def test_constant_image_reports_nothing_to_stretch(self):
        self.document.current = Image.new("L", (4, 3), 100)
        result = self.run_tool()
        self.assertEqual(result.message, "Nothing to stretch; image unchanged.")
        self.assertEqual(result.details["Stretched range"], "nothing to stretch (image has a single value)")

    def test_empty_image_reports_no_pixels(self):
        self.document.current = Image.new("L", (0, 0))
        result = self.run_tool()
        self.assertEqual(result.details["Stretched range"], "nothing to stretch (no pixels)")


class PercentileDialogTests(unittest.TestCase):
    """Opens the real dialog and drives it from code, like a user would."""

    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display available for Tk")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def dialog_widgets(self, kind):
        dialog = self.root.winfo_children()[0]
        return [widget for widget in dialog.winfo_children() if isinstance(widget, kind)]

    def type_values(self, low, high):
        for entry, value in zip(self.dialog_widgets(tk.Entry), (low, high)):
            entry.delete(0, "end")
            entry.insert(0, value)

    def close_dialog(self):
        self.root.winfo_children()[0].destroy()

    def test_dialog_shows_defaults(self):
        def check_and_close():
            self.shown = [entry.get() for entry in self.dialog_widgets(tk.Entry)]
            self.label = self.dialog_widgets(tk.Label)[0].cget("text")
            self.close_dialog()
        self.root.after(100, check_and_close)
        ContrastStretchingTool().choose_percentiles(self.root)
        self.assertEqual(self.shown, ["2", "98"])
        self.assertIn("defaults: 2 / 98", self.label)

    def test_closing_the_dialog_keeps_the_defaults(self):
        def type_then_close():
            self.type_values("5", "95")  # typed but never confirmed with OK
            self.close_dialog()
        self.root.after(100, type_then_close)
        self.assertEqual(ContrastStretchingTool().choose_percentiles(self.root), DEFAULT_PERCENTILES)

    def test_ok_uses_the_typed_values(self):
        def type_and_ok():
            self.type_values("5", "95")
            self.dialog_widgets(tk.Button)[0].invoke()
        self.root.after(100, type_and_ok)
        self.assertEqual(ContrastStretchingTool().choose_percentiles(self.root), (5.0, 95.0))

    def test_invalid_values_keep_the_dialog_open(self):
        def type_invalid_then_fix():
            self.type_values("abc", "95")
            self.dialog_widgets(tk.Button)[0].invoke()   # rejected: dialog stays open
            self.still_open = self.root.winfo_children()[0].winfo_exists()
            self.type_values("10", "90")
            self.dialog_widgets(tk.Button)[0].invoke()
        self.root.after(100, type_invalid_then_fix)
        with patch("forensics_app.tools.contrast_stretching.messagebox.showerror") as error_box:
            self.assertEqual(ContrastStretchingTool().choose_percentiles(self.root), (10.0, 90.0))
        error_box.assert_called_once()
        self.assertTrue(self.still_open)


if __name__ == "__main__":
    unittest.main()
