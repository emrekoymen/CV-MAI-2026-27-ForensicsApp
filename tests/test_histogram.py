import unittest

import matplotlib

matplotlib.use("Agg")  # render figures off-screen so the tests never open a window

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.histogram import (
    HistogramTool,
    compute_histogram,
    histogram_stats,
    to_8bit,
)


def non_zero(counts):
    """Turn 256 counts into {value: count} for the values that occur, for readable checks."""
    return {int(value): int(counts[value]) for value in np.nonzero(counts)[0]}


class ComputeHistogramTests(unittest.TestCase):
    def test_counts_known_grayscale_values(self):
        pixels = np.array([[10, 10], [10, 200]], dtype=np.uint8)
        counts = compute_histogram(pixels)
        self.assertEqual(list(counts), ["Gray"])
        self.assertEqual(non_zero(counts["Gray"]), {10: 3, 200: 1})

    def test_colour_channels_are_counted_separately(self):
        counts = compute_histogram(Image.new("RGB", (2, 3), "red"))
        self.assertEqual(list(counts), ["Red", "Green", "Blue"])
        self.assertEqual(non_zero(counts["Red"]), {255: 6})
        self.assertEqual(non_zero(counts["Green"]), {0: 6})
        self.assertEqual(non_zero(counts["Blue"]), {0: 6})

    def test_every_channel_counts_every_pixel_once(self):
        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
        counts = compute_histogram(pixels)
        for name, channel_counts in counts.items():
            with self.subTest(channel=name):
                self.assertEqual(len(channel_counts), 256)
                self.assertEqual(channel_counts.sum(), 5 * 7)

    def test_first_and_last_values_have_their_own_bins(self):
        counts = compute_histogram(np.array([[0, 255]], dtype=np.uint8))
        self.assertEqual(non_zero(counts["Gray"]), {0: 1, 255: 1})

    def test_single_pixel_image(self):
        counts = compute_histogram(Image.new("RGB", (1, 1), (1, 2, 3)))
        self.assertEqual([non_zero(c) for c in counts.values()], [{1: 1}, {2: 1}, {3: 1}])

    def test_empty_image_does_not_crash(self):
        counts = compute_histogram(Image.new("L", (0, 0)))
        self.assertEqual(counts["Gray"].sum(), 0)


class ImageModeTests(unittest.TestCase):
    """Image types that gave a wrong histogram before."""

    def test_one_bit_white_is_counted_at_255(self):
        counts = compute_histogram(Image.new("1", (4, 4), 1))
        self.assertEqual(non_zero(counts["Gray"]), {255: 16})

    def test_palette_image_counts_real_colours_not_indices(self):
        image = Image.new("RGB", (4, 4), (200, 10, 10)).convert("P", palette=Image.ADAPTIVE)
        counts = compute_histogram(image)
        self.assertEqual(non_zero(counts["Red"]), {200: 16})
        self.assertEqual(non_zero(counts["Green"]), {10: 16})
        self.assertEqual(non_zero(counts["Blue"]), {10: 16})

    def test_grayscale_alpha_ignores_alpha(self):
        counts = compute_histogram(Image.new("LA", (4, 4), (50, 255)))
        self.assertEqual(non_zero(counts["Gray"]), {50: 16})

    def test_rgba_ignores_alpha(self):
        counts = compute_histogram(Image.new("RGBA", (2, 2), (1, 2, 3, 0)))
        self.assertEqual(list(counts), ["Red", "Green", "Blue"])
        self.assertEqual(non_zero(counts["Blue"]), {3: 4})

    def test_16_bit_image_is_scaled(self):
        counts = compute_histogram(Image.new("I;16", (4, 4), 40000))
        self.assertEqual(non_zero(counts["Gray"]), {156: 16})  # 40000 / 65535 * 255 = 155.6

    def test_float_image_is_scaled(self):
        counts = compute_histogram(Image.new("F", (4, 4), 0.5))
        self.assertEqual(non_zero(counts["Gray"]), {128: 16})  # 0.5 * 255 = 127.5


class To8BitTests(unittest.TestCase):
    def test_8_bit_values_are_unchanged(self):
        pixels = np.array([0, 100, 255], dtype=np.uint8)
        result = to_8bit(pixels)
        np.testing.assert_array_equal(result, pixels)

    def test_16_bit_ends_map_to_black_and_white(self):
        result = to_8bit(np.array([0, 65535], dtype=np.uint16))
        np.testing.assert_array_equal(result, [0, 255])

    def test_out_of_range_values_are_clipped_not_dropped(self):
        result = to_8bit(np.array([-5, 70000], dtype=np.int32))
        np.testing.assert_array_equal(result, [0, 255])
        result = to_8bit(np.array([-0.5, 1.5]))
        np.testing.assert_array_equal(result, [0, 255])

    def test_bool_is_black_and_white(self):
        result = to_8bit(np.array([False, True]))
        np.testing.assert_array_equal(result, [0, 255])


class HistogramStatsTests(unittest.TestCase):
    def test_min_max_and_mean(self):
        counts = np.zeros(256, dtype=int)
        counts[10], counts[30] = 3, 1  # mean = (3 * 10 + 30) / 4 = 15
        stats = histogram_stats(counts)
        self.assertEqual((stats["min"], stats["max"], stats["mean"]), (10, 30, 15.0))

    def test_comb_histogram_has_empty_bins(self):
        # A contrast-stretched image only uses every other value
        counts = np.zeros(256, dtype=int)
        counts[0:255:2] = 1
        stats = histogram_stats(counts)
        self.assertEqual(stats["empty bins"], 127)
        self.assertEqual(stats["range bins"], 255)

    def test_smooth_histogram_has_no_empty_bins(self):
        counts = np.zeros(256, dtype=int)
        counts[50:101] = 4
        stats = histogram_stats(counts)
        self.assertEqual(stats["empty bins"], 0)
        self.assertEqual(stats["range bins"], 51)

    def test_no_pixels_gives_no_stats(self):
        self.assertEqual(histogram_stats(np.zeros(256, dtype=int)), {})


class HistogramToolTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()

    def test_tool_returns_rgb_plot_without_changing_document(self):
        self.document.current = Image.new("RGB", (4, 3), (10, 20, 30))
        result = HistogramTool().run(None, self.document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, (600, 400))  # figsize (6, 4) at dpi 100
        self.assertEqual(self.document.current.getpixel((0, 0)), (10, 20, 30))
        self.assertFalse(self.document.can_undo)

    def test_colour_details_have_two_rows_per_channel(self):
        self.document.current = Image.new("RGB", (4, 3), (10, 20, 30))
        details = HistogramTool().run(None, self.document).details
        self.assertEqual(details["Red min/max/mean"], "10/10/10.0")
        self.assertEqual(details["Blue empty bins"], "0 of 1")
        self.assertEqual(len(details), 1 + 3 * 2)  # Operation + 2 rows for each channel

    def test_empty_image_reports_no_pixels(self):
        self.document.current = Image.new("L", (0, 0))
        result = HistogramTool().run(None, self.document)
        self.assertEqual(result.details["Gray"], "no pixels")
        self.assertEqual(result.image.size, (600, 400))


if __name__ == "__main__":
    unittest.main()
