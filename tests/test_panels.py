import unittest

import numpy as np

from forensics_app.tools.panels import BACKGROUND, grid_layout, label_font_size, make_grid


class GridLayoutTests(unittest.TestCase):
    def test_2x2_grid_has_no_white_space(self):
        # Canvas = panels + one label strip per row + small gaps, nothing else
        size, origins = grid_layout(400, 300, count=4, columns=2)
        strip, gap = int(label_font_size(400) * 1.6), 2
        self.assertEqual(size, (2 * 400 + gap, 2 * (strip + 300) + gap))
        self.assertEqual(origins, [
            (0, strip), (400 + gap, strip),
            (0, 2 * strip + 300 + gap), (400 + gap, 2 * strip + 300 + gap),
        ])

    def test_single_row(self):
        size, origins = grid_layout(100, 50, count=2, columns=2)
        self.assertEqual(size[1], origins[0][1] + 50)  # just one strip and one panel high
        self.assertEqual(origins[0][1], origins[1][1])

    def test_panels_never_overlap(self):
        _, origins = grid_layout(37, 23, count=4, columns=2)
        (x0, y0), (x1, _), (_, y2), _ = origins
        self.assertGreaterEqual(x1, x0 + 37)
        self.assertGreaterEqual(y2, y0 + 23)

    def test_label_grows_with_panel(self):
        self.assertEqual(label_font_size(400), 20)
        self.assertEqual(label_font_size(4000), 200)
        self.assertEqual(label_font_size(50), 10)  # minimum, so small panels stay legible


class MakeGridTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.colour = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)

    def test_panels_are_pasted_pixel_for_pixel(self):
        grey = self.colour[:, :, 1]
        grid = np.array(make_grid([("Colour", self.colour), ("Grey", grey)], columns=2))
        (x0, y0), (x1, y1) = grid_layout(40, 30, count=2, columns=2)[1]
        np.testing.assert_array_equal(grid[y0:y0 + 30, x0:x0 + 40], self.colour)
        np.testing.assert_array_equal(grid[y1:y1 + 30, x1:x1 + 40], np.dstack([grey] * 3))

    def test_gap_and_strip_use_background_colour(self):
        grid = np.array(make_grid([("A", self.colour), ("B", self.colour)], columns=2))
        (x0, y0), _ = grid_layout(40, 30, count=2, columns=2)[1]
        self.assertEqual(tuple(grid[y0 + 10, 40]), BACKGROUND)  # gap between the panels
        self.assertEqual(tuple(grid[0, 0]), BACKGROUND)         # corner of the label strip

    def test_label_is_drawn_in_the_strip(self):
        grid = np.array(make_grid([("Label", self.colour)], columns=1))
        strip = grid[:grid_layout(40, 30, count=1, columns=1)[1][0][1]]
        self.assertTrue((strip != BACKGROUND).any())  # some text pixels

    def test_long_label_stays_inside_its_panel(self):
        # The text must be cut off at the panel edge, not spill over the gap or neighbour
        label = "A very long label that does not fit"
        grid = np.array(make_grid([(label, self.colour), ("", self.colour)], columns=2))
        strip_height = grid_layout(40, 30, count=2, columns=2)[1][0][1]
        self.assertTrue((grid[:strip_height, 40:] == BACKGROUND).all())

    def test_tiny_and_empty_panels_do_not_crash(self):
        for shape in ((1, 1, 3), (0, 0, 3)):
            with self.subTest(shape=shape):
                pixels = np.zeros(shape, dtype=np.uint8)
                grid = make_grid([("Original", pixels), ("Red channel", pixels[:, :, 0])], columns=2)
                self.assertEqual(grid.size, grid_layout(shape[1], shape[0], 2, 2)[0])


if __name__ == "__main__":
    unittest.main()
