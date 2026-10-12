import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.filters import FiltersTool, compute_filter


class FiltersTests(unittest.TestCase):
    def test_constant_image(self):
        image = Image.new("L", (16, 16), 100)
        self.assertFalse(np.array(compute_filter(image, "Sobel")).any())
        self.assertTrue((np.array(compute_filter(image, "Gaussian")) == 100).all())

    def test_cancel(self):
        with patch.object(FiltersTool, "choose_filter", return_value=None):
            self.assertIsNone(FiltersTool().run(None, ImageDocument()))


if __name__ == "__main__":
    unittest.main()
