import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.edges import EdgeTool


class EdgeTests(unittest.TestCase):
    def test_edges_are_drawn_in_red(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = 255
        document = ImageDocument()
        document.current = Image.fromarray(image)
        with patch("tkinter.simpledialog.askfloat", return_value=1.0):
            result = EdgeTool().run(None, document)
        red = np.all(np.array(result.image) == [255, 0, 0], axis=2)
        self.assertTrue(red.any())
        self.assertFalse(red[12:20, 12:20].any())

    def test_cancel(self):
        document = ImageDocument()
        document.current = Image.new("L", (8, 8))
        with patch("tkinter.simpledialog.askfloat", return_value=None):
            self.assertIsNone(EdgeTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
