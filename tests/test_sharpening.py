import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.sharpening import SharpeningTool


class SharpeningTests(unittest.TestCase):
    def test_sharpening_keeps_size_and_alpha(self):
        document = ImageDocument()
        document.current = Image.new("RGBA", (32, 32), (100, 150, 200, 128))
        with patch("tkinter.simpledialog.askfloat", return_value=2.0):
            result = SharpeningTool().run(None, document)
        self.assertEqual(result.image.mode, "RGBA")
        self.assertEqual(result.image.size, document.current.size)
        self.assertTrue((np.array(result.image)[:, :, 3] == 128).all())

    def test_cancel(self):
        document = ImageDocument()
        document.current = Image.new("L", (8, 8))
        with patch("tkinter.simpledialog.askfloat", return_value=None):
            self.assertIsNone(SharpeningTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
