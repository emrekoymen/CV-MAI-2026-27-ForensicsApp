import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.hog import HogTool


class HogTests(unittest.TestCase):
    def test_feature_length_follows_parameters(self):
        document = ImageDocument()
        document.current = Image.new("RGB", (48, 32))
        with patch("tkinter.simpledialog.askinteger", side_effect=[9, 8, 2]):
            result = HogTool().run(None, document)
        self.assertIn(f"{3 * 5 * 2 * 2 * 9} HOG features", result.message)  # blocks * cells * orientations


if __name__ == "__main__":
    unittest.main()
