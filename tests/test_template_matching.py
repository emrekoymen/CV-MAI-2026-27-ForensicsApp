import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.template_matching import TemplateMatchingTool, compute_ncc


class TemplateMatchingTests(unittest.TestCase):
    def setUp(self):
        scene = np.random.default_rng(0).integers(0, 256, (40, 50), dtype=np.uint8)
        self.scene, self.template = Image.fromarray(scene), Image.fromarray(scene[10:19, 20:27])

    def test_peak_is_at_template_centre(self):
        _, ncc = compute_ncc(self.scene, self.template)
        self.assertEqual(np.unravel_index(np.argmax(ncc), ncc.shape), (14, 23))

    def test_match_is_marked_red(self):
        document = ImageDocument()
        document.current = self.scene
        with patch("forensics_app.tools.template_matching.open_image", return_value=self.template), \
                patch("tkinter.simpledialog.askfloat", side_effect=[0.0, 0.99]):
            result = TemplateMatchingTool().run(None, document)
        self.assertEqual(result.image.getpixel((23, 14)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
