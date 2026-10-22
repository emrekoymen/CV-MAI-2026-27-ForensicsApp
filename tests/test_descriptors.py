import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools.descriptors import DescriptorsTool, extract_orb, extract_sift


def camera():
    return data.camera() / 255


def run_tool(extractor, n_keypoints=50, sift_values=(4, 3, 1.6)):
    document = ImageDocument()
    document.current = Image.fromarray(data.camera())
    with patch.object(DescriptorsTool, "choose_extractor", return_value=extractor), \
            patch("tkinter.simpledialog.askinteger", return_value=n_keypoints), \
            patch.object(DescriptorsTool, "choose_siftvalues", return_value=sift_values):
        return DescriptorsTool().run(None, document), document


class DescriptorsTests(unittest.TestCase):
    def test_orb_finds_at_most_the_requested_keypoints(self):
        for n in (10, 50):
            with self.subTest(n_keypoints=n):
                keypoints = extract_orb(camera(), n)
                self.assertGreater(len(keypoints), 0)
                self.assertLessEqual(len(keypoints), n)

    def test_keypoints_are_inside_the_image(self):
        image = camera()
        for name, keypoints in (("ORB", extract_orb(image)), ("SIFT", extract_sift(image))):
            with self.subTest(extractor=name):
                rows, cols = keypoints[:, 0], keypoints[:, 1]
                self.assertTrue((rows >= 0).all() and (rows < image.shape[0]).all())
                self.assertTrue((cols >= 0).all() and (cols < image.shape[1]).all())

    def test_shifted_image_gives_shifted_keypoints(self):
        # Moving the content 20 px down should move the same keypoints 20 px down
        image = np.zeros((300, 300))
        image[50:250, 50:250] = camera()[150:350, 150:350]
        shifted = np.roll(image, 20, axis=0)
        original = {tuple(p) for p in np.round(extract_orb(image, 50)).astype(int)}
        moved = {tuple(p) for p in np.round(extract_orb(shifted, 50) - [20, 0]).astype(int)}
        self.assertGreater(len(original & moved), len(original) // 2)

    def test_tool_with_orb_reports_keypoints(self):
        result, document = run_tool("ORB", n_keypoints=50)
        self.assertEqual(result.details["Extractor"], "ORB")
        self.assertLessEqual(result.details["Keypoints"], 50)
        self.assertEqual(result.image.width, document.current.width)

    def test_tool_with_sift_reports_its_settings(self):
        result, _ = run_tool("SIFT", sift_values=(4, 3, 1.6))
        self.assertEqual(result.details["Octaves"], 4)
        self.assertGreater(result.details["Keypoints"], 0)

    def test_tool_does_not_change_document(self):
        _, document = run_tool("ORB")
        np.testing.assert_array_equal(np.array(document.current), data.camera())

    def test_cancel_returns_none(self):
        self.assertIsNone(run_tool(None)[0])
        self.assertIsNone(run_tool("ORB", n_keypoints=None)[0])
        self.assertIsNone(run_tool("SIFT", sift_values=None)[0])


if __name__ == "__main__":
    unittest.main()
