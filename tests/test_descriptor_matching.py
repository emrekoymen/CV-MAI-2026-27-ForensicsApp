import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import data, transform
from skimage.color import rgb2gray

from forensics_app.core import ImageDocument
from forensics_app.tools.descriptor_matching import (
    DescriptorMatchingTool, descriptor_match, descriptor_orb, descriptor_sift)


def camera():
    return data.camera() / 255


def run_tool(second_image):
    document = ImageDocument()
    document.current = Image.fromarray(data.camera())
    with patch("forensics_app.tools.descriptor_matching.open_image", return_value=second_image):
        return DescriptorMatchingTool().run(None, document)


class DescriptorMatchingTests(unittest.TestCase):
    def test_one_descriptor_per_keypoint(self):
        for name, extract, length in (("ORB", descriptor_orb, 256), ("SIFT", descriptor_sift, 128)):
            with self.subTest(extractor=name):
                keypoints, descriptors = extract(camera())
                self.assertEqual(len(keypoints), len(descriptors))
                self.assertEqual(descriptors.shape[1], length)

    def test_image_matches_itself(self):
        # Every keypoint should be matched with the same keypoint in the copy
        _, descriptors = descriptor_orb(camera(), 100)
        matches = descriptor_match(descriptors, descriptors)
        self.assertGreater(np.mean(matches[:, 0] == matches[:, 1]), 0.9)

    def test_matches_of_shifted_image_are_shifted(self):
        image = np.zeros((300, 300))
        image[50:250, 50:250] = camera()[150:350, 150:350]
        shifted = np.roll(image, 20, axis=1)
        keypoints1, descriptors1 = descriptor_orb(image, 100)
        keypoints2, descriptors2 = descriptor_orb(shifted, 100)
        matches = descriptor_match(descriptors1, descriptors2)
        offsets = keypoints2[matches[:, 1]] - keypoints1[matches[:, 0]]
        # Most matches should be 20 px to the right and 0 px down (allowing 1 px of rounding)
        correct = (np.abs(offsets[:, 0]) <= 1) & (np.abs(offsets[:, 1] - 20) <= 1)
        self.assertGreater(np.mean(correct), 0.5)

    def test_lower_ratio_gives_fewer_matches(self):
        _, descriptors1 = descriptor_sift(camera())
        _, descriptors2 = descriptor_sift(np.fliplr(camera()))
        loose = descriptor_match(descriptors1, descriptors2, max_ratio=1.0)
        strict = descriptor_match(descriptors1, descriptors2, max_ratio=0.6)
        self.assertLess(len(strict), len(loose))

    def test_tool_reports_matches(self):
        result = run_tool(Image.fromarray(data.camera()))
        self.assertGreater(result.details["Matches"], 0)
        self.assertLessEqual(result.details["Matches"], result.details["Keypoints (image 1)"])

    def test_ransac_keeps_most_matches_of_a_rotated_copy(self):
        # A rotated and shifted copy fits one transform, so almost every match should be an inlier
        rotated = transform.warp(camera(), transform.AffineTransform(rotation=0.1, translation=(30, -10)).inverse)
        result = run_tool(Image.fromarray((rotated * 255).astype(np.uint8)))
        self.assertGreater(result.details["RANSAC inliers"], 0.8 * result.details["Matches"])

    def test_ransac_rejects_most_matches_of_an_unrelated_image(self):
        # Matches between different images are mostly coincidences that fit no common transform
        astronaut = (rgb2gray(data.astronaut()) * 255).astype(np.uint8)
        result = run_tool(Image.fromarray(astronaut))
        self.assertLess(result.details["RANSAC inliers"], 0.5 * result.details["Matches"])

    def test_cancel_returns_none(self):
        self.assertIsNone(run_tool(None))


if __name__ == "__main__":
    unittest.main()
