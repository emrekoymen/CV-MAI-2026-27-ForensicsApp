import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools.descriptor_matching import descriptor_sift
from forensics_app.tools.k_means import CODEBOOK_FILE, N_CLUSTERS, KMeansTool, split_dataset
from forensics_app.tools.word_histogram import WordHistogramTool, word_histogram


def make_dataset(folder):
    """Two classes of 4 small crops each."""
    for name, pixels in (("astronaut", data.astronaut()), ("coffee", data.coffee())):
        (folder / name).mkdir()
        for i in range(4):
            crop = Image.fromarray(pixels[i * 40:i * 40 + 300, i * 40:i * 40 + 300]).resize((160, 160))
            crop.save(folder / name / f"image_{i + 1:04}.png")


def run_tool(folder, path):
    document = ImageDocument()
    document.load(path)
    with patch("tkinter.filedialog.askdirectory", return_value=folder), \
            patch("tkinter.messagebox.showinfo") as info:
        return WordHistogramTool().run(None, document), info


class WordHistogramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name) / "dataset"
        cls.folder.mkdir()
        make_dataset(cls.folder)
        with patch("tkinter.filedialog.askdirectory", return_value=str(cls.folder)):
            KMeansTool().run(None, ImageDocument())
        cls.kmeans = joblib.load(cls.folder / CODEBOOK_FILE)["kmeans"]
        cls.train, cls.test = split_dataset(cls.folder)
        # An image from outside the dataset, saved next to it
        cls.outside = Path(cls.temp.name) / "camera.png"
        Image.fromarray(data.camera()).resize((160, 160)).save(cls.outside)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frequencies_sum_to_one(self):
        image = np.array(Image.open(self.outside), dtype=float) / 255
        _, descriptors = descriptor_sift(image)
        words, frequencies = word_histogram(self.kmeans, descriptors)
        self.assertEqual(len(words), len(descriptors))
        self.assertEqual(len(frequencies), N_CLUSTERS)
        self.assertAlmostEqual(frequencies.sum(), 1)

    def test_test_image_gets_a_histogram(self):
        result, _ = run_tool(str(self.folder), self.test[0][0])
        self.assertEqual(result.details["Image in dataset"], "Test")
        self.assertGreater(result.details["Keypoints"], 0)
        self.assertIsNotNone(result.image)

    def test_image_outside_dataset_gets_a_histogram(self):
        result, _ = run_tool(str(self.folder), self.outside)
        self.assertEqual(result.details["Image in dataset"], "Not in dataset")

    def test_training_image_is_refused(self):
        result, info = run_tool(str(self.folder), self.train[0][0])
        self.assertIsNone(result)
        self.assertEqual(info.call_args[0][0], "Training image")

    def test_folder_without_codebook_returns_none(self):
        with tempfile.TemporaryDirectory() as folder:
            result, info = run_tool(folder, self.outside)
        self.assertIsNone(result)
        self.assertEqual(info.call_args[0][0], "No codebook")

    def test_cancel_returns_none(self):
        self.assertIsNone(run_tool("", self.outside)[0])


if __name__ == "__main__":
    unittest.main()
