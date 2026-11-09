import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools.k_means import (CODEBOOK_FILE, N_CLUSTERS, PATCH_SIZE, KMeansTool, crop_patch, file_hash,
    split_dataset)


def make_dataset(folder):
    """Two classes of 4 small crops each, plus a text file and an empty folder that should be ignored."""
    for name, pixels in (("astronaut", data.astronaut()), ("coffee", data.coffee())):
        (folder / name).mkdir()
        for i in range(4):
            crop = Image.fromarray(pixels[i * 40:i * 40 + 300, i * 40:i * 40 + 300]).resize((160, 160))
            crop.save(folder / name / f"image_{i + 1:04}.png")
    (folder / "notes.txt").write_text("not an image")
    (folder / "empty").mkdir()


def run_tool(folder):
    with patch("tkinter.filedialog.askdirectory", return_value=folder), patch("tkinter.messagebox.showinfo"):
        return KMeansTool().run(None, ImageDocument())


class KMeansTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The codebook is built once, since SIFT on every image is the slow part
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        make_dataset(cls.folder)
        cls.result = run_tool(str(cls.folder))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_split_is_the_same_every_time(self):
        self.assertEqual(split_dataset(self.folder), split_dataset(self.folder))

    def test_patches_near_the_border_keep_their_size(self):
        path = self.folder / "coffee" / "image_0001.png"
        for row, col in ((0, 0), (80, 80), (159, 159)):
            self.assertEqual(crop_patch(path, row, col).shape, (PATCH_SIZE, PATCH_SIZE))

    def test_builds_and_saves_codebook(self):
        self.assertIn("Built", self.result.message)
        self.assertEqual(self.result.details["Clusters"], N_CLUSTERS)
        self.assertEqual(self.result.details["Train images"], 6)
        saved = joblib.load(self.folder / CODEBOOK_FILE)
        self.assertEqual(saved["kmeans"].cluster_centers_.shape, (N_CLUSTERS, 128))
        self.assertEqual(len(saved["panels"]), N_CLUSTERS)

    def test_saved_split_marks_train_and_test_images(self):
        split = joblib.load(self.folder / CODEBOOK_FILE)["split"]
        train, test = split_dataset(self.folder)
        self.assertTrue(all(split[file_hash(p)] == "Train" for p, _ in train))
        self.assertTrue(all(split[file_hash(p)] == "Test" for p, _ in test))

    def test_second_run_loads_saved_codebook_without_extracting(self):
        with patch("forensics_app.tools.k_means.extract_all") as extract:
            result = run_tool(str(self.folder))
        extract.assert_not_called()
        self.assertIn("Loaded", result.message)
        self.assertEqual(result.image.size, self.result.image.size)

    def test_folder_without_classes_returns_none(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(run_tool(folder))

    def test_cancel_returns_none(self):
        self.assertIsNone(run_tool(""))


if __name__ == "__main__":
    unittest.main()
