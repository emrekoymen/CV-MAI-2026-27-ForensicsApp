import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools.classification import (BANNER_HEIGHT, CLASSIFIER_FILE, ClassificationTool, load_or_train,
    train_classifier)
from forensics_app.tools.k_means import CODEBOOK_FILE, KMeansTool, split_dataset
from forensics_app.tools.word_histogram import WordHistogramTool


def make_dataset(folder):
    """Three classes of 8 small crops each (the probability calibration needs at least 5 training images per class)."""
    for name, pixels in (("astronaut", data.astronaut()), ("coffee", data.coffee()), ("chelsea", data.chelsea())):
        (folder / name).mkdir()
        for i in range(8):
            crop = Image.fromarray(pixels[i * 12:i * 12 + 200, i * 12:i * 12 + 200]).resize((160, 160))
            crop.save(folder / name / f"image_{i + 1:04}.png")


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        make_dataset(cls.folder)
        with patch("tkinter.filedialog.askdirectory", return_value=str(cls.folder)):
            KMeansTool().run(None, ImageDocument())
        cls.kmeans = joblib.load(cls.folder / CODEBOOK_FILE)["kmeans"]
        cls.classifier = load_or_train(cls.folder, cls.kmeans)
        cls.test_image = split_dataset(cls.folder)[1][0][0]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_tool(self, prediction):
        """Run the tool with the classify dialog replaced by a fixed answer."""
        document = ImageDocument()
        document.load(self.test_image)
        with patch("tkinter.filedialog.askdirectory", return_value=str(self.folder)), \
                patch.object(ClassificationTool, "classify_dialog", return_value=prediction):
            return ClassificationTool().run(None, document)

    def test_one_svm_per_class_and_probabilities_sum_to_one(self):
        histograms = np.random.default_rng(0).dirichlet(np.ones(30), size=18)
        labels = np.repeat(["a", "b", "c"], 6)
        model = train_classifier(histograms, labels)
        self.assertEqual(len(model.estimators_), 3)
        self.assertTrue(np.allclose(model.predict_proba(histograms).sum(axis=1), 1))

    def test_trained_model_is_saved_with_its_accuracy(self):
        self.assertTrue((self.folder / CLASSIFIER_FILE).exists())
        self.assertEqual(list(self.classifier["model"].classes_), ["astronaut", "chelsea", "coffee"])
        self.assertTrue(0 <= self.classifier["test accuracy"] <= 1)

    def test_saved_model_is_loaded_without_training(self):
        with patch("forensics_app.tools.classification.image_histograms") as histograms:
            load_or_train(self.folder, self.kmeans)
        histograms.assert_not_called()

    def test_model_is_retrained_when_codebook_is_newer(self):
        codebook = self.folder / CODEBOOK_FILE
        original = codebook.stat()
        newer = (self.folder / CLASSIFIER_FILE).stat().st_mtime + 10
        os.utime(codebook, (newer, newer))
        try:
            # The new model isn't saved, so the real one stays on disk for the other tests
            with patch("forensics_app.tools.classification.train_classifier") as train, \
                    patch("forensics_app.tools.classification.joblib.dump"):
                load_or_train(self.folder, self.kmeans)
            train.assert_called_once()
        finally:
            os.utime(codebook, (original.st_atime, original.st_mtime))

    def test_prediction_is_reported(self):
        result = self.run_tool(("coffee", 0.9))
        self.assertEqual(result.details["Predicted class"], "coffee")
        self.assertEqual(result.details["Probability"], 0.9)
        self.assertIn("coffee", result.message)
        # The prediction banner is added on top of the histogram
        document = ImageDocument()
        document.load(self.test_image)
        with patch("tkinter.filedialog.askdirectory", return_value=str(self.folder)):
            histogram = WordHistogramTool().run(None, document)
        self.assertEqual(result.image.height, histogram.image.height + BANNER_HEIGHT)

    def test_closing_dialog_without_classifying_shows_only_histogram(self):
        result = self.run_tool(None)
        self.assertNotIn("Predicted class", result.details)
        self.assertIn("Test accuracy", result.details)


if __name__ == "__main__":
    unittest.main()
