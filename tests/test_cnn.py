import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import silhouette_score

from forensics_app.tools.cnn_classification import (
    MODELS_DIR, classify, load_passport, passport_figure, passport_summary, preprocess)
from forensics_app.tools.feature_projection import project, projection_figure, subsample

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
PASSPORTS = sorted(MODELS_DIR.glob("*.passport.json"))
# Keras caches the dataset here after the first download (the folder name differs between versions)
CIFAR_CACHED = any((Path.home() / ".keras" / "datasets").glob("cifar-10-batches-py*"))


def make_passport(size=32, formula="x / 255.0"):
    history = {key: [1.0, 0.8, 0.6, 0.5] for key in ("loss", "val_loss", "accuracy", "val_accuracy")}
    return {
        "name": "test_model", "class_names": CLASSES,
        "dataset": {"name": "CIFAR-10", "samples": {"train": 10, "validation": 2, "test": 5}},
        "preprocessing": {"color_mode": "RGB", "input_size": [size, size], "formula": formula},
        "architecture": {"parameters": 1000},
        "training": {"epochs_run": 4, "history": {**history, "phase": [1, 1, 2, 2]}},
        "evaluation": {"test_accuracy": 0.7, "per_class_accuracy": {c: 0.7 for c in CLASSES}},
    }


class PassportTests(unittest.TestCase):
    def test_preprocess_resizes_and_scales_to_0_1(self):
        x = preprocess(Image.new("RGB", (100, 60), (255, 0, 51)), make_passport(32, "x / 255.0"))
        self.assertEqual((x.shape, x.dtype), ((1, 32, 32, 3), np.float32))
        np.testing.assert_allclose(x[0, 10, 10], [1.0, 0.0, 0.2], atol=1e-6)

    def test_preprocess_inception_scaling_and_grayscale_input(self):
        x = preprocess(Image.new("L", (40, 40), 0), make_passport(139, "x / 127.5 - 1.0"))
        self.assertEqual(x.shape, (1, 139, 139, 3))
        np.testing.assert_allclose(x, -1.0)

    def test_unknown_scaling_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocess(Image.new("RGB", (8, 8)), make_passport(formula="x - mean"))

    def test_summary_and_figure(self):
        passport = make_passport()
        self.assertEqual(passport_summary(passport)["Samples (train/val/test)"], "10 / 2 / 5")
        self.assertGreater(passport_figure(passport).width, 500)


@unittest.skipUnless(PASSPORTS, "Set 8 models not in models/set8_models")
class TrainedModelTests(unittest.TestCase):
    def test_probabilities_cover_all_classes(self):
        for path in PASSPORTS:
            probabilities = classify(Image.new("RGB", (64, 48), (90, 140, 200)), load_passport(path))
            self.assertEqual(len(probabilities), 10)
            self.assertAlmostEqual(probabilities.sum(), 1.0, places=4)

    @unittest.skipUnless(CIFAR_CACHED, "CIFAR-10 not downloaded by Keras")
    def test_accuracy_on_cifar_test_images_matches_passport(self):
        import keras
        (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
        keep = np.random.default_rng(0).choice(len(x_test), 200, replace=False)
        for path in PASSPORTS:
            passport = load_passport(path)
            predictions = [classify(Image.fromarray(x_test[i]), passport).argmax() for i in keep]
            accuracy = np.mean(np.array(predictions) == y_test.ravel()[keep])
            self.assertGreater(accuracy, passport["evaluation"]["test_accuracy"] - 0.08, passport["name"])


class ProjectionTests(unittest.TestCase):
    def test_subsample(self):
        self.assertEqual(len(subsample(100, 30)), 30)
        self.assertTrue(np.all(np.diff(subsample(100, 30)) > 0))
        np.testing.assert_array_equal(subsample(5, 30), np.arange(5))

    def test_tsne_and_umap_separate_clear_clusters(self):
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(3), 40)
        features = rng.normal(0, 10, (3, 16))[labels] + rng.normal(0, 0.5, (120, 16))
        projections = project(features)
        for method, xy in projections.items():
            self.assertEqual(xy.shape, (120, 2))
            self.assertGreater(silhouette_score(xy, labels), 0.5, method)
        self.assertGreater(projection_figure(projections, labels, ["a", "b", "c"], "test").width, 1000)


if __name__ == "__main__":
    unittest.main()
