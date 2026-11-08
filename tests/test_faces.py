import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import data

from forensics_app.tools.eigenfaces import build_model, load_faces
from forensics_app.tools.face_detection import MODEL_PATH, detect_faces, make_detector
from forensics_app.tools.face_recognition import closest_faces, find_model


@unittest.skipUnless(MODEL_PATH.exists(), "YuNet model not in models/")
class FaceDetectionTests(unittest.TestCase):
    def test_finds_the_astronaut(self):
        bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
        faces = detect_faces(make_detector(bgr.shape[1], bgr.shape[0]), bgr)
        self.assertEqual(len(faces), 1)

    def test_no_face_in_noise(self):
        bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        self.assertEqual(len(detect_faces(make_detector(200, 200), bgr)), 0)


class EigenfaceTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(0).uniform(0, 255, (20, 48))
        self.model = build_model(self.X, n_components=10)

    def test_eigenfaces_are_orthonormal(self):
        U = self.model["eigenfaces"]
        np.testing.assert_allclose(U @ U.T, np.eye(10), atol=1e-10)

    def test_coefficients_are_projections(self):
        expected = (self.X - self.X.mean(axis=0)) @ self.model["eigenfaces"].T
        np.testing.assert_allclose(self.model["coefficients"], expected)

    def test_train_image_is_its_own_closest_face(self):
        closest, distances = closest_faces(self.X[7], self.model, n=3)
        self.assertEqual(closest[0], 7)
        self.assertAlmostEqual(distances[0], 0)
        self.assertTrue(np.all(np.diff(distances) >= 0))

    def test_folder_names_are_labels_and_model_is_found_above_test_image(self):
        with tempfile.TemporaryDirectory() as root:
            for split, name in (("train", "Ada_Lovelace"), ("test", "Ada_Lovelace")):
                (Path(root) / split / name).mkdir(parents=True)
                Image.new("L", (4, 5), 128).save(Path(root) / split / name / "0001.png")
            X, names, _, shape = load_faces(Path(root) / "train")
            self.assertEqual((X.shape, names, shape), ((1, 20), ["Ada Lovelace"], (5, 4)))
            (Path(root) / "eigenfaces.npz").touch()
            image = Path(root) / "test" / "Ada_Lovelace" / "0001.png"
            self.assertEqual(find_model(image), Path(root) / "eigenfaces.npz")


if __name__ == "__main__":
    unittest.main()
