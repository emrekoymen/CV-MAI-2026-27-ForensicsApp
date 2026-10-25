import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools.closest_match import ClosestMatchTool, best_match_in_folder, prepare


# Match at a smaller width than the app does, so the tests run faster
@patch("forensics_app.tools.closest_match.MATCH_WIDTH", 300)
class ClosestMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Folder with the query, a cropped copy of it, two unrelated images and a text file
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.query = Image.fromarray(data.camera())
        cls.query_path = cls.folder / "camera.png"
        cls.query.save(cls.query_path)
        cls.query.crop((50, 50, 450, 450)).save(cls.folder / "camera_crop.png")
        Image.fromarray(data.astronaut()).save(cls.folder / "astronaut.png")
        Image.fromarray(data.coffee()).save(cls.folder / "coffee.jpg")
        (cls.folder / "notes.txt").write_text("not an image")
        with patch("forensics_app.tools.closest_match.MATCH_WIDTH", 300):
            cls.found = best_match_in_folder(cls.query, cls.query_path, cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_tool(self, folder, metric="Total matches"):
        document = ImageDocument()
        document.current = self.query
        document.path = self.query_path
        with patch("tkinter.filedialog.askdirectory", return_value=folder), \
                patch.object(ClosestMatchTool, "choose_metric", return_value=metric), \
                patch("tkinter.messagebox.showinfo"):
            return ClosestMatchTool().run(None, document)

    def test_prepare_resizes_keeping_proportions(self):
        image, array = prepare(Image.new("RGB", (400, 200)))
        self.assertEqual(image.size, (300, 150))
        self.assertEqual(array.shape, (150, 300))

    def test_finds_the_cropped_copy(self):
        _, path, score = self.found
        self.assertEqual(path.name, "camera_crop.png")
        self.assertGreater(score, 0)

    def test_query_is_not_matched_with_itself(self):
        # The query file is in the folder too, and would get a perfect score if it were not skipped
        _, path, _ = self.found
        self.assertNotEqual(path.resolve(), self.query_path.resolve())

    def test_folder_without_other_images_returns_none(self):
        with tempfile.TemporaryDirectory() as folder:
            self.query.save(Path(folder) / "camera.png")
            (Path(folder) / "notes.txt").write_text("not an image")
            self.assertIsNone(best_match_in_folder(self.query, Path(folder) / "camera.png", folder))

    def test_tool_reports_best_match(self):
        result = self.run_tool(str(self.folder))
        self.assertEqual(result.details["Best match"], "camera_crop.png")
        self.assertIn("camera_crop.png", result.message)

    def test_ransac_ranking_also_finds_the_cropped_copy(self):
        result = self.run_tool(str(self.folder), metric="RANSAC inliers")
        self.assertEqual(result.details["Best match"], "camera_crop.png")
        self.assertEqual(result.details["Ranked by"], "RANSAC inliers")
        self.assertGreater(result.details["RANSAC inliers"], 0)

    def test_cancel_returns_none(self):
        self.assertIsNone(self.run_tool(""))


if __name__ == "__main__":
    unittest.main()
