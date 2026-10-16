import unittest

import numpy as np

from forensics_app.tools.person_detection import detect_person


class PersonDetectionTests(unittest.TestCase):
    def test_planted_figure_is_found(self):
        template = np.zeros((32, 16), dtype=np.uint8)
        template[2:10, 5:11] = 255   # head
        template[10:30, 3:13] = 255  # body
        scene = np.zeros((48, 64), dtype=np.uint8)
        scene[8:40, 40:56] = template
        self.assertEqual(detect_person(scene, template, step=4), ((8, 40), (8, 40)))


if __name__ == "__main__":
    unittest.main()
