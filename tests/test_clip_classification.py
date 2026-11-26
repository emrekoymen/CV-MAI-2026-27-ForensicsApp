import importlib.util
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools import clip_classification as clip_tool
from forensics_app.tools.clip_classification import (ENSEMBLE_TEMPLATES, PRESETS, ClipZeroShotTool, caption, check_template,
    load_clip, parse_classes, prediction_figure, zero_shot)


def model_is_cached():
    """True when CLIP can be loaded without downloading 600 MB during the tests."""
    if clip_tool.LOCAL_MODEL.is_dir():
        return True
    if importlib.util.find_spec("transformers") is None:
        return False
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(clip_tool.MODEL_NAME, "config.json"), str)


class FakeClip:
    """Embeds every caption by the first known word in it; the image looks like a cat, a bit like a dog."""

    name, device, logit_scale = "fake", "cpu", 100.0
    WORDS = {"cat": [1, 0, 0], "dog": [0, 1, 0], "car": [0, 0, 1]}

    def __init__(self):
        self.texts = []

    def embed_images(self, images):
        return clip_tool._normalise(np.array([[0.9, 0.4, 0.0]] * len(images)))

    def embed_texts(self, texts):
        self.texts.extend(texts)
        vectors = [next(v for word, v in self.WORDS.items() if word in text) for text in texts]
        return clip_tool._normalise(np.array(vectors, dtype=float))


class InputTests(unittest.TestCase):
    def test_one_class_per_line_ignoring_blank_lines(self):
        self.assertEqual(parse_classes("  one cat \n\ntwo cats\nno cats\n"), ["one cat", "two cats", "no cats"])

    def test_needs_two_classes(self):
        with self.assertRaises(ValueError):
            parse_classes("cat\n\n")

    def test_classes_must_be_different(self):
        with self.assertRaisesRegex(ValueError, "cat"):
            parse_classes("cat\ndog\ncat")

    def test_template_needs_a_placeholder(self):
        self.assertEqual(check_template("  a photo of a {}. "), "a photo of a {}.")
        with self.assertRaises(ValueError):
            check_template("a photo")

    def test_presets_are_valid_inputs(self):
        self.assertEqual(list(PRESETS), ["Animals", "# Ducks", "Man/Woman", "# vs animal"])
        for classes, template in PRESETS.values():
            self.assertEqual(parse_classes("\n".join(classes)), classes)
            self.assertEqual(check_template(template), template)
        self.assertEqual(caption(PRESETS["# Ducks"][1], "two ducks"), "a photo of two ducks.")

    def test_caption_keeps_other_braces(self):
        self.assertEqual(caption("a {} in {curly} braces", "cat"), "a cat in {curly} braces")


class ZeroShotTests(unittest.TestCase):
    def setUp(self):
        self.clip = FakeClip()
        self.image = Image.new("RGB", (32, 32))

    def test_probabilities_follow_the_class_order_and_sum_to_one(self):
        probabilities, similarities = zero_shot(self.clip, self.image, ["dog", "car", "cat"], ["a photo of a {}."])
        self.assertAlmostEqual(probabilities.sum(), 1.0)
        self.assertEqual(int(np.argmax(probabilities)), 2)
        self.assertGreater(probabilities[0], probabilities[1])
        np.testing.assert_allclose(similarities, [0.4, 0.0, 0.9] / np.linalg.norm([0.9, 0.4]))

    def test_softmax_uses_the_scaled_cosine_similarity(self):
        probabilities, similarities = zero_shot(self.clip, self.image, ["cat", "dog"], ["{}"])
        logits = 100 * similarities
        self.assertAlmostEqual(probabilities[0], 1 / (1 + np.exp(logits[1] - logits[0])))

    def test_every_class_is_captioned_with_every_template(self):
        zero_shot(self.clip, self.image, ["cat", "dog"], ENSEMBLE_TEMPLATES)
        self.assertEqual(len(self.clip.texts), 2 * len(ENSEMBLE_TEMPLATES))
        self.assertIn("a close-up photo of a dog.", self.clip.texts)

    def test_figure_has_a_bar_per_class(self):
        figure = prediction_figure(self.image, ["a", "b", "c"], np.array([0.2, 0.5, 0.3]), "template")
        self.assertEqual(figure.mode, "RGB")
        self.assertGreater(figure.width, figure.height)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (32, 32))
        self.addCleanup(setattr, ClipZeroShotTool, "classes", ClipZeroShotTool.classes)
        self.addCleanup(setattr, ClipZeroShotTool, "template", ClipZeroShotTool.template)
        self.addCleanup(setattr, ClipZeroShotTool, "ensemble", ClipZeroShotTool.ensemble)

    def run_tool(self, choice):
        tool = ClipZeroShotTool()
        with mock.patch.object(tool, "ask_classes", return_value=choice), \
             mock.patch.object(clip_tool, "load_clip", return_value=FakeClip()):
            return tool.run(None, self.document)

    def test_cancel_returns_none(self):
        self.assertIsNone(self.run_tool(None))

    def test_reports_the_best_class_and_every_class(self):
        result = self.run_tool((["dog", "cat", "car"], "a photo of a {}.", False))
        self.assertIn("cat", result.message)
        self.assertEqual(result.details["Classes"], 3)
        self.assertIn("1. cat", result.details)
        self.assertIn("3. car", result.details)
        self.assertEqual(ClipZeroShotTool.classes, ["dog", "cat", "car"])   # remembered for the next run

    def test_ensembling_uses_all_templates(self):
        result = self.run_tool((["dog", "cat"], "a photo of a {}.", True))
        self.assertEqual(result.details["Templates"], " | ".join(ENSEMBLE_TEMPLATES))


@unittest.skipUnless(model_is_cached(), "CLIP not downloaded yet (run the tool once) or transformers missing")
class RealModelTests(unittest.TestCase):
    def test_chelsea_is_a_cat(self):
        probabilities, _ = zero_shot(load_clip(), Image.fromarray(data.chelsea()),
                                     ["cat", "dog", "tiger", "rabbit", "fox", "owl"], ["a photo of a {}."])
        self.assertEqual(int(np.argmax(probabilities)), 0)

    def test_coffee_with_ensembling(self):
        labels = ["coffee", "tea", "soup", "hot chocolate"]
        probabilities, _ = zero_shot(load_clip(), Image.fromarray(data.coffee()), labels, ENSEMBLE_TEMPLATES)
        self.assertEqual(labels[int(np.argmax(probabilities))], "coffee")


if __name__ == "__main__":
    unittest.main()
