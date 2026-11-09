import tkinter as tk
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from .base import ForensicsTool, ToolResult
from .k_means import CODEBOOK_FILE, extract_all, split_dataset
from .word_histogram import histogram_result, word_histogram

CLASSIFIER_FILE = "classifier.joblib"   # saved inside the dataset folder, next to the codebook
PREVIEW_WIDTH = 900                     # the histogram is shrunk to this width in the classify dialog
BANNER_HEIGHT = 50                      # strip above the histogram with the prediction
BACKGROUND = (40, 40, 40)

def add_prediction_banner(image, label, probability):
    """Return the image with a strip on top that shows the predicted class and its probability."""
    canvas = Image.new("RGB", (image.width, image.height + BANNER_HEIGHT), BACKGROUND)
    canvas.paste(image, (0, BANNER_HEIGHT))
    font = ImageFont.load_default(size=28)
    text = f"Prediction: {label.capitalize()} ({probability:.0%})"
    ImageDraw.Draw(canvas).text((image.width / 2, BANNER_HEIGHT / 2), text, font=font, fill=(255, 255, 255), anchor="mm")
    return canvas

def image_histograms(kmeans, items):
    """Histogram of every (path, class) pair; images without SIFT features are skipped."""
    histograms, labels = [], []
    for path, label in items:
        try:
            descriptors, _ = extract_all([path])
        except ValueError:  # extract_all found no descriptors at all
            continue
        histograms.append(word_histogram(kmeans, descriptors)[1])
        labels.append(label)
    return np.array(histograms), np.array(labels)

def train_classifier(histograms, labels):
    """One SVM per class against the other two, with the chi2 kernel, which suits histograms.

    class_weight="balanced" compensates for each "rest" being about twice as big as the class itself.
    An SVM only gives scores, so each one is calibrated (Platt scaling) to turn its score into a probability.
    """
    svm = SVC(kernel=chi2_kernel, class_weight="balanced")
    return OneVsRestClassifier(CalibratedClassifierCV(svm, ensemble=False)).fit(histograms, labels)

def load_or_train(folder, kmeans):
    """Load the saved classifier, or train and save it if there is none or the codebook is newer."""
    path = Path(folder) / CLASSIFIER_FILE
    if path.exists() and path.stat().st_mtime > (Path(folder) / CODEBOOK_FILE).stat().st_mtime:
        return joblib.load(path)
    train, test = split_dataset(folder)
    model = train_classifier(*image_histograms(kmeans, train))
    test_histograms, test_labels = image_histograms(kmeans, test)
    saved = {"model": model, "test accuracy": model.score(test_histograms, test_labels)}
    joblib.dump(saved, path)
    return saved


class ClassificationTool(ForensicsTool):
    tool_id = "classification"
    title = "Classification"
    category = "Bag-of-words"
    description = "Show the visual word histogram of the image and classify it with one-vs-rest SVMs (chi2 kernel)."

    def classify_dialog(self, parent, image, model, frequencies):
        """Show the histogram with a Classify button; return (class, probability), or None if closed without it."""
        dialog = tk.Toplevel(parent)
        dialog.title("Classify image")
        dialog.transient(parent)
        dialog.grab_set()

        preview = image.resize((PREVIEW_WIDTH, round(image.height * PREVIEW_WIDTH / image.width)))
        photo = ImageTk.PhotoImage(preview)
        tk.Label(dialog, image=photo).pack(padx=12, pady=12)
        answer = tk.StringVar(value="")
        tk.Label(dialog, textvariable=answer, font=("TkDefaultFont", 14)).pack(pady=(0, 6))
        prediction = None

        def classify():
            nonlocal prediction
            probabilities = model.predict_proba([frequencies])[0]
            best = int(probabilities.argmax())
            prediction = (str(model.classes_[best]), float(probabilities[best]))
            answer.set(f"{prediction[0].capitalize()}: {prediction[1]:.0%}")

        tk.Button(dialog, text="Classify", command=classify).pack(pady=(0, 12))
        dialog.wait_window()
        return prediction

    def run(self, parent, document):
        found = histogram_result(parent, document)
        if found is None:
            return None
        folder, codebook, frequencies, result = found
        classifier = load_or_train(folder, codebook["kmeans"])

        prediction = self.classify_dialog(parent, result.image, classifier["model"], frequencies)
        details = {**result.details, "Operation": "Classification",
            "Test accuracy": round(classifier["test accuracy"], 2)}
        image, message = result.image, result.message
        if prediction is not None:
            details["Predicted class"], details["Probability"] = prediction[0], round(prediction[1], 2)
            image = add_prediction_banner(image, *prediction)
            message = f"Classified as {prediction[0]} with probability {prediction[1]:.0%}"
        return ToolResult(image=image, message=message, details=details)
