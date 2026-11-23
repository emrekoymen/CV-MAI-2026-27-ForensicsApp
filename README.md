# ForensicsApp course starter

ForensicsApp is the semester project shell for the MSc Computer Vision course.  
Students add a small, demonstrable image-forensics feature as the course moves  
from classical image processing to deep learning and interpretability.

The starter already provides:

- a Tkinter desktop interface with tool, image, and results areas;
- opening and exporting common image formats;
- a fit-to-window image preview;
- undo, redo, and reset-to-original behavior;
- a tiny plug-in-style contract for adding course functionality;
- one analysis example and one image-processing example.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The complete student handout is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

## Set 6 (faces) setup

- Face detection: copy `face_detection_yunet_2023mar.onnx` from the Set 6 materials into `models/`.
  Webcam: the *Face detection (webcam)* tool, or `python -m forensics_app.tools.face_detection` (press q to quit).
- Eigenface builder: pick a folder (default `datasets/lfw/`); if it has no `train/`, LFW is downloaded
  and saved as `train|test/<Person_Name>/<index>.png`. The model (mean, eigenfaces, coefficients, names)
  is saved there as `eigenfaces.npz`.
- Face recognition: open an image from `datasets/lfw/test/...`; the model is found in the folders above it.

## Set 7 (bag of visual words) setup

- Copy the Caltech 101 subset from the Set 7 materials into `datasets/Caltech_101_subset/`,
  one subfolder per class (`butterfly/`, `dalmatian/`, `kangaroo/`).
- K-means clustering: pick that folder (the default); the codebook is built on first use and saved there
  as `codebook.joblib`. Word histogram and classification reuse it, and the classifier is trained on first
  use and saved as `classifier.joblib`. Neither file is committed; delete them to rebuild.

## Set 8 (CNN classification) setup

- Train the models in Colab with `notebooks/set8_train_models.ipynb` (GPU runtime, Run all). It downloads
  `set8_models.zip`; unzip it into `models/` so the files are in `models/set8_models/`:
  `cifar10_small_cnn` (CNN from scratch) and `cifar10_inceptionv3` (transfer learning), each as `.keras`
  (architecture + weights), `.passport.json` (dataset, sample sizes, class names, preprocessing, training
  history, test accuracy) and `.test_features.npz` (feature vectors of the 10,000 CIFAR-10 test images).
  Only the passports are committed; the `.keras` and `.npz` files come from the notebook.
- *Model passport*: shows the passport and the training curves of a model.
- *CNN classification*: classifies the loaded image, preprocessed exactly as the passport says.
- *Feature projection (t-SNE, UMAP)*: projects the saved test-set features to 2D; needs no TensorFlow.

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.
