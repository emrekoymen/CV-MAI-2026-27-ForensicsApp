"""Set 6.1: face detection with YuNet (OpenCV), on the loaded image or on a webcam stream."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .base import ForensicsTool, ToolResult

# Not committed (models/ is ignored); copy face_detection_yunet_2023mar.onnx from the Set 6 materials here
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx"


def make_detector(width, height, score_threshold=0.6):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YuNet model not found: {MODEL_PATH}")
    return cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (width, height), score_threshold, 0.3, 5000)


def detect_faces(detector, bgr):
    """Return one row per face: x, y, w, h, five (x, y) landmarks, score."""
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = detector.detect(bgr)
    return np.empty((0, 15)) if faces is None else faces


def draw_faces(bgr, faces):
    output = bgr.copy()
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, f"Face: {face[14]:.2f}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for px, py in face[4:14].reshape(5, 2).astype(int):
            cv2.circle(output, (px, py), 4, (255, 0, 0), -1)
    return output


def run_webcam(camera=0):
    """Detect faces on a live webcam stream until 'q' is pressed."""
    capture = cv2.VideoCapture(camera)
    detector = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detector = detector or make_detector(frame.shape[1], frame.shape[0])
        cv2.imshow("YuNet face detection (press q to quit)", draw_faces(frame, detect_faces(detector, frame)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


class FaceDetectionTool(ForensicsTool):
    tool_id = "face_detection"
    title = "Face detection (YuNet)"
    category = "Faces"
    description = "Detect faces and their five landmarks with the YuNet CNN detector."

    def run(self, parent, document):
        bgr = cv2.cvtColor(np.array(document.current.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = detect_faces(make_detector(bgr.shape[1], bgr.shape[0]), bgr)
        result = Image.fromarray(cv2.cvtColor(draw_faces(bgr, faces), cv2.COLOR_BGR2RGB))
        return ToolResult(
            image=result,
            message=f"Detected {len(faces)} face(s).",
            details={"Operation": "Face detection (YuNet)", "Faces": len(faces),
                     "Scores": ", ".join(f"{face[14]:.2f}" for face in faces) or "—"},
        )


class WebcamFaceDetectionTool(ForensicsTool):
    tool_id = "webcam_face_detection"
    title = "Face detection (webcam)"
    category = "Faces"
    description = "Open the webcam in a separate window and detect faces live (press q to quit)."
    requires_image = False

    def run(self, parent, document):
        # Own process: OpenCV windows and the Tk main loop do not share a thread well
        subprocess.Popen([sys.executable, "-m", __name__])
        return ToolResult(message="Webcam face detection started in a separate window (press q to quit).")


if __name__ == "__main__":
    run_webcam()
