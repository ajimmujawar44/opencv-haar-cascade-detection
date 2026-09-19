"""
detector.py
-----------
Core Haar Cascade detection engine used by the project.

Supported detections:
    - Face
    - Eyes
    - Smile
    - Full body / pedestrian

The detector keeps all OpenCV Haar Cascade logic in one place so that
image, webcam, video, and pedestrian detection scripts behave consistently.

Author: Azeem Mujavar
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

CASCADE_DIR = Path(__file__).resolve().parent.parent / "cascades"


# ---------------------------------------------------------------------
# Detection configuration
# ---------------------------------------------------------------------

@dataclass
class DetectionParams:
    """
    Parameters used by Haar Cascade face detection.

    scaleFactor:
        How much the image size is reduced at each scale.
        Smaller values can detect more objects but are slower.

    minNeighbors:
        Number of neighboring detections required to keep a result.
        Higher values reduce false positives.

    minSize:
        Minimum face size to detect.
    """

    scaleFactor: float = 1.1
    minNeighbors: int = 5
    minSize: Tuple[int, int] = (30, 30)

    def __post_init__(self):
        if self.scaleFactor <= 1.0:
            raise ValueError("scaleFactor must be greater than 1.0.")

        if self.minNeighbors < 0:
            raise ValueError("minNeighbors cannot be negative.")

        if len(self.minSize) != 2:
            raise ValueError("minSize must contain width and height.")

        if self.minSize[0] <= 0 or self.minSize[1] <= 0:
            raise ValueError("minSize values must be greater than 0.")


# ---------------------------------------------------------------------
# Haar Cascade Detector
# ---------------------------------------------------------------------

class HaarFaceDetector:
    """
    Central Haar Cascade detection engine.

    Loads all cascade models once and provides methods for:
        - Face detection
        - Eye detection
        - Smile detection
        - Full-body detection
        - Annotating images/video frames
    """

    def __init__(
        self,
        cascade_dir: Path = CASCADE_DIR,
        params: Optional[DetectionParams] = None,
    ):
        self.cascade_dir = Path(cascade_dir)
        self.params = params or DetectionParams()

        # Load Haar Cascade models
        self.face_cascade = self._load(
            self.cascade_dir / "haarcascade_frontalface_default.xml"
        )

        self.eye_cascade = self._load(
            self.cascade_dir / "haarcascade_eye.xml"
        )

        self.smile_cascade = self._load(
            self.cascade_dir / "haarcascade_smile.xml"
        )

        self.body_cascade = self._load(
            self.cascade_dir / "haarcascade_fullbody.xml"
        )

    # -----------------------------------------------------------------
    # Cascade loading
    # -----------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> cv2.CascadeClassifier:
        """
        Load and validate a Haar Cascade XML file.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"\nCascade file not found:\n{path}\n\n"
                f"Run:\n"
                f"python src/download_cascades.py\n"
                f"to download the required cascade files."
            )

        classifier = cv2.CascadeClassifier(str(path))

        if classifier.empty():
            raise IOError(
                f"\nOpenCV could not load the cascade file:\n{path}\n\n"
                f"The XML file may be corrupted or invalid."
            )

        return classifier

    # -----------------------------------------------------------------
    # Detection methods
    # -----------------------------------------------------------------

    def detect_faces(self, gray: np.ndarray):
        """
        Detect faces in a grayscale image.
        """

        if gray is None or gray.size == 0:
            return ()

        if len(gray.shape) != 2:
            raise ValueError("detect_faces() expects a grayscale image.")

        p = self.params

        return self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=p.scaleFactor,
            minNeighbors=p.minNeighbors,
            minSize=p.minSize,
        )

    def detect_eyes(self, gray_roi: np.ndarray):
        """
        Detect eyes inside a face region.
        """

        if gray_roi is None or gray_roi.size == 0:
            return ()

        return self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(15, 15),
        )

    def detect_smiles(self, gray_roi: np.ndarray):
        """
        Detect smiles inside a face region.
        """

        if gray_roi is None or gray_roi.size == 0:
            return ()

        return self.smile_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(20, 10),
        )

    def detect_bodies(self, gray: np.ndarray):
        """
        Detect full bodies / pedestrians.
        """

        if gray is None or gray.size == 0:
            return ()

        return self.body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

    # -----------------------------------------------------------------
    # High-level frame annotation
    # -----------------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        detect_eyes: bool = True,
        detect_smile: bool = False,
    ):
        """
        Detect faces and optionally eyes/smiles.

        Detection is performed on the original frame resolution.

        Returns:
            annotated_frame
            number_of_faces
        """

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        if len(frame.shape) != 3:
            raise ValueError(
                "annotate_frame() expects a BGR color image."
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Slightly improve Haar detection under different lighting.
        gray = cv2.equalizeHist(gray)

        faces = self.detect_faces(gray)

        for (x, y, w, h) in faces:

            # ---------------------------------------------------------
            # Face rectangle
            # ---------------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2,
            )

            # Keep text inside image boundaries.
            text_y = max(y - 10, 20)

            cv2.putText(
                frame,
                "Face",
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # ---------------------------------------------------------
            # Region of Interest
            # ---------------------------------------------------------

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # ---------------------------------------------------------
            # Eye detection
            # ---------------------------------------------------------

            if detect_eyes:

                eyes = self.detect_eyes(roi_gray)

                for (ex, ey, ew, eh) in eyes:

                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        roi_color,
                        "Eye",
                        (ex, max(ey - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # ---------------------------------------------------------
            # Smile detection
            # ---------------------------------------------------------

            if detect_smile:

                smiles = self.detect_smiles(roi_gray)

                for (sx, sy, sw, sh) in smiles:

                    cv2.rectangle(
                        roi_color,
                        (sx, sy),
                        (sx + sw, sy + sh),
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        roi_color,
                        "Smile",
                        (sx, max(sy - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        return frame, len(faces)