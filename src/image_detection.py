"""
image_detection.py
-------------------
Detect one or many faces (optionally eyes / smiles) in a still image.

Examples
--------
Detect faces only, show the result on screen:
    python src/image_detection.py --input sample_media/images/group_photo.jpg

Detect faces + eyes + smiles, save the annotated result instead of showing it
(useful on servers / CI with no display):
    python src/image_detection.py -i photo.jpg --eyes --smile --save --no-show

Tune sensitivity for a crowd photo with small faces:
    python src/image_detection.py -i crowd.jpg --scale-factor 1.05 --min-neighbors 4
"""

import argparse
import sys
from pathlib import Path

import cv2

from detector import DetectionParams, HaarFaceDetector

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Haar cascade face detection on a single image.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image.")
    parser.add_argument("--eyes", action="store_true", help="Also detect eyes inside each face.")
    parser.add_argument("--smile", action="store_true", help="Also detect smiles inside each face.")
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--save", action="store_true", help="Save annotated image to /outputs.")
    parser.add_argument("--no-show", action="store_true", help="Don't open a display window.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.input)

    if not image_path.exists():
        sys.exit(f"Error: image not found -> {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"Error: OpenCV could not read the image (unsupported format?) -> {image_path}")

    params = DetectionParams(scaleFactor=args.scale_factor, minNeighbors=args.min_neighbors)
    detector = HaarFaceDetector(params=params)

    annotated, num_faces = detector.annotate_frame(
        image, detect_eyes=args.eyes, detect_smile=args.smile
    )

    print(f"Faces detected: {num_faces}")

    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / f"annotated_{image_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved annotated image -> {out_path}")

    if not args.no_show:

    # Resize image only for display
            display_image = annotated.copy()

    max_width = 1200
    max_height = 800

    height, width = display_image.shape[:2]

    scale = min(max_width / width, max_height / height, 1.0)

    new_width = int(width * scale)
    new_height = int(height * scale)

    display_image = cv2.resize(
        display_image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(
        f"Face Detection - {num_faces} face(s) found",
        display_image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
