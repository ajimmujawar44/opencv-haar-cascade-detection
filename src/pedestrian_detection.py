"""
pedestrian_detection.py
-----------------------
Full-body / pedestrian detection using the OpenCV Haar Cascade.

Example:
    python src/pedestrian_detection.py \
        --source Sample_media/Video/vtest.avi

Save result:
    python src/pedestrian_detection.py \
        --source Sample_media/Video/vtest.avi \
        --save

Press 'q' to quit.
"""

import argparse
import time
from pathlib import Path

import cv2

from detector import HaarFaceDetector


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Haar Cascade pedestrian detection."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to a video file.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video to outputs/.",
    )

    parser.add_argument(
        "--display-width",
        type=int,
        default=1200,
        help="Maximum display width. Default: 1200.",
    )

    parser.add_argument(
        "--display-height",
        type=int,
        default=800,
        help="Maximum display height. Default: 800.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Display resizing
# ---------------------------------------------------------------------

def resize_for_display(
    frame,
    max_width=1200,
    max_height=800,
):
    """
    Resize a frame only for screen display.

    The original frame is NOT modified.
    """

    height, width = frame.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    if scale < 1.0:
        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    return frame


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    video_path = Path(args.source)

    if not video_path.exists():
        raise SystemExit(
            f"Error: video not found -> {video_path}"
        )

    # -------------------------------------------------------------
    # Open video
    # -------------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise SystemExit(
            f"Error: could not open video -> {video_path}"
        )

    # -------------------------------------------------------------
    # Detector
    # -------------------------------------------------------------

    detector = HaarFaceDetector()

    # -------------------------------------------------------------
    # Video information
    # -------------------------------------------------------------

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps_input = cap.get(cv2.CAP_PROP_FPS)

    if fps_input <= 0:
        fps_input = 20.0

    print()
    print("=" * 60)
    print("Pedestrian Detection")
    print("=" * 60)
    print(f"Video       : {video_path}")
    print(f"Resolution  : {frame_width} x {frame_height}")
    print(f"Input FPS   : {fps_input:.1f}")
    print("=" * 60)

    # -------------------------------------------------------------
    # Optional video writer
    # -------------------------------------------------------------

    writer = None

    if args.save:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = OUTPUT_DIR / "annotated_pedestrians.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps_input,
            (frame_width, frame_height),
        )

        if not writer.isOpened():
            cap.release()
            raise SystemExit(
                "Error: could not create output video."
            )

        print(f"Saving result -> {output_path}")

    # -------------------------------------------------------------
    # FPS calculation
    # -------------------------------------------------------------

    previous_time = time.perf_counter()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("\nEnd of video.")
                break

            # -----------------------------------------------------
            # Detection
            # -----------------------------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            gray = cv2.equalizeHist(gray)

            bodies = detector.detect_bodies(gray)

            # -----------------------------------------------------
            # Draw detections
            # -----------------------------------------------------

            for (x, y, w, h) in bodies:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "Person",
                    (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # -----------------------------------------------------
            # FPS
            # -----------------------------------------------------

            current_time = time.perf_counter()

            elapsed = max(
                current_time - previous_time,
                1e-6,
            )

            fps = 1.0 / elapsed

            previous_time = current_time

            # -----------------------------------------------------
            # Information overlay
            # -----------------------------------------------------

            cv2.putText(
                frame,
                f"Pedestrians: {len(bodies)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # -----------------------------------------------------
            # Save ORIGINAL resolution
            # -----------------------------------------------------

            if writer is not None:
                writer.write(frame)

            # -----------------------------------------------------
            # Display resized copy
            # -----------------------------------------------------

            display_frame = resize_for_display(
                frame,
                args.display_width,
                args.display_height,
            )

            cv2.imshow(
                "Pedestrian Detection - Press Q to quit",
                display_frame,
            )

            # -----------------------------------------------------
            # Quit
            # -----------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nExiting...")
                break

    finally:

        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()