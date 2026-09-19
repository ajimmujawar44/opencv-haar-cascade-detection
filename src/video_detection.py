"""
video_detection.py
------------------
Real-time Haar Cascade face detection using webcam or video.

Features:
    - Face detection
    - Eye detection
    - Smile detection
    - FPS counter
    - Face counter
    - Optional video recording
    - Resizable display window

Controls:
    Q / q -> Quit
"""

import argparse
import time
from pathlib import Path

import cv2

from detector import DetectionParams, HaarFaceDetector


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
CAPTURE_DIR = OUTPUT_DIR / "webcam_captures"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Haar Cascade face detection."
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video path. Default: 0",
    )

    parser.add_argument(
        "--eyes",
        dest="eyes",
        action="store_true",
        default=True,
        help="Enable eye detection.",
    )

    parser.add_argument(
        "--no-eyes",
        dest="eyes",
        action="store_false",
        help="Disable eye detection.",
    )

    parser.add_argument(
        "--smile",
        action="store_true",
        help="Enable smile detection.",
    )

    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Haar scale factor. Default: 1.1",
    )

    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors. Default: 5",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video.",
    )

    parser.add_argument(
        "--display-width",
        type=int,
        default=1200,
        help="Maximum display width.",
    )

    parser.add_argument(
        "--display-height",
        type=int,
        default=800,
        help="Maximum display height.",
    )

    return parser.parse_args()


def resolve_source(source: str):
    """
    Convert webcam number to integer.

    '0' -> 0
    '1' -> 1
    'video.mp4' -> 'video.mp4'
    """

    source = source.strip()

    if source.isdigit():
        return int(source)

    return source


def resize_for_display(frame, max_width=1200, max_height=800):
    """
    Resize only the displayed frame.

    Detection and saving still use the original frame.
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


def main():

    args = parse_args()

    source = resolve_source(args.source)

    print("\nOpening camera/video...")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(
            f"ERROR: Could not open source -> {args.source}"
        )

    # Give webcam a reasonable resolution.
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    params = DetectionParams(
        scaleFactor=args.scale_factor,
        minNeighbors=args.min_neighbors,
    )

    detector = HaarFaceDetector(params=params)

    # -------------------------------------------------------------
    # Video information
    # -------------------------------------------------------------

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ) or 640

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ) or 480

    fps_input = cap.get(cv2.CAP_PROP_FPS)

    if fps_input <= 0:
        fps_input = 20.0

    print("\n" + "=" * 55)
    print("        HAAR CASCADE FACE DETECTION")
    print("=" * 55)
    print(f"Source       : {args.source}")
    print(f"Resolution   : {frame_width} x {frame_height}")
    print(f"Eyes         : {'ON' if args.eyes else 'OFF'}")
    print(f"Smile        : {'ON' if args.smile else 'OFF'}")
    print("=" * 55)
    print()
    print(">>> Press Q to quit <<<")
    print()

    # -------------------------------------------------------------
    # Video writer
    # -------------------------------------------------------------

    writer = None

    if args.save:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = OUTPUT_DIR / "annotated_video.mp4"

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
                "ERROR: Could not create output video."
            )

        print(f"Saving video -> {output_path}")

    # -------------------------------------------------------------
    # Create OpenCV window
    # -------------------------------------------------------------

    window_name = "Haar Face Detection - Press Q to Quit"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        min(frame_width, args.display_width),
        min(frame_height, args.display_height),
    )

    # -------------------------------------------------------------
    # FPS
    # -------------------------------------------------------------

    previous_time = time.perf_counter()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("\nCould not read frame / end of video.")
                break

            # -----------------------------------------------------
            # Detection
            # -----------------------------------------------------

            frame, num_faces = detector.annotate_frame(
                frame,
                detect_eyes=args.eyes,
                detect_smile=args.smile,
            )

            # -----------------------------------------------------
            # FPS calculation
            # -----------------------------------------------------

            current_time = time.perf_counter()

            elapsed = max(
                current_time - previous_time,
                0.000001,
            )

            fps = 1.0 / elapsed

            previous_time = current_time

            # -----------------------------------------------------
            # Information
            # -----------------------------------------------------

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Faces: {num_faces}",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # -----------------------------------------------------
            # Save original resolution
            # -----------------------------------------------------

            if writer is not None:
                writer.write(frame)

            # -----------------------------------------------------
            # Resize ONLY for display
            # -----------------------------------------------------

            display_frame = resize_for_display(
                frame,
                args.display_width,
                args.display_height,
            )

            cv2.imshow(
                window_name,
                display_frame,
            )

            

            # -----------------------------------------------------
            # KEYBOARD CONTROL
            # -----------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            # q OR Q
            if key == ord("q") or key == ord("Q"):

                print("\nQ pressed.")
                print("Stopping webcam...")
                break

            # ESC also works
            if key == 27:

                print("\nESC pressed.")
                print("Stopping webcam...")
                break

            # -----------------------------------------------------
            # Extra safety:
            # If the OpenCV window is closed using X
            # -----------------------------------------------------

            try:
                window_visible = cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE,
                )

                if window_visible < 1:
                    print("\nWindow closed.")
                    break

            except cv2.error:
                break

    except KeyboardInterrupt:

        print("\nKeyboard interrupt received.")
        print("Stopping webcam...")

    finally:

        print("Releasing camera...")

        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        # Give Windows time to close the OpenCV window.
        cv2.waitKey(1)

        print("Webcam stopped successfully.")


if __name__ == "__main__":
    main()