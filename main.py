"""
main.py
-------
Single entry point for the whole project. Dispatches to the right
detection mode so a user (or your resume reviewer / recruiter running
the repo) only has to remember ONE command.

Usage
-----
    python main.py image    -i sample_media/images/your_photo.jpg
    python main.py webcam
    python main.py video    --source sample_media/videos/vtest.avi
    python main.py pedestrian --source sample_media/videos/vtest.avi

Run `python main.py <mode> -h` to see the options for that mode.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

MODES = {
    "image": "image_detection",
    "webcam": "video_detection",
    "video": "video_detection",
    "pedestrian": "pedestrian_detection",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(__doc__)
        print(f"Available modes: {', '.join(MODES)}")
        sys.exit(1)

    mode = sys.argv.pop(1)
    module_name = MODES[mode]

    if mode == "webcam":
        # webcam mode = video_detection.py with source defaulted to 0
        if "--source" not in sys.argv:
            sys.argv += ["--source", "0"]

    module = __import__(module_name)
    module.main()


if __name__ == "__main__":
    main()
