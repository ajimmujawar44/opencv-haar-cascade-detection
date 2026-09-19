"""
download_cascades.py
--------------------
Downloads the required Haar Cascade XML files from the official
OpenCV GitHub repository.

Run:
    python src/download_cascades.py

The files will be stored inside:
    cascades/
"""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

CASCADE_DIR = Path(__file__).resolve().parent.parent / "cascades"

BASE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/"
)

CASCADES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml",
    "haarcascade_smile.xml",
    "haarcascade_fullbody.xml",
]


# ---------------------------------------------------------------------
# Download function
# ---------------------------------------------------------------------

def download_file(url: str, destination: Path) -> bool:
    """
    Download one cascade file.

    Returns:
        True  -> successful
        False -> failed
    """

    try:
        request = Request(
            url,
            headers={
                "User-Agent": "OpenCV-Haar-Cascade-Project"
            },
        )

        with urlopen(request, timeout=30) as response:
            data = response.read()

        if not data:
            print(f"[error] Empty file received: {destination.name}")
            return False

        destination.write_bytes(data)

        return True

    except HTTPError as error:
        print(
            f"[error] HTTP error {error.code} while downloading "
            f"{destination.name}"
        )
        return False

    except URLError as error:
        print(
            f"[error] Network error while downloading "
            f"{destination.name}: {error.reason}"
        )
        return False

    except Exception as error:
        print(
            f"[error] Could not download {destination.name}: {error}"
        )
        return False


# ---------------------------------------------------------------------
# Download all cascades
# ---------------------------------------------------------------------

def download_all():
    """
    Download every required Haar Cascade XML file.
    """

    CASCADE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OpenCV Haar Cascade Downloader")
    print("=" * 60)

    success_count = 0

    for filename in CASCADES:

        destination = CASCADE_DIR / filename

        if destination.exists() and destination.stat().st_size > 0:
            print(f"[skip]  {filename} already exists")
            success_count += 1
            continue

        url = BASE_URL + filename

        print(f"[fetch] {filename}")

        if download_file(url, destination):
            print(f"[done]  Saved -> {destination}")
            success_count += 1

    print()
    print("=" * 60)
    print(
        f"Completed: {success_count}/{len(CASCADES)} cascade files available."
    )
    print(f"Cascade directory: {CASCADE_DIR}")
    print("=" * 60)

    if success_count != len(CASCADES):
        raise SystemExit(
            "\nSome cascade files could not be downloaded.\n"
            "Check your internet connection and run the command again."
        )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    download_all()