"""OpenCV image I/O through Python file objects for Unicode paths."""
import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    with open(path, "rb") as stream:
        encoded = stream.read()
    if not encoded:
        return None
    return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), flags)


def imwrite(path, image, params=None):
    from pathlib import Path
    ok, encoded = cv2.imencode(Path(path).suffix, image, [] if params is None else params)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    with open(path, "wb") as stream:
        stream.write(encoded.tobytes())
    return True
