"""1920x1080 cap_seal frames -> the DEV 640x480 pixel grid, so the validated 640 path runs unchanged on them
(position_map_G.json + its DEV offset, event gate, rule thresholds, crops). numpy + OpenCV only.

The transform file (frame_warp_1080.json, written by register_1080.py) holds M, the 2x3 similarity taking a 1080
pixel to its DEV 640x480 pixel (OpenCV convention: pixel centres at integer coordinates). apply() first area-averages
the frame down to the nominal scale (cv2.resize INTER_AREA; cv2.warpAffine has no area mode), then applies the small
remaining similarity with INTER_LINEAR. Pixels outside the 1080 view are 0.
"""
import hashlib, json, os
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WARP_PATH = os.path.join(HERE, "frame_warp_1080.json")


def h3(M):
    return np.vstack([np.asarray(M, float), [0.0, 0.0, 1.0]])


def resize_matrix(src, dst):
    """cv2.resize(src size -> dst size) as a 2x3 affine on pixel centres: x_dst = (x_src + 0.5) * dst / src - 0.5."""
    kx, ky = dst[0] / src[0], dst[1] / src[1]
    return np.array([[kx, 0, 0.5 * kx - 0.5], [0, ky, 0.5 * ky - 0.5]])


def warp(img, M, size_out):
    """img resampled into the size_out grid by M (input pixel -> output pixel): INTER_AREA to the nominal scale,
    then the remaining affine with INTER_LINEAR."""
    h, w = img.shape[:2]
    s = np.sqrt(abs(np.linalg.det(np.asarray(M, float)[:, :2])))
    mid = (int(round(w * s)), int(round(h * s)))
    rest = (h3(M) @ np.linalg.inv(h3(resize_matrix((w, h), mid))))[:2]
    small = cv2.resize(img, mid, interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(small, rest, tuple(size_out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class FrameWarp:
    def __init__(self, path=WARP_PATH):
        raw = open(path, "rb").read(); d = json.loads(raw)
        self.path, self.sha256 = path, hashlib.sha256(raw).hexdigest()
        self.size_in, self.size_out = tuple(d["size_in"]), tuple(d["size_out"])
        self.M = np.array(d["matrix_in_to_out"], float)
        a, b = self.M[0, 0], self.M[1, 0]
        if self.M.shape != (2, 3) or not np.allclose(self.M[:, :2], [[a, -b], [b, a]], atol=1e-9):
            raise ValueError(f"{path}: matrix_in_to_out is not a 2x3 similarity")

    def apply(self, img):
        if img.shape[1::-1] != self.size_in:
            raise ValueError(f"expected {self.size_in[0]}x{self.size_in[1]}, got {img.shape[1]}x{img.shape[0]}")
        return warp(img, self.M, self.size_out)

    def info(self):
        return {"file": os.path.relpath(self.path, HERE), "sha256": self.sha256, "size_in": list(self.size_in),
                "size_out": list(self.size_out), "matrix_in_to_out": self.M.tolist()}


def load(path=WARP_PATH):
    """The frozen transform, or None when the file is absent (1080 frames are then wrong_size)."""
    return FrameWarp(path) if os.path.exists(path) else None
