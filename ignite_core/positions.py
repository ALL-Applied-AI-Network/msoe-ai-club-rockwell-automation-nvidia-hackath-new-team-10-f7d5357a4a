"""Runtime position map: twin-derived G01-G30 locations -> 30 inspection crops per real cap_seal frame.
numpy + OpenCV only (the twin is read once by export_position_map.py).

Crop geometry is defined here and nowhere else: real crops now and synthetic training crops later must
both come from extract_crops() / crop_for_model(), so a model sees the same framing in both domains.
"""
import json, os
from dataclasses import dataclass, field
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(HERE, "position_map_G.json")
REG_PATH = os.path.join(HERE, "registration_dev.json")
CROP_PAD_PX = 6              # around the tube+cap anchor box; same pad as cut_crops.py (gate crops)
MODEL_SIZE = 64              # square side handed to a learned model (ResNet-18 input is resized from this)
# yellow-cap rule from registration.py / compare_gate.py (Oct 13 registration), kept identical
YELLOW_LO, YELLOW_HI = (18, 110, 120), (38, 255, 255)
CAP_AREA = (120, 900)        # px, one cap blob
MATCH_PX = 15                # blob -> nearest predicted cap center


@dataclass
class PositionCrop:
    position: str                       # "G01"
    index: int                          # 1..30 = fill order
    frame: str                          # source file name
    box: tuple                          # (x0, y0, x1, y1) in the full image, x1/y1 exclusive, after offset + pad
    clipped: bool                       # box hit the image border
    cap_center: tuple                   # expected seated-cap center, crop coordinates
    hole_center: tuple                  # expected empty-hole opening (plate top), crop coordinates
    tube_top_center: tuple              # expected open-vial mouth, crop coordinates
    cap_poly: np.ndarray = field(repr=False)   # expected seated-cap outline, crop coordinates
    image: np.ndarray = field(repr=False)      # BGR crop


def load_offset(path=REG_PATH):
    """Global (dx, dy) px correction from the DEV registration; (0, 0) if not registered yet."""
    if not os.path.exists(path):
        return (0.0, 0.0), "none (registration_dev.json missing)"
    r = json.load(open(path))
    return tuple(r["offset_dx_dy_px"]), f"registration_dev.json ({len(r['frames_used'])} DEV frames)"


class PositionMap:
    def __init__(self, path=MAP_PATH, offset=None):
        m = json.load(open(path))
        self.meta, self.size = m["meta"], tuple(m["image_size"])
        self.offset, self.offset_source = (tuple(offset), "given") if offset is not None else load_offset()
        d = np.array(self.offset, float)
        self.positions = sorted(m["positions"], key=lambda p: p["index"])
        self.ids = [p["position"] for p in self.positions]
        self.cap_centers = np.array([p["cap_center_px"] for p in self.positions]) + d
        self.hole_centers = np.array([p["hole_center_px"] for p in self.positions]) + d
        self.tube_top_centers = np.array([p["tube_top_center_px"] for p in self.positions]) + d
        self.plate_hole_centers = np.array([p.get("plate_hole_center_px", p["hole_center_px"]) for p in self.positions]) + d
        self.cap_hulls = [np.array(p["cap_hull_px"]) + d for p in self.positions]
        dx, dy = int(round(d[0])), int(round(d[1]))
        self.anchor_boxes = [(x0 + dx, y0 + dy, x1 + dx, y1 + dy) for x0, y0, x1, y1 in (p["anchor_bbox_px"] for p in self.positions)]
        self.plate = np.array(m["plate_polygon_px"]) + d

    def crop_box(self, i, pad=CROP_PAD_PX):
        x0, y0, x1, y1 = self.anchor_boxes[i]
        W, H = self.size
        b = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        c = (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))
        return c, c != b

    def plate_mask(self, dilate=41):
        """Fixture region (around the 30 predicted caps), as registration.py restricts its blob search."""
        m = np.zeros(self.size[::-1], np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(self.cap_centers.astype(np.int32)).reshape(-1, 2), 255)
        return cv2.dilate(m, np.ones((dilate, dilate), np.uint8))


def extract_crops(img, pmap, frame="", pad=CROP_PAD_PX):
    H, W = img.shape[:2]
    assert (W, H) == pmap.size, f"{frame}: expected a {pmap.size} lens-applied cap_seal frame, got {(W, H)}"
    out = []
    for i, pid in enumerate(pmap.ids):
        (x0, y0, x1, y1), clipped = pmap.crop_box(i, pad)
        o = np.array([x0, y0], float)
        out.append(PositionCrop(pid, i + 1, frame, (x0, y0, x1, y1), clipped,
                                tuple(pmap.cap_centers[i] - o), tuple(pmap.plate_hole_centers[i] - o),
                                tuple(pmap.tube_top_centers[i] - o), pmap.cap_hulls[i] - o, img[y0:y1, x0:x1].copy()))
    return out


def crop_for_model(crop, size=MODEL_SIZE):
    """Fixed square model input: pad the crop to a square (edge pixels) then resize."""
    im = crop.image; h, w = im.shape[:2]; s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(im, t, s - h - t, l, s - w - l, cv2.BORDER_REPLICATE)
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def yellow_mask(img):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), YELLOW_LO, YELLOW_HI)


def cap_blobs(img, pmap):
    """Yellow cap blobs matched to predicted cap centers (registration.py method).
    Returns [(position, centroid (2,), area)], centroids in full-image px."""
    m = yellow_mask(img) & pmap.plate_mask()
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        a = stats[k, cv2.CC_STAT_AREA]
        if not CAP_AREA[0] < a < CAP_AREA[1]:
            continue
        d = np.linalg.norm(pmap.cap_centers - cents[k], axis=1); j = int(d.argmin())
        if d[j] < MATCH_PX:
            out.append((pmap.ids[j], cents[k], int(a)))
    return out


def residuals(img, pmap):
    """Matched cap blob minus predicted cap center, per matched position: {position: (dx, dy)}."""
    i = {p: k for k, p in enumerate(pmap.ids)}
    return {p: tuple(c - pmap.cap_centers[i[p]]) for p, c, _ in cap_blobs(img, pmap)}


def layout_problems(m):
    """The handoff's numbering claim: G01-G05 = rightmost image column, top->bottom; each next five = the next
    column to the left. Also every anchor box must sit inside the image."""
    W, H = m["image_size"]; P = m["positions"]; out = []
    c = np.array([p["cap_center_px"] for p in P]).reshape(6, 5, 2)          # [column, row, uv]
    if not np.all(np.diff(c[:, :, 1], axis=1) > 0):
        out.append("rows not increasing top->bottom inside a column")
    if not np.all(np.diff(c[:, :, 0].mean(1)) < 0):
        out.append("columns not stepping right->left")
    for p in P:
        x0, y0, x1, y1 = p["anchor_bbox_px"]
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            out.append(f"{p['position']} anchor box outside the image")
    return out


def fixture_cell(index):
    """(row, col) in the physical layout: G01-G05 = rightmost column top->bottom, next five one column left."""
    return (index - 1) % 5, 5 - (index - 1) // 5


def contact_sheet(crops, captions=None, colors=None, scale=4):
    """30 crops arranged like the fixture (6 columns x 5 rows), each upscaled with a caption strip."""
    th = max(c.image.shape[0] for c in crops); tw = max(c.image.shape[1] for c in crops)
    cw, ch, strip = tw * scale, th * scale, 16
    sheet = np.full((5 * (ch + strip), 6 * cw, 3), 40, np.uint8)
    for k, c in enumerate(crops):
        r, col = fixture_cell(c.index)
        y, x = r * (ch + strip), col * cw
        im = cv2.resize(c.image, (c.image.shape[1] * scale, c.image.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        sheet[y + strip:y + strip + im.shape[0], x:x + im.shape[1]] = im
        cap = captions[k] if captions else c.position
        col_ = colors[k] if colors else (255, 255, 255)
        cv2.putText(sheet, cap, (x + 3, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_, 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x, y), (x + cw - 1, y + ch + strip - 1), (90, 90, 90), 1)
    return sheet


def fixture_window(pmap, margin=18):
    """(x0, y0, x1, y1) around the G plate, for zoomed debug / operator views."""
    W, H = pmap.size
    (x0, y0), (x1, y1) = pmap.plate.min(0) - margin, pmap.plate.max(0) + margin
    return max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))


def draw_positions(img, pmap, blobs=(), scale=3):
    """Debug view: anchor boxes (green), expected cap center (red), expected empty-hole center (blue),
    matched yellow blob centroids (yellow cross), G labels. Returns the zoomed fixture region."""
    x0, y0, x1, y1 = fixture_window(pmap)
    big = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    S = lambda u, v: (int(round((u - x0) * scale)), int(round((v - y0) * scale)))
    for i, pid in enumerate(pmap.ids):
        bx0, by0, bx1, by1 = pmap.anchor_boxes[i]
        cv2.rectangle(big, S(bx0, by0), S(bx1, by1), (0, 200, 0), 1)
        cv2.circle(big, S(*pmap.cap_centers[i]), 3, (0, 0, 255), -1)
        cv2.circle(big, S(*pmap.plate_hole_centers[i]), 3, (255, 120, 0), -1)
        cv2.putText(big, pid, S(bx0, by0 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    for _, c, _ in blobs:
        cv2.drawMarker(big, S(*c), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
    return big
