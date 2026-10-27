"""Overlay twin-projected positions on a real cap_seal frame.
   python overlay.py frame.jpg out.png [--z cap_top|tube_top|tube_base]"""
import json, sys, cv2, numpy as np
from capsealcam import CapSealCam
cam = CapSealCam("cameras/CapSealCam.json"); pos = json.load(open("positions_3d.json"))
img = cv2.imread(sys.argv[1]); out = sys.argv[2]
big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
def P(pts): return cam.project(np.array(pts))[0] * 2
for fx, col in (("tray_good_top", (255, 160, 0)), ("tray_scrap_top", (255, 160, 0))):
    q = P(pos[fx]).astype(int); cv2.polylines(big, [q.reshape(-1, 1, 2)], True, col, 1)
for k, v in pos.items():
    if not k[:1] in "GS" or "cap_top" not in v and "tube_top" not in v: continue
    if k.startswith("S"): continue
    base = P([v["tube_base"]])[0]; top = P([v["cap_top"]])[0]
    rim = P(v["cap_rim"]).astype(int)
    cv2.circle(big, tuple(base.astype(int)), 3, (0, 0, 255), -1)            # red = vial base (hole)
    cv2.polylines(big, [rim.reshape(-1, 1, 2)], True, (0, 255, 0), 1)       # green = cap rim at cap-top height
    cv2.putText(big, k[1:], tuple((top + [-8, -16]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
cv2.imwrite(out, big)
