"""Cut labeled crops for the four test positions (or any list) from a 640x480 CapSealCam image.
Empty positions use the seated-vial anchor box, so 'no_vial' gets a crop too.

    python cut_crops.py <frame.png> <labels.json> <out_dir> [G01,G02,G03,G04]
"""
import json, os, sys, cv2

img_path, labels_path, out_dir = sys.argv[1:4]
want = sys.argv[4].split(",") if len(sys.argv) > 4 else ["G01", "G02", "G03", "G04"]
img = cv2.imread(img_path)
assert img is not None and img.shape[:2] == (480, 640), f"{img_path}: expected the lens-applied 640x480 frame"
lab = json.load(open(labels_path)); pos = {r["position"]: r for r in lab["positions"]}
os.makedirs(out_dir, exist_ok=True)
STYLE = {"pass": (0, 170, 0), "missing_cap": (0, 0, 220), "crooked_cap": (0, 120, 255), "no_vial": (200, 0, 200)}
PAD = 6
vis = img.copy(); rows = []
for pid in want:
    r = pos[pid]
    x0, y0, x1, y1 = r.get("bbox_px", r["anchor_bbox_px"])
    ax0, ay0, ax1, ay1 = r["anchor_bbox_px"]                   # union with the anchor so all crops share a frame
    x0, y0, x1, y1 = min(x0, ax0) - PAD, min(y0, ay0) - PAD, max(x1, ax1) + PAD, max(y1, ay1) + PAD
    x0, y0 = max(0, x0), max(0, y0)
    crop = img[y0:y1, x0:x1]
    name = f"{pid}_{r['state']}" + (f"_{int(r['cap_tilt_deg'])}deg" if r.get("cap_tilt_deg") else "") + ".png"
    cv2.imwrite(os.path.join(out_dir, name), crop)
    rows.append(f"{name},{pid},{r['state']},{r.get('sequence','')},{r.get('cap_tilt_deg','')},{x0},{y0},{x1},{y1}")
    cv2.rectangle(vis, (x0, y0), (x1, y1), STYLE[r["state"]], 1)
    cv2.putText(vis, pid, (x0 - 32, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, STYLE[r["state"]], 1)
cv2.imwrite(os.path.join(out_dir, "overlay.png"), vis)
open(os.path.join(out_dir, "crops.csv"), "w").write("file,position,label,sequence,cap_tilt_deg,x0,y0,x1,y1\n" + "\n".join(rows) + "\n")
print(f"{img_path}: {len(rows)} crops -> {out_dir}")
