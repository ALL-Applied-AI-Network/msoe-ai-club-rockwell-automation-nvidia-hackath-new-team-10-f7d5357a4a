"""Twin-to-real registration error: yellow-cap blobs in real frames vs projected twin cap silhouettes.
   python registration.py frame1.jpg frame2.jpg ...   -> per-frame median/max error (px) + overlays"""
import json, sys, cv2, numpy as np
from pxr import Usd, UsdGeom
from capsealcam import CapSealCam
import os
T = os.environ.get('TRAY_USDC', 'stations/tray_inspection/tray_inspection.usdc')
cam = CapSealCam("cameras/CapSealCam.json")
st = Usd.Stage.Open(T); xc = UsdGeom.XformCache()
cap_local = np.array(UsdGeom.Mesh(st.GetPrimAtPath('/TrayInspection/G01_cap/vial_cap')).GetPointsAttr().Get())
pred = {}
for i in range(1, 31):
    p = st.GetPrimAtPath(f'/TrayInspection/G{i:02d}_cap/vial_cap')
    M = np.array(xc.GetLocalToWorldTransform(p))
    W = np.c_[cap_local, np.ones(len(cap_local))] @ M
    uv = cam.project(W[:, :3])[0]
    hull = cv2.convexHull(uv.astype(np.float32))
    m = cv2.moments(hull)
    pred[f"G{i:02d}"] = (np.array([m['m10']/m['m00'], m['m01']/m['m00']]), hull)
gp = np.array([v[0] for v in pred.values()]); names = list(pred)

def yellow_blobs(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 110, 120), (38, 255, 255))
    # restrict to the G fixture region (projected plate, padded)
    plate = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(plate, cv2.convexHull(gp.astype(np.int32)).reshape(-1, 2), 255)
    plate = cv2.dilate(plate, np.ones((41, 41), np.uint8))
    mask &= plate
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    return [(cents[k], stats[k, cv2.CC_STAT_AREA]) for k in range(1, n) if 120 < stats[k, cv2.CC_STAT_AREA] < 900], mask

allerr = []; alloff = []; allnames = []
for f in sys.argv[1:]:
    img = cv2.imread(f); blobs, mask = yellow_blobs(img)
    errs = []; vis = img.copy()
    for c, area in blobs:
        d = np.linalg.norm(gp - c, axis=1); k = d.argmin()
        if d[k] < 15:
            errs.append((names[k], d[k], c - gp[k]))
        cv2.circle(vis, tuple(np.round(c).astype(int)), 2, (0, 0, 255), -1)
    for n_, (c, hull) in pred.items():
        cv2.polylines(vis, [hull.astype(np.int32)], True, (0, 255, 0), 1)
    tag = f.split('/')[-1][9:24]
    if errs:
        e = np.array([x[1] for x in errs]); off = np.array([x[2] for x in errs])
        allerr += list(e); alloff += list(off); allnames += [x[0] for x in errs]
        print(f"{tag}: caps matched {len(errs):2d}/{len(blobs):2d}  median {np.median(e):.1f}px  max {e.max():.1f}px  mean offset (dx,dy)=({off[:,0].mean():+.1f},{off[:,1].mean():+.1f})  worst {max(errs,key=lambda x:x[1])[0]}")
    else:
        print(f"{tag}: no caps matched ({len(blobs)} blobs)")
    cv2.imwrite(f"reg_{tag}.png", cv2.resize(vis[60:380, 170:460], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
if allerr:
    a = np.array(allerr); print(f"\nALL: n={len(a)} median {np.median(a):.2f}px  p90 {np.percentile(a,90):.2f}px  max {a.max():.2f}px")
if alloff:
    off = np.array(alloff); b = np.median(off, 0); r = np.linalg.norm(off - b, axis=1)
    print(f"global bias (median dx,dy) = ({b[0]:+.2f},{b[1]:+.2f}) px")
    print(f"after removing bias: median {np.median(r):.2f}px  p90 {np.percentile(r,90):.2f}px  max {r.max():.2f}px")
    worst = sorted(zip(r, allnames), reverse=True)[:8]
    print("largest residuals:", [(n, round(float(x),1)) for x, n in worst])
    inl = r < 6.0
    print(f"inliers (<6px after bias): {inl.sum()}/{len(r)}  median {np.median(r[inl]):.2f}px  p90 {np.percentile(r[inl],90):.2f}px")
    print("outlier positions (candidate displaced caps / leaning vials):", sorted(set(n for x, n in zip(r, allnames) if x >= 6.0)))
