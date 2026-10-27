"""Pixel displacement of a crooked / lifted cap and of a leaning vial, from real twin geometry + CapSealCam."""
import numpy as np, cv2
from pxr import Usd, UsdGeom
from capsealcam import CapSealCam
import os
T = os.environ.get('TRAY_USDC', 'stations/tray_inspection/tray_inspection.usdc')
cam = CapSealCam("cameras/CapSealCam.json"); st = Usd.Stage.Open(T); xc = UsdGeom.XformCache()
cap_local = np.array(UsdGeom.Mesh(st.GetPrimAtPath('/TrayInspection/G01_cap/vial_cap')).GetPointsAttr().Get())  # inches
R_CAP, H_CAP, H_TUBE = 0.668, 0.75, 4.1

def rot_about(pts, pivot, axis, deg):
    a = np.radians(deg); axis = axis / np.linalg.norm(axis)
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3) + np.sin(a)*K + (1-np.cos(a))*K@K
    return (pts - pivot) @ R.T + pivot

def to_px(pid, pts_local_cap):
    M = np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(f'/TrayInspection/{pid}_cap/vial_cap')))
    W = np.c_[pts_local_cap, np.ones(len(pts_local_cap))] @ M
    return cam.project(W[:, :3])[0]

def mask(uv, shape=(480, 640)):
    m = np.zeros(shape, np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)*8).astype(np.int32).reshape(-1,2)//1, 1, shift=3); return m

def compare(pid, new_local):
    a, b = to_px(pid, cap_local), to_px(pid, new_local)
    ma, mb = mask(a), mask(b)
    inter, union = (ma & mb).sum(), (ma | mb).sum()
    ca, cb = a.mean(0), b.mean(0)
    top_a = to_px(pid, np.array([[0,0,H_CAP]]))[0]
    return dict(centroid_shift=np.linalg.norm(cb-ca), max_vertex=np.max(np.linalg.norm(b-a,axis=1)), iou=inter/union,
                changed_px=int((ma ^ mb).sum()), cap_px=int(ma.sum()))

dirs = {"toward camera axis": None, "+x": np.array([1,0,0.]), "+y": np.array([0,1,0.])}
print("Cap footprint in image: ~%d px area; diameter ~%.1f px" % (mask(to_px('G13', cap_local)).sum(), np.ptp(to_px('G13', cap_local)[:,0])))
print("\nA) CAP TILTED about one rim point (one side seated, other side lifted)")
print(f"{'pos':4} {'tilt':>5} {'dir':>4} {'centroid':>9} {'max edge':>9} {'IoU':>5} {'px changed':>10}")
rows = []
for pid in ("G13", "G01", "G30"):
    for deg in (3, 5, 10, 15, 20, 30):
        worst = None
        for dname, d in (("+x", np.array([1,0,0.])), ("+y", np.array([0,1,0.])), ("-x", np.array([-1,0,0.])), ("-y", np.array([0,-1,0.]))):
            pivot = np.array([d[0]*R_CAP, d[1]*R_CAP, 0.0]); axis = np.cross([0,0,1.], d)
            r = compare(pid, rot_about(cap_local, pivot, axis, deg)); r["dir"] = dname
            if worst is None or r["centroid_shift"] < worst["centroid_shift"]: worst = r   # report the LEAST visible direction
        rows.append((pid, deg, worst))
        print(f"{pid:4} {deg:4d}° {worst['dir']:>4} {worst['centroid_shift']:8.1f}px {worst['max_vertex']:8.1f}px {worst['iou']:5.2f} {worst['changed_px']:7d}/{worst['cap_px']}")
print("\nB) CAP LIFTED straight up (not seated), no tilt")
for mm in (1, 2, 3, 5):
    r = compare("G13", cap_local + [0, 0, mm/25.4])
    print(f"  lift {mm} mm: centroid {r['centroid_shift']:.2f}px  max edge {r['max_vertex']:.2f}px  IoU {r['iou']:.2f}")
print("\nC) WHOLE VIAL LEANING about its base (not a defect by itself), cap stays seated")
for deg in (1, 2, 3, 5):
    pivot = np.array([0, 0, -(H_TUBE - 0.5)])   # tube base is ~3.6 in below cap base (cap sits at tube_top - 0.5in)
    r = compare("G13", rot_about(cap_local, pivot, np.array([1,0,0.]), deg))
    print(f"  lean {deg}°: centroid {r['centroid_shift']:.1f}px  max edge {r['max_vertex']:.1f}px  IoU {r['iou']:.2f}")
