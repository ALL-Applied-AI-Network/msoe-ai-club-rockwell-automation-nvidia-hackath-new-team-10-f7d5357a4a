"""Twin -> runtime position map for the Good fixture: where G01-G30 land in the real 640x480 cap_seal image.
Run once per twin / camera file (needs usd-core and the twin); the runtime then needs only numpy + OpenCV.

    python export_position_map.py <twin_dir> [out.json]      (default out: position_map_G.json next to this file)

Per position, all projected through CapSealCam (capsealcam.py, = the twin's apply_lens.py):
  cap_center_px       centroid of the seated cap silhouette (same as registration.py / compare_gate.py PRED)
  cap_hull_px         seated cap outline
  hole_center_px      vial base (bottom of the vial, below the plate)
  plate_hole_center_px  vial axis meeting the plate top = where an empty hole opening shows
  tube_top_center_px  open vial mouth (what an uncapped vial shows)
  anchor_bbox_px      box around tube + cap meshes (same formula as make_variants.ANCHOR, used for crops)
No real-image offset is applied here; positions.py adds the DEV-registered offset at runtime.
"""
import datetime, hashlib, json, os, sys
import cv2, numpy as np
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from capsealcam import CapSealCam
from twin_positions import load as load_anchors
from positions import layout_problems

TUBE_R = 0.57                                      # inches, as in make_variants.py


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def r2(a):
    return np.round(np.asarray(a, float), 2).tolist()


def export(twin, out):
    station = os.path.join(twin, "stations", "tray_inspection", "tray_inspection.usdc")
    cam_json = os.path.join(twin, "cameras", "CapSealCam.json")
    cam = CapSealCam(cam_json); st = Usd.Stage.Open(station); xc = UsdGeom.XformCache()
    anchors = load_anchors(station)

    def mesh_world(path):                          # = make_variants.mesh_world
        prim = st.GetPrimAtPath(path); mesh = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Mesh)][0]
        P = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get()); M = np.array(xc.GetLocalToWorldTransform(mesh))
        return (np.c_[P, np.ones(len(P))] @ M)[:, :3]

    px = lambda P: cam.project(np.atleast_2d(P))[0]
    plate_z = float(np.mean(np.array(anchors["tray_good_top"])[:, 2]))
    positions = []
    for i in range(1, 31):
        pid = f"G{i:02d}"; a = anchors[pid]
        cap_uv, tube_uv = px(mesh_world(f"/TrayInspection/{pid}_cap")), px(mesh_world(f"/TrayInspection/{pid}_tube"))
        hull = cv2.convexHull(cap_uv.astype(np.float32)).reshape(-1, 2); m = cv2.moments(hull)
        base, axis = np.array(a["tube_base"]), np.array(a["tube_axis"])
        plate_hole = base + axis * (plate_z - base[2]) / axis[2]
        both = np.vstack([tube_uv, cap_uv])
        x0, y0 = np.floor(both.min(0)).astype(int); x1, y1 = np.ceil(both.max(0)).astype(int)
        positions.append({"position": pid, "index": i,
                          "cap_center_px": r2([m["m10"] / m["m00"], m["m01"] / m["m00"]]),
                          "cap_hull_px": r2(hull),
                          "hole_center_px": r2(px(a["tube_base"])[0]),
                          "plate_hole_center_px": r2(px(plate_hole)[0]),
                          "tube_top_center_px": r2(px(a["tube_top"])[0]),
                          "anchor_bbox_px": [int(x0), int(y0), int(x1), int(y1)]})
    out_map = {"meta": {"generated_by": "export_position_map.py", "date": datetime.date.today().isoformat(),
                        "twin_dir": os.path.basename(os.path.normpath(twin)),
                        "station_sha256": sha256(station), "camera_json_sha256": sha256(cam_json),
                        "image": "lens-applied 640x480 cap_seal frame", "offset_applied": None,
                        "note": "pure twin projection; the real-image offset lives in registration_dev.json"},
               "image_size": [cam.W, cam.H],
               "plate_polygon_px": r2(px(np.array(anchors["tray_good_top"]))),
               "positions": positions}
    problems = layout_problems(out_map)
    out_map["meta"]["layout_check"] = problems or "ok: G01-G05 rightmost column top->bottom, columns step leftwards"
    json.dump(out_map, open(out, "w"), indent=1)
    return out_map, problems


if __name__ == "__main__":
    twin = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "position_map_G.json")
    m, problems = export(twin, out)
    P = m["positions"]
    print(f"{len(P)} positions -> {out}")
    for p in (P[0], P[4], P[5], P[29]):
        print(f"  {p['position']}: cap {p['cap_center_px']}  hole {p['hole_center_px']}  box {p['anchor_bbox_px']}")
    print("layout check:", m["meta"]["layout_check"])
    sys.exit(1 if problems else 0)
