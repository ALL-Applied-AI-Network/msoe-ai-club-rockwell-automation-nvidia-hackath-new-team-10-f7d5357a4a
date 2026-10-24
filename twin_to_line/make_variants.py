"""Author the gate-experiment scenes as small USD override layers over the untouched tray station.
Runs with plain Python + usd-core (no Isaac Sim needed).

    python make_variants.py <TWIN dir> <OUT dir>

Writes OUT/<variant>/scene.usda and OUT/<variant>/labels.json for:
  baseline   station as shipped (all 30 G vials capped), no overrides
  t00        G01 pass, G02 missing cap, G03 cap tilt 0 deg,  G04 no vial, G05-G30 empty
  t10 / t20 / t30   same, G03 cap tilted 10 / 20 / 30 deg (only G03 differs between these four)
  liquid     t20 + liquid in G01 (blue), G02 (orange, uncapped), G03 (blue)
labels.json has, for every G position: state, fill-sequence status, the pixel box of what is drawn
(bbox_px) and the pixel box of the seated vial+cap at that position (anchor_bbox_px, also for empty holes).
"""
import json, os, sys
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capsealcam import CapSealCam

TWIN = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(sys.argv[2])
STATION = os.path.join(TWIN, "stations", "tray_inspection", "tray_inspection.usdc")
cam = CapSealCam(os.path.join(TWIN, "cameras", "CapSealCam.json"))
R_CAP, H_TUBE = 0.668, 4.1                         # inches, from the station meshes
TUBE_R = 0.57
PLACEMENTS = 4                                     # placement attempts so far: G01..G04
RECIPE_RGB = {"blue": (0.03, 0.22, 0.55), "orange": (0.95, 0.42, 0.05)}   # stand-ins for T400 / T200 dyes

base = Usd.Stage.Open(STATION); bxc = UsdGeom.XformCache()

def mesh_world(stage, xc, xform_path):
    prim = stage.GetPrimAtPath(xform_path)
    mesh = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Mesh)][0]
    P = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get()); M = np.array(xc.GetLocalToWorldTransform(mesh))
    return (np.c_[P, np.ones(len(P))] @ M)[:, :3]

def box(uv):
    x0, y0 = np.floor(uv.min(0)).astype(int); x1, y1 = np.ceil(uv.max(0)).astype(int)
    return [int(x0), int(y0), int(x1), int(y1)]

ANCHOR = {f"G{i:02d}": box(cam.project(np.vstack([mesh_world(base, bxc, f"/TrayInspection/G{i:02d}_tube"),
                                                   mesh_world(base, bxc, f"/TrayInspection/G{i:02d}_cap")]))[0])
          for i in range(1, 31)}

def tilt_matrix(local_to_parent, deg, toward=(1.0, 0.0, 0.0)):
    d = np.array(toward, float); d /= np.linalg.norm(d)
    pivot = Gf.Vec3d(*(d * R_CAP)); axis = Gf.Vec3d(*np.cross([0, 0, 1.0], d))
    R = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, deg))
    return Gf.Matrix4d().SetTranslate(-pivot) * R * Gf.Matrix4d().SetTranslate(pivot) * local_to_parent

def liquid_material(stage, name, rgb):
    path = f"/TrayInspection/_materials/liquid_{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.85)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat

def add_liquid(stage, pid, color, fill=0.8):
    """Cheapest credible liquid: a cylinder inside the vial, in the tube's local frame (inches)."""
    cyl = UsdGeom.Cylinder.Define(stage, f"/TrayInspection/{pid}_tube/liquid")
    h = H_TUBE * fill - 0.08
    cyl.CreateRadiusAttr(TUBE_R - 0.04); cyl.CreateHeightAttr(h); cyl.CreateAxisAttr("Z")
    cyl.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.04 + h / 2))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(liquid_material(stage, color, RECIPE_RGB[color]))

def build(name, g03_tilt=None, liquid=None):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "scene.usda")
    if os.path.exists(path):
        os.remove(path)
    layer = Sdf.Layer.CreateNew(path)
    layer.subLayerPaths.append(os.path.relpath(STATION, d))
    stage = Usd.Stage.Open(layer); stage.SetEditTarget(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/TrayInspection"))
    authored = {}
    if g03_tilt is not None:
        authored = {"G01": "pass", "G02": "missing_cap", "G03": "crooked_cap" if g03_tilt > 0 else "pass", "G04": "no_vial"}
        hide = lambda p: UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeInvisible()
        for i in range(5, 31):
            hide(f"/TrayInspection/G{i:02d}_tube"); hide(f"/TrayInspection/G{i:02d}_cap")
        hide("/TrayInspection/G02_cap"); hide("/TrayInspection/G04_tube"); hide("/TrayInspection/G04_cap")
        if g03_tilt > 0:
            cap = UsdGeom.Xformable(stage.GetPrimAtPath("/TrayInspection/G03_cap"))
            M = cap.GetLocalTransformation(); cap.ClearXformOpOrder(); cap.AddTransformOp().Set(tilt_matrix(M, g03_tilt))
    for pid, color in (liquid or {}).items():
        add_liquid(stage, pid, color)
    layer.Save()

    st = Usd.Stage.Open(path); xc = UsdGeom.XformCache()
    vis = lambda p: UsdGeom.Imageable(st.GetPrimAtPath(p)).ComputeVisibility() != UsdGeom.Tokens.invisible
    def tilt(p):
        z = np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(p)))[2, :3]; return float(np.degrees(np.arccos(z[2] / np.linalg.norm(z))))
    labels = []
    for i in range(1, 31):
        pid = f"G{i:02d}"; tp, cp = f"/TrayInspection/{pid}_tube", f"/TrayInspection/{pid}_cap"
        hv, hc = vis(tp), vis(cp)
        rel = round(tilt(cp) - tilt(tp), 2) if hc else None
        state = "no_vial" if not hv else "missing_cap" if not hc else "crooked_cap" if rel > 5 else "pass"
        r = {"position": pid, "state": state, "anchor_bbox_px": ANCHOR[pid]}
        if not hv:
            r["sequence"] = "missed_placement" if i <= PLACEMENTS or g03_tilt is None else "waiting"
        if hv or hc:
            r["bbox_px"] = box(cam.project(np.vstack([mesh_world(st, xc, p) for p, v in ((tp, hv), (cp, hc)) if v]))[0])
        if hc: r["cap_tilt_deg"] = rel
        if pid in (liquid or {}): r["liquid"] = liquid[pid]
        if pid in authored: r["authored"] = authored[pid]
        labels.append(r)
    bad = [r["position"] for r in labels if "authored" in r and r["authored"] != r["state"]]
    assert not bad, f"{name}: labels disagree with authored states at {bad}"
    json.dump({"variant": name, "station": os.path.relpath(STATION, d), "g03_tilt_deg": g03_tilt,
               "liquid": liquid or {}, "placements": PLACEMENTS if g03_tilt is not None else 30, "positions": labels},
              open(os.path.join(d, "labels.json"), "w"), indent=1)
    print(f"{name:9s} {os.path.getsize(path):6d} B  " + "  ".join(f"{r['position']}={r['state']}" + (f"({r['cap_tilt_deg']}°)" if r.get('cap_tilt_deg') else "")
                                                       for r in labels[:4]))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build("baseline")
    for t in (0, 10, 20, 30):
        build(f"t{t:02d}", g03_tilt=t)
    build("liquid", g03_tilt=20, liquid={"G01": "blue", "G02": "orange", "G03": "blue"})
