"""Twin overlay work2 (D040) on top of work1: the floor seen through the G holes gets its own material.

    python twin_fix2.py <WORK1 dir> <OUT dir>            write OUT, an overlay twin over work1 (OUT next to WORK1)
    python twin_fix2.py --check <WORK1 dir> <OUT dir>    compose both; only the new material and one binding may differ

OUT/stations/tray_inspection/tray_inspection.usdc sublayers work1's station by relative path and adds:
  F2a  /TrayInspection/_materials/ignite_hole_floor: UsdPreviewSurface with tw_black_plate's inputs except diffuseColor
       BASE_RGB (warm brown) and roughness BASE_ROUGHNESS. gen_scenes.py's floor block scales the value (and blue) and
       sets the roughness per scene; the 6-scene check render sets the scale (D040)
  F2b  /TrayInspection/tray_good_support/tray_good_support (the G floor under the plate holes) bound to it.
       tw_black_plate stays on the posts and the scrap support
cameras/ and tools/ are copied so OUT works as a twin folder. FIX.json records work1's overlay station as the station this
fix was built on (gen_scenes.py checks it). Real-image targets come from DEV runs 1-3 only (v2_analysis.py, v2_cue.py).
"""
import argparse, hashlib, json, os, shutil, sys
from datetime import datetime, timezone
import numpy as np
from pxr import Usd, UsdShade, Sdf, Gf
from twin_fix import ROOT_INFO, STATION_REL, sha256, values

FIX_ID, BASE_FIX = "work2", "work1"
FLOOR_MESH = "/TrayInspection/tray_good_support/tray_good_support"
SOURCE_MAT = "/TrayInspection/_materials/tw_black_plate"
FLOOR_MAT = "/TrayInspection/_materials/ignite_hole_floor"
BASE_RGB, BASE_ROUGHNESS = (0.30, 0.24, 0.18), 0.75


def expected_changes():
    return {(FLOOR_MESH, "material:binding")}


def build(work1, out):
    work1, out = os.path.abspath(work1), os.path.abspath(out)
    if os.path.dirname(work1) != os.path.dirname(out):
        sys.exit("OUT must sit next to WORK1 (the sublayer path is relative, so the pair must move together)")
    fix1 = json.load(open(os.path.join(work1, "FIX.json")))
    if fix1.get("fix") != BASE_FIX:
        sys.exit(f"{work1} is not the {BASE_FIX} overlay")
    station = os.path.join(work1, STATION_REL); dst = os.path.join(out, STATION_REL)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    src = Sdf.Layer.FindOrOpen(station)
    layer = Sdf.Layer.CreateNew(dst)
    for k in ROOT_INFO:
        layer.pseudoRoot.SetInfo(k, src.pseudoRoot.GetInfo(k))
    layer.subLayerPaths.append(os.path.relpath(station, os.path.dirname(dst)))
    stage = Usd.Stage.Open(layer); stage.SetEditTarget(layer)
    mesh = stage.GetPrimAtPath(FLOOR_MESH)
    bound = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    if not bound or str(bound.GetPath()) != SOURCE_MAT:
        sys.exit(f"{FLOOR_MESH} is bound to {bound.GetPath() if bound else None}, expected {SOURCE_MAT}")
    src_mat = stage.GetPrimAtPath(SOURCE_MAT); src_sh = UsdShade.Shader(stage.GetPrimAtPath(SOURCE_MAT + "/Principled_BSDF"))
    mat = UsdShade.Material.Define(stage, FLOOR_MAT)
    for a in src_mat.GetAttributes():                          # colorSpace etc., as the twin's own materials
        if a.HasAuthoredValue() and not a.GetName().startswith("outputs:"):
            mat.GetPrim().CreateAttribute(a.GetName(), a.GetTypeName()).Set(a.Get())
    sh = UsdShade.Shader.Define(stage, FLOOR_MAT + "/Principled_BSDF"); sh.CreateIdAttr("UsdPreviewSurface")
    inputs = {}
    for i in src_sh.GetInputs():
        if i.Get() is not None:
            sh.CreateInput(i.GetBaseName(), i.GetTypeName()).Set(i.Get()); inputs[i.GetBaseName()] = i.Get()
    sh.GetInput("diffuseColor").Set(Gf.Vec3f(*BASE_RGB)); sh.GetInput("roughness").Set(float(BASE_ROUGHNESS))
    sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(mat)
    layer.Save()
    text = layer.ExportToString()
    open(os.path.join(out, "fix_layer.usda"), "w").write(text)
    for d in ("cameras", "tools"):
        shutil.copytree(os.path.join(work1, d), os.path.join(out, d), dirs_exist_ok=True)
    info = {"fix": FIX_ID, "base_fix": BASE_FIX, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "decision": "D040", "F2a_material": {"path": FLOOR_MAT, "copied_from": SOURCE_MAT, "diffuseColor": list(BASE_RGB),
                                                 "roughness": BASE_ROUGHNESS,
                                                 "other_inputs": {k: (list(v) if hasattr(v, "__len__") else v) for k, v in inputs.items()
                                                                  if k not in ("diffuseColor", "roughness")}},
            "F2b_binding": {"mesh": FLOOR_MESH, "before": SOURCE_MAT, "after": FLOOR_MAT},
            "not_changed": "geometry, camera, lights, every other material and binding, tw_black_plate itself",
            "sublayer": layer.subLayerPaths[0],
            "sha256": {"original_station": sha256(station), "overlay_station": sha256(dst),
                       "base_fix_json": sha256(os.path.join(work1, "FIX.json")),
                       "fix_layer.usda": hashlib.sha256(text.encode()).hexdigest(), "twin_fix2.py": sha256(os.path.abspath(__file__))}}
    json.dump(info, open(os.path.join(out, "FIX.json"), "w"), indent=1)
    print(text)
    print(json.dumps({k: info[k] for k in ("F2a_material", "F2b_binding", "sha256")}, indent=1))


def check(work1, out):
    a = Usd.Stage.Open(os.path.join(os.path.abspath(work1), STATION_REL))
    b = Usd.Stage.Open(os.path.join(os.path.abspath(out), STATION_REL))
    for k in ROOT_INFO:
        assert a.GetRootLayer().pseudoRoot.GetInfo(k) == b.GetRootLayer().pseudoRoot.GetInfo(k), f"root metadata {k} differs"
    va, vb = values(a), values(b)
    only_a, only_b = set(va) - set(vb), set(vb) - set(va)
    diff = {k for k in set(va) & set(vb) if va[k] != vb[k]}
    new_mat = {k for k in only_b if k[0] == FLOOR_MAT or k[0].startswith(FLOOR_MAT + "/")}
    print(f"compared {len(va)} properties on {sum(1 for _ in a.TraverseAll())} prims")
    print("only in work1:", sorted(only_a) or "none")
    print(f"only in work2: {len(new_mat)} properties of {FLOOR_MAT}" + (f", plus {sorted(only_b - new_mat)}" if only_b - new_mat else ""))
    print("values that differ:", sorted(diff) or "none")
    for k in sorted(diff):
        print(f"  {k[0]} {k[1]}: {va[k]} -> {vb[k]}")
    bound = UsdShade.MaterialBindingAPI(b.GetPrimAtPath(FLOOR_MESH)).ComputeBoundMaterial()[0]
    sh = UsdShade.Shader(b.GetPrimAtPath(FLOOR_MAT + "/Principled_BSDF"))
    ok = (not only_a and only_b == new_mat and bool(new_mat) and diff == expected_changes()
          and bound and str(bound.GetPath()) == FLOOR_MAT
          and np.allclose(tuple(sh.GetInput("diffuseColor").Get()), BASE_RGB, atol=1e-6))
    print("CHECK", "OK: only tray_good_support's binding and the new material differ from work1" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work1"); ap.add_argument("out"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.work1, a.out) else 1)
    build(a.work1, a.out)
