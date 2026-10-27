"""Twin-only fix for the gate WORK items (GATE_v2.0.2.md §5): a thin overlay twin over the untouched station file.

    python twin_fix.py <TWIN dir> <OUT dir>            write OUT, an overlay twin (OUT must sit next to TWIN)
    python twin_fix.py --check <TWIN dir> <OUT dir>    compose both stations; only the fixed values may differ

OUT/stations/tray_inspection/tray_inspection.usdc sublayers the original station by relative path and overrides:
  F1a  both RectLights x 2^0.75 (+0.75 EV), colours unchanged: the matte white table panel (B4 region) then matches
       the pooled DEV runs 1-3 luminance (L* 70.0) under the renderer's ACES-like tone curve
  F1b  tw_cap_yellow diffuseColor blue 0.10 -> 0.00: after F1a the tone curve still desaturates the caps (cap b* low,
       HSV saturation under the A1 detector's floor); R and G are unchanged
cameras/ and tools/ are copied so OUT works as a twin folder (render_keep.py --twin OUT, run_gate.sh TWIN=OUT).
The gate kit is not touched (MANIFEST.sha256 unchanged). Values were chosen on gate run 1 and DEV runs 1-3 only.
"""
import argparse, hashlib, json, os, shutil, sys
from datetime import datetime, timezone
import numpy as np
from pxr import Usd, UsdLux, UsdShade, Sdf, Gf

FIX_ID = "work1"
EV = 0.75
LIGHTS = ("/TrayInspection/key_overhead/key_overhead", "/TrayInspection/fill_side/fill_side")
CAP_SHADER = "/TrayInspection/_materials/tw_cap_yellow/Principled_BSDF"
CAP_RGB_BASE, CAP_RGB = (0.95, 0.90, 0.10), (0.95, 0.90, 0.00)
STATION_REL = os.path.join("stations", "tray_inspection", "tray_inspection.usdc")
ROOT_INFO = ("defaultPrim", "upAxis", "metersPerUnit")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_changes():
    return {(p, "inputs:intensity") for p in LIGHTS} | {(CAP_SHADER, "inputs:diffuseColor")}


def build(twin, out):
    twin, out = os.path.abspath(twin), os.path.abspath(out)
    if os.path.dirname(twin) != os.path.dirname(out):
        sys.exit("OUT must sit next to TWIN (the sublayer path is relative, so the pair must move together)")
    station = os.path.join(twin, STATION_REL); dst = os.path.join(out, STATION_REL)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    src = Sdf.Layer.FindOrOpen(station)
    layer = Sdf.Layer.CreateNew(dst)
    for k in ROOT_INFO:
        layer.pseudoRoot.SetInfo(k, src.pseudoRoot.GetInfo(k))
    layer.subLayerPaths.append(os.path.relpath(station, os.path.dirname(dst)))
    stage = Usd.Stage.Open(layer); stage.SetEditTarget(layer)
    lights = {}
    for p in LIGHTS:
        a = UsdLux.RectLight(stage.GetPrimAtPath(p)).GetIntensityAttr(); base = float(a.Get())
        a.Set(float(base * 2 ** EV)); lights[p] = {"base": base, "fixed": float(a.Get())}
    cap = UsdShade.Shader(stage.GetPrimAtPath(CAP_SHADER)).GetInput("diffuseColor"); base = tuple(cap.Get())
    if not np.allclose(base, CAP_RGB_BASE, atol=1e-6):
        sys.exit(f"unexpected base cap colour {base}: not the twin this fix was calibrated on")
    cap.Set(Gf.Vec3f(*CAP_RGB))
    layer.Save()
    text = layer.ExportToString()
    open(os.path.join(out, "fix_layer.usda"), "w").write(text)
    for d in ("cameras", "tools"):
        shutil.copytree(os.path.join(twin, d), os.path.join(out, d), dirs_exist_ok=True)
    info = {"fix": FIX_ID, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gate": "gate-v2.0.2 §5 WORK items B1 cap, B2 plate, B5 frame luminance (run 1); twin-only, kit unchanged",
            "F1a_lights": {"ev": EV, "factor": 2 ** EV, "intensity": lights},
            "F1b_cap_diffuseColor": {"shader": CAP_SHADER, "base": list(base), "fixed": list(CAP_RGB)},
            "calibration": "gate run 1 renders + DEV runs 1-3 only (arm-clear stills; pooled targets equal the gate's "
                           "reference values: cap Lab 85.1/-8/68, plate 91.4/-5/13, table panel L* 70.0)",
            "not_changed": "geometry, camera, variants, thresholds, plate material, light colours, renderer settings",
            "sublayer": layer.subLayerPaths[0],
            "sha256": {"original_station": sha256(station), "overlay_station": sha256(dst),
                       "fix_layer.usda": hashlib.sha256(text.encode()).hexdigest(),
                       "twin_fix.py": sha256(os.path.abspath(__file__))}}
    json.dump(info, open(os.path.join(out, "FIX.json"), "w"), indent=1)
    print(text)
    print(json.dumps({k: info[k] for k in ("F1a_lights", "F1b_cap_diffuseColor", "sha256")}, indent=1))


def values(stage):
    """{(prim path, attribute or relationship name): value} for every authored-or-composed property."""
    out = {}
    for prim in stage.TraverseAll():
        p = str(prim.GetPath())
        out[(p, "<type>")] = (prim.GetTypeName(), prim.IsActive(), tuple(prim.GetAppliedSchemas()))
        for a in prim.GetAttributes():
            v = a.Get()
            if v is not None and hasattr(v, "__len__") and not isinstance(v, str):
                arr = np.asarray(v)
                v = (hashlib.sha1(arr.tobytes()).hexdigest() if arr.dtype.kind not in "OUS" else tuple(map(str, v)), arr.shape)
            out[(p, a.GetName())] = (v, a.GetNumTimeSamples())
        for r in prim.GetRelationships():
            out[(p, r.GetName())] = tuple(map(str, r.GetTargets()))
    return out


def check(twin, out):
    a = Usd.Stage.Open(os.path.join(os.path.abspath(twin), STATION_REL))
    b = Usd.Stage.Open(os.path.join(os.path.abspath(out), STATION_REL))
    for k in ROOT_INFO:
        assert a.GetRootLayer().pseudoRoot.GetInfo(k) == b.GetRootLayer().pseudoRoot.GetInfo(k), f"root metadata {k} differs"
    va, vb = values(a), values(b)
    only = set(va) ^ set(vb)
    diff = {k for k in set(va) & set(vb) if va[k] != vb[k]}
    print(f"compared {len(va)} properties on {sum(1 for _ in a.TraverseAll())} prims")
    print("only in one stage:", sorted(only) or "none")
    print("values that differ:", sorted(diff) or "none")
    for k in sorted(diff):
        print(f"  {k[0]} {k[1]}: {va[k][0]} -> {vb[k][0]}")
    ok = not only and diff == expected_changes()
    print("CHECK", "OK: only the fixed values differ" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("twin"); ap.add_argument("out"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.twin, a.out) else 1)
    build(a.twin, a.out)
