"""Synthetic scene generator for the Good fixture (CPU only, plain Python + usd-core; no Isaac Sim, no GPU).

    python gen_scenes.py <twin_dir> <out_dir> --config configs/smoke.json [--seed 0] [--limit N]
    python gen_scenes.py <twin_dir> --selftest

Each scene is a small USD override layer over the untouched tray station (same mechanism as the frozen gate's
make_variants.py): hidden vials/caps, crooked caps (tilt about a skirt-rim pivot, or lifted), seated caps shifted
off-centre, leaning vials, and per-scene lighting and material variation. Labels are never copied from the spec:
every scene is reopened as a composed stage and the state of all 30 positions is read back from its transforms and
visibility, then checked against the authored spec and against physical-placement rules. Pixel boxes come from the
same CapSealCam model (capsealcam.py) the runtime position map uses.

Writes (no timestamps, so the same commit + config + seed reproduces every byte):
  OUT/scenes/<scene_id>/scene.usda   override layer; sublayers the station by relative path (keep twin + OUT layout)
  OUT/scenes/<scene_id>/labels.json  30 positions; same schema as make_variants.py, so cut_crops.py reads it
  OUT/manifest.csv                   one row per scene-position (the "image" column is filled by the render step)
  OUT/scenes.csv                     one row per scene: placements, lighting and material values (read back)
  OUT/run.json                       config, seed, git commit, generator/config/twin hashes, USD version
  OUT/SHA256SUMS                     hash of every file above

Training configs (D037) declare "twin_fix": "work1" and must then be run on the overlay twin built by twin_fix.py
(the base lights and cap colour of gate run 2); the overlay is checked against its FIX.json and the original station.
Their randomization ranges come from configs/recipe_dev.json (calibrate_recipe.py, DEV runs 1-3 only): one global
exposure per scene for both lights, a light colour temperature, and a material per visible cap drawn from the DEV cap
groups (saturated / pale: diffuse HSV + specular strength through ior).

What this does NOT do: render, apply the lens, cut crops, or say anything about synthetic-to-real transfer. Labels
are the physical truth of the twin scene ("any cap not fully seated" = crooked_cap); whether a state is visible to the
camera is reported separately (cap_shift_px), and the claimed tilt range is decided by Session A, not here.
"""
import argparse, colorsys, csv, hashlib, io, json, os, shutil, subprocess, sys, tempfile
from dataclasses import dataclass

import cv2
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from capsealcam import CapSealCam
from positions import MAP_PATH, PositionMap, fixture_cell

ROOT = "/TrayInspection"
IN_TO_MM = 25.4
STATES = ("pass", "missing_cap", "crooked_cap", "no_vial")       # the visual classes (cut_crops.py, README contract)
LIGHTS = ("key_overhead", "fill_side")
MATS = {"cap": "tw_cap_yellow", "plate": "tw_alu_brushed", "vial": "tw_vial_plastic"}
FLOOR_MAT = "ignite_hole_floor"                                  # work2 overlay (twin_fix2.py, D040)
FLOOR_KEYS = {"val_scale", "roughness", "blue_scale"}
FLOOR_STREAM = 40                                                # its own random stream: other draws stay as they were
CAP_MESH = "vial_cap"           # each G cap's mesh child; its material binding is authored here
SEATED_TILT_TOL_DEG = 0.05      # readback: a cap within these of its as-shipped pose is fully seated
SEATED_GAP_TOL_IN = 0.01 / IN_TO_MM
GEOM_TOL_IN = 0.005             # physical checks (about 0.13 mm)
NEIGHBOR_IN = 2.6               # positions closer than this (axis to axis) are checked for collisions
SCHEMA = 1


class ConfigError(ValueError):
    pass


class PhysicalError(ValueError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def r6(x):
    return None if x is None else float(round(float(x), 6))


# ----------------------------------------------------------------------------------------------- geometry (row vectors)
def rot_about(pivot, axis, deg):
    """4x4 row-vector matrix rotating by deg about an axis through pivot (same convention as make_variants.tilt_matrix)."""
    pivot, axis = Gf.Vec3d(*map(float, pivot)), Gf.Vec3d(*map(float, axis))
    R = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, float(deg)))
    return Gf.Matrix4d().SetTranslate(-pivot) * R * Gf.Matrix4d().SetTranslate(pivot)


def tilt_local(radius, deg, d_local):
    """Tip a cylinder (base at local z=0) by deg about the base-rim point on side d: the top moves toward d,
    the opposite rim lifts by 2*radius*sin(deg)."""
    d = np.array([d_local[0], d_local[1], 0.0]); d /= np.linalg.norm(d)
    return rot_about(d * radius, np.cross([0, 0, 1.0], d), deg)


def np4(m):
    return np.array(m, float)


def gf4(a):
    return Gf.Matrix4d(*[float(x) for x in np.asarray(a, float).ravel()])


def apply(P, M):
    return (np.c_[P, np.ones(len(P))] @ np4(M))[:, :3]


def axis_angle_deg(M):
    """Angle between local +Z mapped by M and +Z (rotation part of a row-vector matrix)."""
    z = np4(M)[2, :3]
    return float(np.degrees(np.arccos(np.clip(z[2] / np.linalg.norm(z), -1, 1))))


def world_dir_to_local(W, azimuth_deg):
    """World-XY azimuth (0 = +X world, CCW from above) -> unit vector in the prim's local XY plane whose world image
    points exactly along that azimuth when seen from above. Exact even for the twin's vials that stand up to ~1.4 deg
    off vertical (local X/Y rows of W, projected to world XY, form an invertible 2x2)."""
    a = np.radians(azimuth_deg)
    v2 = np.array([np.cos(a), np.sin(a)]) @ np.linalg.inv(np4(W)[:2, :2])
    v = np.array([v2[0], v2[1], 0.0])
    return v / np.linalg.norm(v)


def local_dir_to_world_az(W, v):
    w = np.asarray(v, float) @ np4(W)[:3, :3]
    return float(np.degrees(np.arctan2(w[1], w[0])) % 360)


def ring(radius, z, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[radius * np.cos(a), radius * np.sin(a), np.full(n, z)]


def inside_cylinder(P_local, radius, height, tol=GEOM_TOL_IN):
    r = np.hypot(P_local[:, 0], P_local[:, 1])
    return (r < radius - tol) & (P_local[:, 2] > tol) & (P_local[:, 2] < height - tol)


def bbox(uv):
    x0, y0 = np.floor(uv.min(0)).astype(int); x1, y1 = np.ceil(uv.max(0)).astype(int)
    return [int(x0), int(y0), int(x1), int(y1)]


def hull_centroid(uv):
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2); m = cv2.moments(hull)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


# ---------------------------------------------------------------------------------------------------------------- twin
class Twin:
    """Everything read once from the as-shipped station: per-position seated poses, meshes, plate holes, camera."""

    def __init__(self, twin_dir):
        self.dir = os.path.abspath(os.path.expanduser(twin_dir))
        self.station = os.path.join(self.dir, "stations", "tray_inspection", "tray_inspection.usdc")
        self.cam_json = os.path.join(self.dir, "cameras", "CapSealCam.json")
        for p in (self.station, self.cam_json):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"twin file missing: {p}")
        self.cam = CapSealCam(self.cam_json)
        self.station_sha256, self.camera_sha256 = sha256_file(self.station), sha256_file(self.cam_json)
        self.layer = Sdf.Layer.FindOrOpen(self.station)          # held open: every scene stage reuses it
        self.fix, self.base_station_sha256 = None, self.station_sha256
        fix_json = os.path.join(self.dir, "FIX.json")
        if os.path.isfile(fix_json):                              # overlay twin from twin_fix.py
            self.fix = dict(json.load(open(fix_json)), fix_json_sha256=sha256_file(fix_json))
            base = os.path.normpath(os.path.join(os.path.dirname(self.station), self.layer.subLayerPaths[0]))
            if sha256_file(base) != self.fix["sha256"]["original_station"]:
                raise ValueError(f"overlay sublayer {base} is not the station FIX.json was built on")
            while True:                                           # chained overlays (work2 on work1): the twin's own station
                lyr = Sdf.Layer.FindOrOpen(base); sub = list(lyr.subLayerPaths)       # keep the layer alive while reading
                if not sub:
                    break
                base = os.path.normpath(os.path.join(os.path.dirname(base), sub[0]))
            self.base_station_sha256 = sha256_file(base)
            if self.station_sha256 != self.fix["sha256"]["overlay_station"]:
                raise ValueError(f"{self.station} differs from the overlay recorded in FIX.json")
        st = Usd.Stage.Open(self.layer); xc = UsdGeom.XformCache()
        self.base_intensity = {n: float(UsdLux.RectLight(st.GetPrimAtPath(f"{ROOT}/{n}").GetChildren()[0])
                                        .GetIntensityAttr().Get()) for n in LIGHTS}
        fl = st.GetPrimAtPath(f"{ROOT}/_materials/{FLOOR_MAT}/Principled_BSDF")
        self.floor_base = tuple(UsdShade.Shader(fl).GetInput("diffuseColor").Get()) if fl.IsValid() else None
        cap_sh = UsdShade.Shader(st.GetPrimAtPath(f"{ROOT}/_materials/{MATS['cap']}/Principled_BSDF"))
        self.cap_inputs = [(i.GetBaseName(), i.GetTypeName(), i.Get()) for i in cap_sh.GetInputs()]
        self.cap_hue = colorsys.rgb_to_hsv(*cap_sh.GetInput("diffuseColor").Get())[0]
        self.parent_world = np4(xc.GetLocalToWorldTransform(st.GetPrimAtPath(ROOT)))
        self.ids = [f"G{i:02d}" for i in range(1, 31)]
        self.pos = {}
        for i, pid in enumerate(self.ids, 1):
            d = {"index": i}
            for part in ("tube", "cap"):
                prim = st.GetPrimAtPath(f"{ROOT}/{pid}_{part}")
                meshes = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Mesh)]
                assert len(meshes) == 1, f"{pid}_{part}: expected one mesh child"
                assert not UsdGeom.Xformable(meshes[0]).GetOrderedXformOps(), f"{pid}_{part}: mesh has its own xform"
                d[f"{part}_pts"] = np.array(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get(), float)
                d[f"M_{part}"] = UsdGeom.Xformable(prim).GetLocalTransformation()           # local -> /TrayInspection
                d[f"W_{part}"] = np4(xc.GetLocalToWorldTransform(prim))
            d["C_rel"] = d["M_cap"] * d["M_tube"].GetInverse()                               # cap local -> tube local
            self.pos[pid] = d
        g = self.pos["G01"]
        self.R_TUBE = float(np.hypot(*g["tube_pts"][:, :2].T).max()); self.H_TUBE = float(g["tube_pts"][:, 2].max())
        self.R_CAP = float(np.hypot(*g["cap_pts"][:, :2].T).max()); self.H_CAP = float(g["cap_pts"][:, 2].max())
        for pid, d in self.pos.items():
            assert np.allclose(axis_angle_deg(d["C_rel"]), 0, atol=0.01), f"{pid}: cap not coaxial with its tube"
            d["seat_z"] = float(np4(d["C_rel"])[3, 2])                                       # cap base, tube frame
        self.seat_z = float(np.median([d["seat_z"] for d in self.pos.values()]))
        self._plate_holes(st, xc)
        self._seated_projections()
        cells = {pid: fixture_cell(d["index"]) for pid, d in self.pos.items()}
        base = {pid: d["W_tube"][3, :3] / 0.0254 for pid, d in self.pos.items()}
        self.neighbors = {p: [q for q in self.ids if q != p and np.linalg.norm(base[p] - base[q]) < NEIGHBOR_IN]
                          for p in self.ids}
        self.cells = cells
        self.grid_neighbors = {p: [q for q in self.ids if abs(cells[p][0] - cells[q][0]) + abs(cells[p][1] - cells[q][1]) == 1]
                               for p in self.ids}

    def _plate_holes(self, st, xc):
        """Fit the Good plate's hole around each vial (circle in the seated tube frame) for the lean check."""
        plate = st.GetPrimAtPath(f"{ROOT}/tray_good_top/tray_good_top")
        P = np.array(UsdGeom.Mesh(plate).GetPointsAttr().Get(), float)
        Wp = np.c_[P, np.ones(len(P))] @ np4(xc.GetLocalToWorldTransform(plate))
        for pid, d in self.pos.items():
            L = (Wp @ np.linalg.inv(d["W_tube"]))[:, :3]
            q = L[np.hypot(L[:, 0], L[:, 1]) < 0.9]
            if len(q) < 12:
                raise RuntimeError(f"{pid}: no plate hole found around the vial")
            A = np.c_[2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))]
            cx, cy, c = np.linalg.lstsq(A, (q[:, :2] ** 2).sum(1), rcond=None)[0]
            dist = np.hypot(q[:, 0] - cx, q[:, 1] - cy)
            d["hole"] = {"center": (float(cx), float(cy)), "radius": float(dist.min()),
                         "z": (float(q[:, 2].min()), float(q[:, 2].max()))}

    def _seated_projections(self):
        for pid, d in self.pos.items():
            cap_w = apply(d["cap_pts"], gf4(d["W_cap"])); tube_w = apply(d["tube_pts"], gf4(d["W_tube"]))
            cap_uv = self.cam.project(cap_w)[0]
            d["seat_cap_uv"] = cap_uv
            d["seat_cap_center_px"] = hull_centroid(cap_uv)
            d["anchor_bbox_px"] = bbox(self.cam.project(np.vstack([tube_w, cap_w]))[0])

    # -- physical limits ------------------------------------------------------------------------------------------
    def lean_matrix(self, deg, d_local):
        """Vial pivots on the base edge on side d (base stays on the support); its top moves toward d."""
        return tilt_local(self.R_TUBE, deg, d_local)

    def lean_violations(self, pid, L):
        """Tube surface vs the plate hole wall (at plate bottom and top), and base vs the support plane."""
        h = self.pos[pid]["hole"]; out = []
        for z in h["z"]:
            p = apply(ring(self.R_TUBE, z), L)
            gap = np.hypot(p[:, 0] - h["center"][0], p[:, 1] - h["center"][1]).max() - h["radius"]
            if gap > GEOM_TOL_IN:
                out.append(f"{pid}: vial wall {gap * IN_TO_MM:.2f} mm into the plate hole wall at z={z:.2f} in")
        base = apply(ring(self.R_TUBE, 0.0), L)
        if base[:, 2].min() < -GEOM_TOL_IN:
            out.append(f"{pid}: vial base {-base[:, 2].min() * IN_TO_MM:.2f} mm below the support")
        return out

    def lean_limit_deg(self, pid, azimuth_deg, hi=20.0):
        """Largest lean (deg) toward a world azimuth that keeps the vial inside its plate hole (bisection)."""
        d = world_dir_to_local(self.pos[pid]["W_tube"], azimuth_deg)
        lo, up = 0.0, hi
        if self.lean_violations(pid, gf4(np.eye(4))):
            return 0.0
        for _ in range(40):
            mid = (lo + up) / 2
            (lo, up) = (mid, up) if not self.lean_violations(pid, self.lean_matrix(mid, d)) else (lo, mid)
        return lo

    @property
    def offset_limit_mm(self):
        return (self.R_CAP - self.R_TUBE) * IN_TO_MM            # skirt still around the vial wall

    @property
    def lift_limit_mm(self):
        return (self.H_TUBE - self.seat_z) * IN_TO_MM           # beyond this the cap no longer covers the vial top


# ------------------------------------------------------------------------------------------------------- scene spec
@dataclass
class PosSpec:
    position: str
    state: str                       # pass | missing_cap | crooked_cap | no_vial
    variant: str                     # seated | offset | tilt | lift | uncapped | empty
    sequence: str = None             # no_vial only: missed_placement | waiting
    tilt_deg: float = 0.0
    tilt_dir_deg: float = None       # world azimuth the cap top moves toward
    lift_mm: float = 0.0
    offset_mm: float = 0.0
    offset_dir_deg: float = None
    lean_deg: float = 0.0
    lean_dir_deg: float = None
    lean_limit_deg: float = None
    cap_group: str = None            # per-cap material (training recipe); None = the twin's cap material
    cap_rgb: tuple = None
    cap_ior: float = None
    cap_roughness: float = None


@dataclass
class SceneSpec:
    scene_id: str
    index: int
    seed: int
    placements: int
    positions: list
    lighting: dict = None            # None = the twin's own lights
    materials: dict = None           # None = the twin's own materials
    floor: dict = None               # None = the twin's own hole floor (D040 floor block on the work2 overlay)


VARIANTS = {"pass": ("seated", "offset"), "crooked_cap": ("tilt", "lift"), "missing_cap": ("uncapped",),
            "no_vial": ("empty",)}


def make_pos(pid, state, variant=None, **kw):
    variant = variant or VARIANTS[state][0]
    if variant not in VARIANTS[state]:
        raise ConfigError(f"{pid}: variant {variant} does not belong to {state}")
    return PosSpec(pid, state, variant, **kw)


# ------------------------------------------------------------------------------------------------------------ config
RANDOM_KEYS = {"name", "purpose", "schema", "mode", "n_scenes", "fixture", "placements", "states", "pass_variants",
               "crooked_variants", "lean_fraction", "tilt_deg", "lift_mm", "offset_mm", "lean_deg", "lighting",
               "materials", "twin_fix", "crooked_adjacent", "recipe", "placements_mix", "floor"}
SWEEP_KEYS = {"name", "purpose", "schema", "mode", "fixture", "angles_deg", "directions_deg", "pattern", "lighting",
              "materials", "twin_fix", "recipe", "floor"}
LIGHT_KEYS = {"intensity_scale", "color_temperature_k", "rotate_deg"}
LIGHT_KEYS_EV = LIGHT_KEYS | {"exposure_ev"}            # + one global exposure per scene (training recipe)
MAT_KEYS = {"cap_hue_shift", "cap_sat_scale", "cap_val_scale", "cap_roughness", "plate_val_scale", "plate_roughness",
            "vial_roughness"}
MAT_KEYS_CAPS = {"cap_groups", "cap_roughness", "plate_val_scale", "plate_roughness", "vial_roughness"}   # per-cap materials
CAP_GROUP_KEYS = {"weight", "hue_shift", "sat", "val", "spec_scale"}


def _range(cfg, key, lo_ok=None, hi_ok=None):
    v = cfg[key]
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and v[0] <= v[1]):
        raise ConfigError(f"{key}: expected [lo, hi] with lo <= hi, got {v!r}")
    if (lo_ok is not None and v[0] < lo_ok) or (hi_ok is not None and v[1] > hi_ok):
        raise ConfigError(f"{key}: {v} outside the allowed range [{lo_ok}, {hi_ok}]")
    return v


def _weights(cfg, key, allowed):
    w = cfg[key]
    if not isinstance(w, dict) or set(w) - set(allowed) or any((not isinstance(x, (int, float))) or x < 0 for x in w.values()):
        raise ConfigError(f"{key}: expected non-negative weights for {sorted(allowed)}, got {w!r}")
    if sum(w.values()) <= 0:
        raise ConfigError(f"{key}: weights sum to zero")
    return w


def load_config(path, twin=None):
    raw = open(path, "rb").read()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: not valid JSON ({e})")
    validate_config(cfg, twin)
    return cfg, sha256_bytes(raw)


def validate_config(cfg, twin=None):
    if cfg.get("schema") != SCHEMA:
        raise ConfigError(f"schema must be {SCHEMA}")
    if cfg.get("fixture", "G") != "G":
        raise ConfigError("only the Good fixture (G01-G30) is supported")
    mode = cfg.get("mode")
    keys = RANDOM_KEYS if mode == "random" else SWEEP_KEYS if mode == "tilt_sweep" else None
    if keys is None:
        raise ConfigError("mode must be 'random' or 'tilt_sweep'")
    unknown = set(cfg) - keys
    if unknown:
        raise ConfigError(f"unknown config keys for mode {mode}: {sorted(unknown)}")
    if not isinstance(cfg.get("name"), str) or not cfg["name"].replace("_", "").isalnum():
        raise ConfigError("name: letters, digits and _ only")
    for blk, options in (("lighting", (LIGHT_KEYS, LIGHT_KEYS_EV)), ("materials", (MAT_KEYS, MAT_KEYS_CAPS))):
        b = cfg.get(blk)
        if b is None:
            continue
        if not isinstance(b, dict) or set(b) not in options:
            raise ConfigError(f"{blk}: expected exactly one of {[sorted(o) for o in options]} (or null for the twin's own values)")
        for k in b:
            if k != "cap_groups":
                _range(b, k)
        if "cap_groups" in b:
            _cap_groups(b["cap_groups"])
    tf = cfg.get("twin_fix")
    if tf is not None and not isinstance(tf, str):
        raise ConfigError("twin_fix: the fix id of the overlay twin (e.g. 'work1')")
    if twin is not None and tf != (twin.fix["fix"] if twin.fix else None):
        raise ConfigError(f"config needs twin_fix {tf!r} but the twin has {twin.fix['fix'] if twin.fix else None!r} "
                          "(use the overlay twin from twin_fix.py, or the original twin for configs without twin_fix)")
    if "floor" in cfg:
        fb = cfg["floor"]
        if not isinstance(fb, dict) or set(fb) != FLOOR_KEYS:
            raise ConfigError(f"floor: expected exactly {sorted(FLOOR_KEYS)}")
        _range(fb, "val_scale", 0.0); _range(fb, "roughness", 0.0, 1.0)
        if not isinstance(fb["blue_scale"], (int, float)) or fb["blue_scale"] <= 0:
            raise ConfigError("floor: blue_scale must be a positive number")
        if twin is not None and twin.floor_base is None:
            raise ConfigError(f"floor needs the work2 overlay twin ({FLOOR_MAT}, twin_fix2.py)")
    if "crooked_adjacent" in cfg and not isinstance(cfg["crooked_adjacent"], bool):
        raise ConfigError("crooked_adjacent must be true or false")
    if "recipe" in cfg:
        rc = cfg["recipe"]
        if not (isinstance(rc, dict) and set(rc) == {"file", "sha256"}):
            raise ConfigError("recipe: {'file': <path under twin_to_line>, 'sha256': <hex>}")
        path = os.path.join(HERE, rc["file"])
        if not os.path.isfile(path) or sha256_file(path) != rc["sha256"]:
            raise ConfigError(f"recipe: {rc['file']} is missing or differs from the sha256 the config was written from")
    if mode == "random":
        if not isinstance(cfg.get("n_scenes"), int) or cfg["n_scenes"] < 1:
            raise ConfigError("n_scenes must be a positive integer")
        _range(cfg, "placements", 0, 30)
        if "placements_mix" in cfg:                  # [[weight, [lo, hi]], ...]: fill levels drawn from a mixture
            mix = cfg["placements_mix"]
            if not (isinstance(mix, list) and mix and all(isinstance(c, list) and len(c) == 2 and isinstance(c[0], (int, float))
                                                          and c[0] >= 0 for c in mix)) or sum(c[0] for c in mix) <= 0:
                raise ConfigError("placements_mix: [[weight, [lo, hi]], ...] with non-negative weights")
            for _, rg in mix:
                _range({"placements_mix range": rg}, "placements_mix range", *cfg["placements"])
        _weights(cfg, "states", STATES)
        _weights(cfg, "pass_variants", VARIANTS["pass"])
        _weights(cfg, "crooked_variants", VARIANTS["crooked_cap"])
        if not isinstance(cfg.get("lean_fraction"), (int, float)) or not 0 <= cfg["lean_fraction"] <= 1:
            raise ConfigError("lean_fraction must be in [0, 1]")
        _range(cfg, "tilt_deg", 0, 60); _range(cfg, "lean_deg", 0, 20)
        _range(cfg, "offset_mm", 0, twin.offset_limit_mm if twin else None)
        _range(cfg, "lift_mm", 0, twin.lift_limit_mm if twin else None)
        if cfg["states"].get("crooked_cap", 0) > 0 and cfg["crooked_variants"].get("tilt", 0) > 0 and cfg["tilt_deg"][0] <= 0:
            raise ConfigError("tilt_deg: a crooked tilt must be > 0 deg (0 deg is a seated cap)")
    else:
        a = cfg.get("angles_deg"); dirs = cfg.get("directions_deg")
        if not (isinstance(a, list) and a and all(isinstance(x, (int, float)) and 0 <= x <= 60 for x in a)):
            raise ConfigError("angles_deg: list of angles in [0, 60]")
        if not (isinstance(dirs, list) and dirs and all(isinstance(x, (int, float)) for x in dirs)):
            raise ConfigError("directions_deg: list of world azimuths")
        if cfg.get("pattern") != "checkerboard":
            raise ConfigError("pattern must be 'checkerboard' (no two side-by-side neighbours tilted at once)")


def _cap_groups(groups):
    if not isinstance(groups, dict) or not groups:
        raise ConfigError("cap_groups: {name: {weight, hue_shift, sat, val, spec_scale}}")
    for name, g in groups.items():
        if not isinstance(g, dict) or set(g) != CAP_GROUP_KEYS:
            raise ConfigError(f"cap_groups.{name}: expected exactly {sorted(CAP_GROUP_KEYS)}")
        if not isinstance(g["weight"], (int, float)) or g["weight"] < 0:
            raise ConfigError(f"cap_groups.{name}.weight must be >= 0")
        _range(g, "hue_shift", -0.5, 0.5); _range(g, "sat", 0, 1); _range(g, "val", 0, 1); _range(g, "spec_scale", 0, 20)
    if sum(g["weight"] for g in groups.values()) <= 0:
        raise ConfigError("cap_groups: weights sum to zero")


def ior_for_spec(k):
    """UsdPreviewSurface ior whose normal-incidence reflectance is k x the twin's (ior 1.5 -> F0 = 0.04)."""
    f = (0.04 * k) ** 0.5
    return (1 + f) / (1 - f)


# ----------------------------------------------------------------------------------------------------------- sampling
def _u(rng, r):
    return float(rng.uniform(r[0], r[1])) if r[1] > r[0] else float(r[0])


def _pick(rng, weights):
    keys = sorted(weights); p = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def sample_lighting(rng, L):
    if L is None:
        return None
    if "exposure_ev" in L:                       # one exposure for both lights, then a small per-light jitter
        ev, ct, out = _u(rng, L["exposure_ev"]), _u(rng, L["color_temperature_k"]), {}
        for name in LIGHTS:
            j = _u(rng, L["intensity_scale"])
            out[name] = {"exposure_ev": ev, "jitter": j, "intensity_scale": 2 ** ev * j, "color_temperature_k": ct,
                         "rotate_xy_deg": [_u(rng, L["rotate_deg"]), _u(rng, L["rotate_deg"])]}
        return out
    ct = _u(rng, L["color_temperature_k"])
    return {name: {"intensity_scale": _u(rng, L["intensity_scale"]), "color_temperature_k": ct,
                   "rotate_xy_deg": [_u(rng, L["rotate_deg"]), _u(rng, L["rotate_deg"])]} for name in LIGHTS}


def sample_materials(rng, M):
    if M is None:
        return None
    return {k: _u(rng, M[k]) for k in sorted(MAT_KEYS if "cap_groups" not in M else MAT_KEYS_CAPS - {"cap_groups", "cap_roughness"})}


def sample_cap(rng, M, hue0):
    """One cap's material from the DEV cap groups: diffuse HSV around the twin's cap hue, specular strength as ior."""
    if M is None or "cap_groups" not in M:
        return {}
    name = _pick(rng, {k: g["weight"] for k, g in M["cap_groups"].items()}); g = M["cap_groups"][name]
    dh, s_, v_, k = (_u(rng, g[x]) for x in ("hue_shift", "sat", "val", "spec_scale"))
    rgb = colorsys.hsv_to_rgb((hue0 + dh) % 1.0, s_, v_)
    return {"cap_group": name, "cap_rgb": tuple(r6(x) for x in rgb), "cap_ior": r6(ior_for_spec(k)),
            "cap_roughness": r6(_u(rng, M["cap_roughness"]))}


def scene_rng(seed, index):
    return np.random.default_rng([int(seed), int(index)])          # independent of --limit and generation order


def sample_random_scene(cfg, twin, seed, index):
    rng = scene_rng(seed, index)
    lo, hi = cfg["placements"]
    if "placements_mix" in cfg:
        w = np.array([c[0] for c in cfg["placements_mix"]], float)
        lo, hi = cfg["placements_mix"][int(rng.choice(len(w), p=w / w.sum()))][1]
    K = int(rng.integers(lo, hi + 1))
    specs, crooked = [], set()
    for pid in twin.ids:
        i = twin.pos[pid]["index"]
        if i > K:
            specs.append(make_pos(pid, "no_vial", sequence="waiting")); continue
        state = _pick(rng, cfg["states"])
        if state == "crooked_cap" and not cfg.get("crooked_adjacent", True) and crooked & set(twin.grid_neighbors[pid]):
            state = _pick(rng, {k: w for k, w in cfg["states"].items() if k != "crooked_cap"})   # keep tilted caps apart
        if state == "crooked_cap":
            crooked.add(pid)
        if state == "no_vial":
            specs.append(make_pos(pid, "no_vial", sequence="missed_placement")); continue
        kw = {}
        if state == "pass":
            variant = _pick(rng, cfg["pass_variants"])
            if variant == "offset":
                kw.update(offset_mm=_u(rng, cfg["offset_mm"]), offset_dir_deg=_u(rng, [0, 360]))
        elif state == "crooked_cap":
            variant = _pick(rng, cfg["crooked_variants"])
            if variant == "tilt":
                kw.update(tilt_deg=_u(rng, cfg["tilt_deg"]), tilt_dir_deg=_u(rng, [0, 360]))
            else:
                kw.update(lift_mm=_u(rng, cfg["lift_mm"]))
        else:
            variant = "uncapped"
        if rng.random() < cfg["lean_fraction"]:
            az = _u(rng, [0, 360]); lim = twin.lean_limit_deg(pid, az)
            lo, hi = cfg["lean_deg"]
            if lo > lim:
                raise PhysicalError(f"{pid}: lean_deg lower bound {lo} exceeds the plate-hole limit {lim:.2f} deg "
                                    f"toward {az:.0f} deg")
            kw.update(lean_deg=_u(rng, [lo, min(hi, lim)]), lean_dir_deg=az, lean_limit_deg=lim)
        if state != "missing_cap":
            kw.update(sample_cap(rng, cfg.get("materials"), twin.cap_hue))
        specs.append(make_pos(pid, state, variant, **kw))
    return SceneSpec(f"{cfg['name']}_{index:05d}", index, int(seed), K, specs,
                     sample_lighting(rng, cfg.get("lighting")), sample_materials(rng, cfg.get("materials")))


def sweep_cases(cfg):
    cases = []
    for parity in (0, 1):
        for a in cfg["angles_deg"]:
            for d in (cfg["directions_deg"] if a > 0 else cfg["directions_deg"][:1]):
                cases.append((float(a), float(d), parity))
    return cases


def sweep_scene(cfg, twin, seed, index):
    a, d, parity = sweep_cases(cfg)[index]
    rng = scene_rng(seed, index)
    specs = []
    for pid in twin.ids:
        r, c = twin.cells[pid]
        cap = sample_cap(rng, cfg.get("materials"), twin.cap_hue)
        if (r + c) % 2 == parity and a > 0:
            specs.append(make_pos(pid, "crooked_cap", "tilt", tilt_deg=a, tilt_dir_deg=d, **cap))
        else:
            specs.append(make_pos(pid, "pass", **cap))
    return SceneSpec(f"{cfg['name']}_{index:05d}", index, int(seed), 30, specs,
                     sample_lighting(rng, cfg.get("lighting")), sample_materials(rng, cfg.get("materials")))


def n_scenes(cfg):
    return cfg["n_scenes"] if cfg["mode"] == "random" else len(sweep_cases(cfg))


def sample_floor(fb, base, seed, index):
    """D040 hole floor for one scene, from its own stream (seed, index, FLOOR_STREAM): value x val_scale, blue x
    blue_scale, roughness. Drawn after everything else, so every other per-scene value is unchanged."""
    rng = np.random.default_rng([int(seed), int(index), FLOOR_STREAM])
    v, r = _u(rng, fb["val_scale"]), _u(rng, fb["roughness"])
    rgb = [min(1.0, base[0] * v), min(1.0, base[1] * v), min(1.0, base[2] * v * fb["blue_scale"])]
    return {"val_scale": r6(v), "blue_scale": r6(fb["blue_scale"]), "roughness": r6(r), "rgb": [r6(x) for x in rgb]}


def build_spec(cfg, twin, seed, index):
    spec = (sample_random_scene if cfg["mode"] == "random" else sweep_scene)(cfg, twin, seed, index)
    if cfg.get("floor"):
        spec.floor = sample_floor(cfg["floor"], twin.floor_base, seed, index)
    return spec


# ----------------------------------------------------------------------------------------------------------- authoring
def cap_modification(twin, s):
    """Cap-local change relative to the seated cap: rim-pivot tilt, straight lift, or a lateral seat offset."""
    d = twin.pos[s.position]
    if s.variant == "tilt":
        return tilt_local(twin.R_CAP, s.tilt_deg, world_dir_to_local(d["W_cap"], s.tilt_dir_deg))
    if s.variant == "lift":
        return Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, s.lift_mm / IN_TO_MM))
    if s.variant == "offset":
        v = world_dir_to_local(d["W_cap"], s.offset_dir_deg) * s.offset_mm / IN_TO_MM
        return Gf.Matrix4d().SetTranslate(Gf.Vec3d(*v))
    return Gf.Matrix4d(1.0)


def lean_modification(twin, s):
    if s.lean_deg <= 0 or s.state == "no_vial":
        return Gf.Matrix4d(1.0)
    return twin.lean_matrix(s.lean_deg, world_dir_to_local(twin.pos[s.position]["W_tube"], s.lean_dir_deg))


def set_matrix(stage, path, M):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder(); xf.AddTransformOp().Set(M)


def author_scene(twin, spec, scene_dir):
    os.makedirs(scene_dir, exist_ok=True)
    path = os.path.join(scene_dir, "scene.usda")
    if os.path.exists(path):
        raise FileExistsError(path)
    layer = Sdf.Layer.CreateNew(path)
    layer.subLayerPaths.append(os.path.relpath(twin.station, scene_dir))
    layer.customLayerData = {"ignite_generator": "gen_scenes.py", "ignite_scene_id": spec.scene_id}
    stage = Usd.Stage.Open(layer); stage.SetEditTarget(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT))
    for s in spec.positions:
        d = twin.pos[s.position]; tube, cap = f"{ROOT}/{s.position}_tube", f"{ROOT}/{s.position}_cap"
        if s.state == "no_vial":
            UsdGeom.Imageable(stage.GetPrimAtPath(tube)).MakeInvisible()
            UsdGeom.Imageable(stage.GetPrimAtPath(cap)).MakeInvisible(); continue
        if s.state == "missing_cap":
            UsdGeom.Imageable(stage.GetPrimAtPath(cap)).MakeInvisible()
        L, Mod = lean_modification(twin, s), cap_modification(twin, s)
        if L != Gf.Matrix4d(1.0):
            set_matrix(stage, tube, L * d["M_tube"])
        if s.state != "missing_cap" and (L != Gf.Matrix4d(1.0) or Mod != Gf.Matrix4d(1.0)):
            set_matrix(stage, cap, Mod * d["C_rel"] * L * d["M_tube"])
    if spec.lighting:
        for name in LIGHTS:
            xf = stage.GetPrimAtPath(f"{ROOT}/{name}"); lp = UsdLux.RectLight(xf.GetChildren()[0])
            v = spec.lighting[name]
            base_i = lp.GetIntensityAttr().Get(); base_r = xf.GetAttribute("xformOp:rotateXYZ").Get()
            lp.GetIntensityAttr().Set(float(base_i * v["intensity_scale"]))
            lp.CreateEnableColorTemperatureAttr().Set(True)
            lp.CreateColorTemperatureAttr().Set(float(v["color_temperature_k"]))
            rx, ry = v["rotate_xy_deg"]
            xf.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(base_r[0] + rx, base_r[1] + ry, base_r[2]))
    if spec.materials:
        m = spec.materials
        sh = lambda key: UsdShade.Shader(stage.GetPrimAtPath(f"{ROOT}/_materials/{MATS[key]}/Principled_BSDF"))
        if "cap_hue_shift" in m:                                 # one cap colour per scene (smoke)
            cap = sh("cap"); c0 = cap.GetInput("diffuseColor").Get()
            h, s_, v_ = colorsys.rgb_to_hsv(*c0)
            rgb = colorsys.hsv_to_rgb((h + m["cap_hue_shift"]) % 1.0, min(1.0, s_ * m["cap_sat_scale"]),
                                      min(1.0, v_ * m["cap_val_scale"]))
            cap.GetInput("diffuseColor").Set(Gf.Vec3f(*rgb)); cap.GetInput("roughness").Set(float(m["cap_roughness"]))
        plate = sh("plate"); p0 = plate.GetInput("diffuseColor").Get()
        plate.GetInput("diffuseColor").Set(Gf.Vec3f(*[min(1.0, x * m["plate_val_scale"]) for x in p0]))
        plate.GetInput("roughness").Set(float(m["plate_roughness"]))
        sh("vial").GetInput("roughness").Set(float(m["vial_roughness"]))
    if spec.floor:                                               # D040 hole floor (work2 material)
        fs = UsdShade.Shader(stage.GetPrimAtPath(f"{ROOT}/_materials/{FLOOR_MAT}/Principled_BSDF"))
        fs.GetInput("diffuseColor").Set(Gf.Vec3f(*spec.floor["rgb"])); fs.GetInput("roughness").Set(float(spec.floor["roughness"]))
    for s in spec.positions:                                     # per-cap materials (training recipe)
        if s.cap_rgb is None:
            continue
        mpath = f"{ROOT}/_materials/ignite_cap_{s.position}"
        mat = UsdShade.Material.Define(stage, mpath)
        shd = UsdShade.Shader.Define(stage, mpath + "/Principled_BSDF"); shd.CreateIdAttr("UsdPreviewSurface")
        for name, tname, val in twin.cap_inputs:
            shd.CreateInput(name, tname).Set(val)
        shd.GetInput("diffuseColor").Set(Gf.Vec3f(*s.cap_rgb)); shd.GetInput("ior").Set(float(s.cap_ior))
        shd.GetInput("roughness").Set(float(s.cap_roughness))
        mat.CreateSurfaceOutput().ConnectToSource(shd.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"{ROOT}/{s.position}_cap/{CAP_MESH}")).Bind(mat)
    layer.Save()
    return path


# ------------------------------------------------------------------------------------------------------------ readback
def read_back(twin, scene_path):
    """Labels and physical quantities from the COMPOSED stage only (transforms + visibility), never from the spec."""
    st = Usd.Stage.Open(scene_path); xc = UsdGeom.XformCache()
    vis = lambda p: UsdGeom.Imageable(st.GetPrimAtPath(p)).ComputeVisibility() != UsdGeom.Tokens.invisible
    out = {}
    for pid in twin.ids:
        d = twin.pos[pid]; tp, cp = f"{ROOT}/{pid}_tube", f"{ROOT}/{pid}_cap"
        hv, hc = vis(tp), vis(cp)
        W_t = np4(xc.GetLocalToWorldTransform(st.GetPrimAtPath(tp)))
        W_c = np4(xc.GetLocalToWorldTransform(st.GetPrimAtPath(cp)))
        r = {"position": pid, "index": d["index"], "tube_visible": hv, "cap_visible": hc,
             "W_tube": W_t, "W_cap": W_c}
        # tube: its change in its own (as-shipped) frame. W_t = L @ W_t0, so L = W_t @ inv(W_t0)
        L = W_t @ np.linalg.inv(d["W_tube"])
        r["L"] = L
        r["lean_deg"] = axis_angle_deg(L)
        top = np.array([[0, 0, twin.H_TUBE]])
        dv = (apply(top, gf4(L)) - top)[0]
        r["lean_dir_deg"] = local_dir_to_world_az(d["W_tube"], [dv[0], dv[1], 0]) if r["lean_deg"] > 1e-9 else None
        # cap: its change relative to the seated cap, in cap-local coordinates. W_c = Mod @ C_rel @ W_t
        Mod = W_c @ np.linalg.inv(W_t) @ np.linalg.inv(np4(d["C_rel"]))
        r["Mod"] = Mod
        r["tilt_deg"] = axis_angle_deg(Mod)
        # skirt bottom rim (cap-local circle r=R_CAP, z=0; seated at z=0): exact min/max height over the circle
        zc, zr = Mod[3, 2], twin.R_CAP * np.hypot(Mod[0, 2], Mod[1, 2])
        r["seat_gap_min_mm"] = float((zc - zr) * IN_TO_MM)
        r["seat_gap_max_mm"] = float((zc + zr) * IN_TO_MM)
        o = Mod[3, :3]
        r["base_offset_mm"] = float(np.hypot(o[0], o[1]) * IN_TO_MM) if r["tilt_deg"] < SEATED_TILT_TOL_DEG else None
        r["offset_dir_deg"] = local_dir_to_world_az(d["W_cap"], [o[0], o[1], 0]) \
            if r["base_offset_mm"] is not None and r["base_offset_mm"] > 1e-4 else None
        r["fully_seated"] = r["tilt_deg"] < SEATED_TILT_TOL_DEG and abs(r["seat_gap_max_mm"]) < 0.01 \
            and abs(r["seat_gap_min_mm"]) < 0.01
        r["state"] = "no_vial" if not hv else "missing_cap" if not hc else "pass" if r["fully_seated"] else "crooked_cap"
        if r["state"] == "pass":
            r["variant"] = "offset" if r["base_offset_mm"] > 1e-4 else "seated"
        elif r["state"] == "crooked_cap":
            r["variant"] = "tilt" if r["tilt_deg"] >= SEATED_TILT_TOL_DEG else "lift"
        else:
            r["variant"] = {"missing_cap": "uncapped", "no_vial": "empty"}[r["state"]]
        if r["state"] == "crooked_cap" and r["variant"] == "tilt":
            ct = np.array([[0, 0, twin.H_CAP]])
            mv = (apply(ct, gf4(Mod)) - ct)[0]
            r["tilt_dir_deg"] = local_dir_to_world_az(d["W_cap"], [mv[0], mv[1], 0])
        # image: what is drawn, where the seated vial would be, how far the cap silhouette moved
        r["anchor_bbox_px"] = d["anchor_bbox_px"]
        drawn = []
        if hv:
            drawn.append(apply(d["tube_pts"], gf4(W_t)))
        if hc and hv:
            cap_w = apply(d["cap_pts"], gf4(W_c)); drawn.append(cap_w)
            uv = twin.cam.project(cap_w)[0]
            c = hull_centroid(uv)
            r["cap_center_px"] = c
            r["cap_shift_px"] = float(np.linalg.norm(c - d["seat_cap_center_px"]))
            r["cap_max_shift_px"] = float(np.linalg.norm(uv - d["seat_cap_uv"], axis=1).max())
            top_px = twin.cam.project(apply(np.array([[0, 0, twin.H_CAP]]), gf4(W_c)))[0][0]
            top0_px = twin.cam.project(apply(np.array([[0, 0, twin.H_CAP]]), gf4(d["W_cap"])))[0][0]
            dd = top_px - top0_px
            r["cap_top_dir_img_deg"] = float(np.degrees(np.arctan2(dd[1], dd[0])) % 360) if np.linalg.norm(dd) > 1e-3 else None
        r["bbox_px"] = bbox(twin.cam.project(np.vstack(drawn))[0]) if drawn else None
        if hc and hv:
            mat = UsdShade.MaterialBindingAPI(st.GetPrimAtPath(f"{cp}/{CAP_MESH}")).ComputeBoundMaterial()[0]
            shd = mat.ComputeSurfaceSource()[0]
            r["cap_material"] = str(mat.GetPath())
            r["cap_rgb"] = [float(x) for x in shd.GetInput("diffuseColor").Get()]
            r["cap_ior"], r["cap_roughness"] = float(shd.GetInput("ior").Get()), float(shd.GetInput("roughness").Get())
        out[pid] = r
    lights = {}
    for name in LIGHTS:
        xf = st.GetPrimAtPath(f"{ROOT}/{name}"); lp = UsdLux.RectLight(xf.GetChildren()[0])
        en = lp.GetEnableColorTemperatureAttr().Get()
        lights[name] = {"intensity": float(lp.GetIntensityAttr().Get()),
                        "color_temperature_k": float(lp.GetColorTemperatureAttr().Get()) if en else None,
                        "rotate_xyz_deg": [float(x) for x in xf.GetAttribute("xformOp:rotateXYZ").Get()]}
    mats = {}
    for key, name in list(MATS.items()) + [("floor", FLOOR_MAT)]:
        prim = st.GetPrimAtPath(f"{ROOT}/_materials/{name}/Principled_BSDF")
        if key == "floor" and not prim.IsValid():
            continue                                             # no work2 overlay: nothing to read
        s = UsdShade.Shader(prim)
        mats[key] = {"diffuseColor": [float(x) for x in s.GetInput("diffuseColor").Get()],
                     "roughness": float(s.GetInput("roughness").Get())}
    floor_mesh = st.GetPrimAtPath(f"{ROOT}/tray_good_support/tray_good_support")
    if "floor" in mats and floor_mesh.IsValid():
        mats["floor"]["bound_to_floor_mesh"] = str(UsdShade.MaterialBindingAPI(floor_mesh).ComputeBoundMaterial()[0].GetPath()).endswith(FLOOR_MAT)
    cam = np4(xc.GetLocalToWorldTransform(st.GetPrimAtPath(f"{ROOT}/CapSealCam/CapSealCam")))
    return {"positions": out, "lights": lights, "materials": mats, "camera_world": cam, "stage": st}


# -------------------------------------------------------------------------------------------------- physical checks
def physical_issues(twin, rb):
    """Placement rules on the read-back scene. Returns a list of human-readable violations (empty = OK)."""
    issues = []
    P = rb["positions"]
    for pid, r in P.items():
        if not r["tube_visible"]:
            continue
        issues += twin.lean_violations(pid, gf4(r["L"]))
        if r["cap_visible"]:
            if r["seat_gap_min_mm"] < -GEOM_TOL_IN * IN_TO_MM:
                issues.append(f"{pid}: cap {-r['seat_gap_min_mm']:.2f} mm below its seated height (sunk into the vial)")
            if r["seat_gap_min_mm"] >= twin.lift_limit_mm - GEOM_TOL_IN * IN_TO_MM:
                issues.append(f"{pid}: cap skirt entirely above the vial top (cap is off the vial)")
            if r["base_offset_mm"] is not None and r["base_offset_mm"] > twin.offset_limit_mm + GEOM_TOL_IN * IN_TO_MM:
                issues.append(f"{pid}: cap shifted {r['base_offset_mm']:.2f} mm, more than the skirt allows")
    # collisions between neighbours: vertices of one rigid part inside another part's cylinder
    parts = {}
    for pid, r in P.items():
        d = twin.pos[pid]
        if r["tube_visible"]:
            parts[(pid, "tube")] = (r["W_tube"], d["tube_pts"], twin.R_TUBE, twin.H_TUBE,
                                    not np.allclose(r["W_tube"], d["W_tube"], atol=1e-9))
        if r["tube_visible"] and r["cap_visible"]:
            parts[(pid, "cap")] = (r["W_cap"], d["cap_pts"], twin.R_CAP, twin.H_CAP,
                                   not np.allclose(r["W_cap"], d["W_cap"], atol=1e-9))
    seen = set()
    for (pa, ka), A in parts.items():
        for pb in [pa] + twin.neighbors[pa]:
            for kb in ("tube", "cap"):
                B = parts.get((pb, kb))
                if B is None or pa == pb or not (A[4] or B[4]) or frozenset([(pa, ka), (pb, kb)]) in seen:
                    continue
                seen.add(frozenset([(pa, ka), (pb, kb)]))
                for (pX, kX, X), (pY, kY, Y) in (((pa, ka, A), (pb, kb, B)), ((pb, kb, B), (pa, ka, A))):
                    loc = apply(apply(X[1], gf4(X[0])), gf4(np.linalg.inv(Y[0])))
                    n = int(inside_cylinder(loc, Y[2], Y[3]).sum())
                    if n:
                        issues.append(f"{pX} {kX} intersects {pY} {kY} ({n} vertices inside)"); break
    return issues


def spec_mismatches(twin, spec, rb):
    """Authored spec vs read-back scene, per position (state, variant, sequence-free geometry)."""
    bad = []
    for s in spec.positions:
        r = rb["positions"][s.position]
        if r["state"] != s.state or r["variant"] != s.variant:
            bad.append(f"{s.position}: authored {s.state}/{s.variant}, read back {r['state']}/{r['variant']}"); continue
        if s.state == "no_vial":
            continue
        checks = [("lean_deg", s.lean_deg, r["lean_deg"], 1e-6)]
        if s.state != "missing_cap":
            if not np.allclose(r["Mod"], np4(cap_modification(twin, s)), atol=1e-9):
                bad.append(f"{s.position}: cap transform differs from the authored {s.variant}")
            checks += [("tilt_deg", s.tilt_deg, r["tilt_deg"], 1e-6),
                       ("lift_mm (lowest skirt point)", s.lift_mm, r["seat_gap_min_mm"], 1e-6)]
            if s.variant != "tilt":
                checks.append(("offset_mm", s.offset_mm, r["base_offset_mm"], 1e-6))
            if s.variant == "offset":
                checks.append(("offset_dir_deg", 0.0, ((r["offset_dir_deg"] - s.offset_dir_deg + 180) % 360) - 180, 1e-3))
            if s.variant == "tilt":
                checks.append(("tilt_dir_deg", 0.0, ((r["tilt_dir_deg"] - s.tilt_dir_deg + 180) % 360) - 180, 1e-3))
        if s.lean_deg > 0:
            checks.append(("lean_dir_deg", 0.0, ((r["lean_dir_deg"] - s.lean_dir_deg + 180) % 360) - 180, 1e-3))
        if s.state != "missing_cap":
            want_mat = f"{ROOT}/_materials/" + (f"ignite_cap_{s.position}" if s.cap_rgb is not None else MATS["cap"])
            if r.get("cap_material") != want_mat:
                bad.append(f"{s.position}: cap material {r.get('cap_material')}, authored {want_mat}")
            elif s.cap_rgb is not None:
                checks += [("cap_ior", s.cap_ior, r["cap_ior"], 1e-5), ("cap_roughness", s.cap_roughness, r["cap_roughness"], 1e-6)]
                checks += [(f"cap_rgb[{k}]", s.cap_rgb[k], r["cap_rgb"][k], 1e-6) for k in range(3)]
        for name, want, got, tol in checks:
            if abs(want - got) > tol:
                bad.append(f"{s.position}: {name} authored {want:.6f}, read back {got:.6f}")
    if spec.floor:
        f = rb["materials"].get("floor")
        if not f or not f.get("bound_to_floor_mesh"):
            bad.append(f"hole floor: {FLOOR_MAT} missing or not bound to tray_good_support")
        else:
            for k in range(3):
                if abs(f["diffuseColor"][k] - spec.floor["rgb"][k]) > 1e-6:
                    bad.append(f"hole floor rgb[{k}] authored {spec.floor['rgb'][k]:.6f}, read back {f['diffuseColor'][k]:.6f}")
            if abs(f["roughness"] - spec.floor["roughness"]) > 1e-6:
                bad.append(f"hole floor roughness authored {spec.floor['roughness']:.6f}, read back {f['roughness']:.6f}")
    if spec.lighting:
        for name in LIGHTS:
            got, want = rb["lights"][name], spec.lighting[name]
            if got["color_temperature_k"] is None or abs(got["color_temperature_k"] - want["color_temperature_k"]) > 0.01:
                bad.append(f"{name}: colour temperature not read back")
            if abs(got["intensity"] - twin.base_intensity[name] * want["intensity_scale"]) > 1e-6 * got["intensity"]:
                bad.append(f"{name}: intensity {got['intensity']:.3f}, authored {twin.base_intensity[name] * want['intensity_scale']:.3f}")
    return bad


def camera_issues(twin, rb):
    """The USD camera Isaac renders through must be the one the labels project with (capsealcam.py + JSON)."""
    out = []
    W = rb["camera_world"]
    c = W[3, :3]
    if np.linalg.norm(c - twin.cam.C) > 1e-4:
        out.append(f"camera centre differs from CapSealCam.json by {np.linalg.norm(c - twin.cam.C) * 1000:.2f} mm")
    ax = W[:3, :3] / np.linalg.norm(W[:3, :3], axis=1, keepdims=True)
    R_cv = np.stack([ax[0], -ax[1], -ax[2]])                       # USD camera (-Z forward, +Y up) -> OpenCV axes
    if np.abs(R_cv - twin.cam.R).max() > 1e-4:
        out.append(f"camera orientation differs from CapSealCam.json (max {np.abs(R_cv - twin.cam.R).max():.2e})")
    camprim = UsdGeom.Camera(rb["stage"].GetPrimAtPath(f"{ROOT}/CapSealCam/CapSealCam"))
    f_px = camprim.GetFocalLengthAttr().Get() / camprim.GetHorizontalApertureAttr().Get() * twin.cam.rw
    if abs(f_px - twin.cam.f) > 0.05:
        out.append(f"USD focal {f_px:.3f} px vs CapSealCam.json {twin.cam.f:.3f} px")
    return out


# -------------------------------------------------------------------------------------------------------- outputs
MANIFEST_COLS = ["image", "scene_id", "scene_file", "position", "index", "label", "sequence", "variant", "placements",
                 "tilt_deg", "tilt_dir_deg", "cap_top_dir_img_deg", "lift_mm", "seat_gap_max_mm", "offset_mm",
                 "lean_deg", "lean_dir_deg", "lean_limit_deg", "cap_shift_px", "cap_max_shift_px",
                 "bbox_px", "anchor_bbox_px", "bbox_in_crop", "light_id", "material_id", "cap_group", "seed", "config_sha12",
                 "generator_sha12"]


def short_id(obj):
    return "twin_default" if obj is None else sha256_bytes(json.dumps(obj, sort_keys=True).encode())[:10]


def labels_json(spec, rb, crop_boxes, station_rel):
    rows = []
    for s in spec.positions:
        r = rb["positions"][s.position]
        o = {"position": s.position, "state": r["state"], "variant": r["variant"], "anchor_bbox_px": r["anchor_bbox_px"]}
        if s.sequence:
            o["sequence"] = s.sequence
        if r["bbox_px"]:
            o["bbox_px"] = r["bbox_px"]
            x0, y0, x1, y1 = crop_boxes[r["index"] - 1]
            b = r["bbox_px"]; o["bbox_in_crop"] = bool(b[0] >= x0 and b[1] >= y0 and b[2] <= x1 and b[3] <= y1)
        if r["state"] == "crooked_cap" and r["variant"] == "tilt":
            o["cap_tilt_deg"] = r6(r["tilt_deg"]); o["tilt_dir_deg"] = r6(r["tilt_dir_deg"])
        if r["state"] == "crooked_cap" and r["variant"] == "lift":
            o["lift_mm"] = r6(r["seat_gap_min_mm"])
        if r["state"] == "crooked_cap":
            o["seat_gap_max_mm"] = r6(r["seat_gap_max_mm"])          # highest point of the skirt above its seat
        if r["variant"] == "offset":
            o["offset_mm"] = r6(r["base_offset_mm"]); o["offset_dir_deg"] = r6(r["offset_dir_deg"])
        if r["tube_visible"] and r["lean_deg"] > 1e-9:
            o["lean_deg"] = r6(r["lean_deg"]); o["lean_dir_deg"] = r6(r["lean_dir_deg"])
            o["lean_limit_deg"] = r6(s.lean_limit_deg)
        if "cap_shift_px" in r:
            o["cap_shift_px"] = r6(r["cap_shift_px"]); o["cap_max_shift_px"] = r6(r["cap_max_shift_px"])
            o["cap_top_dir_img_deg"] = r6(r["cap_top_dir_img_deg"])
        if s.cap_rgb is not None and r["state"] != "missing_cap":
            o["cap_group"] = s.cap_group; o["cap_rgb"] = [r6(x) for x in r["cap_rgb"]]
            o["cap_ior"] = r6(r["cap_ior"]); o["cap_roughness"] = r6(r["cap_roughness"])
        o["authored"] = s.state
        rows.append(o)
    out = {"scene_id": spec.scene_id, "index": spec.index, "seed": spec.seed, "station": station_rel,
           "placements": spec.placements, "lighting": spec.lighting, "materials": spec.materials,
           "light_id": short_id(spec.lighting), "material_id": short_id(spec.materials), "positions": rows}
    if spec.floor:                                               # only with a floor block: v1 labels are unchanged
        out["floor"] = spec.floor
    return out


def manifest_rows(spec, lab, scene_file, cfg_sha, gen_sha):
    out = []
    for o in lab["positions"]:
        g = lambda k: "" if o.get(k) is None else o[k]
        out.append({"image": "", "scene_id": spec.scene_id, "scene_file": scene_file, "position": o["position"],
                    "index": int(o["position"][1:]), "label": o["state"], "sequence": g("sequence"),
                    "variant": o["variant"], "placements": spec.placements, "tilt_deg": g("cap_tilt_deg"),
                    "tilt_dir_deg": g("tilt_dir_deg"), "cap_top_dir_img_deg": g("cap_top_dir_img_deg"),
                    "lift_mm": g("lift_mm"), "seat_gap_max_mm": g("seat_gap_max_mm"), "offset_mm": g("offset_mm"),
                    "lean_deg": g("lean_deg"), "lean_dir_deg": g("lean_dir_deg"), "lean_limit_deg": g("lean_limit_deg"),
                    "cap_shift_px": g("cap_shift_px"), "cap_max_shift_px": g("cap_max_shift_px"),
                    "bbox_px": " ".join(map(str, o["bbox_px"])) if o.get("bbox_px") else "",
                    "anchor_bbox_px": " ".join(map(str, o["anchor_bbox_px"])), "bbox_in_crop": g("bbox_in_crop"),
                    "light_id": lab["light_id"], "material_id": lab["material_id"], "cap_group": g("cap_group"), "seed": spec.seed,
                    "config_sha12": cfg_sha[:12], "generator_sha12": gen_sha[:12]})
    return out


def git_state():
    try:
        c = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        return c, dirty
    except Exception:
        return "unknown", None


def generate(twin, cfg, cfg_sha, out_dir, seed=0, limit=None, log=print, indices=None):
    """Author, read back and check every scene; write labels, manifest and provenance. Raises on any mismatch."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"{out_dir} is not empty; pick a new output folder")
    os.makedirs(out_dir, exist_ok=True)
    gen_sha = sha256_file(os.path.abspath(__file__))
    pmap = PositionMap(MAP_PATH, offset=(0, 0))
    crop_boxes = [pmap.crop_box(i)[0] for i in range(30)]
    N = n_scenes(cfg) if limit is None else min(limit, n_scenes(cfg))
    todo = list(range(N)) if indices is None else list(indices)
    if indices is not None and (not todo or min(todo) < 0 or max(todo) >= n_scenes(cfg) or len(set(todo)) != len(todo)):
        raise ConfigError(f"indices must be distinct and within 0..{n_scenes(cfg) - 1}")
    N = len(todo)
    manifest, scenes = [], []
    counts = {k: 0 for k in STATES}
    for k_, idx in enumerate(todo):
        spec = build_spec(cfg, twin, seed, idx)
        sdir = os.path.join(out_dir, "scenes", spec.scene_id)
        path = author_scene(twin, spec, sdir)
        rb = read_back(twin, path)
        problems = spec_mismatches(twin, spec, rb) + physical_issues(twin, rb)
        if problems:
            raise PhysicalError(f"{spec.scene_id}: " + "; ".join(problems[:8]))
        lab = labels_json(spec, rb, crop_boxes, os.path.relpath(twin.station, sdir))
        with open(os.path.join(sdir, "labels.json"), "w") as f:
            json.dump(lab, f, indent=1)
        manifest += manifest_rows(spec, lab, os.path.relpath(path, out_dir), cfg_sha, gen_sha)
        for o in lab["positions"]:
            counts[o["state"]] += 1
        sc = {"scene_id": spec.scene_id, "index": idx, "seed": seed, "placements": spec.placements,
              "light_id": lab["light_id"], "material_id": lab["material_id"]}
        if spec.lighting and "exposure_ev" in spec.lighting[LIGHTS[0]]:
            sc["exposure_ev"] = r6(spec.lighting[LIGHTS[0]]["exposure_ev"])
        if any(s.cap_group for s in spec.positions):
            sc["pale_caps"] = sum(o.get("cap_group") == "pale" for o in lab["positions"])
        for name in LIGHTS:
            L = rb["lights"][name]
            sc[f"{name}_intensity"] = r6(L["intensity"]); sc[f"{name}_ct_k"] = r6(L["color_temperature_k"])
            sc[f"{name}_rot_xyz"] = " ".join(f"{x:.4f}" for x in L["rotate_xyz_deg"])
        for key in list(MATS) + (["floor"] if spec.floor else []):
            M = rb["materials"][key]
            sc[f"{key}_rgb"] = " ".join(f"{x:.5f}" for x in M["diffuseColor"]); sc[f"{key}_roughness"] = r6(M["roughness"])
        scenes.append(sc)
        if (k_ + 1) % 25 == 0 or k_ + 1 == N:
            log(f"  {k_ + 1}/{N} scenes")
    _write_csv(os.path.join(out_dir, "manifest.csv"), MANIFEST_COLS, manifest)
    _write_csv(os.path.join(out_dir, "scenes.csv"), list(scenes[0]), scenes)
    commit, dirty = git_state()
    run = {"generator": "gen_scenes.py", "generator_sha256": gen_sha, "config": cfg, "config_sha256": cfg_sha,
           "seed": seed, "scenes": N, "labels": N * 30, **({"indices": todo} if indices is not None else {}), "label_counts": counts, "git_commit": commit,
           "git_dirty_tracked": dirty, "station_sha256": twin.station_sha256, "camera_sha256": twin.camera_sha256,
           "base_station_sha256": twin.base_station_sha256,
           "twin_fix": None if twin.fix is None else {k: twin.fix[k] for k in ("fix", "fix_json_sha256", "sha256")},
           "position_map_sha256": sha256_file(MAP_PATH), "usd_version": ".".join(map(str, Usd.GetVersion())),
           "numpy_version": np.__version__, "python_version": sys.version.split()[0],
           "reproducible_given": "same generator, config, seed, twin files, USD and NumPy versions",
           "rendered": False, "note": "scenes generated and label-checked; not rendered; no transfer claim"}
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=1)
    sums = []
    for root, _, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            if fn != "SHA256SUMS":
                sums.append(f"{sha256_file(p)}  {os.path.relpath(p, out_dir)}")
    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sorted(sums, key=lambda s: s[66:])) + "\n")
    return run


def _write_csv(path, cols, rows):
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    with open(path, "w", newline="") as f:
        f.write(buf.getvalue())


# ---------------------------------------------------------------------------------------------------------- selftest
def selftest_spec(twin):
    """One scene that exercises every state and variant at fixed positions (14 placements; G15-G30 waiting)."""
    lim = lambda pid, az: twin.lean_limit_deg(pid, az)
    S = [make_pos("G01", "pass"), make_pos("G02", "missing_cap"),
         make_pos("G03", "crooked_cap", "tilt", tilt_deg=20, tilt_dir_deg=0),
         make_pos("G04", "no_vial", sequence="missed_placement"),
         make_pos("G05", "crooked_cap", "lift", lift_mm=3.0),
         make_pos("G06", "pass", "offset", offset_mm=1.0, offset_dir_deg=90),
         make_pos("G07", "pass", lean_deg=min(1.0, 0.8 * lim("G07", 45)), lean_dir_deg=45, lean_limit_deg=lim("G07", 45)),
         make_pos("G08", "crooked_cap", "tilt", tilt_deg=30, tilt_dir_deg=225),
         make_pos("G09", "missing_cap", lean_deg=min(1.0, 0.8 * lim("G09", 270)), lean_dir_deg=270,
                  lean_limit_deg=lim("G09", 270)),
         make_pos("G10", "crooked_cap", "tilt", tilt_deg=10, tilt_dir_deg=135, lean_deg=min(0.8, 0.8 * lim("G10", 135)),
                  lean_dir_deg=135, lean_limit_deg=lim("G10", 135)),
         make_pos("G11", "pass"), make_pos("G12", "crooked_cap", "tilt", tilt_deg=5, tilt_dir_deg=315),
         make_pos("G13", "pass"), make_pos("G14", "pass")]
    S += [make_pos(f"G{i:02d}", "no_vial", sequence="waiting") for i in range(15, 31)]
    light = {n: {"intensity_scale": 0.8, "color_temperature_k": 5600.0, "rotate_xy_deg": [3.0, -2.0]} for n in LIGHTS}
    mats = {"cap_hue_shift": 0.01, "cap_sat_scale": 0.95, "cap_val_scale": 0.9, "cap_roughness": 0.35,
            "plate_val_scale": 1.1, "plate_roughness": 0.5, "vial_roughness": 0.06}
    return SceneSpec("selftest_00000", 0, 0, 14, S, light, mats)


def selftest(twin, log=print):
    fails = []
    tmp = tempfile.mkdtemp(prefix="gen_scenes_selftest_")
    # 1) as-shipped scene: everything seated, projection agrees with the frozen runtime map, camera agrees with JSON
    base = SceneSpec("selftest_base", 0, 0, 30, [make_pos(p, "pass") for p in twin.ids])
    rb = read_back(twin, author_scene(twin, base, os.path.join(tmp, "base")))
    fails += [f"base: {m}" for m in spec_mismatches(twin, base, rb) + physical_issues(twin, rb) + camera_issues(twin, rb)]
    pm = json.load(open(MAP_PATH))
    for p in pm["positions"]:
        r = rb["positions"][p["position"]]
        if np.abs(r["cap_center_px"] - np.array(p["cap_center_px"])).max() > 0.01:
            fails.append(f"{p['position']}: seated cap centre {r['cap_center_px'].round(2)} vs map {p['cap_center_px']}")
        if r["anchor_bbox_px"] != p["anchor_bbox_px"]:
            fails.append(f"{p['position']}: anchor box {r['anchor_bbox_px']} vs map {p['anchor_bbox_px']}")
    if pm["meta"].get("station_sha256") != twin.base_station_sha256:
        fails.append("position_map_G.json was exported from a different station file")
    # 2) every state and variant, read back and checked
    spec = selftest_spec(twin)
    rb2 = read_back(twin, author_scene(twin, spec, os.path.join(tmp, "all")))
    fails += [f"all-states: {m}" for m in spec_mismatches(twin, spec, rb2) + physical_issues(twin, rb2)]
    # 3) the checks must catch broken scenes, not just pass good ones
    bad = SceneSpec("selftest_bad", 0, 0, 30, [make_pos(p, "pass") for p in twin.ids])
    bad.positions[0] = make_pos("G01", "pass", lean_deg=6.0, lean_dir_deg=0.0)        # far past the plate-hole limit
    rb3 = read_back(twin, author_scene(twin, bad, os.path.join(tmp, "bad_lean")))
    if not any("plate hole" in m for m in physical_issues(twin, rb3)):
        fails.append("lean check did not flag a 6 deg lean")
    sunk = SceneSpec("selftest_sunk", 0, 0, 30, [make_pos(p, "pass") for p in twin.ids])
    p = author_scene(twin, sunk, os.path.join(tmp, "sunk"))
    lay = Sdf.Layer.FindOrOpen(p); st = Usd.Stage.Open(lay); st.SetEditTarget(lay)
    set_matrix(st, f"{ROOT}/G13_cap", Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, -0.1)) * twin.pos["G13"]["M_cap"])
    lay.Save()
    if not any("below its seated height" in m for m in physical_issues(twin, read_back(twin, p))):
        fails.append("seat check did not flag a cap pushed 2.5 mm into its vial")
    clash = SceneSpec("selftest_clash", 0, 0, 30, [make_pos(p, "pass") for p in twin.ids])
    p = author_scene(twin, clash, os.path.join(tmp, "clash"))
    lay = Sdf.Layer.FindOrOpen(p); st = Usd.Stage.Open(lay); st.SetEditTarget(lay)
    d13, d14 = twin.pos["G13"], twin.pos["G14"]
    shift = (np4(d14["M_cap"])[3, :3] - np4(d13["M_cap"])[3, :3]) * 0.6                # G13 cap 60% of the way to G14
    set_matrix(st, f"{ROOT}/G13_cap", d13["M_cap"] * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, shift))))
    lay.Save()
    if not any("intersects" in m for m in physical_issues(twin, read_back(twin, p))):
        fails.append("collision check did not flag overlapping neighbour caps")
    # 4) byte-identical re-authoring
    a = author_scene(twin, spec, os.path.join(tmp, "repeat_a", "x")); b = author_scene(twin, spec, os.path.join(tmp, "repeat_b", "x"))
    if sha256_file(a) != sha256_file(b):
        fails.append("the same spec authored twice gave different bytes")
    # reference only (not pass/fail): measured image shift of a crooked cap and a leaning vial at G13
    log("reference (measured, not asserted): G13 crooked cap vs seated, over 8 tilt directions "
        "(silhouette centroid shift | largest vertex shift)")
    for t in (10, 15, 20, 30):
        cs, ms = [], []
        for az in range(0, 360, 45):
            sp = SceneSpec("ref", 0, 0, 30, [make_pos(p, "pass") for p in twin.ids])
            sp.positions[12] = make_pos("G13", "crooked_cap", "tilt", tilt_deg=t, tilt_dir_deg=az)
            r = read_back(twin, author_scene(twin, sp, os.path.join(tmp, f"ref_{t}_{az}")))["positions"]["G13"]
            cs.append(r["cap_shift_px"]); ms.append(r["cap_max_shift_px"])
        log(f"  {t:2d} deg: centroid {min(cs):.2f}-{max(cs):.2f} px | largest vertex {min(ms):.2f}-{max(ms):.2f} px")
    lims = [twin.lean_limit_deg(pid, az) for pid in twin.ids for az in range(0, 360, 45)]
    log(f"  plate-hole lean limit over G01-G30 x 8 directions: {min(lims):.2f}-{max(lims):.2f} deg "
        f"(vial pivots on its base edge; hole r={twin.pos['G13']['hole']['radius'] * IN_TO_MM:.2f} mm, "
        f"vial r={twin.R_TUBE * IN_TO_MM:.2f} mm)")
    log(f"  limits: offset <= {twin.offset_limit_mm:.2f} mm, lift < {twin.lift_limit_mm:.2f} mm")
    return fails, tmp


# --------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("twin", help="csi_digital_twin_v1 folder")
    ap.add_argument("out", nargs="?", help="output folder (must be new or empty)")
    ap.add_argument("--config", help="configs/smoke.json, configs/tilt_sweep.json, ...")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="only the first N scenes (scene i is the same with or without it)")
    ap.add_argument("--indices", help="only these scene indices, comma-separated (scene i is the same either way)")
    ap.add_argument("--selftest", action="store_true", help="check authoring, readback and physical rules on the twin")
    a = ap.parse_args(argv)
    twin = Twin(a.twin)
    if a.selftest:
        fails, tmp = selftest(twin)
        for f in fails:
            print("FAIL", f)
        if fails:
            print(f"selftest FAILED ({len(fails)}); scenes kept in {tmp}")
            return 1
        shutil.rmtree(tmp, ignore_errors=True)
        print("selftest passed")
        return 0
    if not (a.out and a.config):
        ap.error("give an output folder and --config (or --selftest)")
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1")
    try:
        cfg, cfg_sha = load_config(a.config, twin)
    except ConfigError as e:
        ap.error(f"config: {e}")
    idx = [int(x) for x in a.indices.split(",")] if a.indices else None
    print(f"{cfg['name']}: {len(idx) if idx else n_scenes(cfg) if a.limit is None else min(a.limit, n_scenes(cfg))} scenes, seed {a.seed}")
    try:
        run = generate(twin, cfg, cfg_sha, a.out, a.seed, a.limit, indices=idx)
    except ConfigError as e:
        ap.error(f"indices: {e}")
    print(f"labels OK: {run['scenes']}x30 read back from the composed stages and match the authored states")
    print("physical checks OK (seat height, cap on vial, plate hole, neighbour collisions)")
    print("label counts:", ", ".join(f"{k} {v}" for k, v in run["label_counts"].items()))
    print(f"wrote {a.out}/manifest.csv, scenes.csv, run.json, SHA256SUMS  (git {run['git_commit'][:12]}"
          f"{' dirty' if run['git_dirty_tracked'] else ''}); not rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
