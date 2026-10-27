"""Scene generator: geometry, config, readback, physical rules, reproducibility, compatibility with the frozen gate
and the runtime crop contract. Twin-dependent tests use IGNITE_TWIN (default ~/Downloads/csi_digital_twin/
csi_digital_twin_v1) and are skipped when the twin is not on this machine. Pixel displacements are never asserted
against estimates; only invariants (zero at seated, monotonic in tilt, agreement with the exported runtime map)."""
import ast, csv, hashlib, json, os, shutil, subprocess, sys, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
try:
    from pxr import Gf
    import gen_scenes as G
    HAVE_USD = True
except ImportError:                                  # usd-core missing: the whole module is skipped
    HAVE_USD = False

TWIN = os.path.expanduser(os.environ.get("IGNITE_TWIN", "~/Downloads/csi_digital_twin/csi_digital_twin_v1"))
HAVE_TWIN = os.path.isfile(os.path.join(TWIN, "stations", "tray_inspection", "tray_inspection.usdc"))
CFG = os.path.join(HERE, "configs")
_twin = None


def twin():
    global _twin
    if _twin is None:
        _twin = G.Twin(TWIN)
    return _twin


def gate_tilt_matrix():
    """The frozen gate's tilt_matrix(), extracted from make_variants.py without running that script."""
    src = open(os.path.join(HERE, "make_variants.py")).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tilt_matrix"]
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "R_CAP" or
              (isinstance(t, ast.Tuple) and any(getattr(e, "id", None) == "R_CAP" for e in t.elts)) for t in n.targets)]
    ns = {"np": np, "Gf": Gf}
    exec(compile(ast.Module(body=consts + keep, type_ignores=[]), "make_variants.py", "exec"), ns)
    return ns["tilt_matrix"], ns["R_CAP"]


@unittest.skipUnless(HAVE_USD, "usd-core not installed")
class Geometry(unittest.TestCase):
    def test_rotation_keeps_pivot_and_has_requested_angle(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            pivot = rng.normal(size=3); d = rng.normal(size=2); deg = rng.uniform(1, 40)
            axis = np.cross([0, 0, 1.0], [d[0], d[1], 0])
            M = G.rot_about(pivot, axis, deg)
            np.testing.assert_allclose(G.apply(pivot[None], M)[0], pivot, atol=1e-12)
            self.assertAlmostEqual(G.axis_angle_deg(M), deg, places=9)

    def test_tilt_matches_frozen_gate_convention(self):
        gate_tilt, r_cap = gate_tilt_matrix()
        for deg in (5, 10, 20, 30):
            for az in (0, 37, 90, 200):
                d = (np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0)
                np.testing.assert_allclose(G.np4(G.tilt_local(r_cap, deg, d)), G.np4(gate_tilt(Gf.Matrix4d(1.0), deg, d)),
                                           atol=1e-12)

    def test_rim_pivot_tilt_invariants(self):
        R, H = 0.668, 0.75
        for deg in (5, 20, 40):
            for az in (0, 120, 250):
                d = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0])
                M = G.tilt_local(R, deg, d)
                rim = G.apply(G.ring(R, 0.0, 3600), M)
                self.assertGreaterEqual(rim[:, 2].min(), -1e-12)                   # never sinks below the seat
                self.assertAlmostEqual(rim[:, 2].max(), 2 * R * np.sin(np.radians(deg)), places=5)
                mv = G.apply(np.array([[0, 0, H]]), M)[0] - [0, 0, H]
                self.assertGreater(np.dot(mv[:2], d[:2]), 0)                         # top moves toward d
                self.assertAlmostEqual(mv[0] * d[1] - mv[1] * d[0], 0, places=12)          # and only toward d

    def test_world_direction_roundtrip_on_tilted_prims(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            yaw, tx, ty = rng.uniform(0, 360), rng.uniform(-3, 3), rng.uniform(-3, 3)
            M = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), tx) * Gf.Rotation(Gf.Vec3d(0, 1, 0), ty) *
                                        Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw))
            W = G.np4(M) * np.array([[0.0254], [0.0254], [0.0254], [1]])
            for az in (0, 45, 133.7, 300):
                v = G.world_dir_to_local(W, az)
                self.assertAlmostEqual(v[2], 0.0)
                back = G.local_dir_to_world_az(W, v)
                self.assertAlmostEqual(((back - az + 180) % 360) - 180, 0.0, places=8)


@unittest.skipUnless(HAVE_USD, "usd-core not installed")
class Config(unittest.TestCase):
    def cfg(self, name):
        return json.load(open(os.path.join(CFG, name)))

    def test_shipped_configs_are_valid(self):
        for name in ("smoke.json", "tilt_sweep.json"):
            G.validate_config(self.cfg(name))

    def test_smoke_has_no_crooked_lean_or_offset(self):
        c = self.cfg("smoke.json")
        self.assertEqual(c["states"]["crooked_cap"], 0); self.assertEqual(c["lean_fraction"], 0)
        self.assertEqual(c["pass_variants"]["offset"], 0)

    def test_sweep_covers_every_angle_direction_and_parity(self):
        c = self.cfg("tilt_sweep.json"); cases = G.sweep_cases(c)
        n_dir = len(c["directions_deg"]); n_ang = len([a for a in c["angles_deg"] if a > 0])
        self.assertEqual(len(cases), 2 * (1 + n_ang * n_dir))
        self.assertEqual(len({x for x in cases}), len(cases))

    def test_rejects_bad_configs(self):
        base = self.cfg("smoke.json")
        bad = [dict(base, extra=1), dict(base, mode="grid"), dict(base, fixture="S"), dict(base, schema=2),
               dict(base, tilt_deg=[30, 10]), dict(base, n_scenes=0), dict(base, lean_fraction=1.5),
               dict(base, states={"pass": 1, "chipped": 1}), dict(base, lighting={"intensity_scale": [1, 2]}),
               dict(base, states={"pass": 0.5, "crooked_cap": 0.5}, tilt_deg=[0, 10]),
               dict(self.cfg("tilt_sweep.json"), pattern="all")]
        for c in bad:
            with self.assertRaises(G.ConfigError):
                G.validate_config(c)

    @unittest.skipUnless(HAVE_TWIN, "twin not found (set IGNITE_TWIN)")
    def test_limits_from_twin_geometry(self):
        c = dict(self.cfg("smoke.json"), offset_mm=[0.5, 10.0])
        with self.assertRaises(G.ConfigError):
            G.validate_config(c, twin())
        c = dict(self.cfg("smoke.json"), lift_mm=[1, 20])
        with self.assertRaises(G.ConfigError):
            G.validate_config(c, twin())


@unittest.skipUnless(HAVE_USD and HAVE_TWIN, "usd-core or twin not available (set IGNITE_TWIN)")
class OnTwin(unittest.TestCase):
    def setUp(self):
        self.t = twin(); self.tmp = tempfile.mkdtemp(prefix="test_gen_scenes_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def scene(self, spec, name="s"):
        return G.read_back(self.t, G.author_scene(self.t, spec, os.path.join(self.tmp, name)))

    def all_pass(self, sid="x"):
        return G.SceneSpec(sid, 0, 0, 30, [G.make_pos(p, "pass") for p in self.t.ids])

    def test_twin_dimensions_match_frozen_gate_constants(self):
        _, r_cap = gate_tilt_matrix()
        self.assertAlmostEqual(self.t.R_CAP, r_cap, places=3)
        self.assertAlmostEqual(self.t.R_TUBE, 0.57, places=3); self.assertAlmostEqual(self.t.H_TUBE, 4.1, places=3)

    def test_selftest(self):
        fails, tmp = G.selftest(self.t, log=lambda *a: None)
        shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(fails, [])

    def test_seated_scene_matches_runtime_map_and_camera(self):
        rb = self.scene(self.all_pass())
        self.assertEqual(G.camera_issues(self.t, rb), [])
        pm = json.load(open(G.MAP_PATH))
        self.assertEqual(pm["meta"]["station_sha256"], self.t.station_sha256)
        for p in pm["positions"]:
            r = rb["positions"][p["position"]]
            self.assertEqual(r["state"], "pass")
            np.testing.assert_allclose(r["cap_center_px"], p["cap_center_px"], atol=0.01)
            self.assertEqual(r["anchor_bbox_px"], p["anchor_bbox_px"])
            self.assertEqual(r["cap_shift_px"], 0.0)

    def test_every_state_reads_back_from_the_composed_stage(self):
        spec = G.selftest_spec(self.t); rb = self.scene(spec)
        self.assertEqual(G.spec_mismatches(self.t, spec, rb), [])
        self.assertEqual(G.physical_issues(self.t, rb), [])
        P = rb["positions"]
        self.assertEqual((P["G02"]["state"], P["G04"]["state"], P["G20"]["state"]), ("missing_cap", "no_vial", "no_vial"))
        self.assertEqual((P["G03"]["state"], P["G03"]["variant"]), ("crooked_cap", "tilt"))
        self.assertAlmostEqual(P["G03"]["tilt_deg"], 20, places=6)
        self.assertEqual((P["G05"]["state"], P["G05"]["variant"]), ("crooked_cap", "lift"))
        self.assertAlmostEqual(P["G05"]["seat_gap_min_mm"], 3.0, places=6)
        self.assertEqual((P["G06"]["state"], P["G06"]["variant"]), ("pass", "offset"))
        self.assertAlmostEqual(P["G06"]["base_offset_mm"], 1.0, places=6)
        self.assertEqual(P["G07"]["state"], "pass"); self.assertGreater(P["G07"]["lean_deg"], 0)
        self.assertEqual(P["G09"]["state"], "missing_cap"); self.assertGreater(P["G09"]["lean_deg"], 0)
        self.assertEqual(P["G12"]["state"], "crooked_cap")                        # 5 deg: not fully seated = crooked
        self.assertLess(P["G12"]["cap_shift_px"], P["G03"]["cap_shift_px"])

    def test_tilt_shift_is_zero_when_seated_and_grows_with_angle(self):
        for pid, az in (("G13", 0), ("G01", 135), ("G30", 270)):
            shifts = []
            for deg in (0, 5, 10, 20, 30, 40):
                spec = self.all_pass(f"t{deg}")
                i = self.t.ids.index(pid)
                if deg:
                    spec.positions[i] = G.make_pos(pid, "crooked_cap", "tilt", tilt_deg=deg, tilt_dir_deg=az)
                shifts.append(self.scene(spec, f"{pid}_{deg}")["positions"][pid]["cap_shift_px"])
            self.assertEqual(shifts[0], 0.0)
            self.assertTrue(all(b > a for a, b in zip(shifts, shifts[1:])), f"{pid}: {shifts}")

    def test_physical_rules_flag_violations(self):
        lim = self.t.lean_limit_deg("G13", 90)
        ok, bad = self.all_pass("ok"), self.all_pass("bad")
        ok.positions[12] = G.make_pos("G13", "pass", lean_deg=0.9 * lim, lean_dir_deg=90)
        bad.positions[12] = G.make_pos("G13", "pass", lean_deg=1.2 * lim, lean_dir_deg=90)
        self.assertEqual(G.physical_issues(self.t, self.scene(ok, "ok")), [])
        self.assertTrue(any("plate hole" in m for m in G.physical_issues(self.t, self.scene(bad, "bad"))))

    def test_sampling_refuses_leans_the_plate_cannot_hold(self):
        c = json.load(open(os.path.join(CFG, "smoke.json")))
        c.update(lean_fraction=1.0, lean_deg=[6.0, 7.0])
        with self.assertRaises(G.PhysicalError):
            G.build_spec(c, self.t, 0, 0)

    def test_layer_touches_only_positions_lights_and_materials(self):
        from pxr import Sdf
        spec = G.selftest_spec(self.t)
        path = G.author_scene(self.t, spec, os.path.join(self.tmp, "layer"))
        paths = []
        Sdf.Layer.FindOrOpen(path).Traverse("/", lambda p: paths.append(p) if p.IsPrimPath() else None)
        allowed = {G.ROOT} | {f"{G.ROOT}/{p}_{k}" for p in self.t.ids for k in ("tube", "cap")} \
            | {f"{G.ROOT}/{n}" for n in G.LIGHTS} | {f"{G.ROOT}/{n}/{n}" for n in G.LIGHTS} \
            | {f"{G.ROOT}/_materials"} | {f"{G.ROOT}/_materials/{m}" for m in G.MATS.values()} \
            | {f"{G.ROOT}/_materials/{m}/Principled_BSDF" for m in G.MATS.values()}
        self.assertTrue({str(p) for p in paths} <= allowed, sorted({str(p) for p in paths} - allowed))
        self.assertNotIn(f"{G.ROOT}/CapSealCam", {str(p) for p in paths})
        untouched = [p for p in spec.positions if p.state == "pass" and p.variant == "seated" and not p.lean_deg]
        for p in untouched:
            self.assertNotIn(f"{G.ROOT}/{p.position}_cap", {str(x) for x in paths})

    def test_lights_and_materials_read_back(self):
        spec = G.selftest_spec(self.t); rb = self.scene(spec)
        base = self.scene(self.all_pass("base"), "base")
        for n in G.LIGHTS:
            want = base["lights"][n]["intensity"] * spec.lighting[n]["intensity_scale"]
            self.assertAlmostEqual(rb["lights"][n]["intensity"], want, delta=want * 1e-6)
            self.assertAlmostEqual(rb["lights"][n]["color_temperature_k"], 5600.0, places=2)
            self.assertIsNone(base["lights"][n]["color_temperature_k"])
        self.assertAlmostEqual(rb["materials"]["cap"]["roughness"], 0.35, places=6)
        self.assertNotEqual(rb["materials"]["cap"]["diffuseColor"], base["materials"]["cap"]["diffuseColor"])

    def test_generate_is_reproducible_and_limit_independent(self):
        cfg, sha = G.load_config(os.path.join(CFG, "smoke.json"), self.t)
        a, b, c, s1 = (os.path.join(self.tmp, x) for x in ("a", "b", "c", "s1"))
        quiet = lambda *x: None
        G.generate(self.t, cfg, sha, a, 0, 5, log=quiet); G.generate(self.t, cfg, sha, b, 0, 5, log=quiet)
        self.assertEqual(open(os.path.join(a, "SHA256SUMS")).read(), open(os.path.join(b, "SHA256SUMS")).read())
        G.generate(self.t, cfg, sha, c, 0, 3, log=quiet)
        for i in range(3):
            f = f"scenes/smoke_{i:05d}/scene.usda"
            self.assertEqual(open(os.path.join(a, f), "rb").read(), open(os.path.join(c, f), "rb").read())
        G.generate(self.t, cfg, sha, s1, 1, 1, log=quiet)
        f = "scenes/smoke_00000/labels.json"
        self.assertNotEqual(open(os.path.join(a, f)).read(), open(os.path.join(s1, f)).read())
        run = json.load(open(os.path.join(a, "run.json")))
        self.assertEqual((run["seed"], run["config_sha256"], run["rendered"]), (0, sha, False))
        rows = list(csv.DictReader(open(os.path.join(a, "manifest.csv"))))
        self.assertEqual(len(rows), 5 * 30)
        self.assertTrue(all(r["image"] == "" for r in rows))
        with self.assertRaises(FileExistsError):
            G.generate(self.t, cfg, sha, a, 0, 1, log=quiet)

    def test_station_file_is_never_modified(self):
        before = self.t.station_sha256
        cfg, sha = G.load_config(os.path.join(CFG, "tilt_sweep.json"), self.t)
        G.generate(self.t, cfg, sha, os.path.join(self.tmp, "sw"), 0, 4, log=lambda *x: None)
        self.assertEqual(G.sha256_file(self.t.station), before)

    def test_labels_work_with_cut_crops_and_runtime_crops(self):
        import cv2
        import positions as P
        spec = G.selftest_spec(self.t)
        sdir = os.path.join(self.tmp, "cc")
        rb = G.read_back(self.t, G.author_scene(self.t, spec, sdir))
        pm = P.PositionMap(G.MAP_PATH, offset=(0, 0))
        lab = G.labels_json(spec, rb, [pm.crop_box(i)[0] for i in range(30)], "x")
        lp = os.path.join(sdir, "labels.json"); json.dump(lab, open(lp, "w"))
        img = os.path.join(self.tmp, "black.png"); cv2.imwrite(img, np.zeros((480, 640, 3), np.uint8))
        want = ",".join(o["position"] for o in lab["positions"])
        out = subprocess.run([sys.executable, os.path.join(HERE, "cut_crops.py"), img, lp, os.path.join(self.tmp, "crops"), want],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        rows = list(csv.DictReader(open(os.path.join(self.tmp, "crops", "crops.csv"))))
        self.assertEqual([r["label"] for r in rows], [o["state"] for o in lab["positions"]])
        crops = P.extract_crops(np.zeros((480, 640, 3), np.uint8), pm, "synthetic")
        self.assertEqual(len(crops), 30)
        for o, c in zip(lab["positions"], crops):
            if "bbox_px" in o:
                b, (x0, y0, x1, y1) = o["bbox_px"], c.box
                self.assertEqual(o["bbox_in_crop"], b[0] >= x0 and b[1] >= y0 and b[2] <= x1 and b[3] <= y1)


TWIN_FIX = os.path.expanduser(os.environ.get("IGNITE_TWIN_FIX", "~/Downloads/csi_digital_twin/twin_fix1"))
HAVE_FIX = os.path.isfile(os.path.join(TWIN_FIX, "FIX.json"))
_fix = None


def fix_twin():
    global _fix
    if _fix is None:
        _fix = G.Twin(TWIN_FIX)
    return _fix


@unittest.skipUnless(HAVE_USD and HAVE_TWIN and HAVE_FIX, "needs the twin and its work1 overlay (IGNITE_TWIN_FIX)")
class TrainingRecipe(unittest.TestCase):
    """Training configs run on the work1 overlay twin with the DEV-derived recipe (D037)."""

    def setUp(self):
        self.t = fix_twin(); self.tmp = tempfile.mkdtemp(prefix="test_recipe_")
        self.cfg, self.sha = G.load_config(os.path.join(CFG, "train_v1.json"), self.t)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_overlay_twin_is_the_gate_run2_base(self):
        fix = json.load(open(os.path.join(TWIN_FIX, "FIX.json")))
        self.assertEqual(self.t.fix["fix"], "work1")
        self.assertEqual(self.t.base_station_sha256, json.load(open(G.MAP_PATH))["meta"]["station_sha256"])
        for n in G.LIGHTS:
            self.assertAlmostEqual(self.t.base_intensity[n], fix["F1a_lights"]["intensity"][f"{G.ROOT}/{n}/{n}"]["fixed"], places=2)
        self.assertEqual(dict((k, v) for k, _, v in self.t.cap_inputs)["diffuseColor"][2], 0.0)

    def test_configs_and_twins_must_match(self):
        for name in ("train_v1.json", "tilt_train.json"):
            G.load_config(os.path.join(CFG, name), self.t)
            with self.assertRaises(G.ConfigError):
                G.load_config(os.path.join(CFG, name), twin())                    # original twin: refused
        with self.assertRaises(G.ConfigError):
            G.load_config(os.path.join(CFG, "smoke.json"), self.t)              # old config on the overlay: refused
        with self.assertRaises(G.ConfigError):
            G.validate_config(dict(self.cfg, recipe={"file": "configs/recipe_dev.json", "sha256": "0" * 64}), self.t)

    def test_generated_scenes_follow_the_recipe(self):
        G.generate(self.t, self.cfg, self.sha, os.path.join(self.tmp, "g"), 0, 6, log=lambda *a: None)
        L, M = self.cfg["lighting"], self.cfg["materials"]
        for i in range(6):
            sdir = os.path.join(self.tmp, "g", "scenes", f"train_v1_{i:05d}")
            lab = json.load(open(os.path.join(sdir, "labels.json"))); rb = G.read_back(self.t, os.path.join(sdir, "scene.usda"))
            ev = lab["lighting"]["key_overhead"]["exposure_ev"]
            self.assertTrue(L["exposure_ev"][0] <= ev <= L["exposure_ev"][1])
            for n in G.LIGHTS:
                v = lab["lighting"][n]
                self.assertEqual(v["exposure_ev"], ev)
                self.assertAlmostEqual(rb["lights"][n]["intensity"], self.t.base_intensity[n] * 2 ** ev * v["jitter"], delta=1e-3)
                self.assertTrue(L["color_temperature_k"][0] <= rb["lights"][n]["color_temperature_k"] <= L["color_temperature_k"][1])
            crooked = set()
            for o in lab["positions"]:
                if o["state"] in ("pass", "crooked_cap"):
                    g = M["cap_groups"][o["cap_group"]]
                    self.assertEqual(rb["positions"][o["position"]]["cap_material"], f"{G.ROOT}/_materials/ignite_cap_{o['position']}")
                    k = ((o["cap_ior"] - 1) / (o["cap_ior"] + 1)) ** 2 / 0.04
                    self.assertTrue(g["spec_scale"][0] - 1e-4 <= k <= g["spec_scale"][1] + 1e-4, (o["position"], k))
                    self.assertTrue(M["cap_roughness"][0] <= o["cap_roughness"] <= M["cap_roughness"][1])
                else:
                    self.assertNotIn("cap_group", o)
                if o["state"] == "crooked_cap":
                    self.assertTrue(20 <= o["cap_tilt_deg"] <= 30 and o["bbox_in_crop"]); crooked.add(o["position"])
            for p in crooked:
                self.assertFalse(crooked & set(self.t.grid_neighbors[p]), f"scene {i}: {p} has a crooked neighbour")

    def test_per_cap_material_tampering_is_caught(self):
        from pxr import Sdf, Usd, UsdShade
        spec = G.build_spec(self.cfg, self.t, 0, 3)
        path = G.author_scene(self.t, spec, os.path.join(self.tmp, "tamper"))
        self.assertEqual(G.spec_mismatches(self.t, spec, G.read_back(self.t, path)), [])
        p = next(s.position for s in spec.positions if s.cap_rgb is not None)
        lay = Sdf.Layer.FindOrOpen(path); st = Usd.Stage.Open(lay); st.SetEditTarget(lay)
        UsdShade.Shader(st.GetPrimAtPath(f"{G.ROOT}/_materials/ignite_cap_{p}/Principled_BSDF")).GetInput("ior").Set(1.5)
        lay.Save()
        self.assertTrue(any("cap_ior" in m for m in G.spec_mismatches(self.t, spec, G.read_back(self.t, path))))

    def test_sampled_mix_matches_the_recipe(self):
        specs = [G.build_spec(self.cfg, self.t, 0, i) for i in range(120)]
        caps = [s for sp in specs for s in sp.positions if s.cap_group]
        pale = sum(s.cap_group == "pale" for s in caps) / len(caps)
        self.assertAlmostEqual(pale, self.cfg["materials"]["cap_groups"]["pale"]["weight"], delta=0.04)
        empty = sum(s.state == "no_vial" for sp in specs for s in sp.positions) / (30 * len(specs))
        self.assertLess(empty, 0.3)                                                # rebalanced away from smoke's 58 %
        K = [sp.placements for sp in specs]
        self.assertTrue(min(K) <= 3 and max(K) == 30 and sum(k <= 10 for k in K) >= 5, K)   # empty-start fills covered
        with self.assertRaises(G.ConfigError):
            G.validate_config(dict(self.cfg, placements_mix=[[1.0, [0, 40]]]), self.t)

    def test_pass_offsets_are_sideways_and_fully_seated(self):
        spec = next(sp for sp in (G.build_spec(self.cfg, self.t, 0, i) for i in range(40))
                    if sum(s.variant == "offset" for s in sp.positions) >= 3)
        rb = G.read_back(self.t, G.author_scene(self.t, spec, os.path.join(self.tmp, "off")))
        for s in spec.positions:
            if s.variant == "offset":
                r = rb["positions"][s.position]
                self.assertEqual((r["state"], r["variant"]), ("pass", "offset"))
                self.assertLess(max(abs(r["seat_gap_min_mm"]), abs(r["seat_gap_max_mm"]), r["tilt_deg"]), 1e-9)
                self.assertAlmostEqual(r["base_offset_mm"], s.offset_mm, places=6)


class FrozenGate(unittest.TestCase):
    def test_gate_kit_matches_manifest(self):
        for line in open(os.path.join(HERE, "MANIFEST.sha256")):
            digest, name = line.split()
            got = hashlib.sha256(open(os.path.join(HERE, name), "rb").read()).hexdigest()
            self.assertEqual(got, digest, f"{name} differs from the frozen gate-v2.0.2 kit")


if __name__ == "__main__":
    unittest.main()
