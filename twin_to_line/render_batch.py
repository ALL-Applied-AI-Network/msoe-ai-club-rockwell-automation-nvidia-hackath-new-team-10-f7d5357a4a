"""Batch renderer: many gen_scenes.py scenes in ONE Isaac Sim session (render_keep.py starts Isaac once per scene,
~30 s each). Per scene it does exactly what render_keep.py does: open the stage, wait for loading, re-sync the scaled
CapSealCam's clipping range, two orchestrator steps of --subframes path-tracing subframes, read the rgb annotator,
keep the 768x576 pinhole image and apply the twin's lens (tools/apply_lens.py arithmetic, run in-process; the first
scene is also run through apply_lens.py itself and must match it exactly).

    <isaac>/python.sh render_batch.py --twin TWIN --scenes OUT/scenes --out RENDERS [--subframes 64] [--only id,id]

Writes RENDERS/<scene_id>.png (640x480), <scene_id>_pinhole768.png, <scene_id>.json (timings), render_log.csv.
Scenes whose .png already exists are skipped, so a stopped run resumes where it left off. Check it against
render_keep.py on a few scenes before trusting a large batch (same framing, differences only from sampling noise).
"""
import argparse, csv, json, os, subprocess, sys, time, traceback
T0 = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("--twin", required=True); ap.add_argument("--scenes", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--camera", default="CapSealCam"); ap.add_argument("--subframes", type=int, default=64)
ap.add_argument("--only", help="comma-separated scene ids (default: every scene folder)")
a = ap.parse_args()
TWIN = os.path.abspath(os.path.expanduser(a.twin)); OUT = os.path.abspath(a.out); os.makedirs(OUT, exist_ok=True)
cam = json.load(open(os.path.join(TWIN, "cameras", a.camera + ".json")))
rw, rh = cam["render_px"]
ids = a.only.split(",") if a.only else sorted(d for d in os.listdir(a.scenes) if os.path.isfile(os.path.join(a.scenes, d, "scene.usda")))
todo = [s for s in ids if not os.path.exists(os.path.join(OUT, s + ".png"))]
print(f"{len(ids)} scenes, {len(ids) - len(todo)} already rendered, {len(todo)} to render", flush=True)

from isaacsim import SimulationApp
T_IMPORT = time.time()
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline, omni.kit.app
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    W, H = cam["final_image_px"]; f, k1 = cam["focal_px_final"], cam["lens_k1"]           # tools/apply_lens.py
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]); D = np.array([k1, 0, 0, 0, 0.0])
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    und = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1)[:, None, :], K, D).reshape(H, W, 2)
    MAPX = (und[..., 0] * f + rw / 2).astype(np.float32); MAPY = (und[..., 1] * f + rh / 2).astype(np.float32)
    lens = lambda img: cv2.remap(img, MAPX, MAPY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    ctx = omni.usd.get_context()
    log = open(os.path.join(OUT, "render_log.csv"), "a", newline=""); w = csv.writer(log)
    if log.tell() == 0:
        w.writerow(["scene_id", "status", "load_s", "render_s", "total_s", "error"])
    ver = omni.kit.app.get_app().get_build_version()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"Isaac {ver} | {gpu} | app start {time.time() - T_IMPORT:.1f}s", flush=True)
    for n, sid in enumerate(todo, 1):
        t0 = time.time(); stage_path = os.path.join(os.path.abspath(a.scenes), sid, "scene.usda")
        try:
            if not ctx.open_stage(stage_path):
                raise RuntimeError(f"could not open stage: {stage_path}")
            while ctx.get_stage_loading_status()[2] > 0:
                kit.update()
            for _ in range(10):
                kit.update()
            stage = ctx.get_stage()
            prim = next((p for p in [cam["prim_in_csi_line"], cam["prim_in_station_file"]] if stage.GetPrimAtPath(p).IsValid()), None)
            if prim is None:
                raise RuntimeError(f"camera {a.camera} not found in {stage_path}")
            omni.timeline.get_timeline_interface().set_current_time(1 / stage.GetTimeCodesPerSecond()); kit.update()
            rp = rep.create.render_product(prim, (rw, rh))
            ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
            rep.orchestrator.set_capture_on_play(False); kit.update()
            clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr(); near_far = clip.Get()
            stage.SetEditTarget(stage.GetSessionLayer())                     # render_keep.py's clipping re-sync
            clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1])); kit.update(); clip.Set(near_far); kit.update()
            t1 = time.time()
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
            data = np.asarray(ann.get_data())
            if data.ndim != 3 or data.shape[:2] != (rh, rw) or data.shape[2] < 3:
                raise RuntimeError(f"RGB capture returned shape {data.shape}")
            img = data[..., :3][..., ::-1].copy(); t2 = time.time()
            pin = os.path.join(OUT, sid + "_pinhole768.png"); final = os.path.join(OUT, sid + ".png")
            if not cv2.imwrite(pin, img) or not cv2.imwrite(final, lens(img)):
                raise OSError(f"could not write {final}")
            if n == 1:                                                         # in-process lens == the twin's tool
                ref = os.path.join(OUT, sid + "_apply_lens_check.png")
                subprocess.run([sys.executable, os.path.join(TWIN, "tools", "apply_lens.py"), pin,
                                os.path.join(TWIN, "cameras", a.camera + ".json"), ref], check=True, capture_output=True)
                if not np.array_equal(cv2.imread(ref), cv2.imread(final)):
                    raise RuntimeError("in-process lens differs from tools/apply_lens.py")
                os.remove(ref)
            ann.detach([rp]); rp.destroy()
            json.dump({"stage": stage_path, "camera": a.camera, "camera_prim": prim, "renderer": "PathTracing",
                       "rt_subframes": a.subframes, "orchestrator_steps": 2, "render_px": [rw, rh], "final_px": [W, H],
                       "isaac_kit_version": ver, "gpu": gpu, "renderer_script": "render_batch.py",
                       "seconds": {"load": round(t1 - t0, 2), "render": round(t2 - t1, 2), "total": round(time.time() - t0, 2)}},
                      open(os.path.join(OUT, sid + ".json"), "w"), indent=1)
            w.writerow([sid, "ok", round(t1 - t0, 2), round(t2 - t1, 2), round(time.time() - t0, 2), ""]); log.flush()
        except Exception as e:
            traceback.print_exc()
            w.writerow([sid, "error", "", "", round(time.time() - t0, 2), f"{type(e).__name__}: {e}"]); log.flush()
        if n % 10 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} {sid} {time.time() - t0:.1f}s (elapsed {time.time() - T0:.0f}s)", flush=True)
    log.close()
    print(f"done in {time.time() - T0:.0f}s", flush=True)
except BaseException:
    traceback.print_exc()
    kit.close(exit_code=1)
    raise
kit.close()
