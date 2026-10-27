"""Team Ignite copy of the twin's tools/render_example.py. Changes, all marked "IGNITE":
  - twin folder comes from --twin (the kit lives outside the twin folder)
  - the undistorted 768x576 pinhole render is kept next to the output (out_pinhole768.png)
  - render settings, Isaac Sim version, GPU and wall-clock timings go to out.json

Render one image from a camera of the digital twin with Isaac Sim (headless).

    ./python.sh tools/render_example.py --camera CapSealCam --out cap_seal.png
    ./python.sh tools/render_example.py --camera FillCam --frame 301 --out fill.png
    ./python.sh tools/render_example.py --camera CapSealCam --stage stations/tray_inspection/tray_inspection.usdc --out cap_seal.png
    ./python.sh tools/render_example.py --camera FillCam --stage stations/fill_station/fill_station.usda --out fill.png

Run with Isaac Sim's python (python.sh in the Isaac Sim install). By default it opens the full line
(csi_line.usda); --stage opens a single station file instead. The camera settings come from cameras/<name>.json.
CapSealCam is rendered with overscan and then given the real lens distortion (tools/apply_lens.py).
"""
import argparse, json, os, subprocess, sys, time, traceback
T0 = time.time()  # IGNITE
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--twin", required=True, help="csi_digital_twin_v1 folder")  # IGNITE
ap.add_argument("--camera", required=True, choices=["CapSealCam", "FillCam"],
                help="camera name; settings come from cameras/<name>.json")
ap.add_argument("--stage", default="csi_line.usda", help="csi_line.usda or one station file")
ap.add_argument("--frame", type=int, default=1, help="time code (fill station animation: 1-686 at 30 fps)")
ap.add_argument("--out", default="render.png")
ap.add_argument("--subframes", type=int, default=32, help="path-tracing subframes (quality)")
ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim / Kit argument (repeatable)")
a = ap.parse_args()
HERE = os.path.abspath(os.path.expanduser(a.twin))  # IGNITE

cam = json.load(open(os.path.join(HERE, "cameras", a.camera + ".json")))
stage_path = Path(a.stage).expanduser()
if not stage_path.is_absolute():
    stage_path = Path(HERE) / stage_path
stage_path = str(stage_path.resolve())
if not os.path.isfile(stage_path):
    ap.error(f"stage does not exist: {stage_path}")
if a.frame < 1 or (a.camera == "FillCam" and a.frame > 686):
    ap.error("use a positive time code; the fill animation covers 1-686")
if a.subframes < 1:
    ap.error("--subframes must be positive")
Path(a.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
a.out = str(Path(a.out).expanduser().resolve())
rw, rh = cam["render_px"]

from isaacsim import SimulationApp
T_IMPORT = time.time()  # IGNITE
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh, "extra_args": a.kit_arg})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_path):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    prim = next((p for p in [cam["prim_in_csi_line"], cam["prim_in_station_file"]]
                 if stage.GetPrimAtPath(p).IsValid()), None)
    if prim is None:
        raise RuntimeError(f"camera {a.camera} not found in {stage_path}")

    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(a.frame / stage.GetTimeCodesPerSecond())
    kit.update()

    rp = rep.create.render_product(prim, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()
    T_STAGE = time.time()  # IGNITE: app started + stage loaded

    # RTX's first sync of a scaled camera (CapSealCam carries the 0.0254 inch scale) misreads its
    # clippingRange and renders black. Changing and restoring the value in the session layer forces a
    # correct re-sync; the authored camera in the USD file is untouched.
    from pxr import Gf, UsdGeom
    clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
    kit.update()
    clip.Set(near_far)
    kit.update()

    T_R0 = time.time()  # IGNITE
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
    data = np.asarray(ann.get_data())
    if data.ndim != 3 or data.shape[:2] != (rh, rw) or data.shape[2] < 3:
        raise RuntimeError(f"RGB capture returned shape {data.shape}; check GPU/RTX startup errors")
    img = data[..., :3].copy()
    T_R1 = time.time()  # IGNITE

    # Save and post-process before close(): Isaac's default fast shutdown exits
    # the process and will never execute statements after kit.close().
    if cam.get("overscan", 1.0) != 1.0:
        tmp = a.out.replace(".png", "_pinhole768.png")  # IGNITE: kept, not deleted
        if not cv2.imwrite(tmp, img[..., ::-1]):
            raise OSError(f"could not write {tmp}")
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "tools", "apply_lens.py"), tmp,
                            os.path.join(HERE, "cameras", a.camera + ".json"), a.out], check=True)
        finally:
            pass  # IGNITE: keep the undistorted render
    elif not cv2.imwrite(a.out, img[..., ::-1]):
        raise OSError(f"could not write {a.out}")
    try:  # IGNITE: record what produced this image
        import carb, omni.kit.app
        ver = omni.kit.app.get_app().get_build_version()
    except Exception as e:
        ver = f"unknown ({e})"
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except Exception as e:
        gpu = f"unknown ({e})"
    json.dump({"stage": stage_path, "camera": a.camera, "camera_prim": prim, "renderer": "PathTracing",
               "rt_subframes": a.subframes, "orchestrator_steps": 2, "render_px": [rw, rh],
               "final_px": cam["final_image_px"], "lens_k1": cam.get("lens_k1"), "focal_px_final": cam.get("focal_px_final"),
               "isaac_kit_version": ver, "gpu": gpu,
               "seconds": {"app_start_and_stage_load": round(T_STAGE - T_IMPORT, 1), "render": round(T_R1 - T_R0, 1),
                           "total_process": round(time.time() - T0, 1)}},
              open(a.out.replace(".png", ".json"), "w"), indent=1)
    print("wrote", a.out, flush=True)
except BaseException:
    traceback.print_exc()
    # Preserve failure status even when Kit's fast shutdown terminates Python.
    kit.close(exit_code=1)
    raise
kit.close()
