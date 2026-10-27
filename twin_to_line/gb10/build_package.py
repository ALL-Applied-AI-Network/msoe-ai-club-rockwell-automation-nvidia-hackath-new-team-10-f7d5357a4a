"""Build the GB10 package (aarch64 Linux, rule inspector only) from one commit, on the Mac. LIVE_RUNBOOK.md section 7.

    python gb10/build_package.py [--out results/gb10_pkg]          (always from HEAD; runtime files must be committed)

Writes <out>/ttl_gb10_<sha>/ and <out>/ttl_gb10_<sha>.tar (+ .sha256), never into git:
  code.tar      git archive of the commit (ARCHIVE_COMMIT carries the sha); holds every runtime file: maps,
                registration_dev.json, rule_thresholds.json, frame_warp_1080.json
  runtime/      standalone CPython 3.11 for linux aarch64 (python-build-standalone, pinned, sha256 checked)
  wheels/cpXY/  linux aarch64 wheels for exactly the runtime's third-party imports (numpy, OpenCV headless) and a
                requirements.txt with the pins: cp311 = the Mac's versions; cp310 / cp312 for a system Python
  preflight/    the CSI 1080 still, 10 DEV run-1 event frames (640) and their 1080 copies, and the Mac's predictions for
                all three (expected/), produced here by the same live_inspect code from the working tree, which must
                equal the commit
  data_index/   filename/timestamp/run metadata only, to refuse protected stills without the real dataset
  bootstrap.sh, README.txt, IMPORTS.txt, FILES.txt, SHA256SUMS
Beside the tar: start.sh, stage.sh, seal.sh, live.sh, export.sh, monday_ops.py, monday.sha256 (copy all to USB root).
Only DEV run 1 frames are bundled (real_data refuses anything else).
"""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, tarfile, urllib.request
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import cv2
import real_data as R
import frame_warp as FW
import live_inspect as L
from replay_1080 import warp_up

PBS = {"tag": "20261009", "name": "cpython-3.11.17+20261009-aarch64-unknown-linux-gnu-install_only.tar.gz",
       "sha256": "693ef5c3064e13af98178fba648c4757be92b842546a0ee2dae19ce1e3d863ba", "max_mb": 200}
PINS = {"cp311": ["numpy==2.4.6", "opencv-python-headless==5.0.0.93"],          # = the Mac venv
        "cp312": ["numpy==2.4.6", "opencv-python-headless==5.0.0.93"],
        "cp310": ["numpy==2.2.6", "opencv-python-headless==5.0.0.93"]}         # numpy >= 2.3 needs Python 3.11
PLATFORMS = ("manylinux_2_28_aarch64", "manylinux_2_17_aarch64", "manylinux2014_aarch64")
STILL = os.path.join(HERE, "results", "csi_1080", "csi_1080_empty_G.jpg")
REPLAY1 = os.path.join(HERE, "results", "live", "replay_run1")
EVENTS1 = os.path.join(HERE, "results", "prototype", "events", "events_run1_video.csv")
N_FRAMES = 10
MONDAY_FILES = ("start.sh", "stage.sh", "seal.sh", "live.sh", "export.sh", "monday_ops.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout.strip()


def python_runtime(dst):
    url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS['tag']}/{PBS['name'].replace('+', '%2B')}"
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        mb = int(r.headers["Content-Length"]) / 1e6
    if mb > PBS["max_mb"]:
        raise SystemExit(f"{PBS['name']} is {mb:.0f} MB (> {PBS['max_mb']} MB): ask before downloading")
    path = os.path.join(dst, PBS["name"]); urllib.request.urlretrieve(url, path)
    if sha256(path) != PBS["sha256"]:
        raise SystemExit(f"{PBS['name']}: sha256 differs from the release's SHA256SUMS")
    return path


def wheels(dst):
    for tag, pins in PINS.items():
        d = os.path.join(dst, tag); os.makedirs(d)
        cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--disable-pip-version-check", "--only-binary=:all:", "--implementation", "cp",
               "--python-version", tag[2] + "." + tag[3:], "--dest", d, *sum((["--platform", p] for p in PLATFORMS), []), *pins]
        subprocess.run(cmd, check=True)
        open(os.path.join(d, "requirements.txt"), "w").write("\n".join(pins) + "\n")


def package_index(dst):
    """Protection metadata only: no images or videos are opened or copied."""
    ix = R.Index()
    os.makedirs(dst)
    with open(os.path.join(dst, "image_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "timestamp", "video_id"])
        w.writeheader()
        for name, run in sorted(ix.run_of.items()):
            w.writerow({"filename": name, "timestamp": ix.time_of[name].isoformat(), "video_id": run})


def preflight_data(dst):
    """The bundled frames and the Mac's predictions for them (the same code as live_inspect --preflight)."""
    ix = R.Index(); events = [r["frame"] for r in csv.DictReader(open(EVENTS1))][-N_FRAMES:]
    for name, _ in L.PREFLIGHT_SETS:
        os.makedirs(os.path.join(dst, name))
    shutil.copy2(STILL, os.path.join(dst, "still_1080"))
    fw = FW.FrameWarp()
    for f in events:
        R.check_allowed(ix.run_of_source(f))                                   # DEV only
        img = cv2.imread(os.path.join(REPLAY1, f + ".png"))
        cv2.imwrite(os.path.join(dst, "run1_640", f + ".png"), img)
        cv2.imwrite(os.path.join(dst, "run1_1080", f + ".png"), warp_up(img, fw))
    exp = os.path.join(dst, "expected"); os.makedirs(exp); tmp = os.path.join(dst, "..", "_mac_preflight")
    for name, _ in L.PREFLIGHT_SETS:
        L.score_folder(os.path.join(dst, name), os.path.join(tmp, name))
        shutil.copy2(os.path.join(tmp, name, "predictions.csv"), os.path.join(exp, name + ".csv"))
    json.dump({**L.environment(), "code": L.code_version()}, open(os.path.join(exp, "mac.json"), "w"), indent=1)
    shutil.rmtree(tmp)
    return events


def import_trace(pkg):
    """Third-party modules imported during a preflight run, not counting interpreter start-up (must be covered by the wheels)."""
    code = ("import runpy, sys, sysconfig\nbefore = set(sys.modules)\nsys.argv = ['live_inspect.py', '--preflight', sys.argv[1], '--out', sys.argv[2]]\n"
            "try:\n    runpy.run_path('live_inspect.py', run_name='__main__')\nexcept SystemExit as e:\n    ok = e.code == 0\n"
            "sp = sysconfig.get_paths()['purelib']\n"
            "mods = sorted({m.split('.')[0] for m, v in list(sys.modules.items()) if m not in before and (getattr(v, '__file__', None) or '').startswith(sp)})\n"
            "print('PREFLIGHT', 'OK' if ok else 'FAILED'); print(' '.join(mods))")
    out = subprocess.run([sys.executable, "-c", code, os.path.join(pkg, "preflight"), os.path.join(pkg, "..", "_trace")],
                         cwd=HERE, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    shutil.rmtree(os.path.join(pkg, "..", "_trace"))
    return out[-2], out[-1].split()


def copy_monday(code_archive, destination):
    """Put the committed operator kit beside the tar, with its own USB checksum file."""
    with tarfile.open(code_archive) as archive:
        for name in MONDAY_FILES:
            data = archive.extractfile("twin_to_line/monday/" + name).read()
            with open(os.path.join(destination, name), "xb") as f:
                f.write(data)
    with open(os.path.join(destination, "monday.sha256"), "x") as f:
        for name in MONDAY_FILES:
            f.write(f"{sha256(os.path.join(destination, name))}  {name}\n")


def main():
    ap = argparse.ArgumentParser(description="build the GB10 package from one commit")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "gb10_pkg"))
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)                            # git -C must not reinterpret a caller-relative output
    commit = git("rev-parse", "HEAD"); short = commit[:7]
    changed = git("status", "--porcelain", "--", *L.RUNTIME_FILES, "replay_1080.py", "gb10", "monday", "ARCHIVE_COMMIT", ".gitattributes")
    if changed:
        raise SystemExit(f"uncommitted runtime/package files:\n{changed}\ncommit first (expected results must come from the packaged code)")
    if any(os.path.lexists(os.path.join(a.out, name)) for name in (*MONDAY_FILES, "monday.sha256")):
        raise SystemExit("operator kit already exists in output; use a fresh --out folder")
    pkg = os.path.join(a.out, f"ttl_gb10_{short}")
    if os.path.exists(pkg):
        raise SystemExit(f"{pkg} exists; remove it to rebuild")
    os.makedirs(os.path.join(pkg, "runtime"))
    top = git("rev-parse", "--show-toplevel")                             # archive the whole repo, not this subfolder
    subprocess.run(["git", "-C", top, "archive", "--format=tar", "-o", os.path.join(pkg, "code.tar"), commit], check=True)
    with tarfile.open(os.path.join(pkg, "code.tar")) as t:
        names = set(t.getnames())
        missing = [f for f in L.RUNTIME_FILES if f"twin_to_line/{f}" not in names]
        stamp = t.extractfile("twin_to_line/ARCHIVE_COMMIT").read().decode().strip()
    if missing or stamp != commit:
        raise SystemExit(f"code.tar: missing {missing} or ARCHIVE_COMMIT {stamp!r} != {commit}")
    python_runtime(os.path.join(pkg, "runtime"))
    wheels(os.path.join(pkg, "wheels"))
    package_index(os.path.join(pkg, "data_index"))
    events = preflight_data(os.path.join(pkg, "preflight"))
    shutil.copy2(os.path.join(HERE, "gb10", "bootstrap.sh"), pkg)
    status, mods = import_trace(pkg)
    if status != "PREFLIGHT OK" or not set(mods) <= {"numpy", "cv2"}:
        raise SystemExit(f"import trace / Mac preflight: {status}, third-party modules {mods}")
    open(os.path.join(pkg, "IMPORTS.txt"), "w").write(
        f"third-party modules imported by live_inspect.py --preflight (rule inspector): {' '.join(mods)}\n"
        "wheels: numpy, opencv-python-headless (cv2); everything else is the standard library\n")
    open(os.path.join(pkg, "README.txt"), "w").write(
        f"Twin-to-Line GB10 package, commit {commit}\n"
        "Instructions: LIVE_RUNBOOK.md section 7 (inside code.tar: twin_to_line/LIVE_RUNBOOK.md).\n"
        f"  tar -xf ttl_gb10_{short}.tar && cd ttl_gb10_{short} && bash bootstrap.sh     -> BOOTSTRAP OK\n"
        f"Bundled: CPython 3.11.17 aarch64 ({PBS['tag']}); wheels {PINS}; preflight frames: CSI 1080 still, DEV run 1 "
        f"{events[0]} .. {events[-1]} ({len(events)}) at 640 and as 1080 copies.\n")
    files = sorted(os.path.relpath(os.path.join(d, f), pkg) for d, _, fs in os.walk(pkg) for f in fs)
    with open(os.path.join(pkg, "FILES.txt"), "w") as f:
        for p in files:
            f.write(f"{os.path.getsize(os.path.join(pkg, p)):>11d}  {p}\n")
    files.append("FILES.txt")
    with open(os.path.join(pkg, "SHA256SUMS"), "w") as f:
        for p in files:
            f.write(f"{sha256(os.path.join(pkg, p))}  {p}\n")
    tar = pkg + ".tar"
    with tarfile.open(tar, "w") as t:
        t.add(pkg, arcname=os.path.basename(pkg))
    open(tar + ".sha256", "w").write(f"{sha256(tar)}  {os.path.basename(tar)}\n")
    copy_monday(os.path.join(pkg, "code.tar"), a.out)
    total = sum(os.path.getsize(os.path.join(pkg, p)) for p in files)
    print(f"package {pkg}: {len(files) + 1} files, {total / 1e6:.1f} MB; Mac preflight {status}; imports {mods}")
    print(f"tar {tar}: {os.path.getsize(tar) / 1e6:.1f} MB  sha256 {sha256(tar)}")


if __name__ == "__main__":
    main()
