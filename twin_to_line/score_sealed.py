"""Once-only offline staged-still scoring; labels must be locked before any model is loaded.

Five fixed arms, unchanged live_inspect.Session gate=each / LIVE_FILL / track=False.
The persistent results/sealed_score_locks ledger refuses the same image bytes even
under another --out or with renamed input files. Failed attempts consume the lock;
do not delete it to repeat an evaluation. Labels and raw image bytes are snapshotted
separately from predictions. This is not a threshold fitter or an arm selector.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from datetime import datetime, timezone

import cv2
import numpy as np
import live_inspect as L
from fill_state import LIVE_FILL
from rules import RuleInspector

HERE = Path(__file__).resolve().parent
LOCKS = HERE / "results" / "sealed_score_locks"
ARMS = ("A_d038", "A_preregistered", "B_synthetic", "B_prime", "B_double_prime")
RULE_FILES = ("rule_thresholds_synth.json", "rule_thresholds.json", "rule_thresholds_dev_good_only.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value, mode="w"):
    with path.open(mode) as f:
        json.dump(value, f, indent=2)
        f.write("\n")


class SealedRuleInspector(RuleInspector):
    def __init__(self, filename, provenance):
        super().__init__(str(HERE / filename))
        self.provenance = provenance

    def info(self):
        return {**super().info(), "status": self.provenance}


def load_inspectors(device):
    from resnet_inspector import ResNetInspector
    a = ResNetInspector(str(HERE / "results/train/resnet18_v1"), "d038", device=device)
    pre = ResNetInspector(str(HERE / "results/train/resnet18_v1"), "preregistered", device=device, model=a.model)
    provenance = ("B: synthetic-validation fit (D033)",
                  "B′: DEV-fit reference; organizer permission unresolved",
                  "B″: good-only DEV fit (D041), in-sample reference; organizer permission unresolved")
    inspectors = [a, pre] + [SealedRuleInspector(f, note) for f, note in zip(RULE_FILES, provenance)]
    return dict(zip(ARMS, inspectors))


def inputs(folder, labels, labels_sha256):
    # This must precede image reads and model construction.
    if not re.fullmatch(r"[0-9a-fA-F]{64}", labels_sha256):
        raise ValueError("--labels-sha256 must be a complete SHA256")
    label_bytes = labels.read_bytes()
    if digest(label_bytes) != labels_sha256.lower():
        raise ValueError("locked labels SHA256 mismatch; no scoring started")
    paths = sorted(p for p in folder.iterdir() if L.is_image(p.name))
    if not paths:
        raise ValueError("no staged stills")
    ix = L.dataset_index()
    for path in paths:  # All names checked before opening any image.
        reason = L.refusal(ix, str(path))
        if reason:
            raise L.R.ProtectedDataError(reason)
    frames = [L.frame_id(str(p)) for p in paths]
    if len(set(frames)) != len(frames):
        raise ValueError("duplicate runtime frame IDs")
    rows = list(csv.DictReader(io.StringIO(label_bytes.decode("utf-8-sig"))))
    required = {"frame", "position", "ground_truth", "gt_sequence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("labels require frame, position, ground_truth, gt_sequence")
    keys = [(r["frame"], r["position"]) for r in rows]
    expected = {(f, f"G{i:02d}") for f in frames for i in range(1, 31)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("labels must reconcile exactly once to all 30 positions of every input frame")
    for r in rows:
        if r.get("usable", "1") == "0":
            continue  # Locked ambiguity remains excluded; never invent ground truth.
        if r["ground_truth"] not in {"pass", "no_vial", "crooked_cap", "missing_cap", "damaged_vial"}:
            raise ValueError("invalid ground_truth")
        if r["ground_truth"] == "no_vial" and r["gt_sequence"] not in {"waiting", "missed"}:
            raise ValueError("no_vial requires waiting or missed ground truth")
    images = []
    for p in paths:
        before = L.file_version(str(p))
        data = p.read_bytes()
        if before != L.file_version(str(p)):
            raise ValueError(f"input changed while reading: {p.name}")
        complete = data.endswith(b"IEND\xaeB`\x82") if p.suffix.lower() == ".png" else data.rstrip(b"\x00").endswith(b"\xff\xd9")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if complete else None
        if img is None or img.shape[:2] not in {(480, 640), (1080, 1920)}:
            raise ValueError(f"incomplete, unreadable or unsupported image: {p.name}")
        images.append((p.name, data, img, L.frame_time(str(p))))
    return label_bytes, images, ix


def score(folder, labels, labels_sha256, out, device="cpu"):
    folder, labels, out = (Path(p).resolve() for p in (folder, labels, out))
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; never overwrite or rerun")
    label_bytes, images, ix = inputs(folder, labels, labels_sha256)
    hashes = {name: digest(data) for name, data, _, _ in images}
    # Names/labels/output paths cannot create a second opportunity on identical pixels.
    identity = digest("\n".join(sorted(hashes.values())).encode())
    LOCKS.mkdir(parents=True, exist_ok=True)
    lock = LOCKS / f"{identity}.json"
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "source": str(folder), "out": str(out),
              "labels_sha256": digest(label_bytes), "images_sha256": hashes, "arms": list(ARMS),
              "gate": "each", "mode": LIVE_FILL, "track": False, "status": "started",
              "script_sha256": digest(Path(__file__).read_bytes()), "completed_arms": []}
    try:
        write_json(lock, record, "x")
    except FileExistsError:
        raise FileExistsError(f"set already attempted: {lock}; preserve its evidence, do not repeat") from None
    owned_output = False
    try:
        out.mkdir(parents=True, exist_ok=False)
        owned_output = True
        (out / "inputs").mkdir()
        (out / "labels.locked.csv").write_bytes(label_bytes)
        for name, data, _, _ in images:
            (out / "inputs" / name).write_bytes(data)
        write_json(out / "evaluation.json", record)
        inspectors = load_inspectors(device)
        for arm in ARMS:
            session = L.Session(str(out / arm), source=f"sealed:{folder}", gate="each", mode=LIVE_FILL,
                                track=False, labels=str(out / "labels.locked.csv"), ix=ix,
                                inspector=inspectors[arm], quiet=True)
            session.write_info(sealed_arm=arm, locked_labels_sha256=digest(label_bytes), input_sha256=hashes)
            L.run(session, ((str(out / "inputs" / name), img.copy(), t) for name, _, img, t in images))
            if session.stop_reason != "end of source" or set(session.counts) - {"scored", "arm"}:
                raise RuntimeError(f"{arm} incomplete: {session.stop_reason}, {dict(session.counts)}")
            record["completed_arms"].append(arm)
            record.setdefault("coverage", {})[arm] = dict(session.counts)
            write_json(out / "evaluation.json", record)
            write_json(lock, record)
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(lock, record)
        if owned_output:
            write_json(out / "evaluation.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--labels-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    record = score(args.folder, args.labels, args.labels_sha256, args.out, args.device)
    print(f"SEALED SCORING COMPLETE: {len(record['images_sha256'])} images, {len(ARMS)} operating points -> {args.out}")


if __name__ == "__main__":
    main()
