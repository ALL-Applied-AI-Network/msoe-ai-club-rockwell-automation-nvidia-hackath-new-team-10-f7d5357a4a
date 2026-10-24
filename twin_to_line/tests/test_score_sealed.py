"""Sealed evaluation guards plus actual five-arm parity on three cached DEV stills."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import live_inspect as L
import score_sealed as S


def label_file(folder, path):
    # Test-only labels check plumbing, not model performance or real ground truth.
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "position", "ground_truth", "gt_sequence"])
        w.writeheader()
        for p in sorted(folder.iterdir()):
            if L.is_image(p.name):
                for i in range(1, 31):
                    w.writerow(dict(frame=L.frame_id(str(p)), position=f"G{i:02d}",
                                    ground_truth="no_vial", gt_sequence="waiting"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "LOCKS", tmp_path / "locks")
    folder = tmp_path / "stills"; folder.mkdir()
    cv2.imwrite(str(folder / "setup_00.png"), np.zeros((480, 640, 3), np.uint8))
    labels = tmp_path / "labels.csv"
    sha = label_file(folder, labels)
    return folder, labels, sha, tmp_path / "out"


@pytest.mark.parametrize("problem", ["missing", "mismatch", "malformed_hash", "missing_position", "duplicate_position"])
def test_labels_guard_precedes_pixels_and_models(sample, monkeypatch, problem):
    folder, labels, sha, out = sample
    if problem == "missing": labels.unlink()
    elif problem == "mismatch": sha = "0" * 64
    elif problem == "malformed_hash": sha = "short"
    else:
        lines = labels.read_text().splitlines()
        lines = lines[:-1] if problem == "missing_position" else lines + [lines[-1]]
        labels.write_text("\n".join(lines) + "\n")
        sha = S.digest(labels.read_bytes())
    def forbidden(*a, **kw): raise AssertionError("pixels/model touched before label validation")
    monkeypatch.setattr(S.cv2, "imdecode", forbidden)
    monkeypatch.setattr(S, "load_inspectors", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        S.score(folder, labels, sha, out)
    assert not out.exists() and not S.LOCKS.exists()


def test_protected_frame_name_refused_before_pixels(sample, monkeypatch):
    folder, labels, _, out = sample
    # Fabricated filename, synthetic bytes only; never a dataset holdout access.
    (folder / "setup_00.png").rename(folder / "cap_seal_video_4.mp4@t0000.0.png")
    sha = label_file(folder, labels)
    monkeypatch.setattr(S.cv2, "imdecode", lambda *a: pytest.fail("protected name decoded"))
    with pytest.raises(L.R.ProtectedDataError): S.score(folder, labels, sha, out)
    assert not out.exists()


def test_bad_image_does_not_start_attempt(sample):
    folder, labels, sha, out = sample
    (folder / "setup_00.png").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="incomplete"): S.score(folder, labels, sha, out)
    assert not S.LOCKS.exists()


def test_existing_output_preserved(sample):
    folder, labels, sha, out = sample
    out.mkdir(); (out / "predictions.csv").write_text("preserve")
    with pytest.raises(FileExistsError): S.score(folder, labels, sha, out)
    assert (out / "predictions.csv").read_text() == "preserve"


def test_failed_attempt_blocks_new_output_and_renamed_inputs(sample, monkeypatch):
    folder, labels, sha, out = sample
    def fail(*a): raise RuntimeError("model unavailable")
    monkeypatch.setattr(S, "load_inspectors", fail)
    with pytest.raises(RuntimeError, match="model unavailable"): S.score(folder, labels, sha, out)
    assert json.loads((out / "evaluation.json").read_text())["status"] == "failed"
    original = (out / "evaluation.json").read_bytes()
    (folder / "setup_00.png").rename(folder / "renamed.png")
    sha = label_file(folder, labels)
    with pytest.raises(FileExistsError, match="already attempted"):
        S.score(folder, labels, sha, out.parent / "second")
    assert (out / "evaluation.json").read_bytes() == original


def test_output_creation_race_preserves_other_attempt(sample, monkeypatch):
    folder, labels, sha, out = sample
    original = S.write_json
    def racing(path, value, mode="w"):
        original(path, value, mode)
        if mode == "x":
            out.mkdir()
            (out / "evaluation.json").write_text("other process")
    monkeypatch.setattr(S, "write_json", racing)
    with pytest.raises(FileExistsError): S.score(folder, labels, sha, out)
    assert (out / "evaluation.json").read_text() == "other process"


def test_five_arms_on_three_dev_stills_match_unchanged_pipeline(tmp_path, monkeypatch):
    source = HERE / "results/rc_1024/inputs/staged_3"
    paths = sorted(p for p in source.iterdir() if L.is_image(p.name))
    assert len(paths) == 3, "three cached DEV stills required, no silent skip"
    monkeypatch.setattr(S, "LOCKS", tmp_path / "locks")
    folder = tmp_path / "stills"; folder.mkdir()
    ix = L.dataset_index()
    for p in paths:
        assert L.refusal(ix, str(p)) is None
        # RC metadata identifies these original stills as DEV run 1.
        assert ix is not None and ix.run_of_source(str(p)) == "1"
        shutil.copyfile(p, folder / p.name)
    labels = tmp_path / "labels.csv"; sha = label_file(folder, labels)
    out = tmp_path / "scored"
    report = S.score(folder, labels, sha, out)
    assert report["status"] == "complete" and report["completed_arms"] == list(S.ARMS)
    assert S.digest((out / "labels.locked.csv").read_bytes()) == sha
    inspectors = S.load_inspectors("cpu")
    for arm, inspector in inspectors.items():
        reference = tmp_path / f"reference_{arm}"
        session = L.Session(str(reference), gate="each", mode=S.LIVE_FILL, track=False,
                            inspector=inspector, quiet=True, ix=ix)
        L.run(session, L.replay(str(folder), ix=ix))
        with (out / arm / "predictions.csv").open() as f: actual = list(csv.DictReader(f))
        with (reference / "predictions.csv").open() as f: expected = list(csv.DictReader(f))
        assert actual == expected
        assert report["coverage"][arm] == {"scored": 2, "arm": 1}
        metadata = json.loads((out / arm / "session.json").read_text())
        assert metadata["gate"] == "each" and metadata["mode"] == S.LIVE_FILL and not metadata["track"]
        assert metadata["locked_labels_sha256"] == sha
        if arm in S.ARMS[2:]:
            file = S.RULE_FILES[S.ARMS[2:].index(arm)]
            assert metadata["inspector"]["files"] == {file: S.digest((HERE / file).read_bytes())}
            assert "plumbing placeholder" not in metadata["inspector"]["status"]
            if arm == "B_synthetic": assert "synthetic-validation" in metadata["inspector"]["status"]
            if arm == "B_double_prime": assert "D041" in metadata["inspector"]["status"]
        else:
            assert metadata["inspector"]["threshold"] == ("d038" if arm == "A_d038" else "preregistered")
    with pytest.raises(FileExistsError, match="already attempted"):
        S.score(folder, labels, sha, tmp_path / "again")
