"""Operator safety boundaries; synthetic images only. Full hardware rehearsal is separate."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('monday_operator', HERE / 'monday/monday_ops.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


@pytest.fixture
def visit(tmp_path, monkeypatch):
    root = tmp_path / 'ttl'; root.mkdir()
    day = root / 'visit_test'; day.mkdir()
    for name in ('staged', 'records', 'attempts', 'live_in'):
        (day / name).mkdir()
    usb = tmp_path / 'usb with spaces'; usb.mkdir()
    camera = tmp_path / 'camera with spaces'; camera.mkdir()
    with (day / 'records/staging_log.csv').open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=M.FIELDS).writeheader()
    ctx = dict(DAY=str(day), STAGED=str(day / 'staged'), USB=str(usb), READY='1', PY=sys.executable,
               INSTALL=str(day / 'install'), PKG='ttl_gb10_abcdef0')
    monkeypatch.setattr(M, 'ROOT', root)
    M.save_context(ctx)
    return ctx, camera


def stage_one(ctx, camera, monkeypatch):
    def reply(prompt):
        if prompt.startswith('Camera'):
            return str(camera)
        if prompt.startswith('setup 00'):
            cv2.imwrite(str(camera / 'capture.png'), np.zeros((480, 640, 3), np.uint8))
            return ''
        return 'q'
    monkeypatch.setattr('builtins.input', reply)
    M.stage(ctx)


def test_stage_newest_skip_resume_and_rewritten_name(visit, monkeypatch):
    ctx, camera = visit
    old = camera / 'old.png'; cv2.imwrite(str(old), np.ones((480, 640, 3), np.uint8))
    original = old.read_bytes()
    replies = iter([str(camera), '', 's', 'q'])
    def reply(prompt):
        answer = next(replies)
        if prompt.startswith('setup 00'):
            for i in (1, 2):
                p = camera / f'new{i}.png'
                cv2.imwrite(str(p), np.full((480, 640, 3), i, np.uint8))
                os.utime(p, (100 + i, 100 + i))
        return answer
    monkeypatch.setattr('builtins.input', reply)
    assert '1 setups staged' in M.stage(ctx)
    assert [r['status'] for r in M.rows(ctx)] == ['staged', 'skipped']
    assert M.rows(ctx)[0]['source_name'] == 'new2.png'
    assert old.read_bytes() == original
    def resume(prompt):
        assert not prompt.startswith('Camera')
        if prompt.startswith('setup 02'):
            cv2.imwrite(str(camera / 'new2.png'), np.full((480, 640, 3), 3, np.uint8))
            return ''
        return 'q'
    monkeypatch.setattr('builtins.input', resume)
    assert '2 setups staged' in M.stage(M.read_context())
    assert len(M.reconcile(ctx)) == 2


@pytest.mark.parametrize('change', ['extra', 'missing', 'changed', 'duplicate'])
def test_seal_reconciliation_rejects_bad_evidence(visit, monkeypatch, change):
    ctx, camera = visit
    stage_one(ctx, camera, monkeypatch)
    p = Path(ctx['STAGED']) / 'setup_00.png'
    if change == 'extra': (p.parent / 'unlogged.png').write_bytes(p.read_bytes())
    if change == 'missing': p.rename(p.parent.parent / 'retained.png')
    if change == 'changed': p.write_bytes(b'changed')
    if change == 'duplicate':
        log = Path(ctx['DAY']) / 'records/staging_log.csv'
        with log.open('a') as f: f.write(log.read_text().splitlines()[1] + '\n')
    with pytest.raises(ValueError): M.seal(ctx)
    assert not ctx.get('SEALED')


def test_disk_seal_when_usb_missing_and_staging_closed(visit, monkeypatch):
    ctx, camera = visit; stage_one(ctx, camera, monkeypatch)
    Path(ctx['USB']).rename(Path(ctx['USB'] + '_unmounted'))
    assert 'SEALED DISK OK: USB FAILED' in M.seal(ctx)
    assert M.digest(ctx['SEALED']) == ctx['SEALED_SHA256']
    with pytest.raises(ValueError, match='already sealed'): M.stage(ctx)


@pytest.mark.parametrize('alias', [False, True])
def test_live_refuses_staged_folder_and_symlink_before_output(visit, monkeypatch, alias):
    ctx, camera = visit; stage_one(ctx, camera, monkeypatch)
    M.seal(ctx)
    folder = Path(ctx['STAGED'])
    if alias:
        folder = folder.parent / 'alias'; folder.symlink_to(ctx['STAGED'])
    monkeypatch.setattr('builtins.input', lambda _: str(folder))
    with pytest.raises(ValueError, match='refused'): M.live(ctx)
    assert not list((Path(ctx['DAY']) / 'attempts').iterdir())


def test_helper_starts_in_fresh_python_without_shadowing_stdlib(tmp_path):
    r = subprocess.run([sys.executable, str(HERE / 'monday/monday_ops.py'), 'stage'],
                       env={**os.environ, 'HOME': str(tmp_path)}, capture_output=True, text=True)
    assert r.returncode == 1 and 'STOP: stage:' in r.stdout
    assert 'Traceback' not in r.stderr


def test_scripts_parse_and_missing_context_stops(tmp_path):
    for script in (HERE / 'monday').glob('*.sh'):
        subprocess.run(['bash', '-n', str(script)], check=True)
    for name in ('stage', 'seal', 'live', 'export'):
        # Child-only HOME models the CSI account, leaving the host environment untouched.
        r = subprocess.run(['bash', str(HERE / 'monday' / (name + '.sh'))],
                           env={**os.environ, 'HOME': str(tmp_path)}, capture_output=True, text=True)
        assert r.returncode and 'STOP:' in r.stderr


def test_builder_copies_kit_from_archive_and_refuses_overwrite(tmp_path):
    from gb10 import build_package as B
    code = tmp_path / 'code.tar'
    with tarfile.open(code, 'w') as t:
        for name in B.MONDAY_FILES:
            t.add(HERE / 'monday' / name, arcname='twin_to_line/monday/' + name)
    out = tmp_path / 'usb'; out.mkdir()
    B.copy_monday(code, out)
    for name in B.MONDAY_FILES:
        assert (out / name).read_bytes() == (HERE / 'monday' / name).read_bytes()
    assert len((out / 'monday.sha256').read_text().splitlines()) == 6
    with pytest.raises(FileExistsError): B.copy_monday(code, out)


def test_zero_staged_seals_warns_and_cannot_reseal(visit, capsys):
    ctx, camera = visit
    assert 'SEALED USB OK' in M.seal(ctx)
    assert 'WARNING: 0 staged' in capsys.readouterr().out
    assert M.live_folder(ctx, str(camera)) == camera
    with pytest.raises(ValueError, match='already sealed'): M.seal(ctx)
    with pytest.raises(ValueError, match='already sealed'): M.stage(ctx)


def test_retakes_preserve_all_attempts_and_seal_last(visit, monkeypatch):
    ctx, camera = visit
    stage_one(ctx, camera, monkeypatch)
    original = (Path(ctx['STAGED']) / 'setup_00.png').read_bytes()
    answers = iter(['r', 'r', 'q'])
    def reply(prompt):
        answer = next(answers)
        if answer == 'r':
            value = 1 + sum(r['status'] == 'retake' for r in M.rows(ctx))
            cv2.imwrite(str(camera / 'capture.png'), np.full((480, 640, 3), value, np.uint8))
        return answer
    monkeypatch.setattr('builtins.input', reply)
    M.stage(ctx)
    assert [r['status'] for r in M.rows(ctx)] == ['staged', 'retake', 'retake']
    assert M.reconcile(ctx)[0]['copy_name'] == 'setup_00_r2.png'
    assert (Path(ctx['STAGED']) / 'setup_00.png').read_bytes() == original
    M.seal(ctx)
    selected = json.loads((Path(ctx['DAY']) / 'records/seal_selection.json').read_text())
    assert [r['copy_name'] for r in selected['earlier_attempts']] == ['setup_00.png', 'setup_00_r1.png']
    assert selected['final'][0]['copy_name'] == 'setup_00_r2.png'


def test_duplicate_and_wrong_size_refused_without_advancing(visit, monkeypatch, capsys):
    ctx, camera = visit
    stage_one(ctx, camera, monkeypatch)
    captures = iter([np.zeros((480, 640, 3), np.uint8), np.ones((240, 320, 3), np.uint8), None])
    def reply(prompt):
        assert prompt.startswith('setup 01')
        image = next(captures)
        if image is None: return 'q'
        cv2.imwrite(str(camera / 'capture.png'), image)
        return ''
    monkeypatch.setattr('builtins.input', reply)
    M.stage(ctx)
    output = capsys.readouterr().out
    assert 'REFUSED: duplicate' in output and 'REFUSED: unsupported size 320x240' in output
    assert len(M.rows(ctx)) == 1 and len(list(Path(ctx['STAGED']).iterdir())) == 1


def test_camera_folder_allowed_sealed_folders_and_overlaps_refused(visit, monkeypatch):
    ctx, camera = visit; stage_one(ctx, camera, monkeypatch); M.seal(ctx)
    assert M.live_folder(ctx, str(camera)) == camera
    protected = [Path(ctx['STAGED']), Path(ctx['SEALED']).parent, Path(ctx['SEALED_USB'])]
    for folder in protected:
        child = folder / 'nested'; child.mkdir()
        for candidate in (folder, folder.parent, child):
            with pytest.raises(ValueError, match='refused'): M.live_folder(ctx, str(candidate))
    alias = camera.parent / 'camera_alias'; alias.symlink_to(camera)
    assert M.live_folder(ctx, str(alias)) == camera


def test_gate_enter_defaults_to_continuous_and_live_requires_seal(visit, monkeypatch):
    ctx, camera = visit
    monkeypatch.setattr('builtins.input', lambda _: '')
    assert M.yes_no('gate', default_no=True) is False
    with pytest.raises(ValueError, match='seal the staged set'): M.live(ctx)


@pytest.mark.parametrize('count', [0, 2])
def test_start_discovers_usb_and_rejects_ambiguous_package_without_input(tmp_path, count):
    script = tmp_path / 'start.sh'; shutil.copy2(HERE / 'monday/start.sh', script)
    for i in range(count): (tmp_path / f'ttl_gb10_abc{i}.tar').write_text('placeholder')
    result = subprocess.run(['bash', str(script)], stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert result.returncode and f'Packages found ({count})' in result.stdout
    assert 'STOP:' in result.stderr
    for i in range(count): assert f'ttl_gb10_abc{i}.tar' in result.stdout
