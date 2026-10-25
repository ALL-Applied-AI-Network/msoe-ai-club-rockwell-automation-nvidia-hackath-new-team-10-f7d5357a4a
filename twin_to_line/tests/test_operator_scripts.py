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
spec = importlib.util.spec_from_file_location('monday_operator', HERE / 'monday/operator.py')
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


@pytest.mark.parametrize('change', ['extra', 'missing', 'changed', 'duplicate', 'empty'])
def test_seal_reconciliation_rejects_bad_evidence(visit, monkeypatch, change):
    ctx, camera = visit
    if change != 'empty':
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


def test_export_all_visits_twice_without_overwrite(visit, monkeypatch):
    ctx, camera = visit; stage_one(ctx, camera, monkeypatch)
    failed = M.ROOT / 'visit_failed'; failed.mkdir(); (failed / 'failure.txt').write_text('failed bootstrap')
    for _ in range(2): assert 'RESULTS USB OK: 2 visits' in M.export(ctx)
    copies = list(Path(ctx['USB']).glob('visit_export_*/evidence.tar'))
    assert len(copies) == 2
    for copy in copies:
        with tarfile.open(copy) as t:
            assert t.extractfile('visit_failed/failure.txt').read() == b'failed bootstrap'
            assert t.extractfile('visit_test/staged/setup_00.png').read() == (camera / 'capture.png').read_bytes()


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
