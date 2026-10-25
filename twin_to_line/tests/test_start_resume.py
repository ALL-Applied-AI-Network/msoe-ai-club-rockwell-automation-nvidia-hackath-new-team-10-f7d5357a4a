"""Execute start's decision table on a tiny fake package; no real bootstrap/runtime changes."""
import csv
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

import pytest

HERE = Path(__file__).resolve().parents[1]
KIT = ('start.sh', 'stage.sh', 'seal.sh', 'live.sh', 'export.sh', 'monday_ops.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def operator(tmp_path):
    home = tmp_path / 'home'; home.mkdir()
    usb = tmp_path / 'USB with space'; usb.mkdir()
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    # macOS has no flock. This stub excludes locking from the decision-table unit test;
    # the real Linux flock is exercised in the non-root Ubuntu end-to-end rehearsal.
    (bin_dir / 'flock').write_text('#!/bin/sh\nexit 0\n'); (bin_dir / 'flock').chmod(0o755)
    for name in KIT: shutil.copy2(HERE / 'monday' / name, usb / name)
    (usb / 'monday.sha256').write_text(''.join(f'{sha(usb/n)}  {n}\n' for n in KIT))
    env = {**os.environ, 'HOME': str(home), 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
           'TTL_TEST_PYTHON': sys.executable, 'TTL_TEST_FAIL': '0'}
    return home, usb, env


def package(usb, variant='one'):
    source = usb.parent / ('source_' + variant); source.mkdir()
    script = source / 'bootstrap.sh'
    script.write_text('''#!/usr/bin/env bash
set -eu
printf 'called\\n' >> bootstrap_calls
if [[ "$TTL_TEST_FAIL" == 1 ]]; then exit 7; fi
mkdir -p venv/bin
ln -s "$TTL_TEST_PYTHON" venv/bin/python
printf 'PREFLIGHT OK\\nBOOTSTRAP OK\\n'
''' + '# variant ' + variant + '\n')
    archive = usb / 'ttl_gb10_abcdef0.tar'
    with tarfile.open(archive, 'w') as t: t.add(script, arcname='ttl_gb10_abcdef0/bootstrap.sh')
    (usb / 'ttl_gb10_abcdef0.tar.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    return archive


def run(operator, *args):
    _, usb, env = operator
    return subprocess.run(['bash', str(usb/'start.sh'), *args], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True)


def context(home):
    result = {}
    for line in (home/'ttl/operator.env').read_text().splitlines():
        key, value = line.split('=', 1); result[key] = shlex.split(value)[0]
    return result


def snapshot(day):
    return {str(p.relative_to(day)): p.read_bytes() for p in day.rglob('*') if p.is_file() and not p.is_symlink()}


def add_two_captures(ctx):
    log = Path(ctx['DAY'])/'records/staging_log.csv'
    for setup in ('00', '01'):
        p = Path(ctx['STAGED'])/f'setup_{setup}.png'; p.write_bytes(('test capture '+setup).encode())
        with log.open('a', newline='') as f:
            csv.writer(f).writerow([setup, 'staged', 'camera.png', p.name, 'test time', sha(p)])


def test_no_current_visit_creates_recorded_success(operator):
    home, usb, env = operator; archive = package(usb)
    r = run(operator)
    assert r.returncode == 0 and 'SETUP OK: preflight passed' in r.stdout
    ctx = context(home)
    assert ctx['READY'] == '1' and ctx['PACKAGE_SHA256'] == sha(archive)
    assert 'SETUP OK' in (Path(ctx['DAY'])/'records/setup_ok.txt').read_text()
    assert len(list((home/'ttl').glob('visit_*'))) == 1


@pytest.mark.parametrize('sealed', [False, True])
def test_same_package_resumes_without_writes_or_bootstrap(operator, sealed):
    home, usb, env = operator; package(usb); assert run(operator).returncode == 0
    ctx = context(home); add_two_captures(ctx)
    day = Path(ctx['DAY'])
    if sealed:
        for p in (day/'operator.env', home/'ttl/operator.env'):
            with p.open('a') as f: f.write('SEALED=sealed_evidence.tar\n')
    before = snapshot(day); context_before = (home/'ttl/operator.env').read_bytes()
    r = run(operator)
    assert r.returncode == 0 and f'(2 staged, sealed {"yes" if sealed else "no"})' in r.stdout
    assert 'SETUP OK: resumed' in r.stdout and 'BOOTSTRAP OK' not in r.stdout
    assert len(list((home/'ttl').glob('visit_*'))) == 1 and snapshot(day) == before
    assert (home/'ttl/operator.env').read_bytes() == context_before


def test_failed_setup_creates_new_visit_and_keeps_failed_evidence(operator):
    home, usb, env = operator; package(usb); env['TTL_TEST_FAIL'] = '1'
    assert run(operator).returncode != 0
    first = Path(context(home)['DAY']); before = snapshot(first)
    assert context(home)['READY'] == '0' and not (first/'records/setup_ok.txt').exists()
    env['TTL_TEST_FAIL'] = '0'; r = run(operator)
    assert r.returncode == 0 and 'resumed' not in r.stdout
    assert Path(context(home)['DAY']) != first and snapshot(first) == before
    assert len(list((home/'ttl').glob('visit_*'))) == 2


def test_different_package_stops_then_explicit_new_keeps_old(operator):
    home, usb, env = operator; package(usb); assert run(operator).returncode == 0
    old = context(home); first = Path(old['DAY']); before = snapshot(first)
    new = package(usb, 'two'); r = run(operator)
    assert r.returncode != 0 and f"used package {old['PACKAGE_SHA256']}; USB has {sha(new)}" in r.stderr
    assert '-> bash start.sh --new-visit' in r.stderr and context(home) == old
    assert snapshot(first) == before and len(list((home/'ttl').glob('visit_*'))) == 1
    r = run(operator, '--new-visit')
    assert r.returncode == 0 and context(home)['DAY'] != old['DAY']
    assert snapshot(first) == before and len(list((home/'ttl').glob('visit_*'))) == 2


def test_explicit_new_with_same_package(operator):
    home, usb, env = operator; package(usb); assert run(operator).returncode == 0
    first = Path(context(home)['DAY']); before = snapshot(first)
    r = run(operator, '--new-visit')
    assert r.returncode == 0 and 'resumed' not in r.stdout
    assert Path(context(home)['DAY']) != first and snapshot(first) == before


def test_corrupt_usb_stops_without_resetting_current_then_fixed_resumes(operator):
    home, usb, env = operator; archive = package(usb); assert run(operator).returncode == 0
    old = context(home); before = snapshot(Path(old['DAY'])); original = archive.read_bytes()
    with archive.open('ab') as f: f.write(b'corrupt')
    r = run(operator)
    assert r.returncode != 0 and 'checksum failed' in r.stderr
    assert context(home) == old and snapshot(Path(old['DAY'])) == before
    archive.write_bytes(original); r = run(operator)
    assert r.returncode == 0 and 'SETUP OK: resumed' in r.stdout
    assert len(list((home/'ttl').glob('visit_*'))) == 1


def test_legacy_success_uses_retained_recorded_checksum(operator):
    home, usb, env = operator; package(usb); assert run(operator).returncode == 0
    ctx = context(home)
    for p in (home/'ttl/operator.env', Path(ctx['DAY'])/'operator.env'):
        p.write_text('\n'.join(line for line in p.read_text().splitlines() if not line.startswith('PACKAGE_SHA256='))+'\n')
    r = run(operator)
    assert r.returncode == 0 and 'SETUP OK: resumed' in r.stdout


def test_bad_options_stop_before_any_visit(operator):
    home, usb, env = operator; package(usb)
    for args in [('--unknown',), ('--new-visit', 'extra')]:
        r = run(operator, *args)
        assert r.returncode != 0 and 'usage:' in r.stderr
    assert not (home/'ttl').exists()
