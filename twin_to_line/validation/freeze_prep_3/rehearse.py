"""Run inside arm64 Ubuntu as UID 1000. Only three explicitly supplied DEV stills are read."""
import csv
import hashlib
import json
import os
import pty
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

HOME = Path.home()
USB = Path('/rehearsal/usb')
IMAGES = sorted(Path('/dev-stills').glob('*.jpg'))
assert len(IMAGES) == 3
assert os.getuid() != 0
print('REHEARSAL: non-root uid=' + str(os.getuid()), flush=True)
subprocess.run(['uname', '-m'], check=True)
print(Path('/etc/os-release').read_text(), flush=True)
PACKAGE = next(USB.glob('ttl_gb10_*.tar')).stem


class Command:
    def __init__(self, script):
        print('\n$ bash ' + str(script), flush=True)
        self.keyboard, terminal = pty.openpty()
        self.proc = subprocess.Popen(['bash', str(script)], stdin=terminal, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        os.close(terminal)
        self.output = ''
        def read():
            while True:
                data = self.proc.stdout.read(1)
                if not data:
                    break
                char = data.decode(errors='replace')
                self.output += char
                print(char, end='', flush=True)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def wait_for(self, value, timeout=90):
        end = time.monotonic() + timeout
        while value not in self.output:
            if self.proc.poll() is not None or time.monotonic() > end:
                raise AssertionError('missing prompt/output: ' + value + '\n' + self.output)
            time.sleep(.05)

    def send(self, value):
        print(value, flush=True)
        os.write(self.keyboard, (value + '\n').encode())

    def finish(self, expected):
        code = self.proc.wait(timeout=180)
        self.reader.join(timeout=3)
        assert code == 0, (code, self.output)
        assert expected in self.output, self.output


start = Command(USB / 'start.sh')
start.wait_for('Mounted USB directory:'); start.send(str(USB))
start.wait_for('without .tar:'); start.send(PACKAGE)
start.finish('SETUP OK')
visits = sorted((HOME / 'ttl').glob('visit_*'))
assert len(visits) == 1
DAY = visits[0]
CAMERA = Path('/rehearsal/camera with spaces'); CAMERA.mkdir(exist_ok=True)
original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in IMAGES}
# Every script is a fresh process: no terminal's exported variables are shared.
stage = Command(HOME / 'ttl/stage.sh')
stage.wait_for('Camera folder'); stage.send(str(CAMERA))
for setup, image in [('00', IMAGES[0]), ('01', None), ('02', IMAGES[1]), ('03', IMAGES[2])]:
    stage.wait_for('setup ' + setup + ': Enter')
    if image is None:
        stage.send('s')
    else:
        shutil.copy2(image, CAMERA / image.name)
        stage.send('')
stage.wait_for('setup 04: Enter'); stage.send('q'); stage.finish('STAGING OK: 3 setups staged')
with (DAY / 'records/staging_log.csv').open() as f:
    log = list(csv.DictReader(f))
assert [r['status'] for r in log] == ['staged', 'skipped', 'staged', 'staged']
for row in log:
    if row['status'] == 'staged':
        assert hashlib.sha256((DAY / 'staged' / row['copy_name']).read_bytes()).hexdigest() == row['sha256']
assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CAMERA.iterdir()} == original
(DAY / 'records/label_sheet.txt').write_text('DEV rehearsal only; no Monday labels.\n')
seal = Command(HOME / 'ttl/seal.sh'); seal.finish('SEALED USB OK')
# Missing USB: a verified local seal must explicitly permit live, without claiming USB success.
USB.rename(USB.with_name('usb_disconnected'))
seal = Command(HOME / 'ttl/seal.sh'); seal.finish('SEALED DISK OK: USB FAILED')
USB.with_name('usb_disconnected').rename(USB)
# A symlink to staged must fail without creating a live attempt.
alias = Path('/rehearsal/staged_alias'); alias.symlink_to(DAY / 'staged')
refuse = Command(HOME / 'ttl/live.sh')
refuse.wait_for('stream URL:'); refuse.send(str(alias))
assert refuse.proc.wait(timeout=15) != 0
refuse.reader.join(timeout=3)
assert 'STOP:' in refuse.output and 'refused' in refuse.output
assert not list((DAY / 'attempts').iterdir())
LIVE = DAY / 'live_in'
live = Command(HOME / 'ttl/live.sh')
live.wait_for('stream URL:'); live.send(str(LIVE))
live.wait_for('independent stills?'); live.send('y')
live.wait_for('buffered frames?'); live.send('y')
live.wait_for('watching ')
out = next((DAY / 'attempts').iterdir())
for i, image in enumerate(IMAGES):
    shutil.copy2(image, LIVE / image.name)
    deadline = time.monotonic() + 20
    while True:
        with (out / 'trial_log.csv').open() as f:
            trial = list(csv.DictReader(f))
        if len(trial) == i + 1:
            break
        assert time.monotonic() < deadline, trial
        time.sleep(.1)
os.kill(live.proc.pid, signal.SIGINT)
live.finish('LIVE OK')
session = json.loads((out / 'session.json').read_text())
assert session['stop_reason'] == 'ctrl-c' and session['frames'] == 3, session
with (out / 'predictions.csv').open() as f:
    predictions = list(csv.DictReader(f))
print(f'LIVE VERIFIED: {len(trial)} DEV stills, {len(predictions)} prediction rows, stop_reason=ctrl-c', flush=True)
# Restart creates a second OUT; empty fresh watcher stops cleanly.
again = Command(HOME / 'ttl/live.sh')
again.wait_for('stream URL:'); again.send(str(LIVE))
again.wait_for('independent stills?'); again.send('n')
again.wait_for('buffered frames?'); again.send('y')
again.wait_for('watching '); os.kill(again.proc.pid, signal.SIGINT); again.finish('LIVE OK')
assert len(list((DAY / 'attempts').iterdir())) == 2
# A real checksum failure creates a failed visit without Python. Export still works.
(USB / 'ttl_gb10_deadbeef.tar').write_bytes(b'invalid archive')
(USB / 'ttl_gb10_deadbeef.tar.sha256').write_text('0' * 64 + '  ttl_gb10_deadbeef.tar\n')
failed = Command(USB / 'start.sh')
failed.wait_for('Mounted USB directory:'); failed.send(str(USB))
failed.wait_for('without .tar:'); failed.send('ttl_gb10_deadbeef')
assert failed.proc.wait(timeout=15) != 0
failed.reader.join(timeout=3)
assert 'STOP:' in failed.output and 'SETUP OK' not in failed.output
assert len(list((HOME / 'ttl').glob('visit_*'))) == 2
for _ in range(2):
    export = Command(HOME / 'ttl/export.sh'); export.finish('RESULTS USB OK: 2 visits')
copies = list(USB.glob('visit_export_*/evidence.tar'))
assert len(copies) == 2
import tarfile
for copy in copies:
    with tarfile.open(copy) as t:
        names = t.getnames()
        assert len({n.split('/')[0] for n in names}) == 2
        assert any(n.endswith('/records/label_sheet.txt') for n in names)
        assert any(n.endswith('/install/ttl_gb10_deadbeef.tar') for n in names)
assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CAMERA.iterdir()} == original
print('\nREHEARSAL OK: all five scripts; 3 staged + 1 skip; source hashes unchanged; USB and disk seals; '
      'staged alias refused; 3 live DEV stills; fresh OUT on restart; 2 visits exported twice with verified SHA256.', flush=True)
