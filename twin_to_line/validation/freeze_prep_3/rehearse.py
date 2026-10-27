"""Round-1 rehearsal inside non-root arm64 Ubuntu, using bundled DEV captures only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import pty
import shlex
import shutil
import signal
import subprocess
import threading
import time

HOME = Path.home()
USB = Path('/rehearsal/usb')
STEPS = []
assert os.getuid() != 0
print(f'REHEARSAL: uid={os.getuid()}', flush=True)
subprocess.run(['uname', '-m'], check=True)
print('$ command -v python3', flush=True)
probe = subprocess.run(['bash', '-c', 'command -v python3'], capture_output=True, text=True)
print(probe.stdout, end='', flush=True)
assert not probe.stdout and probe.returncode == 1
print('(no output; exit 1; before start.sh)', flush=True)


class Command:
    def __init__(self, script, label):
        self.label, self.started = label, time.monotonic()
        print('\n$ bash ' + str(script), flush=True)
        self.keyboard, terminal = pty.openpty()
        self.proc = subprocess.Popen(['bash', str(script)], stdin=terminal, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        os.close(terminal)
        self.output = ''
        def read():
            while True:
                data = self.proc.stdout.read(1)
                if not data: break
                char = data.decode(errors='replace')
                self.output += char
                print(char, end='', flush=True)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def wait_for(self, value, count=1, timeout=90):
        end = time.monotonic() + timeout
        while self.output.count(value) < count:
            if self.proc.poll() is not None or time.monotonic() > end:
                raise AssertionError('missing prompt/output: ' + value + '\n' + self.output)
            time.sleep(.03)

    def send(self, value):
        print(value, flush=True)
        os.write(self.keyboard, (value + '\n').encode())

    def finish(self, expected, success=True):
        code = self.proc.wait(timeout=180)
        self.reader.join(timeout=3)
        os.close(self.keyboard)
        assert (code == 0) == success, (code, self.output)
        assert expected in self.output, self.output
        final = self.output.strip().splitlines()[-1]
        assert '========== ' in final and ('OK:' in final or 'STOP:' in final), final
        step = dict(step=self.label, seconds=round(time.monotonic() - self.started, 3), final=final, exit=code)
        STEPS.append(step)
        print('TIMED STEP: ' + json.dumps(step), flush=True)


def run(name, label, expected, success=True):
    c = Command(USB / name if name == 'start.sh' else HOME / 'ttl' / name, label)
    c.finish(expected, success)
    return c


def context():
    result = {}
    for line in (HOME / 'ttl/operator.env').read_text().splitlines():
        k, v = line.split('=', 1); result[k] = shlex.split(v)[0]
    return result


# Both starts get no typed input at all, and keep distinct visits.
run('start.sh', 'start: no typing', 'SETUP OK')
ZERO = Path(context()['DAY'])
run('start.sh', 'start: again, no typing', 'SETUP OK')
ctx = context(); DAY = Path(ctx['DAY'])
assert DAY != ZERO
run('live.sh', 'live before seal', 'STOP: live: seal the staged set', success=False)
CAMERA = Path('/rehearsal/camera'); CAMERA.mkdir()
frames = sorted((Path(ctx['INSTALL']) / ctx['PKG'] / 'preflight/run1_640').glob('*.png'))
assert len(frames) >= 6
PY = ctx['PY']
# Make a synthetic unsupported image, without resizing any CSI/DEV frame.
subprocess.run([PY, '-c', "import cv2,numpy as np; cv2.imwrite('/rehearsal/small.png',np.zeros((240,320,3),np.uint8))"], check=True)
stage = Command(HOME / 'ttl/stage.sh', 'stage: six setups, one skip, one retake, two refusals')
stage.wait_for('Camera folder'); stage.send(str(CAMERA))
source = CAMERA / 'capture.png'

def save(frame, response, next_output, count=1):
    shutil.copyfile(frame, source)
    stage.send(response)
    stage.wait_for(next_output, count)
    assert source.read_bytes() == frame.read_bytes()

stage.wait_for('setup 00: Enter'); save(frames[0], '', 'setup 01: Enter')
stage.send('s'); stage.wait_for('setup 02: Enter')
save(frames[1], '', 'setup 03: Enter')
save(frames[2], 'r', 'setup 03: Enter', 2)
save(frames[2], '', 'REFUSED: duplicate')
stage.wait_for('setup 03: Enter', 3)
save(Path('/rehearsal/small.png'), '', 'REFUSED: unsupported size 320x240')
stage.wait_for('setup 03: Enter', 4)
save(frames[3], '', 'setup 04: Enter')
save(frames[4], '', 'setup 05: Enter')
save(frames[5], '', 'setup 06: Enter')
stage.send('q'); stage.finish('STAGING OK: 5 setups staged')
with (DAY / 'records/staging_log.csv').open() as f: log = list(csv.DictReader(f))
assert len(log) == 7 and [r['status'] for r in log] == ['staged','skipped','staged','retake','staged','staged','staged']
assert len(list((DAY / 'staged').iterdir())) == 6
assert (DAY / 'staged/setup_02.png').read_bytes() == frames[1].read_bytes()
assert (DAY / 'staged/setup_02_r1.png').read_bytes() == frames[2].read_bytes()

# Select the separate first visit to prove an empty seal is useful without any camera context.
shutil.copyfile(ZERO / 'operator.env', HOME / 'ttl/operator.env')
zero = run('seal.sh', 'seal separate visit: zero staged', 'SEALED USB OK')
assert 'WARNING: 0 staged' in zero.output and 'RECONCILED: 0 setups actually staged; 0 skipped' in zero.output


def live_run(label, answer, feed=False):
    c = Command(HOME / 'ttl/live.sh', label)
    c.wait_for('stream URL:'); c.send(str(CAMERA))
    c.wait_for('Enter = n (continuous camera feed); y only for stills CSI saves on request.'); c.send(answer)
    c.wait_for('buffered frames?'); c.send('y')
    c.wait_for('watching ')
    out = Path(context()['OUT'])
    if feed:
        for frame, target in [(frames[0], source), (Path('/csi1080/csi_1080_empty_G.jpg'), CAMERA / 'live1080.jpg'), (frames[1], source)]:
            with (out / 'trial_log.csv').open() as f: before = len(list(csv.DictReader(f)))
            shutil.copyfile(frame, target)
            end = time.monotonic() + 15
            while True:
                with (out / 'trial_log.csv').open() as f: trial = list(csv.DictReader(f))
                if len(trial) > before: break
                assert time.monotonic() < end, trial
                time.sleep(.05)
        assert len(trial) == 3 and {r['path'] for r in trial} == {'640', '1080->640'}, trial
        assert all(r['status'] not in {'unreadable', 'wrong_size', 'error'} for r in trial), trial
        print('LIVE INPUT CHECK: ' + json.dumps([{'status': r['status'], 'path': r['path']} for r in trial]), flush=True)
    os.kill(c.proc.pid, signal.SIGINT)
    c.finish('LIVE OK')
    session = json.loads((out / 'session.json').read_text())
    assert session['stop_reason'] == 'ctrl-c'
    return out


live_run('live allowed after zero seal: Enter gate', '', feed=False)
shutil.copyfile(DAY / 'operator.env', HOME / 'ttl/operator.env')
seal = run('seal.sh', 'seal: final and earlier attempts', 'SEALED USB OK')
assert 'EARLIER ATTEMPT: setup 02: setup_02.png' in seal.output
assert 'FINAL: setup 02: setup_02_r1.png' in seal.output
selected = json.loads((DAY / 'records/seal_selection.json').read_text())
assert len(selected['final']) == 5 and len(selected['earlier_attempts']) == 1
run('seal.sh', 'seal again', 'already sealed', success=False)
run('stage.sh', 'stage after seal', 'already sealed', success=False)

# Every protected location is rejected before an OUT is created.
ctx = context()
for label, folder in [('staged', Path(ctx['STAGED'])), ('disk seal', Path(ctx['SEALED']).parent), ('USB seal', Path(ctx['SEALED_USB']))]:
    c = Command(HOME / 'ttl/live.sh', 'live refuses ' + label)
    c.wait_for('stream URL:'); c.send(str(folder)); c.finish('refused', success=False)
first = live_run('live same camera: Enter gate, 640 + 1080 + rewritten 640, SIGINT', '', feed=True)
second = live_run('live same camera: y gate, 640 + 1080 + rewritten 640, SIGINT', 'y', feed=True)
assert first != second
run('export.sh', 'export every visit', 'RESULTS USB OK: 2 visits')
copy = next(USB.glob('visit_export_*/evidence.tar'))
t0 = time.monotonic()
r = subprocess.run(['sha256sum', '-c', 'evidence.tar.sha256'], cwd=copy.parent, capture_output=True, text=True)
assert r.returncode == 0, r.stderr
print(r.stdout, end='', flush=True)
STEPS.append(dict(step='re-verify exported archive', seconds=round(time.monotonic()-t0,3), final=r.stdout.strip(), exit=r.returncode))
import tarfile
with tarfile.open(copy) as t:
    assert len({n.split('/')[0] for n in t.getnames()}) == 2
    assert t.extractfile(DAY.name+'/staged/setup_02.png').read() == frames[1].read_bytes()
    assert t.extractfile(DAY.name+'/staged/setup_02_r1.png').read() == frames[2].read_bytes()
# Package discovery failures: no operator input and no visit created.
for count in (0, 2):
    folder = Path('/rehearsal') / ('discovery_' + str(count)); folder.mkdir()
    shutil.copyfile(USB / 'start.sh', folder / 'start.sh')
    for i in range(count): (folder / f'ttl_gb10_abc{i}.tar').write_bytes(b'placeholder')
    c = Command(folder / 'start.sh', f'start with {count} tars')
    c.finish('expected exactly one', success=False)
    assert f'Packages found ({count})' in c.output
Path('/rehearsal/steps.json').write_text(json.dumps(STEPS, indent=2))
print('REHEARSAL OK: round 1; all requested cases verified.', flush=True)
