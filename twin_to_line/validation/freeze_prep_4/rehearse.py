"""Freeze-prep-4 non-root arm64 rehearsal. Only bundled DEV preflight frames are used."""
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
USB = Path('/rehearsal/USB with space')
STEPS = []
assert os.getuid() != 0
print(f'uid={os.getuid()}', flush=True)
subprocess.run(['uname', '-m'], check=True)
print('$ command -v python3', flush=True)
r = subprocess.run(['bash','-c','command -v python3'], capture_output=True, text=True)
print(r.stdout, end='', flush=True)
assert r.returncode == 1 and not r.stdout
print('(no output; exit 1; before start)', flush=True)


class Command:
    def __init__(self, script, label, *args):
        self.label, self.started = label, time.monotonic()
        print('\n$ bash ' + str(script) + (' ' + ' '.join(args) if args else ''), flush=True)
        self.keyboard, terminal = pty.openpty()
        self.proc = subprocess.Popen(['bash', str(script), *args], stdin=terminal, stdout=subprocess.PIPE,
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

    def wait_for(self, value):
        end = time.monotonic() + 90
        while value not in self.output:
            assert self.proc.poll() is None and time.monotonic() < end, self.output
            time.sleep(.03)

    def send(self, value):
        print(value, flush=True)
        os.write(self.keyboard, (value + '\n').encode())

    def finish(self, expected, success=True):
        code = self.proc.wait(timeout=180)
        self.reader.join(timeout=3); os.close(self.keyboard)
        assert (code == 0) == success and expected in self.output, (code, self.output)
        final = self.output.strip().splitlines()[-1]
        assert final.startswith('========== ') and ('OK:' in final or 'STOP:' in final), final
        row = dict(step=self.label, seconds=round(time.monotonic()-self.started,3), final=final, exit=code)
        STEPS.append(row)
        print('TIMED STEP: ' + json.dumps(row), flush=True)
        return self


def run(name, label, expected, *args, success=True):
    script = USB/name if name == 'start.sh' else HOME/'ttl'/name
    return Command(script, label, *args).finish(expected, success)


def context():
    result = {}
    for line in (HOME/'ttl/operator.env').read_text().splitlines():
        key,value = line.split('=',1); result[key] = shlex.split(value)[0]
    return result


def visits():
    return sorted((HOME/'ttl').glob('visit_*'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


run('start.sh', '1 start: new', 'SETUP OK: preflight passed')
ctx=context(); FIRST=Path(ctx['DAY'])
assert len(visits()) == 1 and ctx['PACKAGE_SHA256'] == digest(USB/(ctx['PKG']+'.tar'))
frames=sorted((Path(ctx['INSTALL'])/ctx['PKG']/'preflight/run1_640').glob('*.png'))
CAMERA=Path('/rehearsal/camera'); CAMERA.mkdir()
stage=Command(HOME/'ttl/stage.sh','2 stage two setups, q')
stage.wait_for('Camera folder'); stage.send(str(CAMERA))
for i in range(2):
    stage.wait_for(f'setup {i:02d}: Enter')
    shutil.copyfile(frames[i],CAMERA/'capture.png'); stage.send('')
stage.wait_for('setup 02: Enter'); stage.send('q'); stage.finish('STAGING OK: 2 setups staged')
original={p.name:digest(p) for p in (FIRST/'staged').iterdir()}
bootstrap_before=(FIRST/'records/bootstrap.log').read_bytes()
resumed=run('start.sh','3 start again: resume','(2 staged, sealed no)')
assert 'BOOTSTRAP OK' not in resumed.output and len(visits()) == 1 and context()['DAY']==str(FIRST)
assert (FIRST/'records/bootstrap.log').read_bytes()==bootstrap_before
stage=Command(HOME/'ttl/stage.sh','3 stage resumes at setup 02')
stage.wait_for('setup 02: Enter'); assert 'Camera folder' not in stage.output
stage.send('q'); stage.finish('STAGING OK: 2 setups staged')
run('seal.sh','4 seal first visit','SEALED USB OK')
sealed_context=(HOME/'ttl/operator.env').read_bytes()
resumed=run('start.sh','4 start again: sealed resume','(2 staged, sealed yes)')
assert len(visits())==1 and (HOME/'ttl/operator.env').read_bytes()==sealed_context
assert 'BOOTSTRAP OK' not in resumed.output
live=Command(HOME/'ttl/live.sh','4 live after sealed resume, SIGINT')
live.wait_for('stream URL:'); live.send(str(CAMERA))
live.wait_for('y only for stills CSI saves on request.'); live.send('y')
live.wait_for('buffered frames?'); live.send('y'); live.wait_for('watching ')
out=Path(context()['OUT']); shutil.copyfile(frames[2],CAMERA/'capture.png')
end=time.monotonic()+15
while True:
    with (out/'trial_log.csv').open() as f: trial=list(csv.DictReader(f))
    if trial: break
    assert time.monotonic()<end
    time.sleep(.05)
os.kill(live.proc.pid,signal.SIGINT); live.finish('LIVE OK')
assert json.loads((out/'session.json').read_text())['stop_reason']=='ctrl-c'
run('start.sh','5 explicit new visit, same package','SETUP OK: preflight passed','--new-visit')
assert len(visits())==2 and context()['DAY']!=str(FIRST)
zero=run('seal.sh','5 seal zero staged','SEALED USB OK')
assert 'WARNING: 0 staged' in zero.output
# Preserve every fake-USB version in a history folder; only tar padding changes.
archive=next(USB.glob('ttl_gb10_*.tar')); checksum=archive.with_suffix('.tar.sha256')
history=Path('/rehearsal/archive_history'); history.mkdir()
archive.rename(history/'original.tar'); checksum.rename(history/'original.sha256')
shutil.copyfile(history/'original.tar',archive)
with archive.open('ab') as f: f.write(b'\0'*10240)
checksum.write_text(f'{digest(archive)}  {archive.name}\n')
current_before=(HOME/'ttl/operator.env').read_bytes()
run('start.sh','6 different tar: refuse','-> bash start.sh --new-visit',success=False)
assert len(visits())==2 and (HOME/'ttl/operator.env').read_bytes()==current_before
run('start.sh','6 explicit new visit, different tar','SETUP OK: preflight passed','--new-visit')
assert len(visits())==3
# Bad checksum must preserve the current successful visit. Repair resumes it.
archive.rename(history/'different_valid.tar')
shutil.copyfile(history/'different_valid.tar',archive)
with archive.open('ab') as f: f.write(b'corrupt')
current_before=(HOME/'ttl/operator.env').read_bytes()
run('start.sh','7 corrupt tar: refuse','checksum failed',success=False)
assert len(visits())==3 and (HOME/'ttl/operator.env').read_bytes()==current_before
archive.rename(history/'corrupt.tar'); shutil.copyfile(history/'different_valid.tar',archive)
run('start.sh','7 fixed tar: resume','SETUP OK: resumed')
assert len(visits())==3
assert {p.name:digest(p) for p in (FIRST/'staged').iterdir()}==original
run('export.sh','8 export all three visits','RESULTS USB OK: 3 visits')
copy=next(USB.glob('visit_export_*/evidence.tar'))
t0=time.monotonic()
r=subprocess.run(['sha256sum','-c','evidence.tar.sha256'],cwd=copy.parent,capture_output=True,text=True)
assert r.returncode==0,r.stderr
STEPS.append(dict(step='8 independent export recheck',seconds=round(time.monotonic()-t0,3),final=r.stdout.strip(),exit=0))
print(r.stdout,end='',flush=True)
import tarfile
with tarfile.open(copy) as t:
    assert {m.split('/')[0] for m in t.getnames()}=={v.name for v in visits()}
    assert t.extractfile(FIRST.name+'/staged/setup_00.png').read()==frames[0].read_bytes()
    assert t.extractfile(FIRST.name+'/staged/setup_01.png').read()==frames[1].read_bytes()
Path('/rehearsal/steps.json').write_text(json.dumps(STEPS,indent=2))
print('REHEARSAL OK: all freeze-prep-4 steps passed; three visits retained and exported.',flush=True)
