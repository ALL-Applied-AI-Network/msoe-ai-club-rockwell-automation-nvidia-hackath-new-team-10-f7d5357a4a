"""Monday collection/backup orchestration. Never imports an inspector during staging/sealing."""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path.home() / 'ttl'
FIELDS = ['setup', 'status', 'source_name', 'copy_name', 'time', 'sha256']
EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
        return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_context():
    # The shell writes %q initially; shlex reads ordinary quoted paths including spaces.
    values = {}
    for line in (ROOT / 'operator.env').read_text().splitlines():
        key, value = line.split('=', 1)
        words = shlex.split(value)
        values[key] = words[0] if words else ''
    return values


def save_context(ctx):
    content = ''.join(f'{key}={shlex.quote(str(value))}\n' for key, value in ctx.items())
    for path in (Path(ctx['DAY']) / 'operator.env', ROOT / 'operator.env'):
        # Replace only context; keep images, logs, archives and all attempts.
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='context_', delete=False) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(f.name, path)


def rows(ctx):
    with (Path(ctx['DAY']) / 'records/staging_log.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def reconcile(ctx, minimum=1):
    log = rows(ctx)
    setups = [r['setup'] for r in log]
    require(setups == [f'{i:02d}' for i in range(len(log))], 'setup log has missing/duplicate/out-of-order entries')
    require(all(r['status'] in {'staged', 'skipped'} for r in log), 'invalid staging status')
    staged = [r for r in log if r['status'] == 'staged']
    require(len(staged) >= minimum, 'at least one staged setup is required')
    names = [r['copy_name'] for r in staged]
    folder = Path(ctx['STAGED'])
    require(len(names) == len(set(names)), 'duplicate copy names in log')
    require(set(names) == {p.name for p in folder.iterdir()}, 'staged files do not match log; preserve files and reconcile')
    for r in log:
        if r['status'] == 'skipped':
            require(not any(r[k] for k in ('source_name', 'copy_name', 'sha256')), 'skipped setup has a capture')
            continue
        p = folder / r['copy_name']
        require(p.name == r['copy_name'] and p.stem == 'setup_' + r['setup'] and p.suffix in EXTENSIONS,
                'invalid setup filename')
        require(not p.is_symlink() and p.is_file() and p.stat().st_size > 0, 'missing/empty/symlink staged image')
        require(digest(p) == r['sha256'], f'{p.name}: SHA256 differs from staging log')
    return staged


def camera_snapshot(folder):
    return {p.name: digest(p) for p in sorted(folder.iterdir())
            if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in EXTENSIONS}


def stage(ctx):
    require(not ctx.get('SEALED'), 'visit already sealed; staging is closed')
    reconcile(ctx, minimum=0)
    state_path = Path(ctx['DAY']) / 'records/camera_seen.json'
    if not ctx.get('CAMERA'):
        folder = Path(input('Camera folder (ask CSI; before the first save): ').strip()).expanduser().resolve(strict=True)
        require(folder.is_dir(), 'camera path must be a folder')
        staged = Path(ctx['STAGED']).resolve()
        require(folder != staged and staged not in folder.parents and folder not in staged.parents,
                'camera folder must be separate from the staged folder')
        state_path.write_text(json.dumps(camera_snapshot(folder)))
        ctx['CAMERA'] = str(folder)
        save_context(ctx)
        print('Existing camera images ignored. Ask CSI to save setup 00 now.', flush=True)
    folder = Path(ctx['CAMERA'])
    seen = json.loads(state_path.read_text())
    while True:
        setup = f'{len(rows(ctx)):02d}'
        answer = input(f'setup {setup}: Enter after CSI saves it, s = skip, q = done: ').strip().lower()
        if answer == 'q':
            return f'STAGING OK: {len(reconcile(ctx, minimum=0))} setups staged; skips logged; no inference'
        if answer not in ('', 's'):
            print('Use Enter, s, or q.', flush=True)
            continue
        current = camera_snapshot(folder)
        row = dict.fromkeys(FIELDS, '')
        row.update(setup=setup, status='skipped', time=now())
        if answer == '':
            candidates = [folder / n for n, sha in current.items() if seen.get(n) != sha]
            if not candidates:
                print('No new image yet; ask CSI to finish saving, then press Enter again.', flush=True)
                continue
            source = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
            before = source.stat()
            time.sleep(0.5)
            data = source.read_bytes()
            after = source.stat()
            sha = hashlib.sha256(data).hexdigest()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha != current[source.name]:
                print('Image still changing; wait for CSI, then press Enter again.', flush=True)
                continue
            # Decode dimensions only: no runtime/model/threshold modules imported.
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            require(image is not None, 'newest image cannot be decoded; source left untouched')
            target = Path(ctx['STAGED']) / f'setup_{setup}{source.suffix.lower()}'
            with target.open('xb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            require(digest(target) == sha, 'staged copy checksum failed')
            row.update(status='staged', source_name=source.name, copy_name=target.name, sha256=sha)
            h, w = image.shape[:2]
            print(f'{target.name}: {w}x{h}; sha256 {sha}', flush=True)
            if (w, h) not in {(640, 480), (1920, 1080)}:
                print('Unsupported live size; preserve this capture and ask CSI for supported stills.', flush=True)
        with (Path(ctx['DAY']) / 'records/staging_log.csv').open('a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            f.flush()
            os.fsync(f.fileno())
        # Consume all images visible at this setup, including skipped captures.
        seen = current
        state_path.write_text(json.dumps(seen))
        print(f"setup {setup}: {row['status']}", flush=True)


def archive(directory, names, parent, prefix):
    dest = Path(tempfile.mkdtemp(prefix=prefix + datetime.now().strftime('%Y%m%dT%H%M%S') + '_', dir=parent))
    path = dest / 'evidence.tar'
    with tarfile.open(path, 'x') as t:
        for name in names:
            t.add(directory / name, arcname=name)
    sha = digest(path)
    with path.with_suffix('.tar.sha256').open('x') as f:
        f.write(f'{sha}  {path.name}\n')
    require(digest(path) == sha, 'local archive checksum failed')
    print(f'{path}: SHA256 OK', flush=True)
    return path, sha


def usb_copy(ctx, path, sha, prefix):
    usb = Path(ctx['USB'])
    require(usb.is_dir(), 'USB directory is missing; restore the mount')
    dest = Path(tempfile.mkdtemp(prefix=prefix, dir=usb))
    for src in (path, path.with_suffix('.tar.sha256')):
        with src.open('rb') as source, (dest / src.name).open('xb') as out:
            shutil.copyfileobj(source, out)
            out.flush()
            os.fsync(out.fileno())
    subprocess.run(['sync'], check=True)
    subprocess.run(['sha256sum', '-c', path.name + '.sha256'], cwd=dest, check=True)
    require(digest(dest / path.name) == sha, 'USB copy differs from local archive')
    return dest


def seal(ctx):
    staged = reconcile(ctx)
    print(f'RECONCILED: {len(staged)} setups actually staged; {len(rows(ctx)) - len(staged)} skipped', flush=True)
    path, sha = archive(Path(ctx['DAY']), ['staged', 'records'], Path(ctx['DAY']), 'seal_')
    with tarfile.open(path) as t:
        for row in staged:
            require(hashlib.sha256(t.extractfile('staged/' + row['copy_name']).read()).hexdigest() == row['sha256'],
                    'archived capture differs from staging log')
    reconcile(ctx)
    try:
        dest = usb_copy(ctx, path, sha, 'sealed_')
        result = f'SEALED USB OK: {dest}; reset fixture before live'
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        result = f'SEALED DISK OK: USB FAILED ({e}); verified GB10 copy {path}; live may start; retry USB at export'
    ctx.update(SEALED=str(path), SEALED_SHA256=sha, SEAL_STATUS=result)
    save_context(ctx)
    return result


def yes_no(prompt):
    while True:
        value = input(prompt + ' [y/n]: ').strip().lower()
        if value in ('y', 'yes', 'n', 'no'):
            return value in ('y', 'yes')
        print('Type y or n.', flush=True)


def live(ctx):
    require(ctx.get('SEALED'), 'seal the staged set before live')
    reconcile(ctx)
    require(digest(ctx['SEALED']) == ctx['SEALED_SHA256'], 'GB10 sealed archive checksum changed')
    source = input('Fresh live folder or approved RTSP/HTTP(S) stream URL: ').strip()
    if source.startswith(('rtsp://', 'http://', 'https://')):
        src = 'stream:' + source
    else:
        folder = Path(source).expanduser().resolve(strict=True)
        require(folder.is_dir(), 'live input must be a folder or RTSP/HTTP(S) URL')
        for protected in (Path(ctx['STAGED']).resolve(), Path(ctx['CAMERA']).resolve()):
            require(folder != protected and protected not in folder.parents and folder not in protected.parents,
                    'staged/camera collection folder (or ancestor/descendant) refused; use a fresh live folder')
        # Refuse symlinks/hardlinks to any staged capture, too.
        staged = list(Path(ctx['STAGED']).iterdir())
        require(not any(p.is_file() and any(p.samefile(s) for s in staged) for p in folder.iterdir()),
                'live folder contains a link to a staged capture')
        src = 'folder:' + str(folder)
        ctx['IN'] = str(folder)
    each = yes_no('Use --gate each for independent stills?')
    require(yes_no('CSI reset the fixture and cleared staged/buffered frames?'), 'fixture reset not confirmed')
    out = Path(tempfile.mkdtemp(prefix='live_' + datetime.now().strftime('%Y%m%dT%H%M%S') + '_',
                                dir=Path(ctx['DAY']) / 'attempts'))
    ctx.update(OUT=str(out), GATE='each' if each else 'events')
    save_context(ctx)
    shutil.copy2(Path(ctx['DAY']) / 'operator.env', out / 'operator.env')
    command = [ctx['PY'], '-u', 'live_inspect.py', '--source', src, '--out', str(out)]
    if each:
        command += ['--gate', 'each']
    if src.startswith('stream:'):
        command += ['--every', '1.0']
    print(f'LIVE OUT: {out}\nCtrl-C once to stop. Wait for the frame/event summary.', flush=True)
    # A separate child group lets Ctrl-C reach the runtime exactly once while retaining the final OK/STOP line.
    with (out / 'console.log').open('x') as log:
        child = subprocess.Popen(command, cwd=Path(ctx['INSTALL']) / ctx['PKG'] / 'code/twin_to_line',
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        def stop_child(signum, frame):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        old = signal.signal(signal.SIGINT, stop_child)
        old_term = signal.signal(signal.SIGTERM, stop_child)
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        finally:
            signal.signal(signal.SIGINT, old)
            signal.signal(signal.SIGTERM, old_term)
    status = f'{now()} live exit={code} OUT={out}\n'
    with (Path(ctx['DAY']) / 'records/live_status.txt').open('a') as f:
        f.write(status)
    require(code == 0, f'live runtime failed ({code}); preserve {out}')
    session = json.loads((out / 'session.json').read_text())
    require(session.get('stop_reason') == 'ctrl-c', f"unexpected live stop: {session.get('stop_reason')}; inspect {out}")
    return f'LIVE OK: stopped cleanly; evidence {out}; export every visit'


def main():
    action = sys.argv[1]
    try:
        ctx = read_context()
        require(ctx.get('READY') == '1', 'setup/preflight incomplete; repeat start in a fresh visit')
        with (ROOT / 'operator.lock').open('a') as lock:
            # One active writer: prevents duplicate setup numbers, sealing during staging, or export during live.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            message = globals()[action](ctx)
        print(f'\n========== {message} ==========', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as e:
        reason = str(e) or 'interrupted; saved context and evidence retained'
        print(f'\n========== STOP: {action}: {reason} ==========', flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
