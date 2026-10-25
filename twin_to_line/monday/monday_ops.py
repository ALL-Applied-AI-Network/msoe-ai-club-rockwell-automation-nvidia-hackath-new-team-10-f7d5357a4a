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


def reconcile(ctx):
    """Check every retained attempt; return only the final capture for each setup."""
    log = rows(ctx)
    final, attempts, capture_rows = {}, {}, []
    for row in log:
        setup, status = row['setup'], row['status']
        require(status in {'staged', 'skipped', 'retake'}, 'invalid staging status')
        if status == 'retake':
            require(final and setup == next(reversed(final)), 'retake must belong to the last setup')
            attempts[setup] += 1
            stem = f'setup_{setup}_r{attempts[setup]}'
        else:
            require(setup == f'{len(final):02d}', 'setup log has missing/duplicate/out-of-order entries')
            attempts[setup] = 0
            stem = 'setup_' + setup
        final[setup] = row
        if status == 'skipped':
            require(not any(row[k] for k in ('source_name', 'copy_name', 'sha256')), 'skipped setup has a capture')
            continue
        path = Path(ctx['STAGED']) / row['copy_name']
        require(path.name == row['copy_name'] and path.stem == stem and path.suffix in EXTENSIONS,
                'invalid setup filename')
        require(not path.is_symlink() and path.is_file() and path.stat().st_size > 0,
                'missing/empty/symlink staged image')
        require(digest(path) == row['sha256'], f'{path.name}: SHA256 differs from staging log')
        capture_rows.append(row)
    names = [r['copy_name'] for r in capture_rows]
    require(len(names) == len(set(names)), 'duplicate copy names in log')
    require(set(names) == {p.name for p in Path(ctx['STAGED']).iterdir()},
            'staged files do not match log; preserve files and reconcile')
    return [row for row in final.values() if row['status'] != 'skipped']


def camera_snapshot(folder):
    return {p.name: [digest(p), p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(folder.iterdir())
            if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in EXTENSIONS}


def save_seen(path, seen):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='camera_seen_', delete=False) as f:
        json.dump(seen, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def stage(ctx):
    require(not ctx.get('SEALED'), 'visit already sealed; staging is closed')
    reconcile(ctx)
    state_path = Path(ctx['DAY']) / 'records/camera_seen.json'
    if not ctx.get('CAMERA'):
        folder = Path(input('Camera folder (ask CSI; before the first save): ').strip()).expanduser().resolve(strict=True)
        require(folder.is_dir(), 'camera path must be a folder')
        staged = Path(ctx['STAGED']).resolve()
        require(folder != staged and staged not in folder.parents and folder not in staged.parents,
                'camera folder must be separate from the staged copies')
        save_seen(state_path, camera_snapshot(folder))
        ctx['CAMERA'] = str(folder)
        save_context(ctx)
        print('Existing camera images ignored. Ask CSI to save setup 00 now.', flush=True)
    folder = Path(ctx['CAMERA'])
    seen = json.loads(state_path.read_text())
    while True:
        log = rows(ctx)
        setup = f'{len({r["setup"] for r in log}):02d}'
        answer = input(f'setup {setup}: Enter after CSI saves it, s = skip, r = retake the last setup, q = done: ').strip().lower()
        if answer == 'q':
            return f'STAGING OK: {len(reconcile(ctx))} setups staged; skips and retakes logged; no inference'
        if answer not in ('', 's', 'r'):
            print('Use Enter, s, r, or q.', flush=True)
            continue
        if answer == 'r' and not log:
            print('No previous setup to retake.', flush=True)
            continue
        current = camera_snapshot(folder)
        row = dict.fromkeys(FIELDS, '')
        row.update(setup=setup, status='skipped', time=now())
        if answer != 's':
            candidates = [folder / n for n, version in current.items() if seen.get(n) != version]
            if not candidates:
                print('No new image yet; ask CSI to finish saving, then try again.', flush=True)
                continue
            source = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
            before = source.stat()
            time.sleep(0.5)
            data = source.read_bytes()
            after = source.stat()
            sha = hashlib.sha256(data).hexdigest()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha != current[source.name][0]:
                print('Image still changing; wait for CSI, then try again.', flush=True)
                continue
            # Decode dimensions only; no inspector, thresholds or scoring modules.
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            reason = None
            if sha in {r['sha256'] for r in log if r['sha256']}:
                reason = 'duplicate image SHA256 already logged'
            elif image is None:
                reason = 'image cannot be decoded'
            elif (image.shape[1], image.shape[0]) not in {(640, 480), (1920, 1080)}:
                reason = f'unsupported size {image.shape[1]}x{image.shape[0]}; need 640x480 or 1920x1080'
            if reason:
                print(f'REFUSED: {reason}; source untouched; setup unchanged.', flush=True)
                seen = current
                save_seen(state_path, seen)
                continue
            suffix = ''
            if answer == 'r':
                setup = log[-1]['setup']
                count = sum(r['setup'] == setup and r['status'] == 'retake' for r in log) + 1
                suffix = f'_r{count}'
            target = Path(ctx['STAGED']) / f'setup_{setup}{suffix}{source.suffix.lower()}'
            with target.open('xb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            require(digest(target) == sha, 'staged copy checksum failed')
            row.update(setup=setup, status='retake' if answer == 'r' else 'staged',
                       source_name=source.name, copy_name=target.name, sha256=sha)
            print(f'{target.name}: {image.shape[1]}x{image.shape[0]}; sha256 {sha}', flush=True)
        with (Path(ctx['DAY']) / 'records/staging_log.csv').open('a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            f.flush()
            os.fsync(f.fileno())
        seen = current
        save_seen(state_path, seen)
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
    ctx['SEALED_USB'] = str(dest)  # Protect partial USB seals too if a later write/check fails.
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
    require(not ctx.get('SEALED'), 'visit already sealed; preserve the existing seal')
    final = reconcile(ctx)
    log = rows(ctx)
    count = len({r['setup'] for r in log})
    print(f'RECONCILED: {len(final)} setups actually staged; {count - len(final)} skipped; '
          f'{sum(r["status"] == "retake" for r in log)} retakes', flush=True)
    if not final:
        print('WARNING: 0 staged setups; no staged evaluation images collected; live is allowed after this seal.', flush=True)
    final_names = {r['copy_name'] for r in final}
    earlier = [r for r in log if r['copy_name'] and r['copy_name'] not in final_names]
    for row in earlier:
        print(f'EARLIER ATTEMPT: setup {row["setup"]}: {row["copy_name"]} (retained, not final)', flush=True)
    for row in final:
        print(f'FINAL: setup {row["setup"]}: {row["copy_name"]}', flush=True)
    # Selection is recorded alongside all retained bytes; no attempt is overwritten or discarded.
    selection = Path(ctx['DAY']) / 'records/seal_selection.json'
    with selection.open('x') as f:
        json.dump({'final': final, 'earlier_attempts': earlier, 'skipped': count - len(final)}, f, indent=2)
    path, sha = archive(Path(ctx['DAY']), ['staged', 'records'], Path(ctx['DAY']), 'seal_')
    with tarfile.open(path) as t:
        for row in [r for r in log if r['copy_name']]:
            require(hashlib.sha256(t.extractfile('staged/' + row['copy_name']).read()).hexdigest() == row['sha256'],
                    'archived capture differs from staging log')
    reconcile(ctx)
    try:
        dest = usb_copy(ctx, path, sha, 'sealed_')
        ctx['SEALED_USB'] = str(dest)
        result = f'SEALED USB OK: {dest}; reset fixture before live'
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        result = f'SEALED DISK OK: USB FAILED ({e}); verified GB10 copy {path}; live may start; retry USB at export'
    ctx.update(SEALED=str(path), SEALED_SHA256=sha, SEAL_STATUS=result)
    save_context(ctx)
    return result


def live_folder(ctx, source):
    folder = Path(source).expanduser().resolve(strict=True)
    require(folder.is_dir(), 'live input must be a folder or RTSP/HTTP(S) URL')
    protected = [Path(ctx['STAGED']).resolve(), Path(ctx['SEALED']).resolve().parent]
    if ctx.get('SEALED_USB'):
        protected.append(Path(ctx['SEALED_USB']).resolve())
    for item in protected:
        require(folder != item and item not in folder.parents and folder not in item.parents,
                'staged/sealed folder (or overlapping folder) refused')
    # The camera originals are allowed. A link to a staged copy still refers to protected evidence.
    staged = list(Path(ctx['STAGED']).iterdir())
    require(not any(p.is_file() and any(p.samefile(s) for s in staged) for p in folder.iterdir()),
            'live folder contains a link to a staged copy')
    return folder


def yes_no(prompt, default_no=False):
    while True:
        value = input(prompt + ' [y/n]: ').strip().lower()
        if default_no and not value:
            return False
        if value in ('y', 'yes', 'n', 'no'):
            return value in ('y', 'yes')
        print('Type y or n.', flush=True)


def live(ctx):
    require(ctx.get('SEALED'), 'seal the staged set before live')
    reconcile(ctx)
    require(digest(ctx['SEALED']) == ctx['SEALED_SHA256'], 'GB10 sealed archive checksum changed')
    source = input('Live camera folder or approved RTSP/HTTP(S) stream URL: ').strip()
    if source.startswith(('rtsp://', 'http://', 'https://')):
        src = 'stream:' + source
    else:
        folder = live_folder(ctx, source)
        src = 'folder:' + str(folder)
        ctx['IN'] = str(folder)
    each = yes_no('Enter = n (continuous camera feed); y only for stills CSI saves on request.', default_no=True)
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
