"""Build the current commit offline using SHA256-verified RC runtime/wheel downloads."""
import importlib.util
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('build_package', HERE / 'gb10/build_package.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
CACHE = HERE / 'results/rc_1024/package/ttl_gb10_e597c4b'
# Verify cached dependencies against the RC manifest before reusing download bytes.
for line in (CACHE / 'SHA256SUMS').read_text().splitlines():
    sha, name = line.split('  ', 1)
    if name.startswith(('runtime/', 'wheels/')):
        assert B.sha256(CACHE / name) == sha, name


def runtime(dst):
    src = CACHE / 'runtime' / B.PBS['name']
    assert B.sha256(src) == B.PBS['sha256']
    shutil.copy2(src, dst)
    return str(Path(dst) / src.name)


def wheels(dst):
    shutil.copytree(CACHE / 'wheels', dst)


B.python_runtime = runtime
B.wheels = wheels
sys.argv = ['build_package.py', '--out', sys.argv[1] if len(sys.argv) > 1 else str(HERE / 'validation/freeze_prep_3/scratch/usb')]
B.main()
