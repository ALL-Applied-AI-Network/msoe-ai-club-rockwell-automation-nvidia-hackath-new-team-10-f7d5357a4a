"""Compare all package bytes (including nested code.tar); stop on changed live runtime files."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parents[2]
old, new = map(Path, sys.argv[1:3])

def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}

def code(root):
    with tarfile.open(root / 'code.tar') as t:
        return {p.name: hashlib.sha256(t.extractfile(p).read()).hexdigest() for p in t if p.isfile()}

def changes(a, b):
    return dict(added=sorted(b.keys()-a.keys()), removed=sorted(a.keys()-b.keys()),
                changed=sorted(k for k in a.keys() & b.keys() if a[k] != b[k]))

a, b = code(old), code(new)
runtime = next(ast.literal_eval(node.value) for node in ast.parse((HERE / 'live_inspect.py').read_text()).body
               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RUNTIME_FILES' for t in node.targets))
runtime_changed = [f for f in runtime if a.get('twin_to_line/'+f) != b.get('twin_to_line/'+f)]
report = {'outer_package': changes(files(old), files(new)), 'code.tar': changes(a, b),
          'USB_root_additions': ['start.sh','stage.sh','seal.sh','live.sh','export.sh','monday_ops.py','monday.sha256'],
          'runtime_files_compared': list(runtime), 'runtime_files_differing': runtime_changed}
print(json.dumps(report, indent=2))
if runtime_changed: raise SystemExit('STOP: runtime differs from RC')
