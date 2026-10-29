# Data-independent quality gates

The pull-request workflow runs syntax compilation and the CPU test collection
without downloading staging captures, model weights, USD assets, or protected
evidence:

```bash
python -m pip install --requirement twin_to_line/requirements_ci.txt
python twin_to_line/ci_quality_gate.py
```

Before syntax collection or pytest starts, the runner requires this exact
release/kit contract:

- CPython 3.11.17
- NumPy 2.1.0
- `opencv-python-headless` 4.10.0.84 (`cv2.__version__ == 4.10.0`)
- pytest 8.4.2

A missing or different dependency fails preflight; the runner never continues
to pytest in a convenient but incompatible environment. The GitHub workflow
selects Python 3.11.17 and installs the exact pins in `requirements_ci.txt`.

The gate discovers Python sources through `git ls-files -co
--exclude-standard`, so ignored data and result directories are not read during
the syntax check. It runs `ignite_core` and `twin_to_line` in separate pytest
processes because those legacy trees contain same-named top-level modules and
tests. Pytest failures are not suppressed.

Two checks have declared external prerequisites:

- The archived DEV/model parity test remains strict and is excluded only from
  the data-independent CI command. On the evidence host, run it explicitly:

  ```bash
  python twin_to_line/ci_quality_gate.py --archived-dev-parity
  ```

  Missing archived stills or model artifacts are a hard failure. The command
  does not silently skip them. It uses the same exact runtime contract.

- One legacy floor-stream unit test imports the USD-backed scene generator. It
  is included automatically when `pxr` is importable and is explicitly
  reported as not run otherwise. The existing twin-dependent tests retain
  their own USD/twin prerequisite checks.

The quality workflow does not train a model, score staging data, evaluate the
protected split, submit YAAD work, or build deployment media.

The CPU contract intentionally matches the deployment kit's CPython 3.11.17,
NumPy 2.1.0, and OpenCV 4.10.0.84 versions. The hosted workflow is still a
networked, non-hermetic check: GitHub Actions major-version tags and pip's
transitive test dependencies are not vendored or hash-locked. It must not be
treated as a replacement for the sealed offline kit preflight.
