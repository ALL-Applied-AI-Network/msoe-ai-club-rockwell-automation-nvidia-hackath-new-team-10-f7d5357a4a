#!/usr/bin/env python3
"""Run the repository's data-independent quality gate.

The core gate syntax-checks every tracked/non-ignored Python source, then runs
the complete test collection except for:

* one strict parity test that intentionally requires ignored archived DEV
  inputs and model artifacts; and
* one pure floor-randomization test whose legacy module imports optional USD at
  module load time (it is included automatically when ``pxr`` is installed).

The archived parity test is never silently skipped.  Run it explicitly with
``--archived-dev-parity`` on the evidence host; missing prerequisites remain a
hard test failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_DEV_PARITY_NODE = (
    "twin_to_line/tests/test_score_sealed.py::"
    "test_five_arms_on_three_dev_stills_match_unchanged_pipeline"
)
USD_FLOOR_NODE = (
    "twin_to_line/tests/test_v2.py::FloorStream::"
    "test_floor_draw_is_its_own_stream"
)


@dataclass(frozen=True)
class RuntimeContract:
    implementation: str
    python: str
    numpy: str
    opencv: str
    pytest: str


EXPECTED_RUNTIME = RuntimeContract(
    implementation="CPython",
    python="3.11.17",
    numpy="2.1.0",
    opencv="4.10.0",
    pytest="8.4.2",
)


class GateError(RuntimeError):
    """A quality-gate preflight failed before pytest could run."""


def _imported_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # Import/loader failures are contract failures.
        return f"<unavailable: {type(exc).__name__}>"
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else "<missing __version__>"


def runtime_contract() -> RuntimeContract:
    return RuntimeContract(
        implementation=platform.python_implementation(),
        python=".".join(str(part) for part in sys.version_info[:3]),
        numpy=_imported_version("numpy"),
        opencv=_imported_version("cv2"),
        pytest=_imported_version("pytest"),
    )


def validate_runtime_contract(actual: RuntimeContract | None = None) -> RuntimeContract:
    """Require the release/kit versions before any tests can appear green."""
    actual = runtime_contract() if actual is None else actual
    labels = {
        "implementation": "Python implementation",
        "python": "Python",
        "numpy": "NumPy",
        "opencv": "OpenCV (cv2)",
        "pytest": "pytest",
    }
    mismatches = []
    for field, label in labels.items():
        expected_value = getattr(EXPECTED_RUNTIME, field)
        actual_value = getattr(actual, field)
        if actual_value != expected_value:
            mismatches.append(f"{label}: expected {expected_value}, got {actual_value}")
    if mismatches:
        raise GateError(
            "runtime/dependency contract mismatch; "
            + "; ".join(mismatches)
            + ". Use CPython 3.11.17 with twin_to_line/requirements_ci.txt"
        )
    return actual


def _git_python_sources() -> list[Path]:
    """Return only tracked/non-ignored Python files, never ignored data trees."""
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", "*.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or "git ls-files failed"
        raise GateError(detail)
    paths = [ROOT / line for line in result.stdout.splitlines() if line]
    if not paths:
        raise GateError("no tracked/non-ignored Python sources found")
    return sorted(paths)


def syntax_check() -> int:
    """Compile source text without importing modules or writing bytecode."""
    paths = _git_python_sources()
    for path in paths:
        try:
            source = path.read_bytes()
            compile(source, str(path), "exec")
        except (OSError, SyntaxError) as exc:
            raise GateError(f"syntax check failed for {path.relative_to(ROOT)}: {exc}") from exc
    return len(paths)


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def core_pytest_commands(
    python: str = sys.executable, *, pxr_available: bool | None = None
) -> list[list[str]]:
    """Build isolated component commands; exclusions are exact and declared.

    Both legacy source trees expose top-level modules such as ``positions`` and
    contain same-named test modules. Running both collections in one Python
    process causes import-cache collisions, so the gate intentionally gives
    each component a fresh pytest process.
    """
    if pxr_available is None:
        pxr_available = module_available("pxr")
    common = [python, "-m", "pytest", "-q", "--strict-config", "--strict-markers"]
    ignite_command = [*common, "ignite_core/tests"]
    twin_command = [
        *common,
        "twin_to_line/tests",
        "--deselect",
        ARCHIVED_DEV_PARITY_NODE,
    ]
    if not pxr_available:
        twin_command.extend(["--deselect", USD_FLOOR_NODE])
    return [ignite_command, twin_command]


def archived_parity_command(python: str = sys.executable) -> list[str]:
    """Return the strict, separately invoked archived-evidence gate."""
    return [python, "-m", "pytest", "-q", ARCHIVED_DEV_PARITY_NODE]


def _display(command: Sequence[str]) -> str:
    # All command elements are repository constants or the Python executable;
    # this is display-only.  Execution never uses a shell.
    return " ".join(command)


def _run(command: Sequence[str]) -> int:
    return subprocess.run(list(command), cwd=ROOT, check=False).returncode


def _run_all(commands: Sequence[Sequence[str]]) -> int:
    for command in commands:
        returncode = _run(command)
        if returncode:
            return returncode
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archived-dev-parity",
        action="store_true",
        help="run only the strict archived-DEV/model parity test",
    )
    parser.add_argument(
        "--show-command",
        action="store_true",
        help="print the selected pytest command without running it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        actual_runtime = validate_runtime_contract()
    except GateError as exc:
        print(f"QUALITY GATE FAILED: {exc}", file=sys.stderr)
        return 2
    print(
        "Runtime contract: "
        f"{actual_runtime.implementation} {actual_runtime.python}, "
        f"NumPy {actual_runtime.numpy}, OpenCV {actual_runtime.opencv}, "
        f"pytest {actual_runtime.pytest}"
    )
    if args.archived_dev_parity:
        command = archived_parity_command()
        if args.show_command:
            print(_display(command))
            return 0
        print("STRICT ARCHIVED DEV PARITY GATE")
        print(_display(command))
        return _run(command)

    commands = core_pytest_commands()
    if args.show_command:
        for command in commands:
            print(_display(command))
        return 0
    try:
        checked = syntax_check()
    except GateError as exc:
        print(f"QUALITY GATE FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"Syntax check: {checked} tracked/non-ignored Python files")
    print("NOT RUN in core CI: strict archived DEV/model parity gate")
    print(f"Run separately: {_display(archived_parity_command())}")
    if any(USD_FLOOR_NODE in command for command in commands):
        print("NOT RUN in this environment: one floor-stream test requires optional pxr import")
    for command in commands:
        print(_display(command))
    return _run_all(commands)


if __name__ == "__main__":
    raise SystemExit(main())

