"""Data-independent checks for the repository CI/preflight contract."""

from pathlib import Path
import re
import sys

import pytest


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ci_quality_gate as C


def _deselected(command):
    return [command[i + 1] for i, value in enumerate(command) if value == "--deselect"]


def test_core_gate_excludes_only_declared_nonhermetic_nodes_without_pxr():
    ignite, twin = C.core_pytest_commands("python", pxr_available=False)
    assert ignite[:4] == ["python", "-m", "pytest", "-q"]
    assert ignite[-1] == "ignite_core/tests"
    assert twin[:4] == ["python", "-m", "pytest", "-q"]
    assert "twin_to_line/tests" in twin
    assert _deselected(ignite) == []
    assert _deselected(twin) == [C.ARCHIVED_DEV_PARITY_NODE, C.USD_FLOOR_NODE]
    assert "--ignore" not in ignite + twin
    assert "--continue-on-collection-errors" not in ignite + twin


def test_core_gate_restores_floor_test_when_pxr_is_available():
    ignite, twin = C.core_pytest_commands("python", pxr_available=True)
    assert _deselected(ignite) == []
    assert _deselected(twin) == [C.ARCHIVED_DEV_PARITY_NODE]


def test_archived_parity_gate_is_strict_and_exact():
    command = C.archived_parity_command("python")
    assert command == ["python", "-m", "pytest", "-q", C.ARCHIVED_DEV_PARITY_NODE]
    assert not any(word in command for word in ("--deselect", "--ignore", "--maxfail=0"))


def expected_runtime(**changes):
    values = {
        "implementation": "CPython",
        "python": "3.11.17",
        "numpy": "2.1.0",
        "opencv": "4.10.0",
        "pytest": "8.4.2",
    }
    values.update(changes)
    return C.RuntimeContract(**values)


def test_exact_runtime_contract_is_accepted():
    assert C.validate_runtime_contract(expected_runtime()) == expected_runtime()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"implementation": "PyPy"}, "Python implementation: expected CPython, got PyPy"),
        ({"python": "3.11.16"}, "Python: expected 3.11.17, got 3.11.16"),
        ({"numpy": "2.1.1"}, "NumPy: expected 2.1.0, got 2.1.1"),
        ({"opencv": "4.10.0.84"}, "OpenCV (cv2): expected 4.10.0, got 4.10.0.84"),
        ({"pytest": "9.0.0"}, "pytest: expected 8.4.2, got 9.0.0"),
    ],
)
def test_runtime_contract_rejects_every_mismatch(change, message):
    with pytest.raises(C.GateError, match=re.escape(message)):
        C.validate_runtime_contract(expected_runtime(**change))


def test_main_fails_before_syntax_or_pytest_in_wrong_environment(monkeypatch, capsys):
    monkeypatch.setattr(C, "runtime_contract", lambda: expected_runtime(numpy="2.5.4"))
    monkeypatch.setattr(C, "syntax_check", lambda: pytest.fail("syntax should not run"))
    monkeypatch.setattr(C, "_run_all", lambda commands: pytest.fail("pytest should not run"))
    assert C.main([]) == 2
    assert "NumPy: expected 2.1.0, got 2.5.4" in capsys.readouterr().err


def test_main_reaches_gate_only_after_valid_contract(monkeypatch):
    monkeypatch.setattr(C, "runtime_contract", expected_runtime)
    monkeypatch.setattr(C, "syntax_check", lambda: 3)
    monkeypatch.setattr(C, "_run_all", lambda commands: 0)
    assert C.main([]) == 0


def test_ci_dependencies_are_exactly_pinned():
    lines = [
        line.strip()
        for line in (HERE / "requirements_ci.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", line) for line in lines)
    assert {line.split("==", 1)[0] for line in lines} == {
        "numpy",
        "opencv-python-headless",
        "pytest",
    }
    assert "numpy==2.1.0" in lines
    assert "opencv-python-headless==4.10.0.84" in lines
    assert "pytest==8.4.2" in lines


def test_workflow_runs_gate_without_failure_suppression():
    workflow = (ROOT / ".github/workflows/offline-quality-gates.yml").read_text()
    assert "python twin_to_line/ci_quality_gate.py" in workflow
    assert "twin_to_line/requirements_ci.txt" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert 'python-version: "3.11.17"' in workflow
    forbidden = ("continue-on-error", "|| true", "--ignore", "--maxfail=0")
    assert not any(token in workflow for token in forbidden)

