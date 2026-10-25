"""Execute the export instructions on staging-only/retry fixtures; never use real captures."""
from pathlib import Path
import re
import shlex
import subprocess
import tarfile


HERE = Path(__file__).resolve().parents[1]


def blocks():
    section = (HERE / "LIVE_RUNBOOK.md").read_text().split("## Appendix A.", 1)[1]
    return ["\n".join(line[3:] if line.startswith("   ") else line for line in block.splitlines())
            for block in re.findall(r"```bash\n(.*?)```", section, re.S)]


def test_every_section7_shell_block_parses():
    for block in blocks():
        result = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_staging_only_export_includes_failed_retry_and_never_overwrites(tmp_path):
    ttl = tmp_path / "ttl"; ttl.mkdir()
    usb = tmp_path / "usb with spaces"; usb.mkdir()
    expected = {"visit_20261026T123000/records/labels.csv": "labels",
                "visit_20261026T123000/records/paper_log.txt": "CSI refused live",
                "visit_20261026T123000/staged/setup_00.jpg": "synthetic test bytes",
                "visit_20261026T123001/install_retry/preflight_out/preflight.json": "failed"}
    for name, text in expected.items():
        p = ttl / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    block = blocks()[-1].replace('"$HOME"/ttl', shlex.quote(str(ttl))).replace('$HOME/ttl', str(ttl))
    for _ in range(2):
        result = subprocess.run(["bash", "-c", f"USB={shlex.quote(str(usb))}\n" + block], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "RESULTS USB OK" in result.stdout and ": OK" in result.stdout
    archives = sorted(usb.glob("results_*/*.tar"))
    assert len(archives) == 2
    for archive in archives:
        with tarfile.open(archive) as tar:
            for name, text in expected.items():
                assert tar.extractfile(name).read().decode() == text
        assert archive.with_suffix(".tar.sha256").exists()
