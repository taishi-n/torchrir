"""A green CI job must represent executed tests, not an empty/skip-only run."""

from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_test_report.py"


@pytest.mark.parametrize(
    "cases,strict,ok",
    [
        ("<testcase/>", False, True),
        ('<testcase/><testcase><skipped message="optional"/></testcase>', False, True),
        ("", False, False),
        ("<testcase><skipped/></testcase>", False, False),
        ("<testcase/><testcase><skipped/></testcase>", True, False),
        ("<testcase><failure/></testcase>", False, False),
    ],
)
def test_ci_report_requires_executed_successes(tmp_path, cases, strict, ok):
    report = tmp_path / "report.xml"
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(report), *(["--no-skips"] if strict else [])],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == ok, result.stderr
    assert "CI report" in result.stdout + result.stderr
