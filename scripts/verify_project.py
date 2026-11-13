"""Run local quality gates and save their actual outputs in reports/verification.json."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ["-m", "pip", "check"],
    ["-m", "ruff", "check", "."],
    ["-m", "ruff", "format", "--check", "."],
    ["-m", "pytest", "-q", "--basetemp=.local/pytest-temp"],
]


def main() -> None:
    (ROOT / ".local").mkdir(exist_ok=True)
    results = []
    for arguments in COMMANDS:
        process = subprocess.run(
            [sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True, check=False
        )
        result = {
            "command": "python " + " ".join(arguments),
            "exit_code": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }
        results.append(result)
        print(result["command"])
        print(process.stdout.strip())
        if process.stderr:
            print(process.stderr.strip(), file=sys.stderr)
        if process.returncode:
            break
    report = {
        "status": "passed" if all(result["exit_code"] == 0 for result in results) else "failed",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "hosted_ci_executed": False,
        "checks": results,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
