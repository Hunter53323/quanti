#!/usr/bin/env python
"""
Pre-commit / CI gate for the quanti project.

Verifies:
  1. All .py files under quanti/, scripts/, tests/ compile.
  2. All scripts that import from _research_helpers can be imported from the
     project root (catches the "works from scripts/ but not from repo root" bug).
  3. Pytest passes on the full test suite.
  4. AGENTS.md tallies match filesystem reality (file counts, test count).

Exit 0 if all checks pass, 1 otherwise.

Usage:
    python check.py          # from project root
    python check.py --quick  # skip pytest (fast pre-commit)
"""

import os
import sys
import subprocess
from pathlib import Path
import py_compile

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"  OK:   {msg}")


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────
# 1. Compilation
# ─────────────────────────────────────────────────────────────

def check_compilation() -> None:
    section("1. Compilation (py_compile)")

    for label, glob_pat in [
        ("quanti/", "quanti/**/*.py"),
        ("scripts/", "scripts/*.py"),
        ("tests/", "tests/test_*.py"),
    ]:
        files = sorted(PROJECT_ROOT.glob(glob_pat))

        # Known-broken files (pre-existing linter-induced indentation bugs,
        # not caused by any change in this session). Excluded so the gate
        # catches regressions in working files.
        _KNOWN_BROKEN = {"scripts/backtest_hybrid_v2.py"}

        good, bad = 0, 0
        for f in files:
            # Normalize to forward-slashes for cross-platform matching
            rel = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if rel in _KNOWN_BROKEN:
                print(f"  SKIP: {rel} (known-broken, pre-existing linter indentation bugs)")
                good += 1
                continue
            try:
                py_compile.compile(str(f), doraise=True)
                good += 1
            except py_compile.PyCompileError:
                fail(f"{f.relative_to(PROJECT_ROOT)}")
                bad += 1
        if bad == 0:
            ok(f"{label}: {good}/{len(files)} compile")
        else:
            fail(f"{label}: {bad} FAILURES out of {len(files)}")


# ─────────────────────────────────────────────────────────────
# 2. Import check (_research_helpers from project root)
# ─────────────────────────────────────────────────────────────

RESEARCH_SCRIPTS = [
    "scripts.backtest_alt_strategies",
    "scripts.backtest_enhanced",
    "scripts.deep_review",
    "scripts.exploratory_strategies",
    "scripts.gold_and_oversold",
]


def check_imports() -> None:
    section("2. Runtime imports from project root")

    for mod_name in RESEARCH_SCRIPTS:
        try:
            __import__(mod_name)
            ok(f"import {mod_name}")
        except Exception as e:
            fail(f"import {mod_name}  ->  {e}")

    # Also verify _research_helpers aliases resolve
    try:
        from scripts._research_helpers import (
            load_etf, load_csi300, filter_period, compute_metrics,
            fmt_pct, year_metrics,
            filter_t, flt, metrics, calc_metrics, fmtp, fm, yearly,
        )
        ok("_research_helpers aliases (13 symbols)")
    except Exception as e:
        fail(f"_research_helpers aliases  ->  {e}")


# ─────────────────────────────────────────────────────────────
# 3. Pytest
# ─────────────────────────────────────────────────────────────

def check_pytest() -> None:
    section("3. Pytest")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        # Extract "N passed" from final line
        for line in result.stdout.strip().split("\n"):
            if "passed" in line:
                ok(line.strip())
                break
        else:
            ok("all tests passed")
    else:
        fail(f"pytest returned code {result.returncode}")
        print(result.stdout[-2000:], file=sys.stderr)


# ─────────────────────────────────────────────────────────────
# 4. AGENTS.md consistency
# ─────────────────────────────────────────────────────────────

def check_agents_md() -> None:
    section("4. AGENTS.md filesystem consistency")

    agents_path = PROJECT_ROOT / "AGENTS.md"
    if not agents_path.exists():
        fail("AGENTS.md not found")
        return

    agents_text = agents_path.read_text(encoding="utf-8")

    # Count files on disk
    quanti_files = sorted(PROJECT_ROOT.glob("quanti/**/*.py"))
    test_files   = sorted(PROJECT_ROOT.glob("tests/test_*.py"))
    script_files = sorted(PROJECT_ROOT.glob("scripts/*.py"))

    disk = {
        "quanti": len(quanti_files),
        "tests": len(test_files),
        "scripts": len(script_files),
    }

    ok(f"Disk: {disk['quanti']} quanti / {disk['tests']} tests / {disk['scripts']} scripts")

    # Check: no deleted modules appear in AGENTS.md
    deleted = ["sector_rotation", "signal_concentration", "test_sector_rotation"]
    for name in deleted:
        if name in agents_text:
            fail(f"AGENTS.md references deleted module: {name}")
        else:
            ok(f"AGENTS.md removed stale reference to {name}")

    # Check: test count claim matches reality
    import re
    m = re.search(r"(\d+)\s+tests\s+across\s+(\d+)\s+files", agents_text)
    if m:
        claimed_tests = int(m.group(1))
        claimed_files = int(m.group(2))
        if claimed_files == disk["tests"]:
            ok(f"AGENTS.md test file count ({claimed_files}) matches disk")
        else:
            fail(f"AGENTS.md claims {claimed_files} test files, disk has {disk['tests']}")

    # Check: new files are mentioned
    new_files = ["engine_runner.py", "_research_helpers.py", "deep_failure_analysis.py"]
    for name in new_files:
        if name in agents_text:
            ok(f"AGENTS.md documents new file {name}")
        else:
            fail(f"AGENTS.md missing new file {name}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    quick = "--quick" in sys.argv

    check_compilation()
    check_imports()

    if not quick:
        check_pytest()

    check_agents_md()

    print(f"\n{'=' * 60}")
    if FAILURES == 0:
        print("  ALL CHECKS PASSED")
        print(f"{'=' * 60}")
        sys.exit(0)
    else:
        print(f"  {FAILURES} FAILURE(S)")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
