#!/usr/bin/env python3
"""Tech-Lead Code Review Agent.

Automated pre-commit review that checks:
  1. Ownership boundaries — single-owner changes only
  2. Interface compatibility — BaseStrategy / TradeSignal / Signal contracts
  3. Code quality — ruff lint + type hint coverage
  4. Test validation — test file existence + pytest
  5. Cross-boundary alerts — dangerous multi-area edits

Usage:
    .venv/bin/python scripts/review.py              # uncommitted changes
    .venv/bin/python scripts/review.py --base main   # PR diff vs main
    .venv/bin/python scripts/review.py --role strategy-researcher
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── File Ownership Map ──────────────────────────────────────────────

OWNERSHIP: dict[str, str] = {
    "src/data/": "data-engineer",
    "scripts/collect_data.py": "data-engineer",
    "src/indicators/": "strategy-researcher",
    "src/strategy/": "strategy-researcher",
    "src/backtest/": "strategy-researcher",
    "src/ml/": "ml-engineer",
    "scripts/train_model.py": "ml-engineer",
    "src/execution/": "execution-engineer",
    "src/monitoring/": "execution-engineer",
    "scripts/live_trading.py": "execution-engineer",
}

SHARED: set[str] = {
    "config/",
    "tests/",
    "CLAUDE.md",
    "pyproject.toml",
    "docs/",
    "scripts/review.py",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "README.md",
}

# ── Interface Contracts ─────────────────────────────────────────────

REQUIRED_SIGNAL_VALUES = {"LONG", "SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"}

REQUIRED_TRADESIGNAL_FIELDS = {"signal", "symbol", "price", "timestamp"}

REQUIRED_BASE_METHODS = {
    "generate_signal": ["self", "df"],
    "get_required_indicators": ["self"],
}

REQUIRED_EXPORTS = [
    "BaseStrategy",
    "Signal",
    "TradeSignal",
    "MultiTimeframeFilter",
    "PortfolioStrategy",
    "CrossTimeframePortfolio",
    "RSIMeanReversionStrategy",
    "DonchianTrendStrategy",
    "WilliamsRMeanReversionStrategy",
    "CCIMeanReversionStrategy",
    "VWAPMeanReversionStrategy",
    "BBSqueezeBreakoutStrategy",
]


# ── Data Classes ────────────────────────────────────────────────────

class Status:
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str  # PASS / WARN / FAIL
    details: list[str] = field(default_factory=list)


# ── Utilities ───────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent


def get_owner(filepath: str) -> Optional[str]:
    """Return the owning role for a file path, or None if shared/unknown."""
    for prefix in sorted(OWNERSHIP, key=len, reverse=True):
        if filepath == prefix or filepath.startswith(prefix):
            return OWNERSHIP[prefix]
    return None


def is_shared(filepath: str) -> bool:
    """Return True if the file is in a shared area."""
    for prefix in SHARED:
        if filepath == prefix or filepath.startswith(prefix):
            return True
    return False


def get_changed_files(base: Optional[str] = None) -> list[str]:
    """Get list of changed files relative to repo root."""
    if base:
        cmd = ["git", "diff", "--name-only", base]
    else:
        # Uncommitted: staged + unstaged + untracked
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip().splitlines()
        unstaged = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip().splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip().splitlines()
        combined = set(staged + unstaged + untracked)
        return sorted(f for f in combined if f)

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT,
    )
    return sorted(f for f in result.stdout.strip().splitlines() if f)


# ── Check 1: Ownership Boundaries ──────────────────────────────────

def check_ownership(
    files: list[str], role: Optional[str] = None,
) -> CheckResult:
    """Verify all changes belong to a single owner (or shared areas)."""
    owners: dict[str, list[str]] = {}
    for f in files:
        owner = get_owner(f)
        if owner is None:
            continue  # shared or unknown
        owners.setdefault(owner, []).append(f)

    result = CheckResult(name="Ownership Boundaries", status=Status.PASS)

    if role:
        # Check that changes only touch the specified role's files
        violators = {o: fs for o, fs in owners.items() if o != role}
        if violators:
            result.status = Status.FAIL
            for owner, fs in violators.items():
                for f in fs:
                    result.details.append(
                        f"  {f} belongs to [{owner}], not [{role}]"
                    )
    elif len(owners) > 1:
        result.status = Status.FAIL
        for owner, fs in sorted(owners.items()):
            result.details.append(f"  [{owner}]: {', '.join(fs)}")

    return result


# ── Check 2: Interface Compatibility ───────────────────────────────

def _parse_file(path: Path) -> Optional[ast.Module]:
    """Parse a Python file, return AST or None on failure."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, FileNotFoundError):
        return None


def check_interface(files: list[str]) -> CheckResult:
    """Verify BaseStrategy / TradeSignal / Signal / __init__ contracts."""
    result = CheckResult(name="Interface Compatibility", status=Status.PASS)

    strategy_files = [f for f in files if f.startswith("src/strategy/")]
    if not strategy_files:
        return result

    base_path = ROOT / "src" / "strategy" / "base.py"
    init_path = ROOT / "src" / "strategy" / "__init__.py"

    # ── base.py checks ──
    if "src/strategy/base.py" in files:
        tree = _parse_file(base_path)
        if tree is None:
            result.status = Status.FAIL
            result.details.append("  base.py: parse error")
            return result

        # Signal enum values
        signal_values: set[str] = set()
        tradesignal_fields: set[str] = set()
        base_methods: dict[str, list[str]] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == "Signal":
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    signal_values.add(target.id)

                elif node.name == "TradeSignal":
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            # Only required fields (no default value)
                            if item.value is None:
                                tradesignal_fields.add(item.target.id)

                elif node.name == "BaseStrategy":
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args = [a.arg for a in item.args.args]
                            base_methods[item.name] = args

        # Validate Signal enum
        missing_signals = REQUIRED_SIGNAL_VALUES - signal_values
        if missing_signals:
            result.status = Status.FAIL
            result.details.append(
                f"  Signal enum missing: {', '.join(sorted(missing_signals))}"
            )

        # Validate TradeSignal required fields
        missing_fields = REQUIRED_TRADESIGNAL_FIELDS - tradesignal_fields
        if missing_fields:
            result.status = Status.FAIL
            result.details.append(
                f"  TradeSignal missing required fields: "
                f"{', '.join(sorted(missing_fields))}"
            )

        # Validate BaseStrategy methods
        for method, expected_args in REQUIRED_BASE_METHODS.items():
            if method not in base_methods:
                result.status = Status.FAIL
                result.details.append(
                    f"  BaseStrategy.{method}() removed or renamed"
                )
            else:
                actual = base_methods[method]
                if actual != expected_args:
                    result.status = Status.FAIL
                    result.details.append(
                        f"  BaseStrategy.{method}() signature changed: "
                        f"expected ({', '.join(expected_args)}), "
                        f"got ({', '.join(actual)})"
                    )

    # ── __init__.py export check ──
    if "src/strategy/__init__.py" in files:
        tree = _parse_file(init_path)
        if tree is None:
            result.status = Status.FAIL
            result.details.append("  __init__.py: parse error")
            return result

        current_exports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    current_exports.append(elt.value)

        removed = set(REQUIRED_EXPORTS) - set(current_exports)
        if removed:
            result.status = Status.FAIL
            result.details.append(
                f"  __init__.py exports removed: {', '.join(sorted(removed))}"
            )

    return result


# ── Check 3: Code Quality ──────────────────────────────────────────

def _check_type_hints(filepath: str) -> list[str]:
    """Check public functions for missing type hints."""
    path = ROOT / filepath
    if not path.exists() or not filepath.endswith(".py"):
        return []

    tree = _parse_file(path)
    if tree is None:
        return []

    issues: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue  # skip private

        # Check return annotation
        if node.returns is None:
            issues.append(f"  {filepath}:{node.lineno} {node.name}() missing return type")

        # Check argument annotations (skip self/cls)
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is None:
                issues.append(
                    f"  {filepath}:{node.lineno} {node.name}() "
                    f"param '{arg.arg}' missing type hint"
                )

    return issues


def check_quality(files: list[str]) -> CheckResult:
    """Run ruff (if available) and check type hints on changed files."""
    result = CheckResult(name="Code Quality", status=Status.PASS)
    py_files = [f for f in files if f.endswith(".py") and (ROOT / f).exists()]

    if not py_files:
        return result

    # Ruff lint
    ruff = ROOT / ".venv" / "bin" / "ruff"
    if ruff.exists():
        cmd = [str(ruff), "check", "--select", "E,W,F", "--no-fix", "--quiet"]
        cmd.extend(str(ROOT / f) for f in py_files)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if proc.stdout.strip():
            lines = proc.stdout.strip().splitlines()
            if result.status == Status.PASS:
                result.status = Status.WARN
            for line in lines[:10]:  # cap output
                result.details.append(f"  ruff: {line}")
            if len(lines) > 10:
                result.details.append(f"  ... and {len(lines) - 10} more")

    # Type hint check
    hint_issues: list[str] = []
    for f in py_files:
        hint_issues.extend(_check_type_hints(f))

    if hint_issues:
        if result.status == Status.PASS:
            result.status = Status.WARN
        count = len(hint_issues)
        result.details.append(f"  {count} function(s) missing type hints:")
        for issue in hint_issues[:10]:
            result.details.append(issue)
        if count > 10:
            result.details.append(f"  ... and {count - 10} more")

    return result


# ── Check 4: Test Validation ───────────────────────────────────────

def check_tests(files: list[str]) -> CheckResult:
    """Verify test files exist for changed modules and run pytest."""
    result = CheckResult(name="Test Validation", status=Status.PASS)

    src_files = [
        f for f in files
        if f.startswith("src/") and f.endswith(".py")
        and not f.endswith("__init__.py")
    ]

    missing_tests: list[str] = []
    test_files_to_run: list[str] = []

    for src_file in src_files:
        # src/strategy/foo.py → tests/test_strategy_foo.py or tests/strategy/test_foo.py
        parts = Path(src_file).parts  # ('src', 'strategy', 'foo.py')
        module_parts = parts[1:]  # ('strategy', 'foo.py')
        stem = Path(src_file).stem

        # Convention 1: tests/test_{module}_{stem}.py
        flat_name = f"tests/test_{'_'.join(module_parts[:-1])}_{stem}.py"
        # Convention 2: tests/{module}/test_{stem}.py
        nested_name = f"tests/{'_'.join(module_parts[:-1])}/test_{stem}.py"

        flat_exists = (ROOT / flat_name).exists()
        nested_exists = (ROOT / nested_name).exists()

        if flat_exists:
            test_files_to_run.append(str(ROOT / flat_name))
        elif nested_exists:
            test_files_to_run.append(str(ROOT / nested_name))
        else:
            missing_tests.append(src_file)

    if missing_tests:
        result.status = Status.WARN
        for f in missing_tests:
            result.details.append(f"  No test file for {f}")

    # Run pytest on existing test files
    if test_files_to_run:
        pytest_bin = ROOT / ".venv" / "bin" / "pytest"
        if pytest_bin.exists():
            cmd = [str(pytest_bin), "-x", "-q", "--tb=short", "--no-header"]
            cmd.extend(test_files_to_run)
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=ROOT, timeout=120,
            )
            if proc.returncode != 0:
                result.status = Status.FAIL
                output = (proc.stdout + proc.stderr).strip().splitlines()
                for line in output[-10:]:
                    result.details.append(f"  pytest: {line}")

    return result


# ── Check 5: Cross-Boundary Alert ──────────────────────────────────

# Dangerous pairs: modifying both sides of an interface boundary
DANGEROUS_PAIRS: list[tuple[str, str, str]] = [
    ("src/strategy/base.py", "src/execution/", "base.py + execution/* simultaneous"),
    ("src/strategy/base.py", "src/ml/", "base.py + ml/* simultaneous"),
    ("src/backtest/engine.py", "src/execution/", "engine.py + execution/* simultaneous"),
]


def check_cross_boundary(files: list[str]) -> CheckResult:
    """Detect dangerous cross-boundary modifications."""
    result = CheckResult(name="Cross-Boundary Alert", status=Status.PASS)

    owners: dict[str, list[str]] = {}
    for f in files:
        owner = get_owner(f)
        if owner:
            owners.setdefault(owner, []).append(f)

    # Warn if 2+ ownership areas modified
    if len(owners) > 1:
        if result.status == Status.PASS:
            result.status = Status.WARN
        areas = ", ".join(sorted(owners.keys()))
        result.details.append(f"  Multiple ownership areas modified: {areas}")

    # FAIL on dangerous pairs
    for file_a, prefix_b, message in DANGEROUS_PAIRS:
        has_a = file_a in files
        has_b = any(f.startswith(prefix_b) for f in files)
        if has_a and has_b:
            result.status = Status.FAIL
            result.details.append(f"  {message}")

    return result


# ── Main ────────────────────────────────────────────────────────────

def format_status(status: str) -> str:
    """Format status with fixed-width alignment."""
    colors = {Status.PASS: "\033[32m", Status.WARN: "\033[33m", Status.FAIL: "\033[31m"}
    reset = "\033[0m"
    return f"{colors.get(status, '')}{status}{reset}"


def run_review(
    base: Optional[str] = None, role: Optional[str] = None,
) -> int:
    """Run all checks and print results. Returns exit code (1 if any FAIL)."""
    files = get_changed_files(base)

    if not files:
        print("No changed files detected.")
        return 0

    print("=== Tech-Lead Code Review ===")
    print(f"Changed: {len(files)} files")
    if role:
        print(f"Role: {role}")
    print()

    checks = [
        check_ownership(files, role),
        check_interface(files),
        check_quality(files),
        check_tests(files),
        check_cross_boundary(files),
    ]

    for i, check in enumerate(checks, 1):
        status_str = format_status(check.status)
        print(f"{i}. {check.name:<25s} [{status_str}]")
        for detail in check.details:
            print(detail)

    # Summary
    counts = {Status.PASS: 0, Status.WARN: 0, Status.FAIL: 0}
    for check in checks:
        counts[check.status] += 1

    print()
    print(
        f"Summary: "
        f"{format_status(Status.PASS)}={counts[Status.PASS]} | "
        f"{format_status(Status.WARN)}={counts[Status.WARN]} | "
        f"{format_status(Status.FAIL)}={counts[Status.FAIL]}"
    )

    has_fail = counts[Status.FAIL] > 0
    if has_fail:
        print("\nReview FAILED — fix issues before committing.")
    else:
        print("\nReview passed.")

    return 1 if has_fail else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Tech-Lead Code Review Agent")
    parser.add_argument(
        "--base",
        help="Git ref to diff against (e.g. main, HEAD~1). "
        "Default: review uncommitted changes.",
    )
    parser.add_argument(
        "--role",
        choices=[
            "data-engineer",
            "strategy-researcher",
            "ml-engineer",
            "execution-engineer",
        ],
        help="Validate changes belong to this role only.",
    )
    args = parser.parse_args()
    sys.exit(run_review(base=args.base, role=args.role))


if __name__ == "__main__":
    main()
