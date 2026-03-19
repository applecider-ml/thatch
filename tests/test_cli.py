"""Tests for thatch.cli."""

import subprocess
import sys


def test_cli_help():
    """thatch --help should exit cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "thatch.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "THATCH" in result.stdout


def test_cli_no_args():
    """thatch with no args should print help."""
    result = subprocess.run([sys.executable, "-m", "thatch.cli"], capture_output=True, text=True)
    assert result.returncode == 0


def test_cli_catalog_help():
    """thatch catalog --help should work."""
    result = subprocess.run(
        [sys.executable, "-m", "thatch.cli", "catalog", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "datadir" in result.stdout
