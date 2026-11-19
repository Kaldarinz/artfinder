"""Optional OCR dependency for tests that exercise raster captions."""

import subprocess
from shutil import which

import pytest


@pytest.fixture(scope="session")
def tesseract() -> None:
    """Skip the OCR-dependent cases if Tesseract or its English data are missing."""
    executable = which("tesseract")
    if executable is None:
        pytest.skip("Tesseract is required for raster caption extraction")
    result = subprocess.run([executable, "--list-langs"], capture_output=True, text=True)
    if result.returncode or "eng" not in result.stdout.splitlines():
        pytest.skip("Tesseract's English language data is required")
