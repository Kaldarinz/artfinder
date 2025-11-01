"""
Tests for ArticlePDF figure rectangle extraction.
"""

import json
import pytest
from pathlib import Path
from pytest import approx
from artfinder.article_pdf import ArticlePDF


# Get the test PDFs directory
TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"
RECTANGLES_JSON = TEST_PDFS_DIR / "figure_rects.json"


@pytest.fixture
def expected_rectangles():
    """Load expected figure rectangles from JSON file."""
    with open(RECTANGLES_JSON, "r") as f:
        return json.load(f)


@pytest.fixture
def pdf_files():
    """Return list of all test PDF files."""
    return list(TEST_PDFS_DIR.glob("*.pdf"))


class TestFigureRectangles:
    """Tests for figure rectangle extraction from PDF files."""

    def test_rectangles_json_exists(self):
        """Test that the rectangles JSON file exists."""
        assert RECTANGLES_JSON.exists(), f"Rectangles file not found: {RECTANGLES_JSON}"

    def test_rectangles_json_format(self, expected_rectangles):
        """Test that rectangles JSON has correct format."""
        assert isinstance(expected_rectangles, dict)
        assert len(expected_rectangles) > 0

        # Check structure of first entry
        first_pdf = next(iter(expected_rectangles.keys()))
        assert isinstance(expected_rectangles[first_pdf], dict)

    def test_figure_rectangles_consistency(self, pdf_files):
        """Test that multiple calls to figures return consistent rectangles."""
        if not pdf_files:
            pytest.skip("No PDF files found in test directory")

        with ArticlePDF(pdf_files[0]) as article:
            figures1 = article.figures
            figures2 = article.figures

            assert (
                figures1 == figures2
            ), "Multiple calls to figures should return same results"
            assert figures1 is figures2, "figures should be cached"

    def test_all_pdfs_rectangle_extraction(self, expected_rectangles):
        """Test rectangle extraction for all PDFs in the test directory."""
        results = []

        for pdf_name, expected in expected_rectangles.items():
            pdf_path = TEST_PDFS_DIR / pdf_name

            if not pdf_path.exists():
                results.append((pdf_name, "SKIP", "File not found"))
                continue

            try:
                with ArticlePDF(pdf_path) as article:
                    figures = article.figures

                    # Check figure count
                    if len(figures) != len(expected):
                        results.append(
                            (
                                pdf_name,
                                "FAIL",
                                f"Count mismatch: expected {len(expected)}, got {len(figures)}",
                            )
                        )
                        continue

                    # Check figure numbers
                    expected_nums = set(int(k) for k in expected.keys())
                    actual_nums = set(figures.keys())
                    if actual_nums != expected_nums:
                        results.append(
                            (
                                pdf_name,
                                "FAIL",
                                f"Figure numbers mismatch: expected {expected_nums}, got {actual_nums}",
                            )
                        )
                        continue

                    # Check rectangle coordinates (with tolerance for rounding errors)
                    all_match = True
                    for fig_num_str, expected_rect in expected.items():
                        fig_num = int(fig_num_str)
                        extracted_rect = tuple(figures[fig_num].rect)
                        expected_rect_tuple = tuple(expected_rect)

                        # Use pytest.approx for floating-point comparison
                        if extracted_rect != approx(expected_rect_tuple):
                            all_match = False
                            break

                    if all_match:
                        results.append((pdf_name, "PASS", ""))
                    else:
                        results.append(
                            (
                                pdf_name,
                                "FAIL",
                                f"Rectangle coordinates mismatch. Expected: {expected_rect_tuple}, Got: {extracted_rect}",
                            )
                        )

            except Exception as e:
                results.append((pdf_name, "ERROR", str(e)))

        # Report results
        failed = [r for r in results if r[1] in ["FAIL", "ERROR"]]

        if failed:
            fail_msg = "\n".join(
                [f"{name}: {status} - {msg}" for name, status, msg in failed]
            )
            pytest.fail(
                f"Rectangle extraction failed for {len(failed)}/{len(expected_rectangles)} PDFs:\n{fail_msg}"
            )
