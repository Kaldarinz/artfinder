"""
Tests for ArticlePDF figure rectangle extraction.
"""

import json
import pytest
from collections.abc import Iterator
from pathlib import Path
from pytest import approx
from pymupdf import Rect

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

                    # Check figure labels
                    expected_labels = set(expected.keys())
                    actual_labels = set(figures.keys())
                    if actual_labels != expected_labels:
                        results.append(
                            (
                                pdf_name,
                                "FAIL",
                                f"Figure labels mismatch: expected {expected_labels}, got {actual_labels}",
                            )
                        )
                        continue

                    # Check rectangle coordinates (with tolerance for rounding errors)
                    all_match = True
                    for fig_label, expected_rect in expected.items():
                        extracted_rect = tuple(figures[fig_label].rect)
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


FRAMED_FIGURES_PDF = (
    "laser ablation-based one-step generation and bio-functionalization of gold"
    " nanoparticles conjugated with aptamers.pdf"
)


class TestFramePieces:
    """A frame around a figure and its caption, drawn as separate shapes."""

    # BMC's rounded box: hairline edges and 4 x 4 pt corner arcs.
    FRAME = [
        Rect(60.7, 536.1, 286.6, 536.4),
        Rect(286.6, 536.1, 290.6, 540.1),
        Rect(290.3, 540.1, 290.6, 726.2),
        Rect(286.6, 726.2, 290.6, 730.2),
        Rect(60.7, 730.0, 286.6, 730.2),
        Rect(56.7, 726.2, 60.7, 730.2),
        Rect(56.7, 540.1, 56.9, 726.2),
        Rect(56.7, 536.1, 60.7, 540.1),
    ]
    CAPTION = Rect(62.9, 697.9, 284.3, 725.8)

    @pytest.fixture
    def article(self) -> Iterator[ArticlePDF]:
        with ArticlePDF(TEST_PDFS_DIR / FRAMED_FIGURES_PDF) as article:
            yield article

    def test_pieces_framing_a_caption(self, article: ArticlePDF) -> None:
        """Every piece of an outline holding a caption is found."""
        assert article._frame_pieces(self.FRAME, [self.CAPTION]) == set(range(8))

    def test_pieces_framing_no_text(self, article: ArticlePDF) -> None:
        """An outline holding no caption or paragraph — the axes of a plot — stays."""
        assert article._frame_pieces(self.FRAME, [Rect(0, 0, 10, 10)]) == set()

    def test_figure_inside_the_frame_stays(self, article: ArticlePDF) -> None:
        """A drawing within the outline, off its border, is no piece of it."""
        inside = Rect(70, 600, 72, 650)
        found = article._frame_pieces([*self.FRAME, inside], [self.CAPTION])
        assert found == set(range(8))

    def test_figures_sit_above_their_captions(self, article: ArticlePDF) -> None:
        """No figure is taken for the bottom of its frame, just under its caption."""
        assert len(article.figures) == 8
        for figure in article.figures.values():
            assert figure.rect.y1 <= figure.caption.rect.y0
            assert figure.rect.height > ArticlePDF.MIN_FIGURE_SIDE
