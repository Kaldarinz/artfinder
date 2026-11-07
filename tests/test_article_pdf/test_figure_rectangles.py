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
EXPECTED_RECTANGLES: dict[str, dict[str, list[float]]] = json.loads(
    RECTANGLES_JSON.read_text(encoding="utf-8")
)


@pytest.fixture
def expected_rectangles() -> dict[str, dict[str, list[float]]]:
    """Return the golden rectangles.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Rectangle coordinates by PDF name and figure label.
    """
    return EXPECTED_RECTANGLES


@pytest.fixture
def pdf_files() -> list[Path]:
    """Return the test PDF paths in deterministic order.

    Returns
    -------
    list[Path]
        PDFs in the golden corpus.
    """
    return sorted(TEST_PDFS_DIR.glob("*.pdf"))


class TestFigureRectangles:
    """Tests for figure rectangle extraction from PDF files."""

    def test_rectangles_json_exists(self) -> None:
        """Test that the rectangles JSON file exists."""
        assert RECTANGLES_JSON.exists(), f"Rectangles file not found: {RECTANGLES_JSON}"

    def test_rectangles_json_format(
        self, expected_rectangles: dict[str, dict[str, list[float]]]
    ) -> None:
        """Test that rectangles JSON has correct format."""
        assert isinstance(expected_rectangles, dict)
        assert len(expected_rectangles) > 0

        # Check structure of first entry
        first_pdf = next(iter(expected_rectangles.keys()))
        assert isinstance(expected_rectangles[first_pdf], dict)

    def test_figure_rectangles_consistency(self, pdf_files: list[Path]) -> None:
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

    @pytest.mark.parametrize("pdf_name", sorted(EXPECTED_RECTANGLES))
    def test_all_pdfs_rectangle_extraction(
        self, pdf_name: str, expected_rectangles: dict[str, dict[str, list[float]]]
    ) -> None:
        """Compare every label and rectangle, scheduling each PDF separately."""
        pdf_path = TEST_PDFS_DIR / pdf_name
        assert pdf_path.is_file(), f"Golden PDF not found: {pdf_path}"
        expected = expected_rectangles[pdf_name]
        with ArticlePDF(pdf_path) as article:
            figures = article.figures
            assert set(figures) == set(expected)
            for fig_label, expected_rect in expected.items():
                assert tuple(figures[fig_label].rect) == approx(
                    tuple(expected_rect)
                ), f"Rectangle mismatch for figure {fig_label}"


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
