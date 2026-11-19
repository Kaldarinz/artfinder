"""Tests for ArticlePDF figure captions extraction."""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from artfinder.article_pdf import ArticlePDF


TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"
CAPTIONS_JSON = TEST_PDFS_DIR / "all_fig_captions.json"
RASTER_CAPTIONS_PDF = (
    "laser assisted synthesis of carbon nanoparticles with controlled viscosities"
    " for printing applications.pdf"
)
EXPECTED_CAPTIONS: dict[str, dict[str, str]] = json.loads(
    CAPTIONS_JSON.read_text(encoding="utf-8")
)


@pytest.fixture
def expected_captions() -> dict[str, dict[str, str]]:
    """Return the golden captions.

    Returns
    -------
    dict[str, dict[str, str]]
        Caption text by PDF name and figure label.
    """
    return EXPECTED_CAPTIONS


@pytest.fixture
def pdf_files() -> list[Path]:
    """Return the test PDF paths in deterministic order.

    Returns
    -------
    list[Path]
        PDFs in the golden corpus.
    """
    return sorted(TEST_PDFS_DIR.glob("*.pdf"))


class TestFigureCaptions:
    """Tests for figure captions extraction from PDF files."""

    def test_captions_json_exists(self) -> None:
        """Test that the captions JSON file exists."""
        assert CAPTIONS_JSON.exists(), f"Captions file not found: {CAPTIONS_JSON}"

    def test_captions_json_format(
        self, expected_captions: dict[str, dict[str, str]]
    ) -> None:
        """Test that captions JSON has correct format."""
        assert isinstance(expected_captions, dict)
        assert len(expected_captions) > 0
        first_pdf = next(iter(expected_captions))
        assert isinstance(expected_captions[first_pdf], dict)

    def test_figure_captions_keys_are_labels(self, pdf_files: list[Path]) -> None:
        """Test that figure_captions keys are figure labels."""
        if not pdf_files:
            pytest.skip("No PDF files found in test directory")

        with ArticlePDF(pdf_files[0]) as article:
            for key in article.figure_captions:
                assert isinstance(key, str), f"Expected str key, got {type(key)}"
                assert re.fullmatch(
                    r"[A-Z]?\d+(?:\.\d+)*[a-rt-z]?", key
                ), f"Expected a figure label like '1', 'S1', 'A2.1' or '1a', got {key!r}"

    def test_figure_captions_values_are_strings(self, pdf_files: list[Path]) -> None:
        """Test that figure_captions values are strings."""
        if not pdf_files:
            pytest.skip("No PDF files found in test directory")

        with ArticlePDF(pdf_files[0]) as article:
            for value in article.figure_captions.values():
                assert isinstance(value, str), f"Expected str value, got {type(value)}"

    def test_figure_captions_consistency(self, pdf_files: list[Path]) -> None:
        """Test that multiple calls to figure_captions return consistent results."""
        if not pdf_files:
            pytest.skip("No PDF files found in test directory")

        with ArticlePDF(pdf_files[0]) as article:
            captions1 = article.figure_captions
            captions2 = article.figure_captions
            assert captions1 == captions2
            assert captions1 is captions2, "figure_captions should be cached"

    def test_empty_pdf_captions(self) -> None:
        """Placeholder for a PDF without figures."""
        pass

    @pytest.mark.parametrize("pdf_name", sorted(EXPECTED_CAPTIONS))
    def test_all_pdfs_caption_extraction(
        self, pdf_name: str, expected_captions: dict[str, dict[str, str]],
        request: pytest.FixtureRequest,
    ) -> None:
        """Compare every label and full caption, scheduling each PDF separately."""
        pdf_path = TEST_PDFS_DIR / pdf_name
        assert pdf_path.is_file(), f"Golden PDF not found: {pdf_path}"
        expected = expected_captions[pdf_name]
        if pdf_name == RASTER_CAPTIONS_PDF:
            request.getfixturevalue("tesseract")
        with ArticlePDF(pdf_path) as article:
            if pdf_name != RASTER_CAPTIONS_PDF:
                assert article.figure_captions == expected
                return

            def letters_and_digits(text: str) -> str:
                return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum())

            # The reference is a transcription, not Tesseract output. Allow
            # small OCR differences in spacing and low-resolution glyphs.
            assert article.figure_captions.keys() == expected.keys()
            for label, text in expected.items():
                actual = article.figure_captions[label]
                assert SequenceMatcher(
                    None, letters_and_digits(actual), letters_and_digits(text), autojunk=False
                ).ratio() >= 0.95, (label, actual)

            pages = {3: ["1"], 4: ["2"], 5: ["3"], 6: ["4"],
                     7: ["5", "6"], 8: ["7"], 9: ["8"]}
            line_counts = [3, 4, 1, 2, 1, 2, 3, 2]
            for page_no in range(article.file.page_count):
                captions = article.get_figure_caption_rects(page_no)
                labels = pages.get(page_no, [])
                assert len(captions) == len(labels)
                images = article.get_image_rects(page_no) if labels else []
                for label, rect, image in zip(labels, captions, images, strict=True):
                    figure = article.figures[label]
                    assert figure.caption.rect == rect
                    assert figure.caption.lines_no == line_counts[int(label) - 1]
                    assert image.contains(rect) and image.contains(figure.rect)
                    assert 2 < rect.y0 - figure.rect.y1 < 8
