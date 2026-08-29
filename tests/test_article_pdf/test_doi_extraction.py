"""
Tests for DOI extraction from article text and PDFs.
"""

from pathlib import Path

import pymupdf
import pytest

from artfinder.article_pdf import ArticlePDF

TEST_PDFS_DIR = Path(__file__).parent / "article_pdfs"

# Verbatim from the PNAS sample: the DOI is wrapped after its "." separator, so a
# naive first-match search yields the truncated prefix "10.1073/pnas".
WRAPPED_DOI_TEXT = (
    "contains supporting information online at\n"
    "http://www.pnas.org/lookup/suppl/doi:10.1073/pnas.\n"
    "2208830119/-/DCSupplemental.\n"
    "Published September 19, 2022.\n"
)


class TestExtractDoiFromText:
    """Tests for ArticlePDF.extract_doi_from_text."""

    def test_doi_prefix_dot_is_literal(self) -> None:
        """A digit or letter in place of the prefix dot is not a DOI."""
        assert ArticlePDF.extract_doi_from_text("1011073/pnas.2208830119") is None
        assert ArticlePDF.extract_doi_from_text("doi 1000/abc.123") is None

    def test_plain_doi_is_found(self) -> None:
        """A well-formed DOI is returned lowercased."""
        text = "https://doi.org/10.1073/PNAS.2208830119\n"
        assert ArticlePDF.extract_doi_from_text(text) == "10.1073/pnas.2208830119"

    def test_line_wrapped_doi_is_rejoined(self) -> None:
        """A DOI broken across a line is not truncated to its prefix."""
        assert (
            ArticlePDF.extract_doi_from_text(WRAPPED_DOI_TEXT)
            == "10.1073/pnas.2208830119"
        )

    def test_truncated_candidate_loses_to_complete_one(self) -> None:
        """A candidate that is a strict prefix of another is discarded."""
        text = "doi:10.1073/pnas\nand also https://doi.org/10.1073/pnas.2208830119\n"
        assert ArticlePDF.extract_doi_from_text(text) == "10.1073/pnas.2208830119"

    def test_trailing_punctuation_is_stripped(self) -> None:
        """Sentence punctuation adjacent to a DOI is not part of it."""
        text = "(see https://doi.org/10.1007/s00396-014-3397-3).\n"
        assert ArticlePDF.extract_doi_from_text(text) == "10.1007/s00396-014-3397-3"

    def test_supplemental_suffix_is_dropped(self) -> None:
        """The supplement address is not the article's DOI."""
        text = "doi:10.1073/pnas.2208830119/-/DCSupplemental\n"
        assert ArticlePDF.extract_doi_from_text(text) == "10.1073/pnas.2208830119"

    def test_no_doi_returns_none(self) -> None:
        """Text without a DOI yields None."""
        assert ArticlePDF.extract_doi_from_text("no identifier here\n") is None

    def test_prose_line_break_is_not_joined(self) -> None:
        """Rejoining lines must not glue a DOI to the following word."""
        text = "https://doi.org/10.1038/srep25380\n1 of 7\nRESEARCH ARTICLE\n"
        assert ArticlePDF.extract_doi_from_text(text) == "10.1038/srep25380"


class TestExtractDoiCandidates:
    """Tests for ArticlePDF.extract_doi_candidates."""

    def test_multiple_distinct_dois_are_all_returned(self) -> None:
        """Unrelated DOIs are kept, in order of appearance."""
        text = "10.1038/srep25380 and 10.1073/pnas.2208830119\n"
        assert ArticlePDF.extract_doi_candidates(text) == [
            "10.1038/srep25380",
            "10.1073/pnas.2208830119",
        ]

    def test_duplicates_are_collapsed(self) -> None:
        """The same DOI repeated in a page footer appears once."""
        text = "10.1038/srep25380\nfoo\n10.1038/srep25380\n"
        assert ArticlePDF.extract_doi_candidates(text) == ["10.1038/srep25380"]


class TestReferencesHeading:
    """Tests for the bibliography guard in ArticlePDF._get_doi."""

    @pytest.mark.parametrize(
        "heading",
        ["References", "REFERENCES", "Bibliography", "1. References", "Literature Cited"],
    )
    def test_headings_are_recognized(self, heading: str) -> None:
        """A standalone bibliography heading is matched in its usual spellings."""
        text = f"body text\n{heading}\n1. Some Author, J. Foo 1, 1 (2020).\n"
        assert ArticlePDF.REFERENCES_HEADING_PATTERN.search(text) is not None

    def test_inline_mention_is_not_a_heading(self) -> None:
        """The word "references" inside a sentence is not a heading."""
        text = "as shown in the references above, the effect is small\n"
        assert ArticlePDF.REFERENCES_HEADING_PATTERN.search(text) is None


def _make_pdf(pages: list[str]) -> bytes:
    """
    Build an in-memory PDF with one text block per page.

    Parameters
    ----------
    pages : list[str]
        Text to write on each page.

    Returns
    -------
    bytes
        The PDF file contents.
    """

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=9)  # type: ignore[attr-defined]
    data = doc.tobytes()
    doc.close()
    return bytes(data)


class TestBibliographyGuard:
    """The DOIs of cited works must never be taken for the article's own."""

    def test_reference_doi_after_heading_is_not_used(self) -> None:
        """With no DOI in the front matter, the result is None, not a cited DOI."""
        pdf = _make_pdf(
            [
                "A Title\nSome abstract text with no identifier.",
                "References\n1. Author, https://doi.org/10.1016/j.foo.2019.03.001\n",
            ]
        )
        with pytest.warns(UserWarning, match="DOI not found"):
            with ArticlePDF(pdf, identifier="test") as article:
                assert article.doi is None

    def test_doi_before_heading_is_still_found(self) -> None:
        """The guard must not hide a DOI printed above the heading."""
        pdf = _make_pdf(
            [
                "A Title\nhttps://doi.org/10.1038/srep25380\nbody text",
                "References\n1. Author, https://doi.org/10.1016/j.foo.2019.03.001\n",
            ]
        )
        with ArticlePDF(pdf, identifier="test") as article:
            assert article.doi == "10.1038/srep25380"

    def test_doi_on_same_page_before_heading_is_found(self) -> None:
        """A short paper carries its DOI and its references on one page."""
        pdf = _make_pdf(
            [
                "A Title\nhttps://doi.org/10.1038/srep25380\nbody\n"
                "References\n1. Author, https://doi.org/10.1016/j.foo.2019.03.001\n"
            ]
        )
        with ArticlePDF(pdf, identifier="test") as article:
            assert article.doi == "10.1038/srep25380"


class TestArticlePDFDoi:
    """End-to-end DOI extraction from a real PDF."""

    def test_wrapped_doi_in_real_pdf(self) -> None:
        """The PNAS sample resolves to the complete DOI, not the wrapped prefix."""
        path = TEST_PDFS_DIR / (
            "tselikov-et-al-2022-transition-metal-dichalcogenide-nanospheres-"
            "for-high-refractive-index-nanophotonics-and-biomedical.pdf"
        )
        if not path.exists():
            pytest.skip(f"Sample PDF not found: {path}")
        with ArticlePDF(path) as article:
            assert article.doi == "10.1073/pnas.2208830119"
